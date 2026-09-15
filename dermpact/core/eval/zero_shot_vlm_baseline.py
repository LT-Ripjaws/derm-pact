"""Development-only DermLIP zero-shot VLM diagnosis ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score

from dermpact.core import memory
from dermpact.core.eval.memory_diagnosis_route import _collapse_diagnosis, _ece, _softmax


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase6/zero_shot_vlm_baseline_v1"
TRUTH_PATH = ROOT / "research/experiments/dermpact_phase2_4b/paired_photo_v1/canonical_oof_raw.json"
VLM_DIR = ROOT / "research/evidence/derm7pt_v1/features_phase1/dermlip_zeroshot5"
VLM_PATH = VLM_DIR / "features.npz"
VLM_MANIFEST = VLM_DIR / "feature_manifest.json"
CLASS_ORDER: Tuple[str, ...] = ("BCC", "NEV", "MEL", "SK", "MISC")
GLOBAL_TO_FIVE = {
    "basal_cell_carcinoma": "BCC", "nevus": "NEV", "melanoma": "MEL",
    "seborrheic_keratosis_or_benign_keratosis": "SK", "other": "MISC",
    "vascular_lesion": "MISC", "dermatofibroma": "MISC",
}
SEED = 2026090801


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    Path(str(path) + ".sha256").write_text(_sha(path) + "\n", encoding="ascii")


def _metrics(probability: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    prediction = probability.argmax(axis=1)
    one_hot = np.eye(len(CLASS_ORDER))[targets]
    per_class = {label: float(roc_auc_score(one_hot[:, i], probability[:, i])) for i, label in enumerate(CLASS_ORDER)}
    return {
        "accuracy": float(accuracy_score(targets, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, prediction)),
        "macro_f1": float(f1_score(targets, prediction, average="macro")),
        "macro_ovr_auroc": float(np.mean(list(per_class.values()))),
        "melanoma_auroc": per_class["MEL"],
        "per_class_auroc": per_class,
        "nll": float(log_loss(targets, probability, labels=range(len(CLASS_ORDER)))),
        "multiclass_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "ece_10bin": float(_ece(probability, targets)),
        "confusion_matrix": confusion_matrix(targets, prediction, labels=list(range(len(CLASS_ORDER)))).tolist(),
        "support": {label: int(np.sum(targets == i)) for i, label in enumerate(CLASS_ORDER)},
    }


def _bootstrap(targets: np.ndarray, rows: Mapping[str, np.ndarray], n_boot: int) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    values = {name: {"accuracy": [], "balanced_accuracy": [], "macro_ovr_auroc": []} for name in rows}
    n = len(targets)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(targets[idx])) != len(CLASS_ORDER):
            continue
        for name, probability in rows.items():
            metric = _metrics(probability[idx], targets[idx])
            for key in values[name]:
                values[name][key].append(metric[key])
    return {
        name: {
            key: {"ci_low": float(np.quantile(series, 0.025)), "ci_high": float(np.quantile(series, 0.975)),
                  "n_boot_valid": len(series)}
            for key, series in metric_rows.items()
        }
        for name, metric_rows in values.items()
    }


def run(*, out_dir: Path = OUTPUT_DIR, n_boot: int = 2000) -> Dict[str, Any]:
    truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))["truth"]
    if len(truth["ids"]) != 616:
        raise AssertionError("canonical OOF truth must contain exactly 616 development cases")
    targets = np.asarray([CLASS_ORDER.index(GLOBAL_TO_FIVE[label]) for label in truth["diagnosis_targets"]], dtype=np.int64)

    manifest = json.loads(VLM_MANIFEST.read_text(encoding="utf-8"))
    if tuple(manifest["class_order"]) != CLASS_ORDER:
        raise AssertionError("zero-shot class order drift")
    archive = np.load(VLM_PATH, allow_pickle=False)
    case_numbers = np.asarray(archive["case_num"], dtype=np.int64)
    logits = np.asarray(archive["features"], dtype=np.float64)
    by_number = {int(value): index for index, value in enumerate(case_numbers)}
    development_numbers = [int(str(case_id).rsplit("-", 1)[1]) for case_id in truth["ids"]]
    if len(set(development_numbers)) != 616 or any(number not in by_number for number in development_numbers):
        raise AssertionError("development case ids do not map one-to-one onto cached VLM logits")
    vlm_probability = _softmax(np.asarray([logits[by_number[number]] for number in development_numbers]))

    selected_archive = np.load(memory.STAGE2_OOF, allow_pickle=False)
    selected_native = _collapse_diagnosis(_softmax(selected_archive["derm_diagnosis_logits"]))
    selected_probability = selected_native[:, [2, 1, 0, 3, 4]]
    rows = {"selected_phase2_5": selected_probability, "dermlip_zero_shot_vlm": vlm_probability}
    metrics = {name: _metrics(probability, targets) for name, probability in rows.items()}
    intervals = _bootstrap(targets, rows, n_boot)
    result = {
        "schema_version": "dermpact-zero-shot-vlm-baseline-v1",
        "status": "PASS",
        "dataset_role": "Derm7pt development-only",
        "n_cases_scored": 616,
        "class_order": list(CLASS_ORDER),
        "rows": metrics,
        "confidence_intervals": intervals,
        "delta_selected_minus_vlm": {
            key: metrics["selected_phase2_5"][key] - metrics["dermlip_zero_shot_vlm"][key]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "melanoma_auroc")
        },
        "vlm_protocol": {
            "model": "DermLIP cached image encoder plus text-tower zero-shot similarity",
            "prompt_template": manifest["prompt_template"], "prompts": manifest["prompts"],
            "logit_scale": manifest["logit_scale_used"], "supervised_fit": False,
        },
        "sources": {
            "development_truth": {"path": TRUTH_PATH.relative_to(ROOT).as_posix(), "sha256": _sha(TRUTH_PATH)},
            "zero_shot_manifest": {"path": VLM_MANIFEST.relative_to(ROOT).as_posix(), "sha256": _sha(VLM_MANIFEST)},
            "zero_shot_logits": {"path": VLM_PATH.relative_to(ROOT).as_posix(), "sha256": _sha(VLM_PATH)},
            "selected_oof": {"path": memory.STAGE2_OOF.relative_to(ROOT).as_posix(), "sha256": _sha(memory.STAGE2_OOF)},
        },
        "access_attestation": {
            "raw_images_loaded": 0, "official_or_final_labels_loaded": 0,
            "official_or_final_rows_scored": 0,
            "legacy_feature_archive_rows": int(len(case_numbers)),
            "development_feature_rows_indexed_for_scoring": 616,
            "nondevelopment_feature_rows_not_indexed_for_scoring": int(len(case_numbers) - 616),
            "note": "The pre-existing historical feature archive covers 1,011 cases; only the 616 IDs in the canonical development OOF truth artifact are indexed and scored.",
        },
    }
    _atomic_json(out_dir / "results.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    result = run(out_dir=args.out_dir, n_boot=args.n_boot)
    print(json.dumps({"status": result["status"], "n": result["n_cases_scored"],
                      "rows": result["rows"], "delta": result["delta_selected_minus_vlm"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
