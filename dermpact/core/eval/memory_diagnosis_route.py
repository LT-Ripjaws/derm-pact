"""Fold-local, calibrated diagnosis-only memory route for DermPact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from dermpact.core import memory
from dermpact.eval import dermpact_phase2_4b_inputs as inputs_mod
from dermpact.eval import dermpact_phase2_baselines as baselines
from dermpact.eval.dermpact_phase4_benchmark import GLOBAL_CLASSES


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase6/memory_diagnosis_route_v1"
SEED = 2026090707
FIVE_CLASSES: Tuple[str, ...] = (
    "melanoma",
    "nevus",
    "basal_cell_carcinoma",
    "seborrheic_keratosis",
    "miscellaneous",
)
GLOBAL_TO_FIVE = {
    "melanoma": "melanoma",
    "nevus": "nevus",
    "basal_cell_carcinoma": "basal_cell_carcinoma",
    "seborrheic_keratosis_or_benign_keratosis": "seborrheic_keratosis",
    "squamous_cell_carcinoma": "miscellaneous",
    "actinic_keratosis_or_intraepithelial_carcinoma": "miscellaneous",
    "dermatofibroma": "miscellaneous",
    "vascular_lesion": "miscellaneous",
    "other": "miscellaneous",
}
RAW_STATE_KEYS: Tuple[str, ...] = (
    "pigment_network", "streaks", "pigmentation", "regression_structures",
    "dots_and_globules", "blue_whitish_veil", "vascular_structures",
)


def _sha256(path: Path) -> str:
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
    Path(str(path) + ".sha256").write_text(_sha256(path) + "\n", encoding="ascii")


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _collapse_diagnosis(probabilities: np.ndarray) -> np.ndarray:
    out = np.zeros((len(probabilities), len(FIVE_CLASSES)), dtype=np.float64)
    for column, source_label in enumerate(GLOBAL_CLASSES):
        out[:, FIVE_CLASSES.index(GLOBAL_TO_FIVE[source_label])] += probabilities[:, column]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return np.asarray(probabilities, dtype=np.float64).copy()
    logp = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    return _softmax(logp)


def _fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    grid = (0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8)
    losses = [(log_loss(targets, _temperature_scale(probabilities, value), labels=range(len(FIVE_CLASSES))), value)
              for value in grid]
    return float(min(losses)[1])


def _ece(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float((prediction[mask] == targets[mask]).mean()) - float(confidence[mask].mean()))
    return total


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(len(FIVE_CLASSES), dtype=np.float64)[targets]
    per_class_auc = {
        label: float(roc_auc_score(one_hot[:, i], probabilities[:, i]))
        for i, label in enumerate(FIVE_CLASSES)
    }
    return {
        "accuracy": float(accuracy_score(targets, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, prediction)),
        "macro_f1": float(f1_score(targets, prediction, average="macro")),
        "macro_ovr_auroc": float(np.mean(list(per_class_auc.values()))),
        "per_class_auroc": per_class_auc,
        "log_loss": float(log_loss(targets, probabilities, labels=range(len(FIVE_CLASSES)))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10bin": float(_ece(probabilities, targets)),
    }


def _standardize(train: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (query - mean) / scale


def _distance_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((query[:, None, :] - reference[None, :, :]) ** 2, axis=2))


def _memory_distribution(
    distances: np.ndarray,
    reference_targets: np.ndarray,
    *,
    k: int,
    prior_strength: float,
    class_prior: np.ndarray,
    distance_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    k = min(int(k), distances.shape[1])
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nearest_dist = np.take_along_axis(distances, nearest, axis=1)
    order = np.argsort(nearest_dist, axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest_dist = np.take_along_axis(nearest_dist, order, axis=1)
    weights = np.exp(-nearest_dist / max(float(distance_scale), 1e-8))
    votes = np.eye(len(FIVE_CLASSES), dtype=np.float64)[reference_targets[nearest]]
    weighted = (votes * weights[:, :, None]).sum(axis=1)
    distribution = (weighted + prior_strength * class_prior[None, :])
    distribution /= np.maximum(distribution.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(distribution * np.log(np.clip(distribution, 1e-12, 1.0)), axis=1) / np.log(len(FIVE_CLASSES))
    density = np.exp(-nearest_dist[:, 0] / max(float(distance_scale), 1e-8))
    quality = np.clip(density * (1.0 - entropy), 0.0, 1.0)
    return distribution, quality


def _fuse(base: np.ndarray, memory_probability: np.ndarray, quality: np.ndarray, *, weight: float, min_quality: float) -> Tuple[np.ndarray, np.ndarray]:
    if float(weight) == 0.0:
        return np.asarray(base, dtype=np.float64).copy(), np.zeros(len(base), dtype=np.float64)
    alpha = float(weight) * quality * (quality >= float(min_quality))
    fused = (1.0 - alpha[:, None]) * base + alpha[:, None] * memory_probability
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12), alpha


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    k: int
    prior_strength: float
    weight: float
    min_quality: float


def _feature_views(base: np.ndarray, raw: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    criterion_binary = _sigmoid(raw["derm_criteria_logits"])
    raw_states = np.concatenate([_softmax(raw[f"derm_raw_logits__{name}"]) for name in RAW_STATE_KEYS], axis=1)
    melanoma = _sigmoid(raw["derm_melanoma_logits"]).reshape(-1, 1)
    morphology = np.concatenate([criterion_binary, raw_states, melanoma], axis=1)
    return {
        "diagnosis_profile": base,
        "criterion_morphology": morphology,
        "joint_profile": np.concatenate([base, morphology], axis=1),
    }


def _load_development() -> Dict[str, Any]:
    data = inputs_mod.resolve_development_inputs(
        inventory_path=memory.FRIEND_CUDA_MANIFESTS / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        fold_manifest_path=memory.FRIEND_CUDA_MANIFESTS / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        clinical_photo_audit_path=memory.FRIEND_CUDA_MANIFESTS / "dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        structured_fact_support_path=memory.FRIEND_CUDA_MANIFESTS / "dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
        require_files=False,
    )
    if len(data.cases) != 616:
        raise AssertionError("expected exactly 616 locked Derm7pt development cases")
    raw_records = {str(row["group_id"]): row for row in baselines._derm7pt_records()}
    labels = [memory.harmonise_diagnosis(raw_records[case.group_id]["diagnosis"]["harmonized5"]) for case in data.cases]
    if any(label not in FIVE_CLASSES for label in labels):
        raise AssertionError("unmapped Derm7pt diagnosis label in development frame")
    targets = np.asarray([FIVE_CLASSES.index(label) for label in labels], dtype=np.int64)
    folds = np.asarray([case.fold for case in data.cases], dtype=np.int64)
    case_ids = np.asarray([case.group_id for case in data.cases], dtype=str)
    archive = np.load(memory.STAGE2_OOF, allow_pickle=False)
    arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files if key.startswith("derm_")}
    base = _collapse_diagnosis(_softmax(arrays["derm_diagnosis_logits"]))
    return {
        "case_ids": case_ids, "folds": folds, "targets": targets, "base": base,
        "features": _feature_views(base, arrays),
        "source_sha256": _sha256(memory.STAGE2_OOF),
    }


def _select_candidate(base: np.ndarray, targets: np.ndarray, feature_views: Mapping[str, np.ndarray]) -> Tuple[Candidate, float, Dict[str, Any]]:
    base_temperature = _fit_temperature(base, targets)
    base_calibrated = _temperature_scale(base, base_temperature)
    base_metrics = _metrics(base_calibrated, targets)
    best: Tuple[float, Candidate, float, Dict[str, Any]] | None = None
    for feature_name, features in feature_views.items():
        standardized, _ = _standardize(features, features)
        distances = _distance_matrix(standardized, standardized)
        np.fill_diagonal(distances, np.inf)
        finite_nearest = np.min(distances, axis=1)
        distance_scale = float(np.median(finite_nearest))
        prior = np.bincount(targets, minlength=len(FIVE_CLASSES)).astype(np.float64)
        prior /= prior.sum()
        for k in (3, 5, 9, 15):
            for prior_strength in (1.0, 3.0, 8.0):
                memory_probability, quality = _memory_distribution(
                    distances, targets, k=k, prior_strength=prior_strength,
                    class_prior=prior, distance_scale=distance_scale,
                )
                for weight in (0.025, 0.05, 0.1, 0.15, 0.2):
                    for min_quality in (0.0, 0.1, 0.2, 0.3):
                        fused, alpha = _fuse(base, memory_probability, quality, weight=weight, min_quality=min_quality)
                        temperature = _fit_temperature(fused, targets)
                        calibrated = _temperature_scale(fused, temperature)
                        metrics = _metrics(calibrated, targets)
                        auc_gain = metrics["macro_ovr_auroc"] - base_metrics["macro_ovr_auroc"]
                        balanced_delta = metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
                        if auc_gain < 0.001 or balanced_delta < -0.005:
                            continue
                        score = auc_gain + 0.05 * balanced_delta - 0.01 * max(0.0, metrics["log_loss"] - base_metrics["log_loss"])
                        candidate = Candidate(feature_name, k, prior_strength, weight, min_quality)
                        detail = {"train_metrics": metrics, "mean_alpha": float(alpha.mean()), "distance_scale": distance_scale}
                        if best is None or score > best[0]:
                            best = (score, candidate, temperature, detail)
    if best is None:
        return Candidate("diagnosis_profile", 5, 3.0, 0.0, 1.0), base_temperature, {
            "train_metrics": base_metrics, "mean_alpha": 0.0, "distance_scale": None,
        }
    return best[1], best[2], best[3]


def _bootstrap_auc_delta(targets: np.ndarray, base: np.ndarray, fused: np.ndarray, n_boot: int = 2000) -> Dict[str, float]:
    rng = np.random.default_rng(SEED)
    deltas = []
    n = len(targets)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(targets[idx])) != len(FIVE_CLASSES):
            continue
        one_hot = np.eye(len(FIVE_CLASSES))[targets[idx]]
        base_auc = roc_auc_score(one_hot, base[idx], average="macro", multi_class="ovr")
        fused_auc = roc_auc_score(one_hot, fused[idx], average="macro", multi_class="ovr")
        deltas.append(float(fused_auc - base_auc))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "estimate": float(_metrics(fused, targets)["macro_ovr_auroc"] - _metrics(base, targets)["macro_ovr_auroc"]),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_boot_valid": int(len(values)),
    }


def run(*, out_dir: Path = OUTPUT_DIR, n_boot: int = 2000) -> Dict[str, Any]:
    data = _load_development()
    targets, folds, base = data["targets"], data["folds"], data["base"]
    base_oof = np.empty_like(base)
    fused_oof = np.empty_like(base)
    alpha_oof = np.zeros(len(base), dtype=np.float64)
    fold_rows = []

    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        train_views = {name: values[train] for name, values in data["features"].items()}
        candidate, fused_temperature, selection = _select_candidate(base[train], targets[train], train_views)
        base_temperature = _fit_temperature(base[train], targets[train])
        base_oof[heldout] = _temperature_scale(base[heldout], base_temperature)

        if candidate.weight == 0.0:
            fused_oof[heldout] = base_oof[heldout]
            fold_alpha = np.zeros(len(heldout), dtype=np.float64)
        else:
            train_features = data["features"][candidate.feature_set][train]
            held_features = data["features"][candidate.feature_set][heldout]
            train_std, held_std = _standardize(train_features, held_features)
            train_distances = _distance_matrix(train_std, train_std)
            np.fill_diagonal(train_distances, np.inf)
            distance_scale = float(np.median(np.min(train_distances, axis=1)))
            held_distances = _distance_matrix(held_std, train_std)
            prior = np.bincount(targets[train], minlength=len(FIVE_CLASSES)).astype(np.float64)
            prior /= prior.sum()
            memory_probability, quality = _memory_distribution(
                held_distances, targets[train], k=candidate.k,
                prior_strength=candidate.prior_strength, class_prior=prior,
                distance_scale=distance_scale,
            )
            held_fused, fold_alpha = _fuse(
                base[heldout], memory_probability, quality,
                weight=candidate.weight, min_quality=candidate.min_quality,
            )
            fused_oof[heldout] = _temperature_scale(held_fused, fused_temperature)
        alpha_oof[heldout] = fold_alpha
        fold_rows.append({
            "fold": fold, "n_train": int(len(train)), "n_heldout": int(len(heldout)),
            "candidate": candidate.__dict__, "base_temperature": base_temperature,
            "fused_temperature": fused_temperature, "selection": selection,
            "heldout_nonzero_alpha_rate": float(np.mean(fold_alpha > 0)),
            "heldout_mean_alpha": float(np.mean(fold_alpha)),
        })

    base_metrics = _metrics(base_oof, targets)
    fused_metrics = _metrics(fused_oof, targets)
    delta = _bootstrap_auc_delta(targets, base_oof, fused_oof, n_boot=n_boot)
    gates = {
        "macro_ovr_auroc_gain_at_least_0p002": delta["estimate"] >= 0.002,
        "macro_ovr_auroc_delta_ci_positive": delta["ci_low"] > 0.0,
        "balanced_accuracy_noninferior_0p005": fused_metrics["balanced_accuracy"] >= base_metrics["balanced_accuracy"] - 0.005,
        "accuracy_noninferior_0p005": fused_metrics["accuracy"] >= base_metrics["accuracy"] - 0.005,
        "log_loss_not_worse": fused_metrics["log_loss"] <= base_metrics["log_loss"],
        "ece_not_worse_0p005": fused_metrics["ece_10bin"] <= base_metrics["ece_10bin"] + 0.005,
        "memory_activated_on_at_least_10pct": float(np.mean(alpha_oof > 0)) >= 0.10,
    }
    promoted = all(gates.values())
    result = {
        "schema_version": "dermpact-phase6-memory-diagnosis-route-v1",
        "status": "PASS",
        "decision": "PROMOTE_BOUNDED_MEMORY_DIAGNOSIS_ROUTE" if promoted else "RETAIN_COMPARISON_ONLY",
        "promoted": promoted,
        "base_calibrated": base_metrics,
        "memory_fused_calibrated": fused_metrics,
        "macro_ovr_auroc_delta": delta,
        "activation": {
            "nonzero_alpha_rate": float(np.mean(alpha_oof > 0)),
            "mean_alpha": float(np.mean(alpha_oof)),
            "max_alpha": float(np.max(alpha_oof)),
        },
        "folds": fold_rows,
        "gates": gates,
        "source": {
            "stage2_oof_path": str(memory.STAGE2_OOF.relative_to(ROOT)).replace("\\", "/"),
            "stage2_oof_sha256": data["source_sha256"],
            "n_development_cases": 616,
            "official_or_final_rows_loaded": 0,
        },
        "interpretation": (
            "This is a diagnosis-only development route. Retrieved-label counts remain descriptive in the agent. "
            "Runtime diagnosis fusion is disabled unless every promotion gate is true."
        ),
    }
    _atomic_json(out_dir / "results.json", result)
    np.savez_compressed(
        out_dir / "oof_predictions.npz", case_ids=data["case_ids"], folds=folds,
        targets=targets, base_calibrated=base_oof, memory_fused_calibrated=fused_oof,
        memory_alpha=alpha_oof,
    )
    Path(str(out_dir / "oof_predictions.npz") + ".sha256").write_text(
        _sha256(out_dir / "oof_predictions.npz") + "\n", encoding="ascii"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    result = run(out_dir=args.out_dir, n_boot=args.n_boot)
    print(json.dumps({
        "decision": result["decision"], "base": result["base_calibrated"],
        "memory_fused": result["memory_fused_calibrated"],
        "delta": result["macro_ovr_auroc_delta"], "gates": result["gates"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
