"""DermPact Phase 2.1 frozen-backbone development comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .derm_encoders import CANDIDATES, load_candidate_encoder


ROOT = Path(__file__).resolve().parents[2]
MODULE = "dermpact.eval.dermpact_phase2_baselines"
HAM_MANIFEST = ROOT / "research/manifests/ham10000_train_calibration_v2.json"
HAM_ROLE_LEDGER = ROOT / "research/manifests/dermpact_phase1_ham_roles_v1.json"
DERM7PT_INVENTORY = (
    ROOT
    / "research/manifests/dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json"
)
DERM7PT_IMAGES = ROOT / "release_v0/images"
LEGACY_FEATURE_ROOT = (
    ROOT / "research/experiments/derm_perception/v2_tournament/features"
)
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_1"
FEATURE_DIR = EXPERIMENT_DIR / "features"
RESULT_DIR = EXPERIMENT_DIR / "results"
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_1"
LOCK_FILE = ROOT / "requirements-p2.lock"
PYVENV_CONFIG = ROOT / "venv_p2/pyvenv.cfg"

SEED = 20260826
N_FOLDS = 5
N_BOOT = 2000
PROBE_C = 1.0
PROTECTION_MARGIN = 0.02
CANDIDATE_KEYS = ("dermlip", "panderm_base", "panderm")
CANDIDATE_DISPLAY = {
    "dermlip": "DermLIP ViT-B/16",
    "panderm_base": "PanDerm-Base ViT-B/16",
    "panderm": "DermLIP–PanDerm Base with PubMedBERT-256",
}
CRITERIA = (
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
)
CRITERION_SHORT = {
    "pigment_network": "PN",
    "streaks": "STR",
    "pigmentation": "PIG",
    "regression_structures": "RS",
    "dots_and_globules": "DaG",
    "blue_whitish_veil": "BWV",
    "vascular_structures": "VS",
}
RAW_ALLOWED = {
    "pigment_network": {"absent", "typical", "atypical"},
    "streaks": {"absent", "regular", "irregular"},
    "pigmentation": {
        "absent", "diffuse regular", "localized regular",
        "diffuse irregular", "localized irregular",
    },
    "regression_structures": {
        "absent", "blue areas", "white areas", "combinations",
    },
    "dots_and_globules": {"absent", "regular", "irregular"},
    "blue_whitish_veil": {"absent", "present"},
    "vascular_structures": {
        "absent", "arborizing", "comma", "dotted", "hairpin",
        "linear irregular", "within regression", "wreath",
    },
}
POSITIVE_VALUES = {
    "pigment_network": {"atypical"},
    "streaks": {"irregular"},
    "pigmentation": {"diffuse irregular", "localized irregular"},
    "regression_structures": {"blue areas", "white areas", "combinations"},
    "dots_and_globules": {"irregular"},
    "blue_whitish_veil": {"present"},
    "vascular_structures": {"dotted", "linear irregular"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    os.replace(temporary, path)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ham_records() -> list[dict[str, Any]]:
    manifest = _load_json(HAM_MANIFEST)
    role_ledger = _load_json(HAM_ROLE_LEDGER)
    suppress = {
        case_id
        for group in role_ledger["exact_duplicate_training_policy"]["groups"].values()
        for case_id in group["suppressed_case_ids"]
    }
    rows = [
        row for row in manifest["records"]
        if row["split"] == "train" and row["case_id"] not in suppress
    ]
    if len(rows) != 7008:
        raise AssertionError(f"expected 7,008 usable HAM training images, got {len(rows)}")
    if len({row["cluster_id"] for row in rows}) != 5228:
        raise AssertionError("HAM training lesion count changed")
    if suppress != {"ISIC_0029938", "ISIC_0029861"}:
        raise AssertionError(f"HAM exact-duplicate suppression changed: {sorted(suppress)}")
    if any(row["split"] != "train" for row in rows):
        raise AssertionError("non-training HAM row entered Phase 2.1")
    return rows


def _derm7pt_records() -> list[dict[str, Any]]:
    inventory = _load_json(DERM7PT_INVENTORY)
    rows = [row for row in inventory["records"] if row["phase1_role"] == "development"]
    if len(rows) != 616 or len({row["group_id"] for row in rows}) != 616:
        raise AssertionError("Derm7pt development case count changed")
    if Counter(int(row["development_fold"]) for row in rows) != {
        0: 127, 1: 123, 2: 122, 3: 122, 4: 122
    }:
        raise AssertionError("Derm7pt locked development folds changed")
    if any(row["source_split"] not in {"train", "valid"} for row in rows):
        raise AssertionError("Derm7pt official test row entered Phase 2.1")
    for row in rows:
        for criterion in CRITERIA:
            raw = row["criteria_raw"][criterion]
            if raw not in RAW_ALLOWED[criterion]:
                raise ValueError(f"unmapped Derm7pt value {criterion}={raw!r}")
    return rows


def _ham_fold_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    labels_by_lesion: dict[str, str] = {}
    for row in rows:
        lesion = str(row["cluster_id"])
        label = str(row["label"])
        previous = labels_by_lesion.setdefault(lesion, label)
        if previous != label:
            raise AssertionError(f"HAM lesion {lesion} has conflicting labels")
    lesions = sorted(labels_by_lesion)
    labels = np.asarray([labels_by_lesion[lesion] for lesion in lesions])
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    mapping: dict[str, int] = {}
    for fold, (_, valid) in enumerate(splitter.split(lesions, labels)):
        for index in valid:
            mapping[lesions[int(index)]] = fold
    if len(mapping) != len(lesions):
        raise AssertionError("HAM fold assignment is incomplete")
    return mapping


def build_protocol() -> dict[str, Any]:
    ham = _ham_records()
    derm = _derm7pt_records()
    ham_folds = _ham_fold_map(ham)
    ham_fold_rows = [
        {
            "cluster_id": lesion,
            "label": next(row["label"] for row in ham if row["cluster_id"] == lesion),
            "fold": fold,
        }
        for lesion, fold in sorted(ham_folds.items())
    ]
    fold_payload = {
        "protocol": "dermpact-phase2-1-development-folds-v1",
        "seed": SEED,
        "ham": {
            "source_manifest_sha256": sha256_file(HAM_MANIFEST),
            "source_role_ledger_sha256": sha256_file(HAM_ROLE_LEDGER),
            "allowed_role": "train only",
            "n_images_after_exact_duplicate_suppression": len(ham),
            "n_lesions": len(ham_fold_rows),
            "fold_assignment": "lesion-level StratifiedKFold",
            "folds": ham_fold_rows,
        },
        "derm7pt": {
            "source_inventory_sha256": sha256_file(DERM7PT_INVENTORY),
            "allowed_roles": ["official train", "official valid"],
            "n_cases": len(derm),
            "fold_assignment": "Phase 1 locked development_fold; not recomputed",
            "folds": [
                {
                    "group_id": row["group_id"],
                    "harmonized5": row["diagnosis"]["harmonized5"],
                    "fold": int(row["development_fold"]),
                }
                for row in derm
            ],
        },
        "excluded": {
            "ham_calibration_images": 1014,
            "derm7pt_official_test_cases": 395,
            "bcn20000": "not loaded",
            "ph2": "not loaded",
        },
    }
    fold_path = OUTPUT_DIR / "dermpact_phase2_1_development_folds_v1.json"
    fold_hash = _write_json(fold_path, fold_payload)
    protocol = {
        "protocol": "dermpact-phase2-1-frozen-baselines-v1",
        "status": "PRESPECIFIED_BEFORE_RESULTS",
        "candidates_in_fixed_order": [
            {"key": key, "display_name": CANDIDATE_DISPLAY[key],
             "model_id": CANDIDATES[key].model_id}
            for key in CANDIDATE_KEYS
        ],
        "input": "dermoscopic image only",
        "head": {
            "type": "L2-regularized logistic regression",
            "C": PROBE_C,
            "solver": "lbfgs",
            "candidate_specific_hyperparameter_search": False,
            "reason": "isolate frozen representation quality",
        },
        "training_weights": {
            "ham": "equal lesion mass, then equal diagnosis-class mass",
            "derm7pt": "equal case mass, then equal task-class mass",
        },
        "primary_selection_endpoint": (
            "Derm7pt development OOF mean AUROC across MEL and the seven "
            "prespecified dermoscopic criteria"
        ),
        "protected_endpoints": [
            "HAM lesion-level seven-class balanced accuracy",
            "HAM lesion-level melanoma AUROC",
            "Derm7pt five-class balanced accuracy",
        ],
        "selection_rule": {
            "protection_margin_absolute": PROTECTION_MARGIN,
            "violation": (
                "point deficit versus the best candidate exceeds 0.02 and the paired "
                "95% interval for candidate-minus-best excludes zero below zero"
            ),
            "choice": "highest primary point estimate among candidates without a violation",
            "uncertainty_claim": (
                "if the paired primary interval versus the runner-up includes zero, report "
                "development preference, not superiority"
            ),
        },
        "uncertainty": {
            "method": "diagnosis-stratified case/lesion bootstrap",
            "replicates": N_BOOT,
            "seed": SEED,
            "note": "development screening; confirmatory final analysis retains 10,000 replicates",
        },
        "criterion_mapping": {
            criterion: {
                "short": CRITERION_SHORT[criterion],
                "positive_raw_values": sorted(POSITIVE_VALUES[criterion]),
                "allowed_raw_values": sorted(RAW_ALLOWED[criterion]),
            }
            for criterion in CRITERIA
        },
        "fold_manifest": {"path": str(fold_path.relative_to(ROOT)), "sha256": fold_hash},
        "hard_exclusions": [
            "HAM calibration labels or predictions",
            "HAM final-test labels or predictions",
            "Derm7pt official-test images, labels, or predictions",
            "BCN20000 images or labels",
            "PH2 images or labels",
        ],
    }
    protocol_path = OUTPUT_DIR / "dermpact_phase2_1_protocol_v1.json"
    _write_json(protocol_path, protocol)
    return protocol


def _dataset_rows(dataset: str) -> list[dict[str, Any]]:
    if dataset == "ham":
        return _ham_records()
    if dataset == "derm7pt":
        return _derm7pt_records()
    raise ValueError(f"unsupported Phase 2.1 dataset: {dataset}")


def _record_id(row: Mapping[str, Any], dataset: str) -> str:
    return str(row["case_id"] if dataset == "ham" else row["group_id"])


def _image_path(row: Mapping[str, Any], dataset: str) -> Path:
    if dataset == "ham":
        return Path(str(row["image_path"]))
    return DERM7PT_IMAGES / row["modalities"]["dermoscopic"]["path"]


def _image_sha(row: Mapping[str, Any], dataset: str) -> str:
    if dataset == "ham":
        return str(row["image_sha256"])
    return str(row["modalities"]["dermoscopic"]["sha256"])


def _legacy_ham_features(key: str, rows: Sequence[Mapping[str, Any]]):
    import numpy as np

    old_manifest = _load_json(HAM_MANIFEST)["records"]
    index = {row["case_id"]: position for position, row in enumerate(old_manifest)}
    source_dir = LEGACY_FEATURE_ROOT / key
    source_meta = _load_json(source_dir / "feature_manifest.json")
    source_path = source_dir / "features.npy"
    if sha256_file(source_path) != source_meta["feature_file_sha256"]:
        raise AssertionError(f"sealed legacy {key} matrix hash mismatch")
    positions = [index[row["case_id"]] for row in rows]
    if any(old_manifest[position]["split"] != "train" for position in positions):
        raise AssertionError("legacy derivation selected a HAM calibration row")
    features = np.asarray(np.load(source_path, mmap_mode="r")[positions], dtype="float32")
    provenance = {
        "mode": "row-exact derivation from Phase 2.0 golden-validated matrix",
        "source_feature_path": str(source_path.relative_to(ROOT)),
        "source_feature_sha256": source_meta["feature_file_sha256"],
        "source_feature_manifest_sha256": sha256_file(source_dir / "feature_manifest.json"),
        "source_provenance": source_meta["provenance"],
    }
    return features, provenance


def extract_features(candidate: str, dataset: str) -> dict[str, Any]:
    import numpy as np

    if candidate not in CANDIDATE_KEYS:
        raise ValueError(f"unknown Phase 2.1 candidate: {candidate}")
    rows = _dataset_rows(dataset)
    output = FEATURE_DIR / candidate / dataset
    output.mkdir(parents=True, exist_ok=True)
    final = output / "features.npy"
    manifest_path = output / "feature_manifest.json"
    identity = [
        {"id": _record_id(row, dataset), "image_sha256": _image_sha(row, dataset)}
        for row in rows
    ]
    identity_sha = _canonical_hash(identity)
    if final.is_file() and manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest["record_identity_sha256"] != identity_sha:
            raise AssertionError("cached Phase 2.1 features have different row identity")
        if manifest["feature_file_sha256"] != sha256_file(final):
            raise AssertionError("cached Phase 2.1 feature hash mismatch")
        return manifest

    started = time.perf_counter()
    if dataset == "ham" and candidate in {"dermlip", "panderm"}:
        features, provenance = _legacy_ham_features(candidate, rows)
    else:
        for row in rows:
            path = _image_path(row, dataset)
            if not path.is_file():
                raise FileNotFoundError(path)
            if dataset == "derm7pt" and sha256_file(path) != _image_sha(row, dataset):
                raise AssertionError(f"Derm7pt image hash mismatch: {_record_id(row, dataset)}")
        encode, provenance = load_candidate_encoder(candidate)
        batch_size = int(CANDIDATES[candidate].batch_size)
        blocks = []
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            block = encode([str(_image_path(row, dataset)) for row in rows[start:stop]])
            if block.ndim != 2 or len(block) != stop - start or not np.isfinite(block).all():
                raise AssertionError(f"invalid feature block {block.shape}")
            blocks.append(np.asarray(block, dtype="float32"))
            if stop == len(rows) or stop % max(1, batch_size * 10) == 0:
                print(
                    f"[{candidate}/{dataset}] {stop}/{len(rows)}",
                    file=sys.stderr, flush=True,
                )
        features = np.concatenate(blocks, axis=0)
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or float(np.max(np.abs(norms - 1.0))) > 2e-5:
        raise AssertionError("features violate finite unit-norm contract")
    temporary = final.with_suffix(".partial.npy")
    np.save(temporary, features)
    os.replace(temporary, final)
    manifest = {
        "protocol": "dermpact-phase2-1-development-features-v1",
        "candidate": candidate,
        "candidate_display": CANDIDATE_DISPLAY[candidate],
        "model_id": CANDIDATES[candidate].model_id,
        "dataset": dataset,
        "allowed_role": "HAM train only" if dataset == "ham" else "Derm7pt development only",
        "n_records": len(rows),
        "feature_dim": int(features.shape[1]),
        "record_identity_sha256": identity_sha,
        "feature_file_sha256": sha256_file(final),
        "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "excluded_counts": {
            "ham_calibration": 1014,
            "derm7pt_official_test": 395,
            "ph2": "not loaded",
        },
        "final_or_external_predictions": 0,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _balanced_weights(labels: Sequence[Any], groups: Sequence[str] | None = None):
    import numpy as np

    labels = [str(value) for value in labels]
    groups = list(groups) if groups is not None else [str(index) for index in range(len(labels))]
    group_counts = Counter(groups)
    base = np.asarray([1.0 / group_counts[group] for group in groups], dtype=float)
    mass: dict[str, float] = defaultdict(float)
    for label, weight in zip(labels, base):
        mass[label] += float(weight)
    total = float(base.sum())
    weights = np.asarray([
        weight * total / (len(mass) * mass[label])
        for label, weight in zip(labels, base)
    ])
    return weights * len(weights) / weights.sum()


def _new_probe():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=PROBE_C, max_iter=5000, solver="lbfgs", tol=1e-6)


def _aligned_probabilities(model: Any, features: Any, classes: Sequence[str]):
    import numpy as np

    observed = list(model.classes_)
    raw = model.predict_proba(features)
    return np.asarray(raw[:, [observed.index(label) for label in classes]], dtype=float)


def _criterion_labels(rows: Sequence[Mapping[str, Any]], criterion: str):
    import numpy as np

    return np.asarray([
        int(row["criteria_raw"][criterion] in POSITIVE_VALUES[criterion]) for row in rows
    ])


def _folded_multiclass(features: Any, labels: Any, folds: Any, weights: Any):
    import numpy as np

    classes = sorted(set(labels.tolist()))
    probabilities = np.zeros((len(labels), len(classes)), dtype=float)
    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        model = _new_probe().fit(features[train], labels[train], sample_weight=weights[train])
        probabilities[valid] = _aligned_probabilities(model, features[valid], classes)
    return classes, probabilities


def _folded_binary(features: Any, labels: Any, folds: Any, weights: Any):
    import numpy as np

    probabilities = np.zeros(len(labels), dtype=float)
    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        if len(set(labels[train].tolist())) != 2:
            raise AssertionError("binary training fold lost a class")
        model = _new_probe().fit(features[train], labels[train], sample_weight=weights[train])
        probabilities[valid] = model.predict_proba(features[valid])[:, list(model.classes_).index(1)]
    return probabilities


def _aggregate_ham(
    rows: Sequence[Mapping[str, Any]], classes: Sequence[str], diag_probability: Any,
    mel_probability: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["cluster_id"])].append(index)
    result = []
    for lesion, indices in sorted(grouped.items()):
        labels = {str(rows[index]["label"]) for index in indices}
        if len(labels) != 1:
            raise AssertionError(f"HAM lesion {lesion} has conflicting labels")
        probs = np.mean(diag_probability[indices], axis=0)
        result.append({
            "id": lesion,
            "label": next(iter(labels)),
            "n_images": len(indices),
            "diagnosis_probability": {
                label: float(probs[position]) for position, label in enumerate(classes)
            },
            "melanoma_probability": float(np.mean(mel_probability[indices])),
        })
    return result


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _multiclass_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    labels = [str(row["label"]) for row in rows]
    classes = sorted(rows[0]["diagnosis_probability"])
    matrix = [[float(row["diagnosis_probability"][label]) for label in classes] for row in rows]
    predictions = [classes[max(range(len(classes)), key=lambda j: matrix[i][j])] for i in range(len(rows))]
    one_hot = [[int(label == candidate) for candidate in classes] for label in labels]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_ovr_auc": float(roc_auc_score(one_hot, matrix, average="macro")),
    }


def _balanced_accuracy_only(rows: Sequence[Mapping[str, Any]]) -> float:
    """Compute only balanced accuracy for bootstrap hot paths."""
    from sklearn.metrics import balanced_accuracy_score

    classes = sorted(rows[0]["diagnosis_probability"])
    labels = [str(row["label"]) for row in rows]
    predictions = [
        max(classes, key=lambda label: float(row["diagnosis_probability"][label]))
        for row in rows
    ]
    return float(balanced_accuracy_score(labels, predictions))


def _stratified_indices(labels: Sequence[str], rng: Any):
    import numpy as np

    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[str(label)].append(index)
    return np.concatenate([
        rng.choice(indices, size=len(indices), replace=True)
        for _, indices in sorted(by_label.items())
    ])


def _interval(
    rows: Sequence[Mapping[str, Any]], metric: Callable[[Sequence[Mapping[str, Any]]], float],
    *, seed_offset: int = 0,
) -> dict[str, float]:
    import numpy as np

    estimate = float(metric(rows))
    rng = np.random.default_rng(SEED + seed_offset)
    labels = [str(row["label"]) for row in rows]
    samples = np.empty(N_BOOT, dtype=float)
    for replicate in range(N_BOOT):
        indices = _stratified_indices(labels, rng)
        samples[replicate] = metric([rows[int(index)] for index in indices])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "n_boot": N_BOOT}


def _ham_primary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def balanced(sample: Sequence[Mapping[str, Any]]) -> float:
        return _balanced_accuracy_only(sample)

    def melanoma(sample: Sequence[Mapping[str, Any]]) -> float:
        return _auc(
            [int(row["label"] == "melanoma") for row in sample],
            [float(row["melanoma_probability"]) for row in sample],
        )

    return {
        "n_lesions": len(rows),
        "n_images": sum(int(row["n_images"]) for row in rows),
        "diagnosis": _multiclass_metrics(rows),
        "balanced_accuracy_interval": _interval(rows, balanced, seed_offset=11),
        "melanoma_auc_interval": _interval(rows, melanoma, seed_offset=12),
    }


def _derm_output_aucs(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result = {
        "MEL": _auc(
            [int(row["label"] == "MEL") for row in rows],
            [float(row["melanoma_probability"]) for row in rows],
        )
    }
    for criterion in CRITERIA:
        short = CRITERION_SHORT[criterion]
        result[short] = _auc(
            [int(row["criterion_true"][criterion]) for row in rows],
            [float(row["criterion_probability"][criterion]) for row in rows],
        )
    return result


def _derm_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean_auc(sample: Sequence[Mapping[str, Any]]) -> float:
        values = _derm_output_aucs(sample)
        return float(sum(values.values()) / len(values))

    def balanced(sample: Sequence[Mapping[str, Any]]) -> float:
        return _balanced_accuracy_only(sample)

    output_aucs = _derm_output_aucs(rows)
    return {
        "n_cases": len(rows),
        "diagnosis": _multiclass_metrics(rows),
        "output_auc": output_aucs,
        "eight_output_mean_auc_interval": _interval(rows, mean_auc, seed_offset=21),
        "diagnosis_balanced_accuracy_interval": _interval(rows, balanced, seed_offset=22),
        "positive_counts": {
            "MEL": sum(row["label"] == "MEL" for row in rows),
            **{
                CRITERION_SHORT[criterion]: sum(
                    bool(row["criterion_true"][criterion]) for row in rows
                )
                for criterion in CRITERIA
            },
        },
    }


def evaluate_candidate(candidate: str) -> dict[str, Any]:
    import numpy as np

    if candidate not in CANDIDATE_KEYS:
        raise ValueError(candidate)
    ham = _ham_records()
    derm = _derm7pt_records()
    ham_features = np.load(FEATURE_DIR / candidate / "ham/features.npy", mmap_mode="r")
    derm_features = np.load(FEATURE_DIR / candidate / "derm7pt/features.npy", mmap_mode="r")
    if len(ham_features) != len(ham) or len(derm_features) != len(derm):
        raise AssertionError("feature/record count mismatch")

    ham_fold_map = _ham_fold_map(ham)
    ham_folds = np.asarray([ham_fold_map[row["cluster_id"]] for row in ham], dtype=int)
    ham_labels = np.asarray([str(row["label"]) for row in ham])
    ham_groups = [str(row["cluster_id"]) for row in ham]
    ham_weights = _balanced_weights(ham_labels, ham_groups)
    ham_classes, ham_diag = _folded_multiclass(
        ham_features, ham_labels, ham_folds, ham_weights
    )
    ham_mel_true = np.asarray([int(label == "melanoma") for label in ham_labels])
    ham_mel = _folded_binary(
        ham_features, ham_mel_true, ham_folds,
        _balanced_weights(ham_mel_true, ham_groups),
    )
    ham_predictions = _aggregate_ham(ham, ham_classes, ham_diag, ham_mel)

    derm_folds = np.asarray([int(row["development_fold"]) for row in derm], dtype=int)
    derm_labels = np.asarray([str(row["diagnosis"]["harmonized5"]) for row in derm])
    derm_classes, derm_diag = _folded_multiclass(
        derm_features, derm_labels, derm_folds, _balanced_weights(derm_labels)
    )
    derm_mel_true = np.asarray([int(label == "MEL") for label in derm_labels])
    derm_mel = _folded_binary(
        derm_features, derm_mel_true, derm_folds, _balanced_weights(derm_mel_true)
    )
    criterion_true: dict[str, Any] = {}
    criterion_probability: dict[str, Any] = {}
    for criterion in CRITERIA:
        truth = _criterion_labels(derm, criterion)
        criterion_true[criterion] = truth
        criterion_probability[criterion] = _folded_binary(
            derm_features, truth, derm_folds, _balanced_weights(truth)
        )
    derm_predictions = []
    for index, row in enumerate(derm):
        derm_predictions.append({
            "id": str(row["group_id"]),
            "label": str(row["diagnosis"]["harmonized5"]),
            "diagnosis_probability": {
                label: float(derm_diag[index, position])
                for position, label in enumerate(derm_classes)
            },
            "melanoma_probability": float(derm_mel[index]),
            "criterion_true": {
                criterion: bool(criterion_true[criterion][index]) for criterion in CRITERIA
            },
            "criterion_probability": {
                criterion: float(criterion_probability[criterion][index])
                for criterion in CRITERIA
            },
        })

    candidate_dir = RESULT_DIR / candidate
    ham_path = candidate_dir / "ham_oof_lesion_predictions.jsonl"
    derm_path = candidate_dir / "derm7pt_oof_predictions.jsonl"
    ham_hash = _write_jsonl(ham_path, ham_predictions)
    derm_hash = _write_jsonl(derm_path, derm_predictions)
    result = {
        "protocol": "dermpact-phase2-1-frozen-baseline-result-v1",
        "candidate": candidate,
        "candidate_display": CANDIDATE_DISPLAY[candidate],
        "model_id": CANDIDATES[candidate].model_id,
        "evidence_level": "development OOF screening only; not final or external evidence",
        "head": {"type": "fixed logistic probe", "C": PROBE_C},
        "ham": _ham_primary_metrics(ham_predictions),
        "derm7pt": _derm_metrics(derm_predictions),
        "predictions": {
            "ham": {"path": str(ham_path.relative_to(ROOT)), "sha256": ham_hash},
            "derm7pt": {"path": str(derm_path.relative_to(ROOT)), "sha256": derm_hash},
        },
        "feature_manifests": {
            dataset: {
                "path": str((FEATURE_DIR / candidate / dataset / "feature_manifest.json").relative_to(ROOT)),
                "sha256": sha256_file(FEATURE_DIR / candidate / dataset / "feature_manifest.json"),
            }
            for dataset in ("ham", "derm7pt")
        },
        "forbidden_records_loaded_or_scored": 0,
    }
    result_path = candidate_dir / "result.json"
    _write_json(result_path, result)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _paired_interval(
    rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]],
    metric: Callable[[Sequence[Mapping[str, Any]]], float], *, seed_offset: int,
) -> dict[str, float]:
    import numpy as np

    if [row["id"] for row in rows_a] != [row["id"] for row in rows_b]:
        raise AssertionError("paired candidate predictions are not case-aligned")
    estimate = float(metric(rows_a) - metric(rows_b))
    rng = np.random.default_rng(SEED + seed_offset)
    labels = [str(row["label"]) for row in rows_a]
    samples = np.empty(N_BOOT, dtype=float)
    for replicate in range(N_BOOT):
        indices = _stratified_indices(labels, rng)
        sample_a = [rows_a[int(index)] for index in indices]
        sample_b = [rows_b[int(index)] for index in indices]
        samples[replicate] = metric(sample_a) - metric(sample_b)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "n_boot": N_BOOT}


def finalize() -> dict[str, Any]:
    results = {key: _load_json(RESULT_DIR / key / "result.json") for key in CANDIDATE_KEYS}
    predictions = {
        key: {
            "ham": _read_jsonl(RESULT_DIR / key / "ham_oof_lesion_predictions.jsonl"),
            "derm7pt": _read_jsonl(RESULT_DIR / key / "derm7pt_oof_predictions.jsonl"),
        }
        for key in CANDIDATE_KEYS
    }

    def derm_mean(rows: Sequence[Mapping[str, Any]]) -> float:
        values = _derm_output_aucs(rows)
        return float(sum(values.values()) / len(values))

    def derm_balanced(rows: Sequence[Mapping[str, Any]]) -> float:
        return _balanced_accuracy_only(rows)

    def ham_balanced(rows: Sequence[Mapping[str, Any]]) -> float:
        return _balanced_accuracy_only(rows)

    def ham_mel(rows: Sequence[Mapping[str, Any]]) -> float:
        return _auc(
            [int(row["label"] == "melanoma") for row in rows],
            [float(row["melanoma_probability"]) for row in rows],
        )

    metric_specs = {
        "derm7pt_eight_output_mean_auc": ("derm7pt", derm_mean),
        "derm7pt_five_class_balanced_accuracy": ("derm7pt", derm_balanced),
        "ham_seven_class_balanced_accuracy": ("ham", ham_balanced),
        "ham_melanoma_auc": ("ham", ham_mel),
    }
    comparisons = []
    seed_offset = 100
    for left_index, left in enumerate(CANDIDATE_KEYS):
        for right in CANDIDATE_KEYS[left_index + 1:]:
            entry = {"left": left, "right": right, "difference": {}}
            for name, (dataset, metric) in metric_specs.items():
                seed_offset += 1
                entry["difference"][name] = _paired_interval(
                    predictions[left][dataset], predictions[right][dataset], metric,
                    seed_offset=seed_offset,
                )
            comparisons.append(entry)

    point = {
        key: {
            "primary": float(results[key]["derm7pt"]["eight_output_mean_auc_interval"]["estimate"]),
            "derm_balanced": float(results[key]["derm7pt"]["diagnosis_balanced_accuracy_interval"]["estimate"]),
            "ham_balanced": float(results[key]["ham"]["balanced_accuracy_interval"]["estimate"]),
            "ham_mel": float(results[key]["ham"]["melanoma_auc_interval"]["estimate"]),
        }
        for key in CANDIDATE_KEYS
    }
    protected = {
        "derm_balanced": ("derm7pt", derm_balanced),
        "ham_balanced": ("ham", ham_balanced),
        "ham_mel": ("ham", ham_mel),
    }
    violations: dict[str, list[dict[str, Any]]] = {key: [] for key in CANDIDATE_KEYS}
    for metric_name, (dataset, metric) in protected.items():
        best = max(CANDIDATE_KEYS, key=lambda key: point[key][metric_name])
        for candidate in CANDIDATE_KEYS:
            if candidate == best:
                continue
            difference = _paired_interval(
                predictions[candidate][dataset], predictions[best][dataset], metric,
                seed_offset=300 + list(protected).index(metric_name) * 10
                + list(CANDIDATE_KEYS).index(candidate),
            )
            deficit = point[best][metric_name] - point[candidate][metric_name]
            if deficit > PROTECTION_MARGIN and difference["ci_high"] < 0.0:
                violations[candidate].append({
                    "endpoint": metric_name,
                    "best_candidate": best,
                    "point_deficit": deficit,
                    "paired_candidate_minus_best": difference,
                })
    eligible = [key for key in CANDIDATE_KEYS if not violations[key]]
    if not eligible:
        raise AssertionError("every frozen candidate violated a protected endpoint")
    selected = max(eligible, key=lambda key: point[key]["primary"])
    runner_up = max(
        [key for key in eligible if key != selected],
        key=lambda key: point[key]["primary"], default=None,
    )
    primary_difference = None
    claim = "only eligible candidate"
    if runner_up is not None:
        primary_difference = _paired_interval(
            predictions[selected]["derm7pt"], predictions[runner_up]["derm7pt"],
            derm_mean, seed_offset=401,
        )
        claim = (
            "development preference, not superiority"
            if primary_difference["ci_low"] <= 0.0 <= primary_difference["ci_high"]
            else "development superiority under the prespecified paired interval"
        )
    decision = {
        "protocol": "dermpact-phase2-1-backbone-decision-v1",
        "selected_candidate": selected,
        "selected_display": CANDIDATE_DISPLAY[selected],
        "eligible_candidates": eligible,
        "protected_violations": violations,
        "point_estimates": point,
        "runner_up": runner_up,
        "selected_minus_runner_up_primary": primary_difference,
        "claim": claim,
        "use_in_next_phase": (
            "controlled backbone for the plain shared multitask ladder; this decision does not "
            "authorize a final/external generalization claim"
        ),
    }
    comparison_path = OUTPUT_DIR / "dermpact_phase2_1_paired_comparisons_v1.json"
    comparison_hash = _write_json(comparison_path, {
        "protocol": "dermpact-phase2-1-paired-comparisons-v1",
        "comparisons": comparisons,
    })
    decision_path = OUTPUT_DIR / "dermpact_phase2_1_backbone_decision_v1.json"
    decision_hash = _write_json(decision_path, decision)
    result_artifacts = {}
    for key in CANDIDATE_KEYS:
        source = RESULT_DIR / key / "result.json"
        target = OUTPUT_DIR / f"dermpact_phase2_1_{key}_result_v1.json"
        result_artifacts[key] = {"path": str(target.relative_to(ROOT)),
                                 "sha256": _write_json(target, _load_json(source))}
    protocol_path = OUTPUT_DIR / "dermpact_phase2_1_protocol_v1.json"
    folds_path = OUTPUT_DIR / "dermpact_phase2_1_development_folds_v1.json"
    receipt = {
        "protocol": "dermpact-phase2-1-receipt-v1",
        "phase": "2.1",
        "status": "PASS",
        "selected_backbone": selected,
        "selected_backbone_display": CANDIDATE_DISPLAY[selected],
        "selection_claim": claim,
        "protocol_artifact": {"path": str(protocol_path.relative_to(ROOT)),
                              "sha256": sha256_file(protocol_path)},
        "fold_artifact": {"path": str(folds_path.relative_to(ROOT)),
                          "sha256": sha256_file(folds_path)},
        "candidate_results": result_artifacts,
        "paired_comparisons": {"path": str(comparison_path.relative_to(ROOT)),
                               "sha256": comparison_hash},
        "decision": {"path": str(decision_path.relative_to(ROOT)),
                     "sha256": decision_hash},
        "scope_completed": [
            "pure PanDerm-Base frozen development baseline",
            "DermLIP frozen development baseline",
            "DermLIP–PanDerm hybrid frozen development baseline",
            "HAM training-only lesion-grouped OOF diagnosis comparison",
            "Derm7pt locked-development-fold diagnosis and seven-criterion comparison",
            "paired protected-endpoint checks and controlled backbone choice",
        ],
        "hard_scope_checks": {
            "ham_calibration_predictions": 0,
            "ham_final_test_predictions": 0,
            "derm7pt_official_test_predictions": 0,
            "bcn20000_records_loaded": 0,
            "ph2_records_loaded": 0,
        },
        "claim_restriction": (
            "All candidates still have unresolved pretraining-overlap status. Phase 2.1 is "
            "development selection evidence only."
        ),
        "next_phase": (
            "Phase 2.2 trains the plain shared multitask baseline with the selected backbone, "
            "masked losses, dataset-balanced batches, and dermoscopic image only."
        ),
    }
    receipt_path = OUTPUT_DIR / "dermpact_phase2_1_receipt_v1.json"
    receipt_hash = _write_json(receipt_path, receipt)
    return {**receipt, "receipt_sha256": receipt_hash}


def audit_runtime_and_bind_receipt() -> dict[str, Any]:
    """Record the runtime relocation discovered before Phase 2.1 execution."""
    import platform

    from .dermpact_phase2_foundation import audit_environment

    audit = audit_environment()
    checks = dict(audit["checks"])
    checks.update({
        "phase2_1_receipt_exists": (
            OUTPUT_DIR / "dermpact_phase2_1_receipt_v1.json"
        ).is_file(),
        "python_patch_is_3_12_13": platform.python_version() == "3.12.13",
        "pyvenv_config_points_to_bundled_runtime": (
            "codex-primary-runtime" in PYVENV_CONFIG.read_text(encoding="utf-8")
        ),
    })
    payload = {
        **audit,
        "protocol": "dermpact-phase2-1-runtime-transition-v1",
        "phase": "2.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "transition": {
            "discovered_before_phase2_1_scoring": True,
            "previous_interpreter": "CPython 3.12.10 at the removed user-local path",
            "phase2_1_interpreter": platform.python_version(),
            "reason": (
                "venv_p2's original base interpreter path no longer existed; the venv "
                "was relocated to the available bundled CPython 3.12 runtime"
            ),
            "locked_site_packages_changed": False,
            "pip_check": "PASS",
            "phase2_0_receipt_remains_historical": True,
        },
        "pyvenv_config": {
            "path": str(PYVENV_CONFIG.relative_to(ROOT)),
            "sha256": sha256_file(PYVENV_CONFIG),
        },
        "environment_lock": {
            "path": str(LOCK_FILE.relative_to(ROOT)),
            "sha256": sha256_file(LOCK_FILE),
        },
    }
    path = OUTPUT_DIR / "dermpact_phase2_1_runtime_audit_v1.json"
    digest = _write_json(path, payload)
    receipt_path = OUTPUT_DIR / "dermpact_phase2_1_receipt_v1.json"
    receipt = _load_json(receipt_path)
    receipt["runtime_audit"] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "status": payload["status"],
    }
    receipt["runtime_note"] = (
        "The locked package set is unchanged; Phase 2.1 ran under CPython 3.12.13 "
        "after the missing 3.12.10 base path was relocated and re-audited."
    )
    receipt_digest = _write_json(receipt_path, receipt)
    return {**payload, "receipt_sha256_after_runtime_binding": receipt_digest}


def _run_child(args: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", MODULE, *args], check=False,
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Phase 2.1 child failed: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if completed.stderr.strip():
        print(completed.stderr, file=sys.stderr, end="")
    return json.loads(completed.stdout)


def run_all() -> dict[str, Any]:
    build_protocol()
    for candidate in CANDIDATE_KEYS:
        for dataset in ("ham", "derm7pt"):
            _run_child(["--extract", "--candidate", candidate, "--dataset", dataset])
        _run_child(["--evaluate", "--candidate", candidate])
    return finalize()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--audit-runtime", action="store_true")
    parser.add_argument("--candidate", choices=CANDIDATE_KEYS)
    parser.add_argument("--dataset", choices=("ham", "derm7pt"))
    args = parser.parse_args(argv)
    if args.audit_runtime:
        payload = audit_runtime_and_bind_receipt()
    elif args.extract:
        if not args.candidate or not args.dataset:
            parser.error("--extract requires --candidate and --dataset")
        payload = extract_features(args.candidate, args.dataset)
    elif args.evaluate:
        if not args.candidate:
            parser.error("--evaluate requires --candidate")
        payload = evaluate_candidate(args.candidate)
    elif args.finalize:
        payload = finalize()
    else:
        payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
