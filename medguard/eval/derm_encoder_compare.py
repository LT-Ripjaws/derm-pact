"""Paired, lesion-clustered comparison of completed derm encoder screens.

This module compares development/calibration artifacts only.  It never opens the
sealed HAM10000 test split, and its inferential outputs are for architecture
selection rather than final paper claims.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

from .derm_encoders import FEATURE_ROOT
from .statistics import (
    delong_auc_test,
    holm_adjust,
    mcnemar_exact,
    paired_clustered_bootstrap_difference,
)


RESULT_ROOT = FEATURE_ROOT.parent / "results"
COMPARISON_ROOT = FEATURE_ROOT.parent / "comparison"
DIAGNOSIS_LABELS = ("akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_candidate(root: Path, key: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    directory = root / key
    result = _read_json(directory / "result.json")
    rows = _read_jsonl(directory / "calibration_predictions.jsonl")
    if result.get("test_records_loaded") != 0:
        raise ValueError(f"{key}: result is not test-redacted")
    if not rows:
        raise ValueError(f"{key}: empty prediction artifact")
    threshold = float(result["operating_points"]["specificity_90"]["threshold"])
    enriched = [
        {
            **row,
            "diagnosis_correct": str(row["prediction"]) == str(row["label"]),
            "malignant_prediction": float(row["malignant_score"]) >= threshold,
        }
        for row in rows
    ]
    return result, enriched


def _align(candidates: dict[str, list[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    if len(candidates) < 2:
        raise ValueError("paired comparison requires at least two candidates")
    aligned: dict[str, list[dict[str, object]]] = {}
    reference_key = sorted(candidates)[0]
    reference = {str(row["case_id"]): row for row in candidates[reference_key]}
    if len(reference) != len(candidates[reference_key]):
        raise ValueError(f"{reference_key}: duplicate case_id")
    case_ids = sorted(reference)
    for key, rows in candidates.items():
        lookup = {str(row["case_id"]): row for row in rows}
        if len(lookup) != len(rows):
            raise ValueError(f"{key}: duplicate case_id")
        if set(lookup) != set(case_ids):
            missing = sorted(set(case_ids) - set(lookup))[:3]
            extra = sorted(set(lookup) - set(case_ids))[:3]
            raise ValueError(f"{key}: case mismatch; missing={missing}, extra={extra}")
        ordered = [lookup[case_id] for case_id in case_ids]
        for ref, row in zip((reference[case_id] for case_id in case_ids), ordered):
            invariant = ("cluster_id", "label", "malignant_true")
            if any(ref[field] != row[field] for field in invariant):
                raise ValueError(f"{key}: target/cluster mismatch for {row['case_id']}")
        aligned[key] = ordered
    return aligned


def _accuracy(rows: Sequence[dict[str, object]]) -> float:
    return sum(bool(row["diagnosis_correct"]) for row in rows) / len(rows)


def _macro_recall(rows: Sequence[dict[str, object]]) -> float:
    recalls = []
    for label in DIAGNOSIS_LABELS:
        labelled = [row for row in rows if str(row["label"]) == label]
        if labelled:
            recalls.append(sum(str(row["prediction"]) == label for row in labelled) / len(labelled))
    return sum(recalls) / len(recalls)


def _melanoma_recall(rows: Sequence[dict[str, object]]) -> float:
    labelled = [row for row in rows if str(row["label"]) == "melanoma"]
    return sum(str(row["prediction"]) == "melanoma" for row in labelled) / len(labelled)


def _auc(rows: Sequence[dict[str, object]]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(
        [bool(row["malignant_true"]) for row in rows],
        [float(row["malignant_score"]) for row in rows],
    ))


def _malignant_recall(rows: Sequence[dict[str, object]]) -> float:
    positives = [row for row in rows if bool(row["malignant_true"])]
    return sum(bool(row["malignant_prediction"]) for row in positives) / len(positives)


def _specificity(rows: Sequence[dict[str, object]]) -> float:
    negatives = [row for row in rows if not bool(row["malignant_true"])]
    return sum(not bool(row["malignant_prediction"]) for row in negatives) / len(negatives)


BOOTSTRAP_METRICS: dict[str, Callable[[Sequence[dict[str, object]]], float]] = {
    "diagnosis_accuracy": _accuracy,
    "diagnosis_macro_recall": _macro_recall,
    "melanoma_diagnosis_recall": _melanoma_recall,
    "malignant_auc": _auc,
    "malignant_recall_at_90_specificity": _malignant_recall,
    "specificity_at_selected_threshold": _specificity,
}


def _metric_snapshot(result: dict[str, object]) -> dict[str, object]:
    metrics = result["calibration_metrics"]
    return {
        "diagnosis_accuracy": metrics["diagnosis_accuracy"]["estimate"],
        "diagnosis_macro_recall": metrics["diagnosis_macro_recall"]["estimate"],
        "melanoma_diagnosis_recall": metrics["melanoma_diagnosis_recall"]["estimate"],
        "malignant_auc": metrics["malignant_auc"]["estimate"],
        "malignant_recall_at_90_specificity": metrics["malignant_recall_at_90_specificity"]["estimate"],
        "specificity_at_selected_threshold": metrics["specificity_at_selected_threshold"]["estimate"],
        "false_reassurance_at_90_specificity": metrics["false_reassurance_at_90_specificity"]["estimate"],
        "diagnosis_aurc": metrics["diagnosis_aurc"],
        "malignant_brier": metrics["malignant_brier"],
        "diagnosis_ece": metrics["diagnosis_ece"],
        "malignant_ece": metrics["malignant_ece"],
    }


def _pareto_front(snapshots: dict[str, dict[str, object]]) -> list[str]:
    maximize = (
        "diagnosis_accuracy", "diagnosis_macro_recall", "malignant_auc",
        "malignant_recall_at_90_specificity",
    )
    front = []
    for candidate, metrics in snapshots.items():
        dominated = False
        for rival, other in snapshots.items():
            if candidate == rival:
                continue
            weakly_better = all(float(other[name]) >= float(metrics[name]) for name in maximize)
            strictly_better = any(float(other[name]) > float(metrics[name]) for name in maximize)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front)


def compare_candidates(
    keys: Sequence[str], *, result_root: Path = RESULT_ROOT,
    output_dir: Path = COMPARISON_ROOT, n_boot: int = 2000,
) -> dict[str, object]:
    if len(set(keys)) != len(keys):
        raise ValueError("candidate keys must be unique")
    loaded = {key: _load_candidate(result_root, key) for key in keys}
    hashes = {str(result["development_manifest_sha256"]) for result, _ in loaded.values()}
    if len(hashes) != 1:
        raise ValueError("candidates use different development manifests")
    aligned = _align({key: rows for key, (_, rows) in loaded.items()})
    snapshots = {key: _metric_snapshot(result) for key, (result, _) in loaded.items()}
    comparisons: dict[str, object] = {}
    delong_p: dict[str, float] = {}
    diagnosis_p: dict[str, float] = {}
    malignant_p: dict[str, float] = {}
    for a, b in combinations(sorted(keys), 2):
        rows_a, rows_b = aligned[a], aligned[b]
        clusters = [str(row["cluster_id"]) for row in rows_a]
        intervals = {
            name: vars(paired_clustered_bootstrap_difference(
                rows_a, rows_b, clusters, metric, n_boot=n_boot,
                seed=20260713,
            ))
            for name, metric in BOOTSTRAP_METRICS.items()
        }
        delong = delong_auc_test(
            [bool(row["malignant_true"]) for row in rows_a],
            [float(row["malignant_score"]) for row in rows_a],
            [float(row["malignant_score"]) for row in rows_b],
        )
        diagnosis_mcnemar = mcnemar_exact(
            [bool(row["diagnosis_correct"]) for row in rows_a],
            [bool(row["diagnosis_correct"]) for row in rows_b],
        )
        malignant_mcnemar = mcnemar_exact(
            [bool(row["malignant_prediction"]) == bool(row["malignant_true"]) for row in rows_a],
            [bool(row["malignant_prediction"]) == bool(row["malignant_true"]) for row in rows_b],
        )
        name = f"{a}_minus_{b}"
        delong_p[name] = float(delong["p_value"])
        diagnosis_p[name] = float(diagnosis_mcnemar[2])
        malignant_p[name] = float(malignant_mcnemar[2])
        comparisons[name] = {
            "direction": "positive estimates favor the first named candidate",
            "paired_lesion_cluster_bootstrap": intervals,
            "delong_malignant_auc": delong,
            "mcnemar_diagnosis_correct": {
                "a_only_correct": diagnosis_mcnemar[0],
                "b_only_correct": diagnosis_mcnemar[1],
                "p_value": diagnosis_mcnemar[2],
            },
            "mcnemar_malignant_operating_point_correct": {
                "a_only_correct": malignant_mcnemar[0],
                "b_only_correct": malignant_mcnemar[1],
                "p_value": malignant_mcnemar[2],
            },
        }
    adjusted = {
        "delong_malignant_auc": holm_adjust(delong_p),
        "mcnemar_diagnosis_correct": holm_adjust(diagnosis_p),
        "mcnemar_malignant_operating_point_correct": holm_adjust(malignant_p),
    }
    for family, values in adjusted.items():
        for name, p_value in values.items():
            comparisons[name].setdefault("holm_adjusted_p", {})[family] = p_value
    output: dict[str, object] = {
        "protocol": "ham10000-derm-encoder-paired-comparison-v2",
        "evidence_level": "development/calibration screening; not final-test evidence",
        "development_manifest_sha256": next(iter(hashes)),
        "n_images": len(next(iter(aligned.values()))),
        "n_lesions": len({str(row["cluster_id"]) for row in next(iter(aligned.values()))}),
        "bootstrap_replicates": n_boot,
        "candidate_metrics": snapshots,
        "pareto_front": _pareto_front(snapshots),
        "comparisons": comparisons,
        "limitations": [
            "operating thresholds and model selection use calibration data",
            "DeLong and McNemar are image-level; lesion-cluster bootstrap is the primary uncertainty analysis",
            "external validation and pretraining-overlap audits remain mandatory",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = compare_candidates(args.models, n_boot=args.bootstrap)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
