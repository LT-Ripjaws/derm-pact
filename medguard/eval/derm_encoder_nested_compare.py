"""Paired lesion-level comparison for the nested derm encoder tournament."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .derm_encoder_nested_tournament import RESULT_ROOT
from .statistics import delong_auc_test, holm_adjust, paired_clustered_bootstrap_difference


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "dermlip"
CANDIDATES = ("dermlip", "panderm", "monet", "biomedclip")
OUTPUT = ROOT / "research/experiments/derm_perception/v4_nested_lesion_tournament/comparison.json"


def _load(model: str) -> tuple[list[dict[str, Any]], float]:
    directory = RESULT_ROOT / model
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result.get("protocol") != "ham10000-derm-encoder-nested-lesion-tournament-v4":
        raise ValueError(f"{model} is not a v4 lesion-level result")
    if result.get("test_records_loaded") != 0:
        raise ValueError(f"{model} reports final-test exposure")
    rows = [
        json.loads(line) for line in
        (directory / "node_validation_lesion_predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    threshold = float(result["operating_point"]["specificity_90"]["threshold"])
    for row in rows:
        row["malignant_prediction"] = float(row["malignant_score"]) >= threshold
    return rows, threshold


def _accuracy(rows: Sequence[dict[str, Any]]) -> float:
    return sum(row["prediction"] == row["label"] for row in rows) / len(rows)


def _actionable_recall(rows: Sequence[dict[str, Any]]) -> float:
    positives = [row for row in rows if row["malignant_true"]]
    return sum(row["malignant_prediction"] for row in positives) / len(positives)


def _specificity(rows: Sequence[dict[str, Any]]) -> float:
    negatives = [row for row in rows if not row["malignant_true"]]
    return sum(not row["malignant_prediction"] for row in negatives) / len(negatives)


def _interval_dict(interval) -> dict[str, float]:
    return {
        "difference": float(interval.estimate),
        "ci_low": float(interval.low),
        "ci_high": float(interval.high),
    }


def _paired_interval(
    rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]],
    metric: Callable[[Sequence[dict[str, Any]]], float],
) -> dict[str, float]:
    return _interval_dict(paired_clustered_bootstrap_difference(
        rows_a, rows_b, [str(row["cluster_id"]) for row in rows_a],
        metric, n_boot=5000, seed=20260714,
    ))


def compare() -> dict[str, Any]:
    loaded = {model: _load(model) for model in CANDIDATES}
    reference_rows, _ = loaded[REFERENCE]
    reference_ids = [row["cluster_id"] for row in reference_rows]
    comparisons: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for model in CANDIDATES:
        if model == REFERENCE:
            continue
        rows, _ = loaded[model]
        if [row["cluster_id"] for row in rows] != reference_ids:
            raise ValueError(f"{model} lesion order does not match {REFERENCE}")
        labels = [bool(row["malignant_true"]) for row in reference_rows]
        auc = delong_auc_test(
            labels,
            [float(row["malignant_score"]) for row in reference_rows],
            [float(row["malignant_score"]) for row in rows],
        )
        p_values[model] = float(auc["p_value"])
        comparisons[model] = {
            "direction": f"{REFERENCE}_minus_{model}",
            "malignant_auc_delong": auc,
            "diagnosis_accuracy": _paired_interval(reference_rows, rows, _accuracy),
            "actionable_recall": _paired_interval(reference_rows, rows, _actionable_recall),
            "specificity": _paired_interval(reference_rows, rows, _specificity),
        }
    adjusted = holm_adjust(p_values)
    for model, value in adjusted.items():
        comparisons[model]["malignant_auc_holm_p"] = float(value)
    return {
        "protocol": "ham10000-derm-encoder-v4-paired-lesion-comparison",
        "reference": REFERENCE,
        "n_lesions": len(reference_rows),
        "test_records_loaded": 0,
        "comparisons": comparisons,
        "decision_rule": (
            "Prefer actionable recall and specificity, then malignant AUC/AURC/calibration; "
            "retain uncertainty and do not claim superiority when paired intervals cross zero."
        ),
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    result = compare()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
