"""Paired lesion-level comparison of frozen HAM encoders on external BCN20000."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

from .derm_encoder_compare import delong_auc_test, holm_adjust, mcnemar_exact
from .derm_encoder_tournament import _threshold_at_specificity
from .derm_external_bcn import ACTIONABLE, EXPERIMENT_ROOT, SHARED_CLASSES, _aggregate_lesions
from .statistics import clustered_bootstrap_ci


OUTPUT = EXPERIMENT_ROOT / "comparison" / "paired_comparison.json"


def _load_model(key: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    directory = EXPERIMENT_ROOT / "results" / key
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    image_rows = [json.loads(line) for line in
                  (directory / "external_predictions.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    lesions = _aggregate_lesions(image_rows)
    threshold = float(result["frozen_malignant_threshold"])
    for row in lesions:
        row["actionable_true"] = str(row["label"]) in ACTIONABLE
        row["actionable_prediction"] = float(row["malignant_score"]) >= threshold
        row["diagnosis_correct"] = row["prediction"] == row["label"]
    return result, lesions


def _align(rows_by_model: dict[str, Sequence[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    ordered: dict[str, dict[str, dict[str, object]]] = {}
    for model, rows in rows_by_model.items():
        mapping = {str(row["case_id"]): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"duplicate lesion id in {model}")
        ordered[model] = mapping
    case_sets = {model: set(mapping) for model, mapping in ordered.items()}
    if len({frozenset(values) for values in case_sets.values()}) != 1:
        raise ValueError("external candidates do not contain identical lesions")
    case_ids = sorted(next(iter(case_sets.values())))
    aligned = {model: [ordered[model][case_id] for case_id in case_ids] for model in ordered}
    reference = next(iter(aligned.values()))
    for model, rows in aligned.items():
        for left, right in zip(reference, rows):
            if left["label"] != right["label"] or left["cluster_id"] != right["cluster_id"]:
                raise ValueError(f"external truth alignment mismatch for {model}")
    return aligned


def _paired_ci(rows_a: Sequence[dict[str, object]], rows_b: Sequence[dict[str, object]],
               metric: Callable[[Sequence[dict[str, object]]], float], *, n_boot: int,
               seed: int) -> dict[str, float]:
    paired = [{"cluster_id": str(a["cluster_id"]), "a": a, "b": b}
              for a, b in zip(rows_a, rows_b)]

    def delta(sample):
        return float(metric([row["a"] for row in sample])
                     - metric([row["b"] for row in sample]))

    interval = clustered_bootstrap_ci(
        paired, [str(row["cluster_id"]) for row in paired], delta,
        n_boot=n_boot, seed=seed,
    )
    return {"estimate": interval.estimate, "low": interval.low, "high": interval.high}


def _metrics() -> dict[str, Callable[[Sequence[dict[str, object]]], float]]:
    from sklearn.metrics import recall_score, roc_auc_score

    def shared(rows):
        return [row for row in rows if str(row["label"]) != "scc"]

    def recall_at_external_specificity_90(rows):
        point = _threshold_at_specificity(
            [bool(row["actionable_true"]) for row in rows],
            [float(row["malignant_score"]) for row in rows], 0.90,
        )
        return float(point["recall"])

    return {
        "diagnosis_accuracy": lambda rows: sum(
            bool(row["diagnosis_correct"]) for row in shared(rows)
        ) / len(shared(rows)),
        "diagnosis_macro_recall": lambda rows: float(recall_score(
            [str(row["label"]) for row in shared(rows)],
            [str(row["prediction"]) for row in shared(rows)],
            labels=list(SHARED_CLASSES), average="macro", zero_division=0,
        )),
        "actionable_auc": lambda rows: float(roc_auc_score(
            [bool(row["actionable_true"]) for row in rows],
            [float(row["malignant_score"]) for row in rows],
        )),
        "actionable_recall_at_frozen_threshold": lambda rows: (
            sum(bool(row["actionable_prediction"]) for row in rows
                if bool(row["actionable_true"]))
            / sum(bool(row["actionable_true"]) for row in rows)
        ),
        "actionable_recall_at_external_specificity_90_curve": recall_at_external_specificity_90,
        "specificity_at_frozen_threshold": lambda rows: (
            sum(not bool(row["actionable_prediction"]) for row in rows
                if not bool(row["actionable_true"]))
            / sum(not bool(row["actionable_true"]) for row in rows)
        ),
        "scc_recall_at_frozen_threshold": lambda rows: (
            sum(bool(row["actionable_prediction"]) for row in rows if row["label"] == "scc")
            / sum(row["label"] == "scc" for row in rows)
        ),
    }


def compare(keys: Sequence[str], *, n_boot: int = 2000, output: Path = OUTPUT) -> dict[str, object]:
    if len(set(keys)) != len(keys) or len(keys) < 2:
        raise ValueError("provide at least two unique external candidates")
    loaded = {key: _load_model(key) for key in keys}
    manifest_hashes = {str(result["external_manifest_sha256"]) for result, _ in loaded.values()}
    if len(manifest_hashes) != 1:
        raise ValueError("external candidates use different frozen manifests")
    aligned = _align({key: rows for key, (_, rows) in loaded.items()})
    metrics = _metrics()
    comparisons: dict[str, object] = {}
    delong_family: dict[str, float] = {}
    diagnosis_family: dict[str, float] = {}
    actionable_family: dict[str, float] = {}
    for pair_index, (a, b) in enumerate(combinations(sorted(keys), 2)):
        rows_a, rows_b = aligned[a], aligned[b]
        name = f"{a}_minus_{b}"
        intervals = {
            metric_name: _paired_ci(rows_a, rows_b, metric, n_boot=n_boot,
                                    seed=20260713 + pair_index * 100 + metric_index)
            for metric_index, (metric_name, metric) in enumerate(metrics.items())
        }
        delong = delong_auc_test(
            [bool(row["actionable_true"]) for row in rows_a],
            [float(row["malignant_score"]) for row in rows_a],
            [float(row["malignant_score"]) for row in rows_b],
        )
        diagnosis_a = [bool(row["diagnosis_correct"]) for row in rows_a if row["label"] != "scc"]
        diagnosis_b = [bool(row["diagnosis_correct"]) for row in rows_b if row["label"] != "scc"]
        diagnosis = mcnemar_exact(diagnosis_a, diagnosis_b)
        binary = mcnemar_exact(
            [bool(row["actionable_prediction"]) == bool(row["actionable_true"]) for row in rows_a],
            [bool(row["actionable_prediction"]) == bool(row["actionable_true"]) for row in rows_b],
        )
        delong_family[name] = float(delong["p_value"])
        diagnosis_family[name] = float(diagnosis[2])
        actionable_family[name] = float(binary[2])
        comparisons[name] = {
            "direction": "positive estimates favor the first named candidate",
            "paired_lesion_bootstrap": intervals,
            "delong_actionable_auc": delong,
            "mcnemar_diagnosis_correct": {
                "a_only_correct": diagnosis[0], "b_only_correct": diagnosis[1],
                "p_value": diagnosis[2],
            },
            "mcnemar_actionable_operating_point_correct": {
                "a_only_correct": binary[0], "b_only_correct": binary[1],
                "p_value": binary[2],
            },
        }
    adjusted = {
        "delong_actionable_auc": holm_adjust(delong_family),
        "mcnemar_diagnosis_correct": holm_adjust(diagnosis_family),
        "mcnemar_actionable_operating_point_correct": holm_adjust(actionable_family),
    }
    for family, values in adjusted.items():
        for name, value in values.items():
            comparisons[name].setdefault("holm_adjusted_p", {})[family] = value
    result = {
        "protocol": "bcn20000-frozen-ham-encoder-paired-comparison-v1",
        "external_manifest_sha256": next(iter(manifest_hashes)),
        "bootstrap_replicates": n_boot,
        "models": list(keys),
        "n_lesions": len(next(iter(aligned.values()))),
        "threshold_policy": "each model uses its HAM calibration specificity-90 threshold; no BCN tuning",
        "matched_specificity_policy": (
            "recall at external specificity 0.90 is a descriptive ROC ordinate recomputed "
            "inside each bootstrap sample and is never used as a deployment threshold"
        ),
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(compare(args.models, n_boot=args.bootstrap), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
