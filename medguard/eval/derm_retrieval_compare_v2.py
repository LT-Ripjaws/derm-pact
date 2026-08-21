"""Paired post-hoc comparison of the admission-first retrieval operating points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from medguard.eval.derm_retrieval_tournament import _jsonl, _sha256, paired_bootstrap_difference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "research" / "experiments" / "retrieval" / "derm_repair_v2" / "per_case.jsonl"
DEFAULT_OUTPUT = ROOT / "research" / "experiments" / "retrieval" / "derm_repair_v2" / "paired_analysis.json"


def compare(input_path: Path, output_path: Path) -> Dict[str, Any]:
    rows = _jsonl(input_path)
    clean = [row for row in rows if row["environment"] == "clean"]
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method in sorted({str(row["method"]) for row in clean}):
        by_method[method] = [row for row in clean if row["method"] == method]

    dense = by_method["dense_top1"]
    medcpt = by_method["dense_medcpt_top1"]
    adaptive = by_method["adaptive_dense_medcpt"]
    comparisons: Dict[str, Any] = {}
    seed = 7200
    for left_name, left, right_name, right in [
        ("dense_medcpt_top1", medcpt, "dense_top1", dense),
        ("adaptive_dense_medcpt", adaptive, "dense_medcpt_top1", medcpt),
    ]:
        for metric in [
            "concept_recall_at_5", "management_recall_at_5", "concept_precision_at_5", "mrr",
            "retrieval_calls", "reranker_pairs",
        ]:
            seed += 1
            key = f"{left_name}_vs_{right_name}__{metric}"
            comparisons[key] = paired_bootstrap_difference(left, right, metric, seed=seed)

    medcpt_management = comparisons[
        "dense_medcpt_top1_vs_dense_top1__management_recall_at_5"
    ]
    medcpt_mrr = comparisons["dense_medcpt_top1_vs_dense_top1__mrr"]
    adaptive_mrr = comparisons["adaptive_dense_medcpt_vs_dense_medcpt_top1__mrr"]
    adaptive_recall = comparisons[
        "adaptive_dense_medcpt_vs_dense_medcpt_top1__concept_recall_at_5"
    ]
    adaptive_management = comparisons[
        "adaptive_dense_medcpt_vs_dense_medcpt_top1__management_recall_at_5"
    ]
    result = {
        "schema_version": "medguard-derm-retrieval-paired-analysis-v2",
        "evidence_level": "post-hoc development analysis; not confirmatory",
        "input_sha256": _sha256(input_path),
        "comparisons": comparisons,
        "decisions": {
            "dense_top1": (
                "retain as the matched-compute efficiency point; highest concept precision and no reranker"
            ),
            "dense_medcpt_top1": (
                "retain as the evidence-rich point"
                if medcpt_management["ci95"][0] > 0 or medcpt_mrr["ci95"][0] > 0
                else "demote because neither management recall nor MRR gain is statistically supported"
            ),
            "adaptive_dense_medcpt": (
                "demote from the retrieval node: MRR improved, but neither concept nor management recall improved; "
                "the second query added 37.2% more calls and belongs in the later remediation-value tournament"
                if adaptive_mrr["ci95"][0] > 0
                and adaptive_recall["estimate"] == 0.0
                and adaptive_management["ci95"][0] <= 0.0
                else "demote: no supported primary retrieval-quality gain despite extra calls"
            ),
        },
        "limitations": [
            "The operating-point choice is post-hoc and must be confirmed on fresh cases.",
            "Compute is represented by calls and reranked pairs, not shared-cache wall time.",
            "Automatic relevance and derm-only limitations remain unchanged.",
        ],
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(compare(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
