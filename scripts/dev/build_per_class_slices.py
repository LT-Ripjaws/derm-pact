"""Per-class descriptive slices over the frozen sealed H2H outputs (RG-033, part 1).

Reporting only.  This script reads already-written artifacts and computes nothing that
requires the sealed label join, so it opens NO sealed access and changes no sealed number.

What it can compute without truth
---------------------------------
Grouped by the frozen probe's committed class: case counts, selective coverage and
abstention, calibrated-confidence distribution, guideline eligibility, citation volume,
and the terminal-action mix -- for the contract arm and for the ungated A4 comparator.

What it cannot compute without truth
------------------------------------
Per-class ECE, per-class risk-coverage/AURC, and per-class selective accuracy all need
per-case correctness.  No per-case truth is persisted in any written artifact (the scored
report keeps only aggregates and the 7x7 confusion matrix), so those require one logged
sealed label-join access.  They are deliberately left out rather than approximated.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]

PERCEPTION = REPO / "research/experiments/medguard_h2h_v1/perception/dermlip_sealed_v1/predictions.jsonl"
SEALED_H2H = REPO / "research/evidence/sealed_h2h/sealed_h2h_1495x11.json"
CATALOG = REPO / "medguard/data/corpus/derm_guideline_recommendations_v1.jsonl"
OUT_JSON = REPO / "research/evidence/sealed_h2h/per_class_slices_sealed.json"

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc"]
ARMS_REPORTED = ["M", "A4", "B1"]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "p25": round(ordered[len(ordered) // 4], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[(3 * len(ordered)) // 4], 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def build() -> Dict[str, Any]:
    perception = {row["case_id"]: row for row in _read_jsonl(PERCEPTION)}
    sealed = json.loads(SEALED_H2H.read_text(encoding="utf-8"))
    catalog = _read_jsonl(CATALOG)

    cases = sealed["cases"]
    if len(cases) != len(perception):
        raise ValueError(f"case-count mismatch: {len(cases)} sealed vs {len(perception)} perception")

    catalog_conditions = sorted({str(record["condition"]) for record in catalog})

    per_class: Dict[str, Dict[str, Any]] = {}
    for name in CLASS_ORDER:
        per_class[name] = {
            "n": 0,
            "confidence": [],
            "confidence_answered": [],
            "confidence_abstained": [],
            "lesions_with_citations": 0,
            "cited_and_answered": 0,
            "citations_total": 0,
            "actions": Counter(),
            "arm_answered": defaultdict(int),
        }

    for case in cases:
        committed = str(case["committed_diagnosis"])
        if committed not in per_class:
            raise ValueError(f"unexpected committed class {committed!r}")
        bucket = per_class[committed]
        probe = perception[case["case_id"]]

        bucket["n"] += 1
        confidence = float(probe["calibrated_confidence"])
        bucket["confidence"].append(confidence)

        arms = case["arms"]
        contract = arms["M"]
        # Coverage is defined by the terminal action, matching the scorer: M answers 1266
        # of 1495 (0.8468).  `diagnostic_abstention` is a narrower, path-dependent flag
        # (51 for M, 102 for A4) and is deliberately not used here.
        abstained = str(contract["final_action"]) == "ABSTAIN"
        (bucket["confidence_abstained"] if abstained else bucket["confidence_answered"]).append(confidence)
        bucket["actions"][str(contract["final_action"])] += 1

        citations = contract.get("guideline_citations") or []
        if citations:
            bucket["lesions_with_citations"] += 1
            bucket["citations_total"] += len(citations)
            if not abstained:
                bucket["cited_and_answered"] += 1

        for arm_id in ARMS_REPORTED:
            arm = arms.get(arm_id)
            if arm is None:
                continue
            if str(arm.get("final_action")) != "ABSTAIN":
                bucket["arm_answered"][arm_id] += 1

    rows = []
    total = len(cases)
    for name in CLASS_ORDER:
        bucket = per_class[name]
        n = bucket["n"]
        if n == 0:
            continue
        eligible = bucket["lesions_with_citations"] > 0
        row = {
            "committed_class": name,
            "n": n,
            "share_of_cohort": round(n / total, 4),
            "guideline_eligible": eligible,
            "lesions_with_citations": bucket["lesions_with_citations"],
            "citation_coverage": round(bucket["lesions_with_citations"] / n, 4),
            # The practically useful set: an answer was delivered AND it carries a
            # resolvable guideline citation.
            "cited_and_answered": bucket["cited_and_answered"],
            "cited_and_answered_share_of_class": round(bucket["cited_and_answered"] / n, 4),
            "citations_total": bucket["citations_total"],
            "citations_per_cited_lesion": (
                round(bucket["citations_total"] / bucket["lesions_with_citations"], 4)
                if bucket["lesions_with_citations"]
                else 0.0
            ),
            "calibrated_confidence": _quantiles(bucket["confidence"]),
            "calibrated_confidence_answered": _quantiles(bucket["confidence_answered"]),
            "calibrated_confidence_abstained": _quantiles(bucket["confidence_abstained"]),
            "terminal_actions": dict(sorted(bucket["actions"].items())),
            "coverage_by_arm": {
                arm_id: round(bucket["arm_answered"][arm_id] / n, 4) for arm_id in ARMS_REPORTED
            },
            "answered_by_arm": {arm_id: bucket["arm_answered"][arm_id] for arm_id in ARMS_REPORTED},
        }
        # Positive = the verifier gate withholds answers this class would otherwise get.
        row["gate_coverage_cost_vs_a4"] = round(
            row["coverage_by_arm"]["A4"] - row["coverage_by_arm"]["M"], 4
        )
        rows.append(row)

    cohort_answered = {
        arm_id: sum(row["answered_by_arm"][arm_id] for row in rows) for arm_id in ARMS_REPORTED
    }
    reconciliation = {
        "m_answered": cohort_answered["M"],
        "m_coverage": round(cohort_answered["M"] / total, 4),
        "m_coverage_matches_scored_report": cohort_answered["M"] == 1266,
        "a4_coverage": round(cohort_answered["A4"] / total, 4),
        "b1_coverage": round(cohort_answered["B1"] / total, 4),
    }

    eligible_classes = [row["committed_class"] for row in rows if row["guideline_eligible"]]
    eligible_n = sum(row["n"] for row in rows if row["guideline_eligible"])
    cited_n = sum(row["lesions_with_citations"] for row in rows)
    cited_answered_n = sum(row["cited_and_answered"] for row in rows)

    return {
        "schema_version": 1,
        "analysis_id": "sealed_per_class_slices_v1",
        "gap": "RG-033 (part 1 of 2: truth-free slices)",
        "source_artifacts": {
            "perception": str(PERCEPTION.relative_to(REPO)).replace("\\", "/"),
            "sealed_h2h": str(SEALED_H2H.relative_to(REPO)).replace("\\", "/"),
            "guideline_catalog": str(CATALOG.relative_to(REPO)).replace("\\", "/"),
        },
        "boundaries": {
            "sealed_access_opened": False,
            "truth_labels_read": False,
            "generation_calls": 0,
            "sealed_numbers_modified": False,
            "paper_role": "descriptive reporting slice; not an inferential comparison",
        },
        "cohort": {
            "lesions": total,
            "catalog_conditions": catalog_conditions,
            "catalog_records": len(catalog),
            "guideline_eligible_classes": eligible_classes,
            "guideline_eligible_lesions": eligible_n,
            "guideline_eligible_share": round(eligible_n / total, 4),
            "lesions_receiving_citations": cited_n,
            "lesions_receiving_citations_share": round(cited_n / total, 4),
            "cited_and_answered": cited_answered_n,
            "cited_and_answered_share_of_cohort": round(cited_answered_n / total, 4),
            "cited_and_answered_share_of_eligible": (
                round(cited_answered_n / cited_n, 4) if cited_n else 0.0
            ),
        },
        "reconciliation_against_scored_report": reconciliation,
        "per_class": rows,
        "requires_sealed_label_join": [
            "per-class expected calibration error",
            "per-class risk-coverage curve and AURC",
            "per-class selective accuracy and over-reliance",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="persist the report to research/evidence/")
    args = parser.parse_args()

    report = build()
    text = json.dumps(report, indent=2, sort_keys=False)
    if args.write:
        OUT_JSON.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT_JSON.relative_to(REPO)}")
    else:
        print(text)


if __name__ == "__main__":
    main()
