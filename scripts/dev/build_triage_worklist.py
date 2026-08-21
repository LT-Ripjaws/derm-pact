"""Order the sealed valid-split contracts into a clinician triage worklist.

Reads the already-built decision contracts (research/evidence/derm7pt_v1/cdss_contracts_valid.json
-- each entry there is a serialised DecisionContract, produced once by the offline pipeline; no
model or classifier runs here). For each one, rebuilds a real DecisionContract via build_contract
(deterministic given case_id/diagnosis/confidence/criteria, so this reproduces the original
exactly) purely so the SAME headline builder the clinician-facing renderer uses
(evidence.render.render_headline) produces the worklist's one-line summary -- no second copy of
that logic lives here.

Ordering is a strict 4-tier partition, matching evidence.contract._build_conflict's own branching
exactly (undecidable is checked first there, then explicit conflict, then agreement) rather than
an independent guess at what "urgent" means:

1. escalate_conflict   -- conflict.present: the two views actively disagree.
2. undecidable         -- not score.decidable: the checklist has not reached a verdict.
3. routine_covered     -- views agree, decidable, and a guideline recommendation is citable.
4. routine_uncovered   -- views agree, decidable, and no guideline recommendation exists.

Within a tier, cases are ordered by seven_point_score.max descending (the higher the worst-case
score, the more melanoma-suspicious the case, even within "routine"), then case_id ascending to
break ties reproducibly. Nothing here depends on wall-clock time, dict iteration order, or any
non-contract input, so the ordering is a pure function of the contracts file.

Output: research/evidence/derm7pt_v1/triage_worklist.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.evidence.render import render_headline

SOURCE = REPO / "research" / "evidence" / "derm7pt_v1" / "cdss_contracts_valid.json"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "triage_worklist.json"

#: Tier order, most urgent first. Matches evidence.contract._build_conflict's own branch order
#: (undecidable checked first, then explicit conflict) -- see this module's docstring.
TIER_ORDER: List[str] = [
    "escalate_conflict",
    "undecidable",
    "routine_covered",
    "routine_uncovered",
]


def _tier(contract) -> str:
    if contract.conflict.present:
        return "escalate_conflict"
    if not contract.score.decidable:
        return "undecidable"
    if contract.guideline.covered:
        return "routine_covered"
    return "routine_uncovered"


def _rebuild_contract(record: Dict[str, Any], catalog: GuidelineCatalog):
    """Reconstruct the DecisionContract a worklist record was serialised from.

    build_contract is a pure function of (case_id, diagnosis, confidence, criteria, catalog), and
    all four are present in the serialised record, so this reproduces the original exactly --
    including for an INDETERMINATE criterion, whose stored ``value`` round-trips as the literal
    string "indeterminate" (see criteria/schema.py's INDETERMINATE constant).
    """
    criteria = {c["name"]: c["value"] for c in record["criteria"]}
    missing = set(CRITERION_NAMES) - set(criteria)
    if missing:
        raise ValueError(f"case {record['case_id']}: criteria table missing {sorted(missing)}")
    return build_contract(
        case_id=record["case_id"],
        classifier_diagnosis=record["classifier"]["diagnosis"],
        classifier_confidence=record["classifier"]["confidence"],
        criteria=criteria,
        catalog=catalog,
        receipt=record.get("receipt"),
    )


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"source not found: {SOURCE}")

    with SOURCE.open(encoding="utf-8") as handle:
        source_data = json.load(handle)

    catalog = GuidelineCatalog.load()
    records = source_data["contracts"]

    entries: List[Dict[str, Any]] = []
    for record in records:
        contract = _rebuild_contract(record, catalog)
        if not contract.safe_to_display:
            # A contract that fails its own verification must not be triaged as if its fields
            # could be trusted -- surface it, don't order it.
            raise SystemExit(
                f"case {contract.case_id} failed verification and cannot be triaged: "
                f"{contract.verification_failures()}"
            )
        entries.append(
            {
                "case_id": contract.case_id,
                "tier": _tier(contract),
                "score_max": contract.score.score_max,
                "headline": render_headline(contract),
            }
        )

    tier_rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    entries.sort(key=lambda e: (tier_rank[e["tier"]], -e["score_max"], e["case_id"]))

    tier_counts = Counter(e["tier"] for e in entries)
    # Every tier appears in the counts even when empty (e.g. this split has zero undecidable
    # cases), so the published denominator is never silently missing a category.
    tier_counts_ordered = {tier: tier_counts.get(tier, 0) for tier in TIER_ORDER}

    result = {
        "protocol": source_data.get("protocol"),
        "source": SOURCE.name,
        "n_cases": len(entries),
        "tier_order": TIER_ORDER,
        "ordering": (
            "escalate_conflict > undecidable > routine_covered > routine_uncovered; "
            "within a tier by seven_point_score.max desc, then case_id asc"
        ),
        "note": (
            "Ordering is deterministic from contract fields only (conflict.present, "
            "seven_point_score.decidable, guideline.covered, seven_point_score.max, case_id) -- "
            "no model output, wall-clock time, or dict/file iteration order is involved. Re-running "
            "this script against an unchanged source file reproduces byte-identical ordering."
        ),
        "tier_counts": tier_counts_ordered,
        "worklist": entries,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUT}")
    print(f"  {len(entries)} cases")
    for tier in TIER_ORDER:
        print(f"  {tier:20s} {tier_counts_ordered[tier]}")


if __name__ == "__main__":
    main()
