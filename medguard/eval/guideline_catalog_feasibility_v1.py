"""Offline feasibility probe for the source-exact guideline catalog.

Answers one question before the authoritative-guideline retrieval channel is
built: can the 23 frozen EADO recommendations actually serve the shared clinical
task, and for which lesions?

This exists because the alternative is discovering the answer after wiring, from
a citation-precision number that looks fine for the wrong reason. Two failure
modes are specifically checked:

1.  Over-reach. If recommendations surface for every lesion regardless of the
    committed condition, any citation metric is measuring nothing and the
    contract binds statements that are true but irrelevant.
2.  Under-service. If the catalog cannot back the claim type the clinical
    question actually asks for, a PASS-eligible contract would be certifying an
    obligation the evidence never discharged.

Runs offline: no generation, no network, no sealed-test access, no truth fields
from the manifest. Lesion class comes from the frozen perception commitment --
the same thing the runtime sees -- not from the label.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_CORPUS = ROOT / "medguard/data/corpus/derm_guideline_recommendations_v1.jsonl"
PERCEPTION = ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1/predictions.jsonl"
REPORT = ROOT / "research/experiments/medguard_h2h_v1/guideline_catalog_feasibility_v1.json"

# The committed diagnosis vocabulary is HAM7; the catalog is keyed by guideline
# condition. Imported rather than redefined: keeping a second copy here silently
# desynchronised this probe from the runtime when actinic keratosis was added.
from medguard.pipeline.guideline_retriever import (  # noqa: E402
    COMMITMENT_TO_CONDITION,
)

# The shared clinical question -- "What diagnosis is most consistent with the
# dermoscopic images, and what evidence supports that assessment?" -- is a
# diagnosis question. An external claim answering it must be diagnosis-eligible.
TASK_CLAIM_TYPE = "diagnosis"

# A claim about what a lesion looks like needs a record that speaks to visual
# assessment. Management text conditioned on a diagnosis cannot establish it.
VISUAL_CLAIM_TYPES = ("assessment", "visual_finding", "differential")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def condition_scoped_candidates(
    committed_diagnosis: str,
    recommendations: Sequence[Mapping[str, Any]],
    claim_type: str | None = None,
) -> List[Mapping[str, Any]]:
    """Recommendations admissible for a committed diagnosis.

    Scoping is keyed on the image-bound commitment, never on the true label.
    A record is admissible only if its condition matches the commitment and, when
    a claim_type is supplied, the record declares that type eligible.
    """
    condition = COMMITMENT_TO_CONDITION.get(committed_diagnosis)
    if condition is None:
        return []
    candidates = [row for row in recommendations if row["condition"] == condition]
    if claim_type is not None:
        candidates = [
            row for row in candidates if claim_type in row["eligible_claim_types"]
        ]
    return candidates


def summarize(
    recommendations: Sequence[Mapping[str, Any]],
    commitments: Mapping[str, str],
) -> Dict[str, Any]:
    by_condition = collections.Counter(row["condition"] for row in recommendations)

    any_reach: Dict[str, int] = collections.Counter()
    task_reach: Dict[str, int] = collections.Counter()
    visual_reach: Dict[str, int] = collections.Counter()
    for diagnosis in commitments.values():
        any_hits = condition_scoped_candidates(diagnosis, recommendations)
        task_hits = condition_scoped_candidates(diagnosis, recommendations, TASK_CLAIM_TYPE)
        visual_hits = [
            row
            for row in any_hits
            if any(kind in row["eligible_claim_types"] for kind in VISUAL_CLAIM_TYPES)
        ]
        any_reach[diagnosis] += 1 if any_hits else 0
        task_reach[diagnosis] += 1 if task_hits else 0
        visual_reach[diagnosis] += 1 if visual_hits else 0

    committed_counts = collections.Counter(commitments.values())
    total = len(commitments)

    per_condition_claims = {}
    for condition in sorted(by_condition):
        rows = [row for row in recommendations if row["condition"] == condition]
        per_condition_claims[condition] = {
            "records": len(rows),
            "diagnosis_eligible": sum(1 for r in rows if "diagnosis" in r["eligible_claim_types"]),
            "assessment_eligible": sum(1 for r in rows if "assessment" in r["eligible_claim_types"]),
            "visual_finding_eligible": sum(1 for r in rows if "visual_finding" in r["eligible_claim_types"]),
            "management_eligible": sum(1 for r in rows if "management" in r["eligible_claim_types"]),
        }

    return {
        "catalog": {
            "records": len(recommendations),
            "by_condition": dict(by_condition),
            "per_condition_claim_eligibility": per_condition_claims,
        },
        "lesions": {
            "total": total,
            "by_committed_diagnosis": dict(committed_counts),
        },
        "reachability": {
            "any_condition_matched_record": sum(any_reach.values()),
            "task_claim_type_backable": sum(task_reach.values()),
            "visual_claim_backable": sum(visual_reach.values()),
            "zero_condition_matched_record": total - sum(any_reach.values()),
            "by_committed_diagnosis": {
                diagnosis: {
                    "lesions": committed_counts[diagnosis],
                    "condition_matched_records": len(
                        condition_scoped_candidates(diagnosis, recommendations)
                    ),
                    "diagnosis_eligible_records": len(
                        condition_scoped_candidates(diagnosis, recommendations, TASK_CLAIM_TYPE)
                    ),
                }
                for diagnosis in sorted(committed_counts)
            },
        },
    }


def build_report() -> Dict[str, Any]:
    recommendations = _load_jsonl(RECOMMENDATION_CORPUS)
    perception = _load_jsonl(PERCEPTION)
    for row in perception:
        if "label" in row or "benign_malignant" in row:
            raise ValueError("perception artifact carries a truth field; refusing to proceed")
    commitments = {row["cluster_id"]: row["committed_diagnosis"] for row in perception}

    summary = summarize(recommendations, commitments)
    reach = summary["reachability"]
    total = summary["lesions"]["total"]

    return {
        "schema_version": "medguard-guideline-catalog-feasibility-v1",
        "status": "offline_feasibility_probe_not_a_paper_result",
        "question": (
            "Can the 23 frozen EADO recommendations serve the shared clinical task, "
            "and for which lesions?"
        ),
        "inputs": {
            "recommendation_corpus": {
                "path": str(RECOMMENDATION_CORPUS.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(RECOMMENDATION_CORPUS),
            },
            "frozen_perception": {
                "path": str(PERCEPTION.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(PERCEPTION),
            },
        },
        "boundaries": {
            "generation_calls": 0,
            "network_access": False,
            "sealed_test_access": False,
            "truth_fields_consumed": False,
            "scoping_key": "frozen image-bound committed diagnosis, never the manifest label",
        },
        "summary": summary,
        "findings": [
            {
                "id": "F1",
                "finding": (
                    f"{reach['any_condition_matched_record']}/{total} lesions have at least one "
                    f"condition-matched recommendation; {reach['zero_condition_matched_record']}/{total} have none."
                ),
                "consequence": (
                    "The claim-scoped output obligation is load-bearing, not a convenience. Without it "
                    "the majority of lesions abstain because the corpus lacks their condition, not "
                    "because the evidence process judged anything. That would make coverage a "
                    "measurement of the catalog's holes."
                ),
            },
            {
                "id": "F2",
                "finding": (
                    "Condition scoping is keyed on the committed diagnosis, so a citation can only ever "
                    "name a record of the committed condition. Precision measured against condition is "
                    "therefore 1.0 by construction."
                ),
                "consequence": (
                    "citation_precision as currently pre-registered would measure nothing. It must be "
                    "redefined as citation integrity -- the cited id resolves to the exact source, "
                    "version, citation and content hash, and was in the offered candidate set -- which "
                    "is checkable and non-trivial, because a generator can hallucinate an id, cite an "
                    "unoffered one, or attach a real id to an unrelated claim. It must NOT be compared "
                    "to MedGraphRAG's human-judged citation precision of 92."
                ),
            },
            {
                "id": "F3",
                "finding": (
                    "Melanoma now has method-licensure records after ingesting EADO Part 1 (Diagnostics): "
                    "4 assessment/visual_finding records -- dermatoscopy (Part-1 recommendation-4, "
                    "'Dermatoscopy shall be used for the assessment of pigmented and non-pigmented skin "
                    "lesions', Grade A), whole-body photography, digital dermatoscopy, and reflectance "
                    "confocal microscopy. Before Part 1 it had zero, and only management text that "
                    "presupposed the diagnosis. Coverage is now symmetric with BCC."
                ),
                "consequence": (
                    "Even with these records, no catalog entry supports the claim that a patient's lesion "
                    "IS melanoma: each is a claim about which diagnostic METHOD is guideline-sanctioned, "
                    "not about this patient. The source-exact route makes the binding exact without "
                    "turning generic text into patient-diagnosis proof, so the C1 image-bound boundary "
                    "holds uniformly across both conditions -- the added records strengthen provenance "
                    "coverage without weakening the separation the contract enforces."
                ),
            },
            {
                "id": "F4",
                "finding": (
                    "The visual records of BCC and melanoma are all method claims, and they carry a genuine "
                    "clinical CONTRAST the arm can surface: BCC box-1 (Grade C) licenses committing a BCC "
                    "diagnosis 'based on clinical examination ... without the need for histological "
                    "confirmation', whereas melanoma Part-1 recommendation-3 (GCP) states 'If a melanoma "
                    "is clinically suspected, it shall be confirmed by histopathology'."
                ),
                "consequence": (
                    "The catalog's honest roles are method licensure -- is this diagnostic procedure "
                    "guideline-sanctioned for this condition -- and conditional management. The BCC/melanoma "
                    "confirmation contrast is a real, citable, condition-specific difference that a "
                    "provenanced arm can report and a naive one cannot. Neither role is patient-diagnosis "
                    "proof."
                ),
            },
            {
                "id": "F5",
                "finding": (
                    "The shared clinical question asks 'what evidence supports that assessment?', which "
                    "invites precisely the conflation the contract forbids: source text does not support "
                    "the assessment of this patient."
                ),
                "consequence": (
                    "The prompt is a protocol artifact and no generation has run, so it is still "
                    "changeable. Left as written, a truthful arm is penalised for declining to do what "
                    "the prompt asks, and an untruthful arm is rewarded for conflating method evidence "
                    "with patient evidence. This is a design decision, not a defect to patch silently."
                ),
            },
        ],
        "verdict": (
            "The catalog is usable but narrow, and its honest scope is not the scope the current prompt "
            "implies. Build the channel: it supplies exact provenance for method and management claims "
            "on 63 of 160 lesions, resolves INDEPENDENT_SOURCE's missing separate index, and gives the "
            "graph two authority tiers to select across. Do not describe it as evidence that a source "
            "proves a patient's diagnosis, and do not report a relevance-style citation precision from it."
        ),
    }


def main() -> int:
    report = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    REPORT.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (REPORT.parent / (REPORT.name + ".sha256")).write_text(digest + "\n", encoding="ascii", newline="\n")

    reach = report["summary"]["reachability"]
    total = report["summary"]["lesions"]["total"]
    print(f"catalog: {report['summary']['catalog']['records']} recommendations "
          f"{report['summary']['catalog']['by_condition']}")
    print(f"lesions with >=1 condition-matched record : {reach['any_condition_matched_record']}/{total}")
    print(f"lesions where the task claim is backable  : {reach['task_claim_type_backable']}/{total}")
    print(f"lesions where a visual claim is backable  : {reach['visual_claim_backable']}/{total}")
    for row in report["findings"]:
        print(f"  [{row['id']}] {row['finding'][:96]}")
    print(f"report: {REPORT.relative_to(ROOT)}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
