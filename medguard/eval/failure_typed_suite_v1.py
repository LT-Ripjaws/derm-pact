"""F1-F4 failure-typed development suite: routing appropriateness, bounds, recovery.

For each frozen development lesion we inject ONE known failure of each type into the
evidence contract, then exercise the deterministic remediation policy/loop:

  F1 sparse/missing evidence          -> MISSING_EVIDENCE      -> EXPAND
  F2 query/evidence misalignment      -> MISALIGNED_EVIDENCE   -> REFORMULATE
  F3 source contradiction             -> SOURCE_CONTRADICTION  -> INDEPENDENT_SOURCE
  F4 image-evidence conflict          -> IMAGE_EVIDENCE_CONFLICT -> ABSTAIN

We measure three mechanism properties per stratum:
  * routing appropriateness -- does the policy pick the correct action for the
    detected failure type (and does verification detect the injected type)?
  * bounded safe termination -- with tools that cannot repair, does the loop take
    at most the budgeted action(s) and then ABSTAIN (never loop, never free-PASS)?
  * recovery -- with tools that repair, does the loop reach PASS at bounded cost?

These are CONSTRUCTED failures. The suite measures the remediation mechanism's
routing/bounds/recovery, NOT how often such failures arise naturally. It is
host-deterministic: zero paid calls, zero model load. ``run_failure_suite`` (below)
measures recovery with fixture repair tools that always succeed by construction --
useful for the routing/bounded-termination properties, but not for "can the system
actually find grounding evidence." ``run_failure_suite_real_retrieval`` is the
separate, heavier pass: same injected failures, but EXPAND/REFORMULATE/
INDEPENDENT_SOURCE are backed by LIVE re-retrieval over the frozen indexes (reusing
matched_full_h2h_run's real contract builder), so recovery can genuinely fail --
that is the point. Still host-deterministic and zero paid calls (retrieval only, no
generation). The injection specification below is the frozen manifest -- freeze it
(hash) before reading any method output.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from medguard.evidence import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    EvidenceContract,
    EvidenceItem,
    FindingState,
    ImageSupport,
    RelationOrigin,
    verify_contract,
)
from medguard.remediation import Action, DeterministicRemediationPolicy, run_remediation


# --- frozen injection specification (the manifest) ---------------------------------
STRATUM_EXPECTED_ACTION: Dict[str, Action] = {
    "F1_sparse_missing_evidence": Action.EXPAND,
    "F2_query_evidence_misalignment": Action.REFORMULATE,
    "F3_source_contradiction": Action.INDEPENDENT_SOURCE,
    "F4_image_evidence_conflict": Action.ABSTAIN,
}
STRATA: Tuple[str, ...] = tuple(STRATUM_EXPECTED_ACTION)

_GUIDELINE_TEXT = "Prompt specialist assessment is indicated for a suspicious pigmented lesion."


def _guideline_evidence() -> List[EvidenceItem]:
    return [EvidenceItem(
        evidence_id="e1", text=_GUIDELINE_TEXT, source_id="NICE-NG12",
        source_version="2025", authority="professional_guideline",
        retrieval_path=("text_dense", "guideline"),
        content_sha256=hashlib.sha256(_GUIDELINE_TEXT.encode()).hexdigest(),
        provenance_uri="https://www.nice.org.uk/guidance/ng12",
        eligible_claim_types=("diagnosis", "assessment", "referral"))]


def _source_claim(*, entailment: EntailmentState, contradiction: bool,
                  with_evidence: bool) -> ClinicalClaim:
    """A source-bound ASSESSMENT claim in a given (failure) relation state."""
    return ClinicalClaim(
        claim_id="s1", claim_type=ClaimType.ASSESSMENT, text="Prompt specialist assessment.",
        image_support=ImageSupport.NOT_APPLICABLE, target_concept="",
        target_state=FindingState.PRESENT, image_finding_ids=(),
        evidence_ids=("e1",) if with_evidence else (), entailment=entailment,
        contradiction=contradiction, evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
        evidence_relation_assessor_id="failure-suite-fixture",
        evidence_relation_receipt_sha256="4" * 64)


def _image_conflict_claim(committed: str, finding_id: str) -> ClinicalClaim:
    """A DIAGNOSIS claim asserting a concept the IMAGE finding contradicts.

    The claim proposes a diagnosis DIFFERENT from the frozen image commitment (an
    evidence-suggested read the image rules out), with image_support=CONTRADICTS.
    Its source relation is otherwise clean (evidence present, entailment SUPPORTED,
    host-deterministic) so the SOLE detected failure is the image-evidence conflict,
    which must route to ABSTAIN rather than be masked by a missing-evidence flag.
    """
    contrast = "nevus" if committed != "nevus" else "melanoma"
    return ClinicalClaim(
        claim_id="d1", claim_type=ClaimType.DIAGNOSIS,
        text=f"Retrieved evidence suggests {contrast}, which the image contradicts.",
        image_support=ImageSupport.CONTRADICTS, target_concept=contrast,
        target_state=FindingState.PRESENT, image_finding_ids=(finding_id,),
        evidence_ids=("e1",), entailment=EntailmentState.SUPPORTED,
        evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
        evidence_relation_assessor_id="failure-suite-fixture",
        evidence_relation_receipt_sha256="4" * 64)


def build_failure_contract(binding: Any, stratum: str, *, case_id: str) -> EvidenceContract:
    """Construct a single-claim contract carrying the injected failure for a lesion."""
    diagnosis = binding.commitment.concept_code
    finding_id = binding.findings[0].finding_id
    if stratum == "F1_sparse_missing_evidence":
        claims, evidence = [_source_claim(
            entailment=EntailmentState.INSUFFICIENT, contradiction=False,
            with_evidence=False)], []
    elif stratum == "F2_query_evidence_misalignment":
        claims, evidence = [_source_claim(
            entailment=EntailmentState.INSUFFICIENT, contradiction=False,
            with_evidence=True)], _guideline_evidence()
    elif stratum == "F3_source_contradiction":
        claims, evidence = [_source_claim(
            entailment=EntailmentState.CONTRADICTED, contradiction=True,
            with_evidence=True)], _guideline_evidence()
    elif stratum == "F4_image_evidence_conflict":
        claims, evidence = [_image_conflict_claim(diagnosis, finding_id)], _guideline_evidence()
    else:
        raise ValueError(f"unknown stratum {stratum!r}")
    return EvidenceContract(
        case_id=case_id, modality="dermoscopy", clinical_question="What is the diagnosis?",
        perception=binding, claims=claims, evidence=evidence)


# --- repair tools: one that recovers, one that cannot (for bounded termination) ----
class _FixingTools:
    """Repairs the injected failure once, so the loop can reach PASS."""

    def __init__(self):
        self.calls: List[str] = []

    def _repair(self, contract: EvidenceContract, name: str) -> EvidenceContract:
        self.calls.append(name)
        value = copy.deepcopy(contract)
        old = value.claims[0]
        image_bound = old.claim_type in {ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL}
        value.claims[0] = ClinicalClaim(
            claim_id=old.claim_id, claim_type=old.claim_type, text=old.text,
            image_support=(ImageSupport.SUPPORTS if image_bound else ImageSupport.NOT_APPLICABLE),
            target_concept=old.target_concept, target_state=old.target_state,
            image_finding_ids=old.image_finding_ids, evidence_ids=("e1",),
            entailment=EntailmentState.SUPPORTED,
            evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
            evidence_relation_assessor_id="failure-suite-fixture",
            evidence_relation_receipt_sha256="4" * 64)
        if not value.evidence:
            value.evidence = _guideline_evidence()
        return value

    def expand(self, c): return self._repair(c, "expand")
    def reformulate(self, c): return self._repair(c, "reformulate")
    def independent_source(self, c): return self._repair(c, "independent_source")


class _NoOpTools(_FixingTools):
    """Cannot repair -> exercises bounded safe termination (ABSTAIN, never loop)."""

    def _repair(self, contract: EvidenceContract, name: str) -> EvidenceContract:
        self.calls.append(name)
        return copy.deepcopy(contract)


# --- suite ------------------------------------------------------------------------
def score_lesion(binding: Any, case_id: str, stratum: str, *,
                 max_steps: int = 4) -> Dict[str, Any]:
    """Routing + bounded-termination + recovery outcome for one lesion x stratum."""
    expected = STRATUM_EXPECTED_ACTION[stratum]
    contract = build_failure_contract(binding, stratum, case_id=case_id)
    report = verify_contract(contract)
    routed = DeterministicRemediationPolicy().decide(report, {})

    noop = _NoOpTools()
    bounded = run_remediation(contract, DeterministicRemediationPolicy(), noop, max_steps=max_steps)
    fixing = _FixingTools()
    recovered = run_remediation(contract, DeterministicRemediationPolicy(), fixing, max_steps=max_steps)

    return {
        "case_id": case_id, "stratum": stratum,
        "detected_failures": sorted(f.value for f in report.failures),
        "routed_action": routed.value,
        "expected_action": expected.value,
        "routing_correct": routed is expected,
        "bounded_final_action": bounded.final_action.value,
        "bounded_steps": len(bounded.trajectory),
        "bounded_calls": list(noop.calls),
        "recovered_final_action": recovered.final_action.value,
        "recovered": recovered.final_action is Action.PASS,
        "recovery_calls": list(fixing.calls),
    }


def run_failure_suite(bindings: Mapping[str, Any]) -> Dict[str, Any]:
    """Run all strata over all lesions; return per-lesion rows + per-stratum aggregate."""
    rows: List[Dict[str, Any]] = []
    for case_id in sorted(bindings):
        for stratum in STRATA:
            rows.append(score_lesion(bindings[case_id], case_id, stratum))

    per_stratum: Dict[str, Any] = {}
    for stratum in STRATA:
        srows = [r for r in rows if r["stratum"] == stratum]
        n = len(srows)
        per_stratum[stratum] = {
            "n": n,
            "expected_action": STRATUM_EXPECTED_ACTION[stratum].value,
            "routing_appropriateness": round(sum(r["routing_correct"] for r in srows) / n, 4) if n else None,
            "bounded_abstain_rate": round(
                sum(r["bounded_final_action"] == Action.ABSTAIN.value for r in srows) / n, 4) if n else None,
            "max_bounded_steps": max((r["bounded_steps"] for r in srows), default=0),
            "recovery_rate_fixture_tools": round(sum(r["recovered"] for r in srows) / n, 4) if n else None,
            "recovery_steps_max": max((len(r["recovery_calls"]) for r in srows), default=0),
        }
    return {
        "schema_version": "medguard-failure-typed-suite-v1",
        "construction": "injected (not natural) failures; measures routing/bounds/recovery",
        "manifest_sha256": manifest_sha256(),
        "strata": {s: STRATUM_EXPECTED_ACTION[s].value for s in STRATA},
        "lesions": len(bindings),
        "per_stratum": per_stratum,
        "rows": rows,
    }


def score_lesion_real_retrieval(
    case: Any, deps: Any, stratum: str, *, max_steps: int = 4,
    image_exemplar_retrieval: bool = True, graph_lineage: bool = True,
) -> Dict[str, Any]:
    """Routing + genuine recovery for one lesion x stratum, using REAL retrieval.

    Same injected failure and routing check as ``score_lesion``; the difference is
    the repair tool. Instead of the fixture's always-succeeds relabel, EXPAND/
    REFORMULATE/INDEPENDENT_SOURCE here rebuild the contract from the SAME live
    frozen-index retrieval the H2H runner's method arm uses
    (``matched_full_h2h_run._RunContractBuilder``/``_RunRemediationTools``), so
    "recovered" means the retrieval system actually found evidence that satisfies
    verification for this lesion -- not that the tool was scripted to satisfy it.
    ``case`` is an ``H2HRunCase`` (or anything with ``.case_id``/``.perception`` and
    what the builder needs); ``deps`` an ``H2HRunDependencies`` wired to the real
    frozen text/image/guideline retrievers. Zero paid calls: retrieval only, no
    generation. F4 (image-evidence conflict) is unaffected -- the policy routes it
    straight to ABSTAIN and never calls a repair tool, fixture or real.
    """
    from medguard.eval.matched_full_h2h_run import _RunContractBuilder, _RunRemediationTools

    expected = STRATUM_EXPECTED_ACTION[stratum]
    contract = build_failure_contract(case.perception, stratum, case_id=case.case_id)
    report = verify_contract(contract)
    routed = DeterministicRemediationPolicy().decide(report, {})

    builder = _RunContractBuilder(
        case, deps, image_exemplar_retrieval=image_exemplar_retrieval,
        graph_lineage=graph_lineage,
    )
    tools = _RunRemediationTools(builder)
    recovered = run_remediation(
        contract, DeterministicRemediationPolicy(), tools, max_steps=max_steps)

    return {
        "case_id": case.case_id, "stratum": stratum,
        "detected_failures": sorted(f.value for f in report.failures),
        "routed_action": routed.value,
        "expected_action": expected.value,
        "routing_correct": routed is expected,
        "recovered_final_action": recovered.final_action.value,
        "recovered": recovered.final_action is Action.PASS,
        "recovery_calls": list(tools.applied),
        "final_evidence_document_ids": [
            item.document_id for item in recovered.contract.evidence],
    }


def run_failure_suite_real_retrieval(
    cases: Sequence[Any], deps: Any, *, max_steps: int = 4,
) -> Dict[str, Any]:
    """Run all strata over real dev lesions with live re-retrieval; genuine recovery RATE.

    A separate, heavier pass from ``run_failure_suite`` (see the module docstring):
    kept side by side rather than replacing it, so the fixture-based routing/bounded-
    termination suite -- and its frozen manifest hash / cached report -- stay
    unchanged. This is the "real EXPAND/REFORMULATE/INDEPENDENT_SOURCE over the
    frozen indexes" pass: host-deterministic (retrieval only), zero paid calls.
    """
    rows: List[Dict[str, Any]] = []
    for case in cases:
        for stratum in STRATA:
            rows.append(score_lesion_real_retrieval(case, deps, stratum, max_steps=max_steps))

    per_stratum: Dict[str, Any] = {}
    for stratum in STRATA:
        srows = [r for r in rows if r["stratum"] == stratum]
        n = len(srows)
        per_stratum[stratum] = {
            "n": n,
            "expected_action": STRATUM_EXPECTED_ACTION[stratum].value,
            "routing_appropriateness": (
                round(sum(r["routing_correct"] for r in srows) / n, 4) if n else None),
            "recovery_rate_real_retrieval": (
                round(sum(r["recovered"] for r in srows) / n, 4) if n else None),
        }
    return {
        "schema_version": "medguard-failure-typed-suite-real-retrieval-v1",
        "construction": (
            "injected (not natural) failures, repaired by LIVE re-retrieval over the "
            "frozen indexes; measures genuine recovery, not routing/bounds (see "
            "run_failure_suite for those, which stay fixture-based)"
        ),
        "manifest_sha256": manifest_sha256(),
        "strata": {s: STRATUM_EXPECTED_ACTION[s].value for s in STRATA},
        "lesions": len(cases),
        "per_stratum": per_stratum,
        "rows": rows,
    }


def manifest_sha256() -> str:
    """Hash of the frozen injection specification (strata + expected routing + evidence)."""
    spec = {
        "strata": {s: STRATUM_EXPECTED_ACTION[s].value for s in STRATA},
        "guideline_text": _GUIDELINE_TEXT,
        "injection": {
            "F1_sparse_missing_evidence": "assessment claim, entailment=insufficient, no evidence",
            "F2_query_evidence_misalignment": "assessment claim, entailment=insufficient, with evidence",
            "F3_source_contradiction": "assessment claim, entailment=contradicted, contradiction=true, with evidence",
            "F4_image_evidence_conflict": "image-bound diagnosis claim, image_support=contradicts",
        },
    }
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
