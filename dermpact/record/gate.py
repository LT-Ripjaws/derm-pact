"""Phase 5: independent, fail-closed verification gate (CONTEXT.md S15.6)."""
from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from dermpact.eval.dermpact_phase4_benchmark import CRITERIA, GLOBAL_CLASSES, criterion_values_from_logits
from dermpact.guidelines.rules import RuleCatalogue, canonical_json
from dermpact.guidelines.engine import ApplicabilityEngine

from dermpact.record.case_record import (
    CLINICIAN_FIELD_TO_ATTR,
    CRITERION_NAMES,
    GATE_POLICY_VERSION,
    NEXT_STEP_PRIORITY,
    SCHEMA_VERSION,
    StructuredCaseRecord,
    _sigmoid,
    _softmax,
    effective_dermoscopic_present,
    expected_effective_criterion_values,
    recompute_derived,
    round_floats,
    verify_chain,
)


@dataclass(frozen=True)
class GateFailure:
    code: str
    detail: str


@dataclass(frozen=True)
class GateWarning:
    code: str
    reason: str
    remaining_usable_output: str
    recovery_action: str


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    failures: Tuple[GateFailure, ...]
    disposition: str  # "display" | "partial_report" | "refusal"
    receipt_sha256: str
    policy_version: str
    checked_at: str


@dataclass(frozen=True)
class AllowedClaims:
    """The closed set of (path, value) leaves an output is permitted to state."""

    claims: FrozenSet[Tuple[str, Any]]

    def __contains__(self, item: Any) -> bool:
        try:
            path, value = item
        except (TypeError, ValueError):
            return False
        return (path, value) in self.claims


#: Every displayable section of the three-layer output (template.py's vocabulary).
ALL_SECTIONS: FrozenSet[str] = frozenset({
    "identity", "warnings", "checklist", "next_step", "agreement",
    "criterion_evidence", "rule_evidence", "source_spans", "amendment_history", "gate_receipt", "versions",
})

_REMOVE_ON_FAILURE: Dict[str, FrozenSet[str]] = {
    "integrity": ALL_SECTIONS - {"identity"},
    "entity_mismatch": ALL_SECTIONS - {"identity"},
    "calculation": frozenset({"checklist", "next_step"}),
    "source_version_hash": frozenset({"source_spans", "next_step", "versions"}),
    "source_stale": frozenset({"source_spans", "next_step", "versions"}),
    "rule_path": frozenset({"next_step", "rule_evidence"}),
    "missing_fact_handling": frozenset({"next_step"}),
    "amendment_consistency": frozenset({"amendment_history"}),
    "required_warnings": frozenset(),
    "generated_claim_scope": frozenset({"next_step"}),
    "timeout": frozenset({"next_step", "rule_evidence", "source_spans"}),
}


def partial_report_scope(verdict: GateVerdict, record: StructuredCaseRecord) -> Dict[str, bool]:
    """Which sections remain showable: ``{section: True}`` if it may be shown."""
    if verdict.disposition == "refusal":
        return {s: (s == "identity") for s in ALL_SECTIONS}
    removed: set = set()
    for f in verdict.failures:
        removed |= _REMOVE_ON_FAILURE.get(f.code, frozenset())
    return {s: (s not in removed) for s in ALL_SECTIONS}


def _walk_leaves(prefix: str, value: Any, out: List[Tuple[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _walk_leaves(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _walk_leaves(f"{prefix}[{i}]", v, out)
    else:
        out.append((prefix, value))


def allowed_claims(record: StructuredCaseRecord) -> AllowedClaims:
    """Every leaf (path, value) actually present in the record: the closed claim set."""
    leaves: List[Tuple[str, Any]] = []
    _walk_leaves("", round_floats(record.to_dict()), leaves)
    return AllowedClaims(claims=frozenset(leaves))


def required_warnings(record: StructuredCaseRecord) -> Tuple[GateWarning, ...]:
    """CONTEXT.md S16.5: every warning carries a reason, remaining usable output, and recovery action."""
    warnings: List[GateWarning] = []
    r = record.reliability

    not_assessed = [
        name for name, value in (
            ("iqa_state", r.iqa_state), ("ood_state", r.ood_state),
            ("view_stability_state", r.view_stability_state), ("selective_abstention", r.selective_abstention),
        ) if value == "not_assessed"
    ]
    if not_assessed:
        warnings.append(GateWarning(
            code="reliability_not_assessed",
            reason=f"Phase 3 reliability assessment not performed for: {', '.join(not_assessed)}",
            remaining_usable_output="perception outputs and the checklist remain displayed but are not certified by a reliability assessment",
            recovery_action="interpret with clinician review",
        ))
    if r.iqa_state == "fail":
        warnings.append(GateWarning(
            code="iqa_fail", reason="image quality assessment failed",
            remaining_usable_output="perception-derived outputs are withheld; identity and input hashes remain displayed",
            recovery_action="reacquire a higher-quality dermoscopic image",
        ))
    if r.ood_state == "out_of_distribution":
        warnings.append(GateWarning(
            code="ood_out_of_distribution", reason="the input was flagged out-of-distribution",
            remaining_usable_output="perception outputs remain displayed but are lower-confidence for this input",
            recovery_action="refer for human review",
        ))
    if r.view_stability_state == "unstable":
        warnings.append(GateWarning(
            code="view_unstable", reason="view stability check failed",
            remaining_usable_output="perception outputs remain displayed but may reflect an unstable capture",
            recovery_action="reacquire a stable dermoscopic image",
        ))
    if r.selective_abstention == "abstained" or record.perception.abstained:
        warnings.append(GateWarning(
            code="abstained", reason="the perception model abstained on this case",
            remaining_usable_output="no direct-classifier class is claimed; the checklist path remains available",
            recovery_action="refer for human review",
        ))
    if not record.checklist.decidable:
        warnings.append(GateWarning(
            code="checklist_undecidable",
            reason="one or more checklist criteria are indeterminate and the score range spans the melanoma-suspicion threshold",
            remaining_usable_output="the checklist score range remains displayed; no point verdict is",
            recovery_action="re-read the indeterminate criteria, or resolve via a clinician assertion",
        ))
    if record.agreement.present:
        warnings.append(GateWarning(
            code="direct_checklist_conflict", reason=record.agreement.description,
            remaining_usable_output="both views remain displayed separately; neither is silently resolved",
            recovery_action="clinician review of the disagreement",
        ))
    warnings.append(GateWarning(
        code="uncalibrated_probabilities",
        reason="displayed probabilities are not calibrated (calibration is a Phase 3 deliverable)",
        remaining_usable_output="relative ranking of classes/criteria remains usable; absolute probability values are not",
        recovery_action="treat probabilities as uncalibrated model scores, not calibrated risk",
    ))
    executable_ids = set(NEXT_STEP_PRIORITY)
    unresolved = any(
        row["rule_id"] in executable_ids and row["state"] in ("applicability_unknown", "no_catalogue_coverage")
        for row in record.applicability.get("results", [])
    )
    if unresolved or record.next_step.status in (
        "conditional", "withheld_unknown", "withheld_no_coverage", "withheld_perception_unavailable",
    ):
        warnings.append(GateWarning(
            code="applicability_unknown_or_no_coverage",
            reason="at least one executable guideline rule could not be resolved to applicable/not_applicable",
            remaining_usable_output="the underlying evidence (checklist, criteria, facts) remains displayed",
            recovery_action="answer the targeted question, or refer for clinician review if none is available",
        ))
    if any(not row.get("checks", {}).get("source_current", True) for row in record.applicability.get("results", [])):
        warnings.append(GateWarning(
            code="source_stale", reason="a cited guideline source was not current as of this record's evaluation date",
            remaining_usable_output="non-stale rule outputs remain displayed",
            recovery_action="re-run against a current catalogue",
        ))
    return tuple(warnings)


def _disposition(failures: Sequence[GateFailure]) -> str:
    codes = {f.code for f in failures}
    if "entity_mismatch" in codes or "integrity" in codes:
        return "refusal"
    if failures:
        return "partial_report"
    return "display"


def _receipt_sha256(verdict: GateVerdict, record: StructuredCaseRecord, expected_case_id: str) -> str:
    payload = {
        "policy_version": verdict.policy_version,
        "checked_at": verdict.checked_at,
        "passed": verdict.passed,
        "disposition": verdict.disposition,
        "failures": [{"code": f.code, "detail": f.detail} for f in verdict.failures],
        "record_sha256": record.record_sha256,
        "expected_case_id": expected_case_id,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class VerificationGate:
    """Independent, fail-closed verification (CONTEXT.md S15.6)."""

    def __init__(self, catalogue: RuleCatalogue, engine: ApplicabilityEngine, *, policy_version: str = GATE_POLICY_VERSION):
        self.catalogue = catalogue
        self.engine = engine
        self.policy_version = policy_version

    # -- individual checks --------------------------------------------------

    def _check_integrity(self, record: StructuredCaseRecord) -> List[GateFailure]:
        failures: List[GateFailure] = []
        if not record.verify_sha256():
            failures.append(GateFailure("integrity", "record_sha256 does not match the recomputed hash"))
        try:
            round_trip = type(record).from_dict(record.to_dict())
            if round_trip.to_dict() != record.to_dict():
                failures.append(GateFailure("integrity", "record does not round-trip byte-identically through to_dict/from_dict"))
        except Exception as exc:  # fail closed on any round-trip error
            failures.append(GateFailure("integrity", f"round-trip failed: {exc}"))
        if record.identity.schema_version != SCHEMA_VERSION:
            failures.append(GateFailure("integrity", f"unknown schema_version {record.identity.schema_version!r}"))
        return failures

    def _check_calculation(self, record: StructuredCaseRecord, recomputed: StructuredCaseRecord) -> List[GateFailure]:
        failures: List[GateFailure] = []
        if record.checklist.to_dict() != recomputed.checklist.to_dict():
            failures.append(GateFailure("calculation", "checklist does not reproduce from the record's own criterion values"))
        if record.agreement.to_dict() != recomputed.agreement.to_dict():
            failures.append(GateFailure("calculation", "direct/checklist agreement does not reproduce"))
        if round_floats(record.facts) != round_floats(recomputed.facts):
            failures.append(GateFailure("calculation", "derived facts do not reproduce from the record's own inputs"))

        p = record.perception
        expected_model_values = criterion_values_from_logits(p.criterion_logits)
        if p.model_criterion_values != expected_model_values:
            failures.append(GateFailure("calculation", "model_criterion_values does not reproduce from criterion_logits"))
        expected_effective = expected_effective_criterion_values(record)
        if p.criterion_values != expected_effective:
            failures.append(GateFailure(
                "calculation",
                "criterion_values do not reproduce from criterion_logits replayed through the record's own criterion_value amendment history",
            ))
        expected_crit_probs = {name: _sigmoid(logit) for name, logit in zip(CRITERIA, p.criterion_logits)}
        if round_floats(expected_crit_probs) != round_floats(p.criterion_probabilities):
            failures.append(GateFailure("calculation", "criterion_probabilities do not reproduce (sigmoid of criterion_logits)"))
        expected_diag_probs = dict(zip(GLOBAL_CLASSES, _softmax(p.diagnosis_logits)))
        if round_floats(expected_diag_probs) != round_floats(p.diagnosis_probabilities):
            failures.append(GateFailure("calculation", "diagnosis_probabilities do not reproduce (softmax of diagnosis_logits)"))
        expected_class = GLOBAL_CLASSES[max(range(len(p.diagnosis_logits)), key=lambda i: p.diagnosis_logits[i])]
        if expected_class != p.provisional_class:
            failures.append(GateFailure("calculation", "provisional_class does not reproduce (argmax of diagnosis_logits)"))
        expected_mel_prob = round(_sigmoid(p.melanoma_logit), 6)
        if expected_mel_prob != round(p.melanoma_probability, 6):
            failures.append(GateFailure("calculation", "melanoma_probability does not reproduce (sigmoid of melanoma_logit)"))
        return failures

    def _check_source_version_hash(self, record: StructuredCaseRecord, now: date) -> List[GateFailure]:
        failures: List[GateFailure] = []
        if record.versions.rule_catalogue_sha256 != self.catalogue.catalogue_sha256:
            failures.append(GateFailure("source_version_hash", "record.versions.rule_catalogue_sha256 does not match the trusted catalogue"))
        cited = {row["rule_id"] for row in record.applicability.get("results", [])}
        for rid in sorted(cited):
            try:
                rule = self.catalogue.get(rid)
            except KeyError:
                failures.append(GateFailure("source_version_hash", f"cited rule {rid} is not in the trusted catalogue"))
                continue
            if not rule.verify():
                failures.append(GateFailure("source_version_hash", f"rule {rid} failed its content-hash verification"))
            if not rule.source.verify_span():
                failures.append(GateFailure("source_version_hash", f"rule {rid} source span failed hash verification"))
            try:
                start = date.fromisoformat(rule.supersession.valid_from)
                end = date.fromisoformat(rule.supersession.valid_until)
            except ValueError:
                failures.append(GateFailure("source_stale", f"rule {rid} has an invalid validity window"))
                continue
            if rule.supersession.status != "current" or not (start <= now <= end):
                failures.append(GateFailure("source_stale", f"rule {rid} is not current as of {now.isoformat()}"))
        empty_pins = [k for k, v in record.versions.to_dict().items() if not v]
        if empty_pins:
            failures.append(GateFailure("source_version_hash", f"empty version pin(s): {empty_pins}"))
        return failures

    def _check_entity(self, record: StructuredCaseRecord, expected_case_id: str) -> List[GateFailure]:
        ids = {record.identity.case_id, record.facts.get("case_id"), record.applicability.get("case_id"), expected_case_id}
        if len(ids) > 1:
            return [GateFailure(
                "entity_mismatch",
                f"case identity mismatch: identity={record.identity.case_id!r} facts={record.facts.get('case_id')!r} "
                f"applicability={record.applicability.get('case_id')!r} expected={expected_case_id!r}",
            )]
        return []

    def _check_rule_path(self, record: StructuredCaseRecord, recomputed: StructuredCaseRecord) -> List[GateFailure]:
        core_keys = ("case_id", "evaluation_date", "jurisdiction", "results",
                     "supported_next_steps", "conditional_next_steps", "targeted_question", "coverage")
        stored_core = {k: record.applicability.get(k) for k in core_keys}
        fresh_core = {k: recomputed.applicability.get(k) for k in core_keys}
        failures: List[GateFailure] = []
        if stored_core != fresh_core:
            failures.append(GateFailure("rule_path", "re-evaluating the engine over the record's own facts does not reproduce the stored applicability"))
        if record.next_step.to_dict() != recomputed.next_step.to_dict():
            failures.append(GateFailure("rule_path", "next_step is not the deterministic function of the stored applicability"))
        return failures

    def _check_missing_fact_handling(self, record: StructuredCaseRecord) -> List[GateFailure]:
        failures: List[GateFailure] = []
        perception_withheld = (record.reliability.iqa_state == "fail") or (not effective_dermoscopic_present(record))
        if perception_withheld:
            if record.next_step.status != "withheld_perception_unavailable":
                failures.append(GateFailure("missing_fact_handling", "perception facts were withheld but next_step.status is not withheld_perception_unavailable"))
            return failures
        results = record.applicability.get("results", [])
        executable_ids = set(NEXT_STEP_PRIORITY)
        unknown_exec = [r for r in results if r["rule_id"] in executable_ids and r["state"] == "applicability_unknown"]
        any_applicable = any(r["state"] == "applicable" for r in results if r["rule_id"] in executable_ids)
        if unknown_exec and not any_applicable:
            if record.applicability.get("targeted_question") is None:
                failures.append(GateFailure("missing_fact_handling", "an executable rule is applicability_unknown but no targeted question is recorded"))
            if record.next_step.status not in ("conditional", "withheld_unknown"):
                failures.append(GateFailure("missing_fact_handling", f"next_step.status={record.next_step.status!r} but an executable rule is applicability_unknown"))
        return failures

    def _check_amendment_consistency(self, record: StructuredCaseRecord, lineage: Sequence[StructuredCaseRecord]) -> List[GateFailure]:
        failures: List[GateFailure] = []
        chain = list(lineage) if lineage else [record]
        verdict = verify_chain(chain, catalogue=self.catalogue, engine=self.engine)
        for f in verdict.failures:
            failures.append(GateFailure("amendment_consistency", f))
        if record.identity.record_version > 1 and not record.identity.parent_record_sha256:
            failures.append(GateFailure("amendment_consistency", "record_version > 1 but parent_record_sha256 is missing"))
        allowed_fields = set(CRITERION_NAMES) | set(CLINICIAN_FIELD_TO_ATTR) | {"image.dermoscopic_present"}
        for entry in record.amendments:
            if entry.field not in allowed_fields:
                failures.append(GateFailure("amendment_consistency", f"amendment field {entry.field!r} is not in the allowed set"))
        return failures

    def _check_required_warnings(self, warnings: Sequence[GateWarning], rendered: Optional[Any]) -> List[GateFailure]:
        if rendered is None:
            return []
        rendered_codes = set(getattr(rendered, "warnings", ()))
        missing = [w.code for w in warnings if w.code not in rendered_codes]
        if missing:
            return [GateFailure("required_warnings", f"rendered output is missing warning code(s): {missing}")]
        return []

    def _check_generated_claim_scope(self, claims: AllowedClaims, rendered: Optional[Any]) -> List[GateFailure]:
        if rendered is None:
            return []
        bad = [tuple(c) for c in getattr(rendered, "claims", ()) if tuple(c) not in claims.claims]
        if bad:
            shown = bad[:5]
            return [GateFailure("generated_claim_scope", f"rendered claim(s) outside the allowed set: {shown}{'...' if len(bad) > 5 else ''}")]
        return []

    # -- entry point ----------------------------------------------------

    def verify(
        self, record: StructuredCaseRecord, *, expected_case_id: str, now: date,
        rendered: Optional[Any] = None, deadline_s: Optional[float] = None,
        lineage: Sequence[StructuredCaseRecord] = (),
    ) -> GateVerdict:
        start = time.perf_counter()
        failures: List[GateFailure] = []

        failures.extend(self._check_integrity(record))

        recomputed: Optional[StructuredCaseRecord]
        try:
            recomputed = recompute_derived(record, catalogue=self.catalogue, engine=self.engine)
        except Exception as exc:
            recomputed = None
            failures.append(GateFailure("calculation", f"recomputation raised: {exc}"))

        if recomputed is not None:
            failures.extend(self._check_calculation(record, recomputed))
        failures.extend(self._check_source_version_hash(record, now))
        failures.extend(self._check_entity(record, expected_case_id))
        if recomputed is not None:
            failures.extend(self._check_rule_path(record, recomputed))
        failures.extend(self._check_missing_fact_handling(record))
        failures.extend(self._check_amendment_consistency(record, lineage))

        warnings = required_warnings(record)
        failures.extend(self._check_required_warnings(warnings, rendered))
        claims = allowed_claims(record)
        failures.extend(self._check_generated_claim_scope(claims, rendered))

        if deadline_s is not None:
            elapsed = time.perf_counter() - start
            if elapsed > deadline_s:
                failures.append(GateFailure("timeout", f"elapsed {elapsed:.6f}s exceeds deadline {deadline_s}s"))

        disposition = _disposition(failures)
        checked_at = now.isoformat()
        verdict = GateVerdict(
            passed=not failures, failures=tuple(failures), disposition=disposition,
            receipt_sha256="", policy_version=self.policy_version, checked_at=checked_at,
        )
        return dataclasses.replace(verdict, receipt_sha256=_receipt_sha256(verdict, record, expected_case_id))


__all__ = [
    "ALL_SECTIONS",
    "AllowedClaims",
    "GateFailure",
    "GateVerdict",
    "GateWarning",
    "VerificationGate",
    "allowed_claims",
    "partial_report_scope",
    "required_warnings",
]
