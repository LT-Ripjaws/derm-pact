"""WP-A1: independent, fail-closed verification gate v2."""
from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from dermpact.guidelines.rules import RuleCatalogue, canonical_json
from dermpact.guidelines.engine import ApplicabilityEngine

from dermpact.record.gate import GateFailure, GateVerdict, GateWarning

from dermpact.core.record_v2 import (
    AmendmentEntryV2,
    CLINICIAN_ASSERTION_FACTS,
    GATE_POLICY_VERSION,
    NEXT_STEP_PRIORITY,
    SCHEMA_VERSION,
    StructuredCaseRecordV2,
    _compute_image_status,
    expected_effective_criterion_values,
    expected_model_criterion_values,
    recompute_derived,
    round_floats,
    verify_chain,
)

FLOAT_TOLERANCE = 2e-6

IQA_STATES = frozenset({"not_flagged", "review_required", "not_assessed"})
OOD_STATES = frozenset({"not_flagged", "flagged", "not_assessed"})
VIEW_STATES = frozenset({"not_flagged", "unstable", "not_assessed"})
SELECTIVE_STATES = frozenset({"answered", "deferred", "not_assessed"})
DISPOSITION_STATES = frozenset({"display", "review", "not_assessed"})
_NON_BLOCKING_WARNING_CODES = frozenset({"unfamiliar_image", "unfamiliarity_unassessed"})


@dataclass(frozen=True)
class AllowedClaimsV2:
    """The closed set of (path, value) leaves an output is permitted to state: every leaf of the hashed record payload, plus a small set of derived display leaves (e.g. rounded percentages) the renderer needs, each recorded with its derivation so an auditor can see exactly how it was produced from an allowed leaf."""

    claims: FrozenSet[Tuple[str, Any]]
    derived_claims: Mapping[Tuple[str, Any], str]

    def __contains__(self, item: Any) -> bool:
        try:
            path, value = item
        except (TypeError, ValueError):
            return False
        return (path, value) in self.claims


#: Every displayable section of the three-layer v2 output (renderer_v2.py's vocabulary).
ALL_SECTIONS: FrozenSet[str] = frozenset({
    "identity", "warnings", "checklist", "next_step", "agreement", "clinician_assertions",
    "image_status", "criterion_evidence", "rule_evidence", "source_spans", "amendment_history",
    "gate_receipt", "versions",
})

#: Which sections a given failing check code removes, unioned across every failing code.
_REMOVE_ON_FAILURE: Dict[str, FrozenSet[str]] = {
    "integrity": ALL_SECTIONS - {"identity"},
    "entity_mismatch": ALL_SECTIONS - {"identity"},
    "calculation": frozenset({"checklist", "next_step", "criterion_evidence"}),
    "source_version_hash": frozenset({"source_spans", "next_step", "versions"}),
    "source_stale": frozenset({"source_spans", "next_step", "versions"}),
    "rule_path": frozenset({"next_step", "rule_evidence"}),
    "missing_fact_handling": frozenset({"next_step"}),
    "amendment_consistency": frozenset({"amendment_history"}),
    "reliability_consistency": frozenset({"warnings"}),
    "image_status_consistency": frozenset({"image_status", "next_step"}),
    "required_warnings": frozenset(),
    "generated_claim_scope": frozenset({"next_step"}),
    "timeout": frozenset({"next_step", "rule_evidence", "source_spans"}),
}


def partial_report_scope(verdict: GateVerdict, record: StructuredCaseRecordV2) -> Dict[str, bool]:
    if verdict.disposition == "refusal":
        return {s: (s == "identity") for s in ALL_SECTIONS}
    removed: set = set()
    for f in verdict.failures:
        removed |= _REMOVE_ON_FAILURE.get(f.code, frozenset())
    return {s: (s not in removed) for s in ALL_SECTIONS}


def _close(a: Any, b: Any, tol: float = FLOAT_TOLERANCE) -> bool:
    """Structural equality for stored-vs-recomputed comparisons, with a ``tol`` absolute-difference allowance on numeric leaves only (see ``FLOAT_TOLERANCE``)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a.keys()) == set(b.keys()) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def _walk_leaves(prefix: str, value: Any, out: List[Tuple[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _walk_leaves(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _walk_leaves(f"{prefix}[{i}]", v, out)
    else:
        out.append((prefix, value))


def _derived_claims(record: StructuredCaseRecordV2) -> Dict[Tuple[str, Any], str]:
    """Rounded-percentage display leaves the renderer needs but the hashed payload does not carry verbatim (it carries the 0-1 probability)."""
    out: Dict[Tuple[str, Any], str] = {}
    mel = float(record.perception.melanoma_head["probability"])
    out[("derived.melanoma_head.probability_percent", round(mel * 100, 1))] = \
        "round(perception.melanoma_head.probability * 100, 1)"
    for cls, p in record.perception.diagnosis_head["probabilities"].items():
        out[(f"derived.diagnosis_head.probabilities.{cls}_percent", round(float(p) * 100, 1))] = \
            f"round(perception.diagnosis_head.probabilities.{cls} * 100, 1)"
    for name, p in record.perception.criterion_probabilities.items():
        out[(f"derived.criterion_probabilities.{name}_percent", round(float(p) * 100, 1))] = \
            f"round(perception.criterion_probabilities.{name} * 100, 1)"
    return out


def allowed_claims_v2(record: StructuredCaseRecordV2) -> AllowedClaimsV2:
    leaves: List[Tuple[str, Any]] = []
    _walk_leaves("", round_floats(record.to_dict()), leaves)
    derived = _derived_claims(record)
    return AllowedClaimsV2(claims=frozenset(leaves) | frozenset(derived.keys()), derived_claims=derived)


def required_warnings_v2(record: StructuredCaseRecordV2) -> Tuple[GateWarning, ...]:
    warnings: List[GateWarning] = []
    r = record.reliability
    for w in r.warnings:
        warnings.append(GateWarning(
            code=str(w.get("code", "")), reason=str(w.get("reason", "")),
            remaining_usable_output=str(w.get("remaining_usable_output", "")),
            recovery_action=str(w.get("recovery_action", "")),
        ))
    if record.perception.perception_mode == "five_member_deployment":
        warnings.append(GateWarning(
            code="ensemble_calibration_unvalidated",
            reason="member-calibrated probabilities are averaged into an aggregate that is not itself independently validated",
            remaining_usable_output="member-wise calibrated evidence and relative ranking remain usable",
            recovery_action="treat the aggregate as a display convention, not a validated ensemble probability",
        ))
    else:
        warnings.append(GateWarning(
            code="oof_single_member_uncalibrated",
            reason="benchmark mode: a single out-of-fold member with raw, uncalibrated probabilities",
            remaining_usable_output="relative ranking remains usable; absolute probabilities are not calibrated",
            recovery_action="treat probabilities as uncalibrated model scores",
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
    executable_ids = set(NEXT_STEP_PRIORITY)
    unresolved = any(
        row["rule_id"] in executable_ids and row["state"] in ("applicability_unknown", "no_catalogue_coverage")
        for row in record.applicability.get("results", [])
    )
    if unresolved or record.next_step.status in ("conditional", "withheld_no_coverage"):
        warnings.append(GateWarning(
            code="applicability_unknown_or_no_coverage",
            reason="at least one executable guideline rule could not be resolved to applicable/not_applicable",
            remaining_usable_output="the underlying evidence (checklist, criteria, facts, clinician assertions) remains displayed",
            recovery_action="answer the targeted question, or refer for clinician review if none is available",
        ))
    if record.image_status.state == "limited_review_flagged":
        warnings.append(GateWarning(
            code="image_limited", reason="image status is limited/review-flagged",
            remaining_usable_output="perception outputs remain displayed with the limitation noted",
            recovery_action=record.image_status.recovery_action,
        ))
    elif record.image_status.state == "unavailable_no_image":
        warnings.append(GateWarning(
            code="image_unavailable", reason="no usable dermoscopic image for this case",
            remaining_usable_output="clinician-assertion-based next steps (e.g. R3/R4/R7) remain available",
            recovery_action=record.image_status.recovery_action,
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


def _receipt_sha256(verdict: GateVerdict, record: StructuredCaseRecordV2, expected_case_id: str) -> str:
    payload = {
        "policy_version": verdict.policy_version, "checked_at": verdict.checked_at, "passed": verdict.passed,
        "disposition": verdict.disposition, "failures": [{"code": f.code, "detail": f.detail} for f in verdict.failures],
        "record_sha256": record.record_sha256, "expected_case_id": expected_case_id,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class VerificationGateV2:
    """Independent, fail-closed verification."""

    def __init__(
        self, catalogue: RuleCatalogue, engine: ApplicabilityEngine, *,
        calibrators: Optional[Mapping[str, Any]] = None, bundle: Optional[Any] = None,
        policy_version: str = GATE_POLICY_VERSION,
    ):
        self.catalogue = catalogue
        self.engine = engine
        self.calibrators = calibrators
        self.bundle = bundle
        self.policy_version = policy_version

    # -- individual checks --------------------------------------------------

    def _check_integrity(self, record: StructuredCaseRecordV2) -> List[GateFailure]:
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

    def _recompute(self, record: StructuredCaseRecordV2) -> Optional[StructuredCaseRecordV2]:
        try:
            return recompute_derived(record, catalogue=self.catalogue, engine=self.engine,
                                     calibrators=self.calibrators, bundle=self.bundle)
        except Exception:
            return None

    def _check_criterion_value_calculation(self, record: StructuredCaseRecordV2) -> List[GateFailure]:
        """Standalone (no ``recompute_derived``, no ``lineage``): re-derive ``model_criterion_values`` from the record's own ``criterion_probabilities`` and the effective ``criterion_values`` from that with the record's own committed ``criterion_value`` amendments replayed, and compare both to what the record claims."""
        failures: List[GateFailure] = []
        p = record.perception
        expected_model_values = expected_model_criterion_values(p.criterion_probabilities)
        if p.model_criterion_values != expected_model_values:
            failures.append(GateFailure("calculation", "model_criterion_values does not reproduce from criterion_probabilities"))
        expected_effective = expected_effective_criterion_values(record)
        if p.criterion_values != expected_effective:
            failures.append(GateFailure(
                "calculation",
                "criterion_values do not reproduce from criterion_probabilities replayed through the record's own criterion_value amendment history",
            ))
        return failures

    def _check_calculation(self, record: StructuredCaseRecordV2, recomputed: StructuredCaseRecordV2) -> List[GateFailure]:
        failures: List[GateFailure] = []
        if record.checklist.to_dict() != recomputed.checklist.to_dict():
            failures.append(GateFailure("calculation", "checklist does not reproduce from the record's own effective criterion values"))
        if record.agreement.to_dict() != recomputed.agreement.to_dict():
            failures.append(GateFailure("calculation", "direct/checklist agreement does not reproduce"))
        if round_floats(record.facts) != round_floats(recomputed.facts):
            failures.append(GateFailure("calculation", "derived facts do not reproduce from the record's own inputs"))

        p = record.perception
        rp = recomputed.perception
        if not _close(p.diagnosis_head, rp.diagnosis_head):
            failures.append(GateFailure("calculation", "diagnosis_head does not reproduce from the record's own member logits"))
        if not _close(p.melanoma_head, rp.melanoma_head):
            failures.append(GateFailure("calculation", "melanoma_head does not reproduce from the record's own member logits"))
        if not _close(p.criterion_probabilities, rp.criterion_probabilities):
            failures.append(GateFailure("calculation", "criterion_probabilities does not reproduce from the record's own member logits"))
        for m, rm in zip(p.members, rp.members):
            if not _close(m.calibrated_diagnosis_probabilities, rm.calibrated_diagnosis_probabilities):
                failures.append(GateFailure("calculation", f"member fold={m.fold} calibrated_diagnosis_probabilities does not reproduce from raw logits + calibrators"))
                break
            if not _close(m.calibrated_binary_probabilities, rm.calibrated_binary_probabilities):
                failures.append(GateFailure("calculation", f"member fold={m.fold} calibrated_binary_probabilities does not reproduce from raw logits + calibrators"))
                break
        return failures

    def _check_source_version_hash(self, record: StructuredCaseRecordV2, now: date) -> List[GateFailure]:
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
        empty_pins = [k for k, v in record.versions.to_dict().items() if not v and k not in ("calibration_sha256", "calibration_state")]
        if empty_pins:
            failures.append(GateFailure("source_version_hash", f"empty version pin(s): {empty_pins}"))
        return failures

    def _check_entity(self, record: StructuredCaseRecordV2, expected_case_id: str) -> List[GateFailure]:
        ids = {record.identity.case_id, record.facts.get("case_id"), record.applicability.get("case_id"), expected_case_id}
        if len(ids) > 1:
            return [GateFailure(
                "entity_mismatch",
                f"case identity mismatch: identity={record.identity.case_id!r} facts={record.facts.get('case_id')!r} "
                f"applicability={record.applicability.get('case_id')!r} expected={expected_case_id!r}",
            )]
        return []

    def _check_rule_path(self, record: StructuredCaseRecordV2, recomputed: StructuredCaseRecordV2) -> List[GateFailure]:
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

    def _check_missing_fact_handling(self, record: StructuredCaseRecordV2) -> List[GateFailure]:
        if record.next_step.status != "conditional":
            return []
        failures: List[GateFailure] = []
        fact = record.next_step.decisive_missing_fact
        if fact is None:
            failures.append(GateFailure("missing_fact_handling", "next_step.status is conditional but no decisive_missing_fact is recorded"))
        elif fact not in CLINICIAN_ASSERTION_FACTS:
            failures.append(GateFailure("missing_fact_handling", f"conditional next_step names decisive fact {fact!r}, which is not an asserted-type clinician fact"))
        return failures

    def _check_amendment_consistency(self, record: StructuredCaseRecordV2, lineage: Sequence[StructuredCaseRecordV2]) -> List[GateFailure]:
        failures: List[GateFailure] = []
        chain = list(lineage) if lineage else [record]
        verdict = verify_chain(chain, catalogue=self.catalogue, engine=self.engine, calibrators=self.calibrators, bundle=self.bundle)
        for f in verdict.failures:
            failures.append(GateFailure("amendment_consistency", f))
        if record.identity.record_version > 1 and not record.identity.parent_record_sha256:
            failures.append(GateFailure("amendment_consistency", "record_version > 1 but parent_record_sha256 is missing"))
        allowed_fields = set(CLINICIAN_ASSERTION_FACTS) | {"image.dermoscopic_present"}
        from dermpact.criteria.schema import CRITERION_NAMES
        allowed_fields |= set(CRITERION_NAMES)
        for entry in record.amendments:
            if entry.field not in allowed_fields:
                failures.append(GateFailure("amendment_consistency", f"amendment field {entry.field!r} is not in the allowed set"))

        last_by_field: Dict[str, AmendmentEntryV2] = {}
        for entry in record.amendments:
            last_by_field[entry.field] = entry  # amendments are append-only: last write wins
        for field, entry in last_by_field.items():
            if field in CRITERION_NAMES:
                actual = record.perception.criterion_values.get(field)
            elif field in CLINICIAN_ASSERTION_FACTS:
                actual = record.clinician_assertions.get(field, {}).get("value")
            elif field == "image.dermoscopic_present":
                actual = record.reliability.dermoscopic_present_override
            else:
                continue  # already flagged above as outside the allowed field set
            if actual != entry.after:
                failures.append(GateFailure(
                    "amendment_consistency",
                    f"last committed amendment for field {field!r} has after={entry.after!r} but the "
                    f"record's current value for that field is {actual!r}",
                ))
        return failures

    def _check_reliability_consistency(self, record: StructuredCaseRecordV2) -> List[GateFailure]:
        failures: List[GateFailure] = []
        r = record.reliability
        for value, vocab, name in (
            (r.iqa_state, IQA_STATES, "iqa_state"), (r.ood_state, OOD_STATES, "ood_state"),
            (r.view_stability_state, VIEW_STATES, "view_stability_state"),
            (r.selective_abstention, SELECTIVE_STATES, "selective_abstention"),
            (r.disposition, DISPOSITION_STATES, "disposition"),
        ):
            if value not in vocab:
                failures.append(GateFailure("reliability_consistency", f"reliability.{name}={value!r} is outside its closed vocabulary"))
        if r.disposition != "not_assessed":
            blocking = any(w.get("code") not in _NON_BLOCKING_WARNING_CODES for w in r.warnings)
            expected = "review" if blocking else "display"
            if r.disposition != expected:
                failures.append(GateFailure(
                    "reliability_consistency",
                    f"reliability.disposition={r.disposition!r} does not reproduce from reliability.warnings under the bundle blocking rule (expected {expected!r})",
                ))
        return failures

    def _check_image_status_consistency(self, record: StructuredCaseRecordV2) -> List[GateFailure]:
        expected = _compute_image_status(
            dermoscopic_input_present=record.inputs.input_mask["dermoscopic"],
            dermoscopic_present_override=record.reliability.dermoscopic_present_override,
            iqa_state=record.reliability.iqa_state, view_stability_state=record.reliability.view_stability_state,
        )
        if expected.to_dict() != record.image_status.to_dict():
            return [GateFailure("image_status_consistency", "image_status does not reproduce from the record's own input mask, override and reliability states")]
        return []

    def _check_required_warnings(self, warnings: Sequence[GateWarning], rendered: Optional[Any]) -> List[GateFailure]:
        if rendered is None:
            return []
        rendered_codes = set(getattr(rendered, "warnings", ()))
        missing = [w.code for w in warnings if w.code not in rendered_codes]
        if missing:
            return [GateFailure("required_warnings", f"rendered output is missing warning code(s): {missing}")]
        return []

    def _check_generated_claim_scope(self, claims: AllowedClaimsV2, rendered: Optional[Any]) -> List[GateFailure]:
        if rendered is None:
            return []
        bad = [tuple(c) for c in getattr(rendered, "claims", ()) if tuple(c) not in claims.claims]
        if bad:
            shown = bad[:5]
            return [GateFailure("generated_claim_scope", f"rendered claim(s) outside the allowed set: {shown}{'...' if len(bad) > 5 else ''}")]
        return []

    # -- entry point ----------------------------------------------------

    def verify(
        self, record: StructuredCaseRecordV2, *, expected_case_id: str, now: date,
        rendered: Optional[Any] = None, deadline_s: Optional[float] = None,
        lineage: Sequence[StructuredCaseRecordV2] = (),
    ) -> GateVerdict:
        start = time.perf_counter()
        failures: List[GateFailure] = []

        failures.extend(self._check_integrity(record))
        failures.extend(self._check_criterion_value_calculation(record))
        recomputed = self._recompute(record)
        if recomputed is None:
            failures.append(GateFailure("calculation", "recomputation raised or failed"))
        else:
            failures.extend(self._check_calculation(record, recomputed))
        failures.extend(self._check_source_version_hash(record, now))
        failures.extend(self._check_entity(record, expected_case_id))
        if recomputed is not None:
            failures.extend(self._check_rule_path(record, recomputed))
        failures.extend(self._check_missing_fact_handling(record))
        failures.extend(self._check_amendment_consistency(record, lineage))
        failures.extend(self._check_reliability_consistency(record))
        failures.extend(self._check_image_status_consistency(record))

        warnings = required_warnings_v2(record)
        failures.extend(self._check_required_warnings(warnings, rendered))
        claims = allowed_claims_v2(record)
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
    "AllowedClaimsV2",
    "GateFailure",
    "GateVerdict",
    "GateWarning",
    "VerificationGateV2",
    "allowed_claims_v2",
    "partial_report_scope",
    "required_warnings_v2",
]
