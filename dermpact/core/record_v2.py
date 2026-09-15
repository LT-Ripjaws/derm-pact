"""WP-A1: the current-version structured case record."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dermpact.eval.dermpact_phase3_calibration import digest as _digest_file
from dermpact.eval.dermpact_phase4_benchmark import (
    CRITERIA,
    GLOBAL_CLASSES,
    INDETERMINATE_BAND,
    NEG_VALUE,
    POS_VALUE,
)
from dermpact.guidelines import reviewed_v2
from dermpact.guidelines.derivation import DERM7PT_INSTRUMENT
from dermpact.guidelines.engine import ApplicabilityEngine, CaseApplicability
from dermpact.guidelines.rules import ApplicabilityState, RuleCatalogue, canonical_json
from dermpact.record.case_record import (
    ACTION_PHRASES,
    ChecklistResult,
    DirectChecklistAgreement,
    PerCriterionScore,
    build_agreement,
    round_floats,
)
from dermpact.reliability.inference import calibrated_fold_probabilities
from dermpact.criteria.aggregate import score_criteria, recommend_management
from dermpact.criteria.schema import CRITERION_NAMES, validate_values

from dermpact.core import versions as V

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = V.RECORD_SCHEMA_VERSION
ENGINE_POLICY_VERSION = V.ENGINE_POLICY_VERSION
GATE_POLICY_VERSION = V.GATE_POLICY_VERSION
RENDERER_VERSION = V.RENDERER_VERSION
PREPROCESSING_VERSION = V.PREPROCESSING_VERSION

#: The nine amendable clinician/patient/site facts.
CLINICIAN_ASSERTION_FACTS: Tuple[str, ...] = (
    "clinician.melanoma_suspected",
    "clinician.equivocal",
    "lesion.under_assessment",
    "patient.high_risk_status",
    "patient.total_nevus_count",
    "patient.high_total_nevus_count",
    "patient.personal_melanoma_history",
    "patient.genetic_predisposition",
    "site.rcm_available",
)

#: Mapping to the ``reviewed_v2.derive_case_facts``/``derive_legacy_facts`` kwarg name.
CLINICIAN_ASSERTION_TO_KWARG: Mapping[str, str] = {
    "clinician.melanoma_suspected": "clinician_melanoma_suspected",
    "clinician.equivocal": "clinician_equivocal",
    "lesion.under_assessment": "lesion_under_assessment",
    "patient.high_risk_status": "high_risk_status",
    "patient.total_nevus_count": "total_nevus_count",
    "patient.high_total_nevus_count": "high_total_nevus_count",
    "patient.personal_melanoma_history": "personal_melanoma_history",
    "patient.genetic_predisposition": "genetic_predisposition",
    "site.rcm_available": "rcm_available",
}

#: Host priority for the single displayed next step: melanoma pathway first.
NEXT_STEP_PRIORITY: Tuple[str, ...] = (
    "eado-mel-p1-2024:R3",
    "eado-mel-p1-2024:R7",
    "eado-mel-p1-2024:R4",
    "eado-mel-p1-2024:R6",
    "eado-mel-p1-2024:R5",
)

RULE_MODALS: Mapping[str, str] = {
    "eado-mel-p1-2024:R3": "shall",
    "eado-mel-p1-2024:R4": "shall",
    "eado-mel-p1-2024:R5": "should",
    "eado-mel-p1-2024:R6": "should",
    "eado-mel-p1-2024:R7": "should",
}

SCOPE_NOTE = "EADO Part 1 diagnostic catalogue only; no treatment, margin, staging, prognosis or survival coverage"
IQA_MEANING = "specified synthetic-corruption warning, not clinician-rated usability"
OOD_MEANING = "advisory unfamiliarity warning; not deferral or invalidity"
FACT_MAPPING_REASON = (
    "reviewed v2 rules consume clinician assertions only; IQA/OOD are advisory host states, not guideline facts"
)
CHECKLIST_IMPLICATION_PREFIX = "checklist evidence only; not a clinical next step. "

SOURCE_LABELS = frozenset({
    "development_allowlisted_case", "synthetic_workflow_fixture", "development_oof_benchmark",
    "evaluation_query_only", "deployment_query_only",
})
CLINICAL_AVAILABILITY_STATES = frozenset({"present", "absent", "duplicate_dermoscopy_excluded", "not_supplied"})


class RecordInputErrorV2(ValueError):
    """A v2 record builder/amendment primitive was given malformed input. Fail closed."""




def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _softmax(xs: Sequence[float]) -> Tuple[float, ...]:
    m = max(float(x) for x in xs)
    exps = [math.exp(float(x) - m) for x in xs]
    s = sum(exps)
    return tuple(e / s for e in exps)


def criterion_values_from_probabilities(probabilities: Mapping[str, float]) -> Dict[str, str]:
    """The v2 categorical reading: same convention as v1's ``criterion_values_from_logits``, applied to an already-probability-space value (the aggregated, member-calibrated ``criterion_probabilities``) rather than to a raw logit."""
    values: Dict[str, str] = {}
    for name in CRITERIA:
        p = float(probabilities[name])
        if abs(p - 0.5) < INDETERMINATE_BAND:
            values[name] = "indeterminate"
        else:
            values[name] = POS_VALUE[name] if p >= 0.5 else NEG_VALUE[name]
    return values


def _calibrate_member(
    diagnosis_logits: Sequence[float], melanoma_logit: float, criterion_logits: Sequence[float],
    fold: int, calibrators: Mapping[str, Any],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Calibrate one member's flat, unbatched raw logits."""
    out = calibrated_fold_probabilities(
        {"diagnosis": [list(diagnosis_logits)], "melanoma": [float(melanoma_logit)], "criteria": [list(criterion_logits)]},
        fold, calibrators,
    )
    diag, binary = out
    return tuple(float(x) for x in diag[0]), tuple(float(x) for x in binary[0])


def _bundle_sha256() -> str:
    return _digest_file(ROOT / V.RELIABILITY_BUNDLE)


def _selection_manifest_sha256() -> str:
    return _digest_file(ROOT / V.SELECTION_MANIFEST)


def _thresholds_from_bundle(bundle: Optional[Any], n_members: int) -> Dict[str, Any]:
    if bundle is None:
        return {}
    per_member = {}
    for fold in range(n_members):
        try:
            per_member[str(fold)] = float(bundle.calibration["folds"][str(fold)]["acceptance_threshold"])
        except (KeyError, TypeError):
            continue
    return {
        "quality_threshold": bundle.quality.get("threshold"),
        "ood_threshold": bundle.threshold,
        "per_member_acceptance_thresholds": per_member,
        "per_fold_view_thresholds": dict(bundle.manifest.get("view_stability", {}).get("per_fold_thresholds", {})),
    }




@dataclass(frozen=True)
class IdentityV2:
    case_id: str
    record_version: int
    parent_record_sha256: Optional[str]
    created_at: str
    schema_version: str = SCHEMA_VERSION
    source_label: str = "development_allowlisted_case"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id, "record_version": self.record_version,
            "parent_record_sha256": self.parent_record_sha256, "created_at": self.created_at,
            "schema_version": self.schema_version, "source_label": self.source_label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityV2":
        return cls(
            case_id=str(payload["case_id"]), record_version=int(payload["record_version"]),
            parent_record_sha256=payload.get("parent_record_sha256"), created_at=str(payload["created_at"]),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            source_label=str(payload.get("source_label", "development_allowlisted_case")),
        )


@dataclass(frozen=True)
class ImageRef:
    input_sha256: Optional[str]
    sanitized_sha256: Optional[str]
    relative_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"input_sha256": self.input_sha256, "sanitized_sha256": self.sanitized_sha256,
                "relative_path": self.relative_path}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImageRef":
        return cls(input_sha256=payload.get("input_sha256"), sanitized_sha256=payload.get("sanitized_sha256"),
                   relative_path=payload.get("relative_path"))


@dataclass(frozen=True)
class InputsV2:
    dermoscopic: ImageRef
    sanitization_receipt: Dict[str, Any]
    clinical: ImageRef
    clinical_availability: str
    input_mask: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dermoscopic": self.dermoscopic.to_dict(), "sanitization_receipt": dict(self.sanitization_receipt),
            "clinical": self.clinical.to_dict(), "clinical_availability": self.clinical_availability,
            "input_mask": dict(self.input_mask),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InputsV2":
        return cls(
            dermoscopic=ImageRef.from_dict(payload["dermoscopic"]),
            sanitization_receipt=dict(payload.get("sanitization_receipt", {})),
            clinical=ImageRef.from_dict(payload["clinical"]),
            clinical_availability=str(payload["clinical_availability"]),
            input_mask=dict(payload["input_mask"]),
        )


@dataclass(frozen=True)
class MemberEvidence:
    fold: Optional[int]
    diagnosis_logits: Tuple[float, ...]
    melanoma_logit: float
    criterion_logits: Tuple[float, ...]
    calibrated_diagnosis_probabilities: Optional[Tuple[float, ...]]
    calibrated_binary_probabilities: Optional[Tuple[float, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold, "diagnosis_logits": list(self.diagnosis_logits), "melanoma_logit": self.melanoma_logit,
            "criterion_logits": list(self.criterion_logits),
            "calibrated_diagnosis_probabilities": (list(self.calibrated_diagnosis_probabilities)
                                                    if self.calibrated_diagnosis_probabilities is not None else None),
            "calibrated_binary_probabilities": (list(self.calibrated_binary_probabilities)
                                                 if self.calibrated_binary_probabilities is not None else None),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemberEvidence":
        cdp = payload.get("calibrated_diagnosis_probabilities")
        cbp = payload.get("calibrated_binary_probabilities")
        return cls(
            fold=payload.get("fold"), diagnosis_logits=tuple(float(x) for x in payload["diagnosis_logits"]),
            melanoma_logit=float(payload["melanoma_logit"]),
            criterion_logits=tuple(float(x) for x in payload["criterion_logits"]),
            calibrated_diagnosis_probabilities=(tuple(float(x) for x in cdp) if cdp is not None else None),
            calibrated_binary_probabilities=(tuple(float(x) for x in cbp) if cbp is not None else None),
        )


@dataclass(frozen=True)
class PerceptionV2:
    perception_mode: str
    model_id: str
    checkpoint_bindings: Tuple[Dict[str, Any], ...]
    calibration_sha256: Optional[str]
    calibration_state: Optional[str]
    members: Tuple[MemberEvidence, ...]
    diagnosis_head: Dict[str, Any]
    melanoma_head: Dict[str, Any]
    criterion_probabilities: Dict[str, float]
    model_criterion_values: Dict[str, str]
    criterion_values: Dict[str, str]
    criterion_convention: Dict[str, Any]
    reported_aggregate: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perception_mode": self.perception_mode, "model_id": self.model_id,
            "checkpoint_bindings": [dict(b) for b in self.checkpoint_bindings],
            "calibration_sha256": self.calibration_sha256, "calibration_state": self.calibration_state,
            "members": [m.to_dict() for m in self.members],
            "diagnosis_head": dict(self.diagnosis_head), "melanoma_head": dict(self.melanoma_head),
            "criterion_probabilities": dict(self.criterion_probabilities),
            "model_criterion_values": dict(self.model_criterion_values),
            "criterion_values": dict(self.criterion_values),
            "criterion_convention": dict(self.criterion_convention),
            "reported_aggregate": {k: list(v) for k, v in self.reported_aggregate.items()},
            "abstained": self.abstained,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerceptionV2":
        return cls(
            perception_mode=str(payload["perception_mode"]), model_id=str(payload["model_id"]),
            checkpoint_bindings=tuple(dict(b) for b in payload["checkpoint_bindings"]),
            calibration_sha256=payload.get("calibration_sha256"), calibration_state=payload.get("calibration_state"),
            members=tuple(MemberEvidence.from_dict(m) for m in payload["members"]),
            diagnosis_head=dict(payload["diagnosis_head"]), melanoma_head=dict(payload["melanoma_head"]),
            criterion_probabilities={str(k): float(v) for k, v in payload["criterion_probabilities"].items()},
            model_criterion_values={str(k): str(v) for k, v in payload["model_criterion_values"].items()},
            criterion_values={str(k): str(v) for k, v in payload["criterion_values"].items()},
            criterion_convention=dict(payload["criterion_convention"]),
            reported_aggregate={str(k): [float(x) for x in v] for k, v in payload.get("reported_aggregate", {}).items()},
            abstained=bool(payload.get("abstained", False)),
        )


@dataclass(frozen=True)
class ReliabilityV2:
    iqa_state: str
    ood_state: str
    view_stability_state: str
    selective_abstention: str
    disposition: str
    scores: Dict[str, Any]
    thresholds: Dict[str, Any]
    warnings: Tuple[Dict[str, Any], ...]
    phase3_bundle_version: str
    phase3_bundle_sha256: str
    meaning: Dict[str, str]
    dermoscopic_present_override: Optional[bool]
    fact_mapping: Dict[str, Optional[str]]
    fact_mapping_reason: str = FACT_MAPPING_REASON

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iqa_state": self.iqa_state, "ood_state": self.ood_state,
            "view_stability_state": self.view_stability_state, "selective_abstention": self.selective_abstention,
            "disposition": self.disposition, "scores": dict(self.scores), "thresholds": dict(self.thresholds),
            "warnings": [dict(w) for w in self.warnings], "phase3_bundle_version": self.phase3_bundle_version,
            "phase3_bundle_sha256": self.phase3_bundle_sha256, "meaning": dict(self.meaning),
            "dermoscopic_present_override": self.dermoscopic_present_override,
            "fact_mapping": dict(self.fact_mapping), "fact_mapping_reason": self.fact_mapping_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReliabilityV2":
        return cls(
            iqa_state=str(payload["iqa_state"]), ood_state=str(payload["ood_state"]),
            view_stability_state=str(payload["view_stability_state"]),
            selective_abstention=str(payload["selective_abstention"]), disposition=str(payload["disposition"]),
            scores=dict(payload.get("scores", {})), thresholds=dict(payload.get("thresholds", {})),
            warnings=tuple(dict(w) for w in payload.get("warnings", ())),
            phase3_bundle_version=str(payload["phase3_bundle_version"]),
            phase3_bundle_sha256=str(payload["phase3_bundle_sha256"]), meaning=dict(payload.get("meaning", {})),
            dermoscopic_present_override=payload.get("dermoscopic_present_override"),
            fact_mapping=dict(payload.get("fact_mapping", {})),
            fact_mapping_reason=str(payload.get("fact_mapping_reason", FACT_MAPPING_REASON)),
        )


@dataclass(frozen=True)
class ImageStatusV2:
    state: str
    reasons: Tuple[str, ...]
    recovery_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "reasons": list(self.reasons), "recovery_action": self.recovery_action}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImageStatusV2":
        return cls(state=str(payload["state"]), reasons=tuple(str(x) for x in payload.get("reasons", ())),
                   recovery_action=str(payload["recovery_action"]))


@dataclass(frozen=True)
class NextStepV2:
    status: str
    rule_id: Optional[str]
    action_code: Optional[str]
    display_text: str
    targeted_question: Optional[str]
    decisive_missing_fact: Optional[str]
    reason: Optional[str]
    image_status_summary: Dict[str, Any]
    scope_note: str = SCOPE_NOTE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "rule_id": self.rule_id, "action_code": self.action_code,
            "display_text": self.display_text, "targeted_question": self.targeted_question,
            "decisive_missing_fact": self.decisive_missing_fact, "reason": self.reason,
            "image_status_summary": dict(self.image_status_summary), "scope_note": self.scope_note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NextStepV2":
        return cls(
            status=str(payload["status"]), rule_id=payload.get("rule_id"), action_code=payload.get("action_code"),
            display_text=str(payload["display_text"]), targeted_question=payload.get("targeted_question"),
            decisive_missing_fact=payload.get("decisive_missing_fact"), reason=payload.get("reason"),
            image_status_summary=dict(payload.get("image_status_summary", {})),
            scope_note=str(payload.get("scope_note", SCOPE_NOTE)),
        )


@dataclass(frozen=True)
class VersionPinsV2:
    model_id: str
    calibration_sha256: Optional[str]
    calibration_state: Optional[str]
    preprocessing_version: str
    rule_catalogue_sha256: str
    derivation_policy_version: str
    engine_policy_version: str
    record_schema_version: str
    gate_policy_version: str
    renderer_version: str
    phase3_bundle_sha256: str
    selection_manifest_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VersionPinsV2":
        return cls(**{f.name: payload.get(f.name) for f in dataclasses.fields(cls)})


@dataclass(frozen=True)
class AssertionInput:
    """One clinician assertion as supplied to the builder."""

    value: Any = None
    actor: Optional[str] = None
    session_id: Optional[str] = None
    asserted_at: Optional[str] = None
    statement: str = ""


@dataclass(frozen=True)
class AmendmentDeltaV2:
    kind: str
    field: str
    before: Any
    after: Any
    actor: str = "unspecified"
    session_id: str = "unspecified"


@dataclass(frozen=True)
class AmendmentEntryV2:
    amendment_id: str
    kind: str
    field: str
    before: Any
    after: Any
    actor: str
    session_id: str
    proposed_at: str
    committed_at: str
    proposal_sha256: str
    token_sha256: str
    consequence_preview_sha256: str
    recomputed_consequences: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amendment_id": self.amendment_id, "kind": self.kind, "field": self.field,
            "before": self.before, "after": self.after, "actor": self.actor, "session_id": self.session_id,
            "proposed_at": self.proposed_at, "committed_at": self.committed_at,
            "proposal_sha256": self.proposal_sha256, "token_sha256": self.token_sha256,
            "consequence_preview_sha256": self.consequence_preview_sha256,
            "recomputed_consequences": dict(self.recomputed_consequences),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AmendmentEntryV2":
        return cls(
            amendment_id=str(payload["amendment_id"]), kind=str(payload["kind"]), field=str(payload["field"]),
            before=payload.get("before"), after=payload.get("after"), actor=str(payload["actor"]),
            session_id=str(payload.get("session_id", "unspecified")),
            proposed_at=str(payload["proposed_at"]), committed_at=str(payload["committed_at"]),
            proposal_sha256=str(payload["proposal_sha256"]), token_sha256=str(payload["token_sha256"]),
            consequence_preview_sha256=str(payload["consequence_preview_sha256"]),
            recomputed_consequences=dict(payload.get("recomputed_consequences", {})),
        )




@dataclass(frozen=True)
class StructuredCaseRecordV2:
    identity: IdentityV2
    inputs: InputsV2
    perception: PerceptionV2
    reliability: ReliabilityV2
    image_status: ImageStatusV2
    checklist: ChecklistResult
    agreement: DirectChecklistAgreement
    clinician_assertions: Dict[str, Any]
    facts: Dict[str, Any]
    applicability: Dict[str, Any]
    next_step: NextStepV2
    versions: VersionPinsV2
    amendments: Tuple[AmendmentEntryV2, ...]
    record_sha256: str
    attachments: Dict[str, Any] = field(
        default_factory=lambda: {"review_episode_ref": None, "xai_receipt_refs": []}
    )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        checklist_dict = self.checklist.to_dict()
        mgmt = checklist_dict.pop("management_implication")
        checklist_dict["checklist_implication"] = {
            "action": mgmt["action"], "reason": CHECKLIST_IMPLICATION_PREFIX + mgmt["reason"],
        }
        payload = {
            "identity": self.identity.to_dict(),
            "inputs": self.inputs.to_dict(),
            "perception": self.perception.to_dict(),
            "reliability": self.reliability.to_dict(),
            "image_status": self.image_status.to_dict(),
            "checklist": checklist_dict,
            "agreement": self.agreement.to_dict(),
            "clinician_assertions": self.clinician_assertions,
            "facts": self.facts,
            "applicability": self.applicability,
            "next_step": self.next_step.to_dict(),
            "versions": self.versions.to_dict(),
            "amendments": [a.to_dict() for a in self.amendments],
            "record_sha256": self.record_sha256,
            "attachments": dict(self.attachments),
        }
        return round_floats(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredCaseRecordV2":
        checklist_payload = dict(payload["checklist"])
        implication = checklist_payload.pop("checklist_implication")
        reason = implication["reason"]
        if reason.startswith(CHECKLIST_IMPLICATION_PREFIX):
            reason = reason[len(CHECKLIST_IMPLICATION_PREFIX):]
        checklist_payload["management_implication"] = {"action": implication["action"], "reason": reason}
        return cls(
            identity=IdentityV2.from_dict(payload["identity"]),
            inputs=InputsV2.from_dict(payload["inputs"]),
            perception=PerceptionV2.from_dict(payload["perception"]),
            reliability=ReliabilityV2.from_dict(payload["reliability"]),
            image_status=ImageStatusV2.from_dict(payload["image_status"]),
            checklist=ChecklistResult.from_dict(checklist_payload),
            agreement=DirectChecklistAgreement.from_dict(payload["agreement"]),
            clinician_assertions=json.loads(json.dumps(payload["clinician_assertions"])),
            facts=json.loads(json.dumps(payload["facts"])),
            applicability=json.loads(json.dumps(payload["applicability"])),
            next_step=NextStepV2.from_dict(payload["next_step"]),
            versions=VersionPinsV2.from_dict(payload["versions"]),
            amendments=tuple(AmendmentEntryV2.from_dict(a) for a in payload.get("amendments", ())),
            record_sha256=str(payload.get("record_sha256", "")),
            attachments=json.loads(json.dumps(payload.get("attachments", {"review_episode_ref": None, "xai_receipt_refs": []}))),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("record_sha256", None)
        payload.pop("attachments", None)
        return hashlib.sha256(canonical_json(payload)).hexdigest()

    def verify_sha256(self) -> bool:
        return bool(self.record_sha256) and self.compute_sha256() == self.record_sha256

    # -- amendment primitive ----------------------------------------------

    def with_amendment(
        self, *, delta: AmendmentDeltaV2, catalogue: RuleCatalogue, engine: ApplicabilityEngine,
        calibrators: Optional[Mapping[str, Any]], created_at: str, bundle: Optional[Any] = None,
    ) -> "StructuredCaseRecordV2":
        """Return a NEW record (version+1) with exactly ``delta`` applied and every derived section recomputed."""
        state = _extract_build_state(self, bundle=bundle)
        state = _apply_delta(state, delta)
        return _assemble(
            state, catalogue, engine, calibrators=calibrators, created_at=created_at,
            record_version=self.identity.record_version + 1, parent_record_sha256=self.record_sha256,
            attachments=self.attachments,
        )




@dataclass(frozen=True)
class _BuildState:
    case_id: str
    source_label: str
    dermoscopic: ImageRef
    dermoscopic_input_present: bool
    sanitization_receipt: Dict[str, Any]
    clinical: ImageRef
    clinical_availability: str
    perception_mode: str
    model_id: str
    checkpoint_bindings: Tuple[Dict[str, Any], ...]
    calibration_sha256: Optional[str]
    calibration_state: Optional[str]
    raw_members: Tuple[Dict[str, Any], ...]  # {fold, diagnosis_logits, melanoma_logit, criterion_logits}
    reported_aggregate: Dict[str, Any]  # worker's own unrounded diagnosis/binary probabilities, audit-only (see PerceptionV2)
    reliability_raw: Dict[str, Any]  # {iqa_state, ood_state, view_stability_state, selective_abstention, disposition, scores, warnings}
    dermoscopic_present_override: Optional[bool]
    bundle: Optional[Any]
    criterion_value_overrides: Dict[str, str]
    clinician_assertions: Dict[str, Any]
    sex: Optional[str]
    site: Optional[str]
    elevation: Optional[str]
    age_years: Optional[float]
    jurisdiction: str
    evaluation_date: date


def _apply_delta(state: _BuildState, delta: AmendmentDeltaV2) -> _BuildState:
    fld = delta.field
    if fld in CRITERION_NAMES:
        overrides = dict(state.criterion_value_overrides)
        overrides[fld] = delta.after
        return dataclasses.replace(state, criterion_value_overrides=overrides)
    if fld in CLINICIAN_ASSERTION_FACTS:
        assertions = dict(state.clinician_assertions)
        assertions[fld] = {
            "value": delta.after,
            "provenance": {"source": ("record" if fld == "lesion.under_assessment" else "asserted"),
                           "actor": delta.actor, "session_id": delta.session_id, "asserted_at": None,
                           "statement": f"amendment: {fld} set to {delta.after!r}"},
        }
        return dataclasses.replace(state, clinician_assertions=assertions)
    if fld == "image.dermoscopic_present":
        return dataclasses.replace(state, dermoscopic_present_override=delta.after)
    raise RecordInputErrorV2(f"with_amendment: unsupported field {fld!r}")


def _extract_build_state(record: StructuredCaseRecordV2, *, bundle: Optional[Any] = None) -> _BuildState:
    p = record.perception
    raw_members = tuple(
        {"fold": m.fold, "diagnosis_logits": list(m.diagnosis_logits), "melanoma_logit": m.melanoma_logit,
         "criteria_logits": list(m.criterion_logits)}
        for m in p.members
    )
    model_values = criterion_values_from_probabilities(p.criterion_probabilities)
    overrides = {name: value for name, value in p.criterion_values.items() if model_values.get(name) != value}

    values = record.facts.get("values", {})
    provenance = record.facts.get("provenance", {})

    def observed(name: str) -> Any:
        prov = provenance.get(name, {})
        if prov.get("source") != "observed":
            return None
        return values.get(name)

    return _BuildState(
        case_id=record.identity.case_id, source_label=record.identity.source_label,
        dermoscopic=record.inputs.dermoscopic, dermoscopic_input_present=record.inputs.input_mask["dermoscopic"],
        sanitization_receipt=dict(record.inputs.sanitization_receipt), clinical=record.inputs.clinical,
        clinical_availability=record.inputs.clinical_availability,
        perception_mode=p.perception_mode, model_id=p.model_id, checkpoint_bindings=p.checkpoint_bindings,
        calibration_sha256=p.calibration_sha256, calibration_state=p.calibration_state,
        raw_members=raw_members, reported_aggregate={k: list(v) for k, v in p.reported_aggregate.items()},
        reliability_raw={
            "iqa_state": record.reliability.iqa_state, "ood_state": record.reliability.ood_state,
            "view_stability_state": record.reliability.view_stability_state,
            "selective_abstention": record.reliability.selective_abstention,
            "disposition": record.reliability.disposition, "scores": dict(record.reliability.scores),
            "warnings": tuple(dict(w) for w in record.reliability.warnings),
        },
        dermoscopic_present_override=record.reliability.dermoscopic_present_override,
        bundle=bundle, criterion_value_overrides=overrides, clinician_assertions=dict(record.clinician_assertions),
        sex=observed("patient.sex"), site=observed("lesion.site"), elevation=observed("lesion.elevation"),
        age_years=observed("patient.age_years"), jurisdiction=str(record.facts["jurisdiction"]),
        evaluation_date=date.fromisoformat(str(record.facts["evaluation_date"])),
    )


def effective_dermoscopic_present(dermoscopic_input_present: bool, override: Optional[bool]) -> bool:
    if override is False:
        return False
    return dermoscopic_input_present


def expected_model_criterion_values(criterion_probabilities: Mapping[str, float]) -> Dict[str, str]:
    return criterion_values_from_probabilities(criterion_probabilities)


def expected_effective_criterion_values(record: StructuredCaseRecordV2) -> Dict[str, str]:
    """The criterion_values a record OUGHT to carry: the model's own reading from the frozen, never-amended, aggregated ``criterion_probabilities``, with every ``criterion_value`` amendment entry in ``record.amendments`` replayed onto it, in order."""
    values = dict(criterion_values_from_probabilities(record.perception.criterion_probabilities))
    for entry in record.amendments:
        if entry.kind == "criterion_value" and entry.field in values:
            values[entry.field] = entry.after
    return values


def current_value(record: StructuredCaseRecordV2, field_name: str) -> Any:
    if field_name in CRITERION_NAMES:
        return record.perception.criterion_values[field_name]
    if field_name in CLINICIAN_ASSERTION_FACTS:
        return record.clinician_assertions[field_name]["value"]
    if field_name == "image.dermoscopic_present":
        return effective_dermoscopic_present(record.inputs.input_mask["dermoscopic"], record.reliability.dermoscopic_present_override)
    raise RecordInputErrorV2(f"unknown amendable field {field_name!r}")


def compute_next_step_v2(applicability: CaseApplicability, *, image_status: ImageStatusV2) -> NextStepV2:
    """Two-pass walk over ``NEXT_STEP_PRIORITY``: pass 1 looks for the first APPLICABLE rule (in priority order) across the WHOLE list and wins outright if found; only when no rule is applicable does pass 2 look for the first UNKNOWN-with-question rule."""
    summary = image_status.to_dict()
    by_id = {r.rule_id: r for r in applicability.results}

    for rid in NEXT_STEP_PRIORITY:
        r = by_id.get(rid)
        if r is not None and r.state is ApplicabilityState.APPLICABLE:
            phrase = ACTION_PHRASES.get(r.action_code or "", r.action_code or "the recommended action")
            modal = RULE_MODALS[rid]
            if modal == "shall":
                text = f"Guideline-supported next step: {phrase} (EADO 2024 {rid}, 'shall')"
            else:
                text = f"Guideline-supported next step: {phrase} ({rid}, '{modal}')"
            return NextStepV2(
                status="recommended", rule_id=rid, action_code=r.action_code, display_text=text,
                targeted_question=None, decisive_missing_fact=None, reason=None, image_status_summary=summary,
            )

    for rid in NEXT_STEP_PRIORITY:
        r = by_id.get(rid)
        if r is not None and r.state is ApplicabilityState.UNKNOWN and r.question:
            text = f"Diagnostic next step is conditional on one missing fact for {rid}: {r.question}"
            return NextStepV2(
                status="conditional", rule_id=rid, action_code=r.action_code, display_text=text,
                targeted_question=r.question, decisive_missing_fact=r.decisive_missing_fact,
                reason="decisive_fact_unknown", image_status_summary=summary,
            )

    return NextStepV2(
        status="withheld_no_coverage", rule_id=None, action_code=None,
        display_text="Diagnostic next step withheld: no catalogue rule is applicable, conditional, or unknown for this case.",
        targeted_question=None, decisive_missing_fact=None, reason="no_catalogue_coverage", image_status_summary=summary,
    )


def _compute_image_status(
    *, dermoscopic_input_present: bool, dermoscopic_present_override: Optional[bool],
    iqa_state: str, view_stability_state: str,
) -> ImageStatusV2:
    effective = effective_dermoscopic_present(dermoscopic_input_present, dermoscopic_present_override)
    if not effective:
        reasons = []
        if dermoscopic_present_override is False:
            reasons.append("reliability_override_no_usable_image")
        if not dermoscopic_input_present:
            reasons.append("no_dermoscopic_image_supplied")
        return ImageStatusV2(state="unavailable_no_image", reasons=tuple(reasons) or ("no_dermoscopic_image_supplied",),
                              recovery_action="acquire_dermoscopic_image")
    if iqa_state == "review_required" or view_stability_state == "unstable":
        reasons = tuple(c for c, flag in (
            ("image_quality_review_required", iqa_state == "review_required"),
            ("view_unstable", view_stability_state == "unstable"),
        ) if flag)
        return ImageStatusV2(state="limited_review_flagged", reasons=reasons, recovery_action="review_image_or_reacquire")
    if iqa_state == "not_assessed" and view_stability_state == "not_assessed":
        return ImageStatusV2(state="not_assessed", reasons=("reliability_not_assessed",),
                              recovery_action="complete_assessment_or_clinician_review")
    return ImageStatusV2(state="usable", reasons=(), recovery_action="none")


def _reliability_raw_from_decision(decision: Optional[Any]) -> Dict[str, Any]:
    """Normalise a real ``ReliabilityDecision`` (or ``None``, meaning Phase 3 was not run for this case) into the plain dict ``_BuildState.reliability_raw`` carries -- the single shape both the initial build and ``recompute_derived`` (which repopulates it from the record's own stored ``reliability`` section) feed into :func:`_build_reliability`."""
    if decision is None:
        return {"iqa_state": "not_assessed", "ood_state": "not_assessed", "view_stability_state": "not_assessed",
                "selective_abstention": "not_assessed", "disposition": "not_assessed", "scores": {}, "warnings": ()}
    return {
        "iqa_state": str(decision.iqa_state), "ood_state": str(decision.ood_state),
        "view_stability_state": str(decision.view_stability_state),
        "selective_abstention": str(decision.selective_abstention), "disposition": str(decision.disposition),
        "scores": dict(decision.scores), "warnings": tuple(dict(w) for w in decision.warnings),
    }


def _build_reliability(
    raw: Mapping[str, Any], bundle: Optional[Any], n_members: int, dermoscopic_present_override: Optional[bool],
) -> ReliabilityV2:
    thresholds = _thresholds_from_bundle(bundle, n_members)
    meaning = {"iqa": IQA_MEANING, "ood": OOD_MEANING}
    fact_mapping = {"image.iqa_state": None, "image.ood_state": None}
    return ReliabilityV2(
        iqa_state=str(raw["iqa_state"]), ood_state=str(raw["ood_state"]),
        view_stability_state=str(raw["view_stability_state"]), selective_abstention=str(raw["selective_abstention"]),
        disposition=str(raw["disposition"]), scores=dict(raw["scores"]), thresholds=thresholds,
        warnings=tuple(dict(w) for w in raw["warnings"]), phase3_bundle_version=V.PHASE3_BUNDLE_VERSION,
        phase3_bundle_sha256=_bundle_sha256(), meaning=meaning, dermoscopic_present_override=dermoscopic_present_override,
        fact_mapping=fact_mapping,
    )


def _assertion_kwargs(assertions: Mapping[str, Any]) -> Dict[str, Any]:
    return {kwarg: assertions[fact]["value"] for fact, kwarg in CLINICIAN_ASSERTION_TO_KWARG.items()}


def _assemble(
    state: _BuildState, catalogue: RuleCatalogue, engine: ApplicabilityEngine, *,
    calibrators: Optional[Mapping[str, Any]], created_at: str, record_version: int,
    parent_record_sha256: Optional[str], attachments: Optional[Mapping[str, Any]] = None,
) -> StructuredCaseRecordV2:
    effective_derm = effective_dermoscopic_present(state.dermoscopic_input_present, state.dermoscopic_present_override)

    members: List[MemberEvidence] = []
    if state.perception_mode == "five_member_deployment":
        if calibrators is None:
            raise RecordInputErrorV2("five_member_deployment requires calibrators")
        for raw in state.raw_members:
            fold = raw["fold"]
            diagnosis_logits = tuple(round_floats(list(raw["diagnosis_logits"])))
            melanoma_logit = round_floats(float(raw["melanoma_logit"]))
            criterion_logits = tuple(round_floats(list(raw["criteria_logits"])))
            calibrated_diag, calibrated_bin = _calibrate_member(
                diagnosis_logits, melanoma_logit, criterion_logits, fold, calibrators,
            )
            members.append(MemberEvidence(
                fold=fold, diagnosis_logits=diagnosis_logits, melanoma_logit=melanoma_logit,
                criterion_logits=criterion_logits, calibrated_diagnosis_probabilities=calibrated_diag,
                calibrated_binary_probabilities=calibrated_bin,
            ))
        diag = np.mean([m.calibrated_diagnosis_probabilities for m in members], axis=0)
        binary = np.mean([m.calibrated_binary_probabilities for m in members], axis=0)
        diag_agg = "arithmetic mean of member-calibrated probabilities"
        mel_agg = "arithmetic mean of member-calibrated binary probabilities"
    elif state.perception_mode == "single_member_oof":
        raw = state.raw_members[0]
        diagnosis_logits = tuple(round_floats(list(raw["diagnosis_logits"])))
        melanoma_logit = round_floats(float(raw["melanoma_logit"]))
        criterion_logits = tuple(round_floats(list(raw["criteria_logits"])))
        members.append(MemberEvidence(
            fold=raw["fold"], diagnosis_logits=diagnosis_logits, melanoma_logit=melanoma_logit,
            criterion_logits=criterion_logits, calibrated_diagnosis_probabilities=None,
            calibrated_binary_probabilities=None,
        ))
        diag = np.array(_softmax(diagnosis_logits))
        binary = np.array([_sigmoid(melanoma_logit), *[_sigmoid(x) for x in criterion_logits]])
        diag_agg = "raw softmax of single OOF member"
        mel_agg = "raw sigmoid of single OOF member"
    else:
        raise RecordInputErrorV2(f"unknown perception_mode {state.perception_mode!r}")

    diagnosis_probabilities = dict(zip(GLOBAL_CLASSES, (float(x) for x in diag)))
    provisional_class = GLOBAL_CLASSES[int(np.argmax(diag))]
    melanoma_probability = float(binary[0])
    criterion_probabilities = {name: float(binary[1 + i]) for i, name in enumerate(CRITERIA)}
    model_criterion_values = criterion_values_from_probabilities(criterion_probabilities)
    effective_criterion_values = dict(model_criterion_values)
    effective_criterion_values.update(state.criterion_value_overrides)

    perception = PerceptionV2(
        perception_mode=state.perception_mode, model_id=state.model_id, checkpoint_bindings=state.checkpoint_bindings,
        calibration_sha256=state.calibration_sha256, calibration_state=state.calibration_state,
        members=tuple(members),
        diagnosis_head={"probabilities": diagnosis_probabilities, "provisional_class": provisional_class, "aggregation": diag_agg},
        melanoma_head={"probability": melanoma_probability, "aggregation": mel_agg},
        criterion_probabilities=criterion_probabilities, model_criterion_values=model_criterion_values,
        criterion_values=effective_criterion_values,
        criterion_convention={"indeterminate_band": INDETERMINATE_BAND, "positive_representative": dict(POS_VALUE),
                              "negative_representative": dict(NEG_VALUE)},
        reported_aggregate={k: list(v) for k, v in state.reported_aggregate.items()},
        abstained=False,
    )

    try:
        normalised = validate_values(effective_criterion_values)
    except ValueError as exc:
        raise RecordInputErrorV2(f"invalid criterion values: {exc}") from exc
    scored = score_criteria(normalised)
    management_action, management_reason = recommend_management(scored)
    checklist = ChecklistResult(
        instrument_id=DERM7PT_INSTRUMENT,
        per_criterion=tuple(PerCriterionScore(c.name, c.value, c.points, c.max_points, c.determinate) for c in scored.per_criterion),
        score_min=scored.score_min, score_max=scored.score_max, decidable=scored.decidable,
        meets_threshold=(scored.meets_threshold if scored.decidable else None),
        indeterminate_criteria=tuple(scored.indeterminate),
        management_implication={"action": management_action, "reason": management_reason},
    )
    agreement = build_agreement(provisional_class, scored)

    image_status = _compute_image_status(
        dermoscopic_input_present=state.dermoscopic_input_present,
        dermoscopic_present_override=state.dermoscopic_present_override,
        iqa_state=str(state.reliability_raw["iqa_state"]), view_stability_state=str(state.reliability_raw["view_stability_state"]),
    )

    assertion_kwargs = _assertion_kwargs(state.clinician_assertions)
    facts = reviewed_v2.derive_case_facts(
        state.case_id,
        clinician_equivocal=assertion_kwargs["clinician_equivocal"],
        lesion_under_assessment=assertion_kwargs["lesion_under_assessment"],
        jurisdiction=state.jurisdiction, evaluation_date=state.evaluation_date,
        dermoscopic_present=effective_derm, iqa_state=None, ood_state=None,
        provisional_class=provisional_class, melanoma_probability=melanoma_probability,
        criterion_values=effective_criterion_values,
        instrument_id=DERM7PT_INSTRUMENT,
        sex=state.sex, site=state.site, elevation=state.elevation, age_years=state.age_years,
        clinician_melanoma_suspected=assertion_kwargs["clinician_melanoma_suspected"],
        high_risk_status=assertion_kwargs["high_risk_status"], total_nevus_count=assertion_kwargs["total_nevus_count"],
        high_total_nevus_count=assertion_kwargs["high_total_nevus_count"],
        personal_melanoma_history=assertion_kwargs["personal_melanoma_history"],
        genetic_predisposition=assertion_kwargs["genetic_predisposition"], rcm_available=assertion_kwargs["rcm_available"],
    )

    applicability_result = engine.evaluate_case(facts, expected_case_id=state.case_id)
    applicability_dict = {
        **applicability_result.to_dict(), "catalogue_sha256": catalogue.catalogue_sha256,
        "engine_policy_version": ENGINE_POLICY_VERSION,
    }
    next_step = compute_next_step_v2(applicability_result, image_status=image_status)

    reliability = _build_reliability(
        state.reliability_raw, state.bundle, len(members), state.dermoscopic_present_override,
    )

    versions = VersionPinsV2(
        model_id=state.model_id, calibration_sha256=state.calibration_sha256, calibration_state=state.calibration_state,
        preprocessing_version=PREPROCESSING_VERSION, rule_catalogue_sha256=catalogue.catalogue_sha256,
        derivation_policy_version=reviewed_v2.DERIVATION_POLICY_VERSION, engine_policy_version=ENGINE_POLICY_VERSION,
        record_schema_version=SCHEMA_VERSION, gate_policy_version=GATE_POLICY_VERSION, renderer_version=RENDERER_VERSION,
        phase3_bundle_sha256=_bundle_sha256(), selection_manifest_sha256=_selection_manifest_sha256(),
    )

    identity = IdentityV2(
        case_id=state.case_id, record_version=record_version, parent_record_sha256=parent_record_sha256,
        created_at=created_at, schema_version=SCHEMA_VERSION, source_label=state.source_label,
    )

    inputs = InputsV2(
        dermoscopic=state.dermoscopic, sanitization_receipt=state.sanitization_receipt, clinical=state.clinical,
        clinical_availability=state.clinical_availability,
        input_mask={"dermoscopic": state.dermoscopic_input_present, "clinical": state.clinical_availability == "present"},
    )

    record = StructuredCaseRecordV2(
        identity=identity, inputs=inputs, perception=perception, reliability=reliability, image_status=image_status,
        checklist=checklist, agreement=agreement, clinician_assertions=dict(state.clinician_assertions), facts=facts.to_dict(),
        applicability=applicability_dict, next_step=next_step, versions=versions, amendments=(), record_sha256="",
        attachments=dict(attachments) if attachments is not None else {"review_episode_ref": None, "xai_receipt_refs": []},
    )
    return dataclasses.replace(record, record_sha256=record.compute_sha256())


def finalize_with_amendments(record: StructuredCaseRecordV2, amendments: Sequence[AmendmentEntryV2]) -> StructuredCaseRecordV2:
    updated = dataclasses.replace(record, amendments=tuple(amendments), record_sha256="")
    return dataclasses.replace(updated, record_sha256=updated.compute_sha256())


def _diff_consequences(before: StructuredCaseRecordV2, after: StructuredCaseRecordV2) -> Dict[str, Any]:
    def checklist(r: StructuredCaseRecordV2) -> Dict[str, Any]:
        return {"score_min": r.checklist.score_min, "score_max": r.checklist.score_max,
                "decidable": r.checklist.decidable, "meets_threshold": r.checklist.meets_threshold}

    def agreement(r: StructuredCaseRecordV2) -> Dict[str, Any]:
        return {"present": r.agreement.present, "escalate": r.agreement.escalate,
                "criteria_path_flags_melanoma": r.agreement.criteria_path_flags_melanoma}

    def next_step(r: StructuredCaseRecordV2) -> Dict[str, Any]:
        return {"status": r.next_step.status, "rule_id": r.next_step.rule_id, "action_code": r.next_step.action_code}

    def applicability(r: StructuredCaseRecordV2) -> Dict[str, str]:
        return {row["rule_id"]: row["state"] for row in r.applicability.get("results", [])}

    def image_status(r: StructuredCaseRecordV2) -> Dict[str, Any]:
        return r.image_status.to_dict()

    ns_before, ns_after = next_step(before), next_step(after)
    return {
        "checklist_before": checklist(before), "checklist_after": checklist(after),
        "agreement_before": agreement(before), "agreement_after": agreement(after),
        "next_step_before": ns_before, "next_step_after": ns_after,
        "applicability_before": applicability(before), "applicability_after": applicability(after),
        "image_status_before": image_status(before), "image_status_after": image_status(after),
        "next_step_changed": ns_before != ns_after,
    }


@dataclass(frozen=True)
class ChainVerdict:
    passed: bool
    failures: Tuple[str, ...]
    n_records: int


def verify_chain(
    records: Sequence[StructuredCaseRecordV2], *, catalogue: RuleCatalogue, engine: ApplicabilityEngine,
    calibrators: Optional[Mapping[str, Any]], bundle: Optional[Any] = None,
) -> ChainVerdict:
    """Independently reconstruct a v2 record lineage from version 1 forward."""
    records = list(records)
    failures: List[str] = []
    if not records:
        return ChainVerdict(passed=True, failures=(), n_records=0)

    case_ids = {r.identity.case_id for r in records}
    if len(case_ids) > 1:
        failures.append("mixed_case_ids")

    versions = [r.identity.record_version for r in records]
    if versions != list(range(versions[0], versions[0] + len(records))):
        failures.append("versions_not_consecutive")

    for r in records:
        for entry in r.amendments:
            if not entry.token_sha256:
                failures.append(f"amendment_missing_token_hash:{entry.amendment_id}")
        expected_model_values = criterion_values_from_probabilities(r.perception.criterion_probabilities)
        if r.perception.model_criterion_values != expected_model_values:
            failures.append(f"model_criterion_values_forged_at_version_{r.identity.record_version}")
        if r.perception.criterion_values != expected_effective_criterion_values(r):
            failures.append(f"criterion_values_forged_at_version_{r.identity.record_version}")

    for i in range(1, len(records)):
        prev, cur = records[i - 1], records[i]
        if cur.identity.parent_record_sha256 != prev.record_sha256:
            failures.append(f"parent_hash_mismatch_at_version_{cur.identity.record_version}")
        if cur.amendments[: len(prev.amendments)] != prev.amendments:
            failures.append(f"amendments_not_prefix_extension_at_version_{cur.identity.record_version}")
        if len(cur.amendments) != len(prev.amendments) + 1:
            failures.append(f"amendments_length_unexpected_at_version_{cur.identity.record_version}")
            continue
        entry = cur.amendments[-1]
        try:
            actual_before = current_value(prev, entry.field)
        except RecordInputErrorV2:
            actual_before = None
        if entry.before != actual_before:
            failures.append(f"before_value_mismatch_at_version_{cur.identity.record_version}")
        delta = AmendmentDeltaV2(kind=entry.kind, field=entry.field, before=entry.before, after=entry.after,
                                 actor=entry.actor, session_id=entry.session_id)
        try:
            rebuilt = prev.with_amendment(delta=delta, catalogue=catalogue, engine=engine, calibrators=calibrators,
                                          created_at=entry.committed_at, bundle=bundle)
            fresh_diff = _diff_consequences(prev, rebuilt)
            if entry.recomputed_consequences != fresh_diff:
                failures.append(f"recomputed_consequences_mismatch_at_version_{cur.identity.record_version}")
            fresh_diff_sha = hashlib.sha256(canonical_json(fresh_diff)).hexdigest()
            if entry.consequence_preview_sha256 != fresh_diff_sha:
                failures.append(f"consequence_preview_sha256_mismatch_at_version_{cur.identity.record_version}")
            expected = finalize_with_amendments(rebuilt, cur.amendments)
        except Exception as exc:  # fail closed: any reconstruction error is a chain failure
            failures.append(f"reconstruction_failed_at_version_{cur.identity.record_version}: {exc}")
            continue
        if expected.record_sha256 != cur.record_sha256:
            failures.append(f"reconstruction_mismatch_at_version_{cur.identity.record_version}")

    return ChainVerdict(passed=not failures, failures=tuple(failures), n_records=len(records))


def recompute_derived(
    record: StructuredCaseRecordV2, *, catalogue: RuleCatalogue, engine: ApplicabilityEngine,
    calibrators: Optional[Mapping[str, Any]], bundle: Optional[Any] = None,
) -> StructuredCaseRecordV2:
    """Rebuild checklist/agreement/facts/applicability/next_step/image_status from the record's own stored perception evidence, clinician assertions and reliability states."""
    state = _extract_build_state(record, bundle=bundle)
    rebuilt = _assemble(
        state, catalogue, engine, calibrators=calibrators, created_at=record.identity.created_at,
        record_version=record.identity.record_version, parent_record_sha256=record.identity.parent_record_sha256,
        attachments=record.attachments,
    )
    if record.amendments:
        rebuilt = finalize_with_amendments(rebuilt, record.amendments)
    return rebuilt




def _clinician_assertions_from_inputs(
    inputs: Mapping[str, AssertionInput], *, lesion_under_assessment_value: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fact in CLINICIAN_ASSERTION_FACTS:
        if fact == "lesion.under_assessment":
            out[fact] = {
                "value": bool(lesion_under_assessment_value),
                "provenance": {"source": "record", "actor": None, "session_id": None, "asserted_at": None,
                               "statement": "host case scope: a lesion is under assessment"},
            }
            continue
        item = inputs.get(fact)
        value = item.value if item is not None else None
        if value is None:
            out[fact] = {"value": None, "provenance": {"source": "not_supplied", "actor": None, "session_id": None,
                                                         "asserted_at": None, "statement": ""}}
        else:
            out[fact] = {"value": value, "provenance": {"source": "asserted", "actor": item.actor,
                                                          "session_id": item.session_id, "asserted_at": item.asserted_at,
                                                          "statement": item.statement}}
    return out


def build_record_from_prediction(
    case_id: str,
    prediction: Mapping[str, Any],
    decision: Optional[Any],
    bundle: Optional[Any],
    clinician_assertions: Mapping[str, AssertionInput],
    observed_facts: Mapping[str, Any],
    evaluation_date: date,
    created_at: str,
    source_label: str = "development_allowlisted_case",
    relative_paths: Optional[Mapping[str, str]] = None,
    *,
    jurisdiction: str = "EU",
    model_id: str = V.PRIMARY_MODEL_ID,
    catalogue: RuleCatalogue,
    engine: ApplicabilityEngine,
    lesion_under_assessment: bool = True,
) -> StructuredCaseRecordV2:
    """Build a version-1 deployment-mode record from a ``SelectedPerception.predict``- shaped ``prediction`` dict and its Phase 3 ``decision`` (or ``None`` when reliability was not assessed for this case)."""
    if source_label not in SOURCE_LABELS:
        raise RecordInputErrorV2(f"unknown source_label {source_label!r}")
    calibrators = bundle.calibration if bundle is not None else None
    if calibrators is None:
        raise RecordInputErrorV2("deployment-mode records require a ReliabilityBundle for calibrators")

    raw = prediction["raw_fold_outputs"]
    if len(raw) != 5:
        raise RecordInputErrorV2("deployment-mode records require exactly five member raw fold outputs")
    raw_members = tuple(
        {"fold": i, "diagnosis_logits": list(raw[i]["diagnosis"]), "melanoma_logit": float(raw[i]["melanoma"]),
         "criteria_logits": list(raw[i]["criteria"])}
        for i in range(5)
    )

    relative_paths = relative_paths or {}
    image_receipt = prediction.get("image_receipt") or {}
    clinical_receipt = prediction.get("clinical_receipt")
    clinical_present = bool(prediction.get("clinical_present"))
    if clinical_present:
        clinical_availability = "present"
    elif clinical_receipt is not None and clinical_receipt.get("availability") == "duplicate_dermoscopy_excluded":
        clinical_availability = "duplicate_dermoscopy_excluded"
    elif "clinical" in relative_paths or clinical_receipt is not None:
        clinical_availability = "absent"
    else:
        clinical_availability = "not_supplied"

    assertions = _clinician_assertions_from_inputs(clinician_assertions, lesion_under_assessment_value=lesion_under_assessment)

    state = _BuildState(
        case_id=case_id, source_label=source_label,
        dermoscopic=ImageRef(image_receipt.get("input_sha256"), image_receipt.get("sanitized_sha256"),
                             relative_paths.get("dermoscopic")),
        dermoscopic_input_present=True, sanitization_receipt=dict(image_receipt),
        clinical=ImageRef((clinical_receipt or {}).get("input_sha256"), (clinical_receipt or {}).get("sanitized_sha256"),
                          relative_paths.get("clinical")),
        clinical_availability=clinical_availability,
        perception_mode="five_member_deployment", model_id=model_id,
        checkpoint_bindings=tuple(dict(b) for b in prediction["checkpoint_bindings"]),
        calibration_sha256=str(prediction["calibration_sha256"]), calibration_state=None,
        raw_members=raw_members,
        reported_aggregate={
            "diagnosis_probabilities": [float(x) for x in prediction["diagnosis_probabilities"]],
            "binary_probabilities": [float(x) for x in prediction["binary_probabilities"]],
        },
        reliability_raw=_reliability_raw_from_decision(decision),
        dermoscopic_present_override=None, bundle=bundle,
        criterion_value_overrides={}, clinician_assertions=assertions,
        sex=observed_facts.get("sex"), site=observed_facts.get("site"), elevation=observed_facts.get("elevation"),
        age_years=observed_facts.get("age_years"), jurisdiction=jurisdiction, evaluation_date=evaluation_date,
    )
    return _assemble(state, catalogue, engine, calibrators=calibrators, created_at=created_at, record_version=1,
                     parent_record_sha256=None)


def build_record_from_oof(
    case_id: str,
    *,
    diagnosis_logits: Sequence[float],
    melanoma_logit: float,
    criterion_logits: Sequence[float],
    oof_manifest: Mapping[str, Any],
    clinician_assertions: Mapping[str, AssertionInput],
    observed_facts: Mapping[str, Any],
    evaluation_date: date,
    created_at: str,
    catalogue: RuleCatalogue,
    engine: ApplicabilityEngine,
    source_label: str = "development_oof_benchmark",
    jurisdiction: str = "EU",
    model_id: str = V.PRIMARY_MODEL_ID,
    dermoscopic_present: bool = True,
    relative_paths: Optional[Mapping[str, str]] = None,
    lesion_under_assessment: bool = True,
) -> StructuredCaseRecordV2:
    """Build a version-1 single-member out-of-fold benchmark record: no image is opened, the perception evidence is the frozen OOF logits, and every downstream probability is raw (uncalibrated)."""
    if source_label not in SOURCE_LABELS:
        raise RecordInputErrorV2(f"unknown source_label {source_label!r}")
    relative_paths = relative_paths or {}
    assertions = _clinician_assertions_from_inputs(clinician_assertions, lesion_under_assessment_value=lesion_under_assessment)

    state = _BuildState(
        case_id=case_id, source_label=source_label,
        dermoscopic=ImageRef(None, None, relative_paths.get("dermoscopic")),
        dermoscopic_input_present=dermoscopic_present, sanitization_receipt={"status": "not_assessed"},
        clinical=ImageRef(None, None, relative_paths.get("clinical")), clinical_availability="not_supplied",
        perception_mode="single_member_oof", model_id=model_id,
        checkpoint_bindings=({"fold": None, "path": oof_manifest.get("path"), "sha256": oof_manifest.get("sha256")},),
        calibration_sha256=None, calibration_state="raw_oof_uncalibrated",
        raw_members=({"fold": None, "diagnosis_logits": list(diagnosis_logits), "melanoma_logit": float(melanoma_logit),
                      "criteria_logits": list(criterion_logits)},),
        reported_aggregate={},
        reliability_raw=_reliability_raw_from_decision(None), dermoscopic_present_override=None, bundle=None,
        criterion_value_overrides={}, clinician_assertions=assertions,
        sex=observed_facts.get("sex"), site=observed_facts.get("site"), elevation=observed_facts.get("elevation"),
        age_years=observed_facts.get("age_years"), jurisdiction=jurisdiction, evaluation_date=evaluation_date,
    )
    return _assemble(state, catalogue, engine, calibrators=None, created_at=created_at, record_version=1,
                     parent_record_sha256=None)


__all__ = [
    "ACTION_PHRASES",
    "AmendmentDeltaV2",
    "AmendmentEntryV2",
    "AssertionInput",
    "CLINICIAN_ASSERTION_FACTS",
    "CLINICIAN_ASSERTION_TO_KWARG",
    "ChainVerdict",
    "ChecklistResult",
    "FACT_MAPPING_REASON",
    "IQA_MEANING",
    "ImageRef",
    "ImageStatusV2",
    "InputsV2",
    "IdentityV2",
    "MemberEvidence",
    "NEXT_STEP_PRIORITY",
    "NextStepV2",
    "OOD_MEANING",
    "PerceptionV2",
    "RULE_MODALS",
    "RecordInputErrorV2",
    "ReliabilityV2",
    "SCHEMA_VERSION",
    "SCOPE_NOTE",
    "StructuredCaseRecordV2",
    "VersionPinsV2",
    "build_agreement",
    "build_record_from_oof",
    "build_record_from_prediction",
    "compute_next_step_v2",
    "criterion_values_from_probabilities",
    "current_value",
    "effective_dermoscopic_present",
    "expected_effective_criterion_values",
    "expected_model_criterion_values",
    "finalize_with_amendments",
    "recompute_derived",
    "round_floats",
    "verify_chain",
]
