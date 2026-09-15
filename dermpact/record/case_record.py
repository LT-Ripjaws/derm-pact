"""Phase 5: the immutable, versioned structured case record (CONTEXT.md S15.3)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from dermpact.eval.dermpact_phase4_benchmark import (
    CRITERIA,
    GLOBAL_CLASSES,
    INDETERMINATE_BAND,
    NEG_VALUE,
    POS_VALUE,
    criterion_values_from_logits,
)
from dermpact.guidelines.derivation import (
    DERIVATION_POLICY_VERSION,
    DERM7PT_INSTRUMENT,
    derive_case_facts,
)
from dermpact.guidelines.engine import ApplicabilityEngine, CaseApplicability
from dermpact.guidelines.rules import ApplicabilityState, RuleCatalogue, canonical_json
from dermpact.criteria.aggregate import SevenPointScore, recommend_management, score_criteria
from dermpact.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD, validate_values


SCHEMA_VERSION = "dermpact-phase5-case-record-v1"
ENGINE_POLICY_VERSION = "dermpact-phase5-engine-policy-v1"
GATE_POLICY_VERSION = "dermpact-phase5-gate-v1"
TEMPLATE_VERSION = "dermpact-phase5-template-v1"
PREPROCESSING_VERSION = "dermpact-phase2-4b-preprocessing-v1"
OOF_MODEL_TAG = "phase2_4b_true_pair_oof"
CALIBRATION_VERSION = "none:phase3_pending"

CLINICIAN_FIELD_TO_ATTR: Mapping[str, str] = {
    "clinician.melanoma_suspected": "clinician_melanoma_suspected",
    "patient.high_risk_status": "high_risk_status",
    "patient.total_nevus_count": "total_nevus_count",
    "patient.high_total_nevus_count": "high_total_nevus_count",
    "patient.personal_melanoma_history": "personal_melanoma_history",
    "patient.genetic_predisposition": "genetic_predisposition",
    "site.rcm_available": "rcm_available",
}
CLINICIAN_FACT_NAMES: Tuple[str, ...] = tuple(CLINICIAN_FIELD_TO_ATTR)

#: Host priority for the single displayed next step: melanoma pathway first.
NEXT_STEP_PRIORITY: Tuple[str, ...] = (
    "eado-mel-p1-2024:R3",
    "eado-mel-p1-2024:R7",
    "eado-mel-p1-2024:R4",
    "eado-mel-p1-2024:R6",
    "eado-mel-p1-2024:R5",
)

#: Short, host-owned phrase per action code -- never generated, always this table.
ACTION_PHRASES: Mapping[str, str] = {
    "histopathological_confirmation": "biopsy/excision (histopathological confirmation)",
    "dermatoscopic_assessment": "dermatoscopic assessment",
    "whole_body_photography_surveillance": "whole-body photography with sequential examinations",
    "sequential_digital_dermatoscopy": "sequential digital dermatoscopy",
    "reflectance_confocal_microscopy": "reflectance confocal microscopy",
}


class RecordInputError(ValueError):
    """A record builder/amendment primitive was given malformed input. Fail closed."""




def _round(value: float) -> float:
    r = round(float(value), 6)
    return 0.0 if r == 0 else r


def round_floats(value: Any) -> Any:
    """Recursively round every float in a JSON-shaped structure to 6 dp."""
    if isinstance(value, float):
        return _round(value)
    if isinstance(value, Mapping):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    return value


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _softmax(xs: Sequence[float]) -> Tuple[float, ...]:
    m = max(float(x) for x in xs)
    exps = [math.exp(float(x) - m) for x in xs]
    s = sum(exps)
    return tuple(e / s for e in exps)


def _tv_to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    return None  # "unknown" or an unrecognised token never round-trips to a bool




@dataclass(frozen=True)
class RecordIdentity:
    case_id: str
    record_version: int
    parent_record_sha256: Optional[str]
    created_at: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "record_version": self.record_version,
            "parent_record_sha256": self.parent_record_sha256,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecordIdentity":
        return cls(
            case_id=str(payload["case_id"]),
            record_version=int(payload["record_version"]),
            parent_record_sha256=payload.get("parent_record_sha256"),
            created_at=str(payload["created_at"]),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class MetadataStrippingReceipt:
    status: str = "not_assessed"
    phase3_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "phase3_version": self.phase3_version}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetadataStrippingReceipt":
        return cls(status=str(payload.get("status", "not_assessed")), phase3_version=payload.get("phase3_version"))


@dataclass(frozen=True)
class InputMask:
    dermoscopic: bool
    clinical: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"dermoscopic": self.dermoscopic, "clinical": self.clinical}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InputMask":
        return cls(dermoscopic=bool(payload["dermoscopic"]), clinical=bool(payload["clinical"]))


@dataclass(frozen=True)
class InputManifest:
    dermoscopic_sha256: str
    dermoscopic_relative_path: str
    clinical_sha256: Optional[str]
    clinical_relative_path: Optional[str]
    clinical_unavailable_reason: Optional[str]
    input_mask: InputMask
    metadata_stripping_receipt: MetadataStrippingReceipt
    inventory_sha256: str
    fold_manifest_sha256: str
    fold: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dermoscopic_sha256": self.dermoscopic_sha256,
            "dermoscopic_relative_path": self.dermoscopic_relative_path,
            "clinical_sha256": self.clinical_sha256,
            "clinical_relative_path": self.clinical_relative_path,
            "clinical_unavailable_reason": self.clinical_unavailable_reason,
            "input_mask": self.input_mask.to_dict(),
            "metadata_stripping_receipt": self.metadata_stripping_receipt.to_dict(),
            "inventory_sha256": self.inventory_sha256,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "fold": self.fold,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InputManifest":
        return cls(
            dermoscopic_sha256=str(payload["dermoscopic_sha256"]),
            dermoscopic_relative_path=str(payload["dermoscopic_relative_path"]),
            clinical_sha256=payload.get("clinical_sha256"),
            clinical_relative_path=payload.get("clinical_relative_path"),
            clinical_unavailable_reason=payload.get("clinical_unavailable_reason"),
            input_mask=InputMask.from_dict(payload["input_mask"]),
            metadata_stripping_receipt=MetadataStrippingReceipt.from_dict(payload["metadata_stripping_receipt"]),
            inventory_sha256=str(payload["inventory_sha256"]),
            fold_manifest_sha256=str(payload["fold_manifest_sha256"]),
            fold=int(payload["fold"]),
        )


@dataclass(frozen=True)
class ReliabilityStates:
    """Phase-3 reliability outputs."""

    iqa_state: str = "not_assessed"
    ood_state: str = "not_assessed"
    view_stability_state: str = "not_assessed"
    selective_abstention: str = "not_assessed"
    calibration_state: str = "uncalibrated"
    phase3_version: Optional[str] = None
    dermoscopic_present_override: Optional[bool] = None
    fact_mapping: Optional[Dict[str, Optional[str]]] = None

    def __post_init__(self) -> None:
        if self.fact_mapping is None:
            def to_fact(state: str) -> Optional[str]:
                return None if state == "not_assessed" else state
            object.__setattr__(self, "fact_mapping", {
                "image.iqa_state": to_fact(self.iqa_state),
                "image.ood_state": to_fact(self.ood_state),
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iqa_state": self.iqa_state,
            "ood_state": self.ood_state,
            "view_stability_state": self.view_stability_state,
            "selective_abstention": self.selective_abstention,
            "calibration_state": self.calibration_state,
            "phase3_version": self.phase3_version,
            "dermoscopic_present_override": self.dermoscopic_present_override,
            "fact_mapping": dict(self.fact_mapping),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReliabilityStates":
        fm = payload.get("fact_mapping")
        return cls(
            iqa_state=str(payload.get("iqa_state", "not_assessed")),
            ood_state=str(payload.get("ood_state", "not_assessed")),
            view_stability_state=str(payload.get("view_stability_state", "not_assessed")),
            selective_abstention=str(payload.get("selective_abstention", "not_assessed")),
            calibration_state=str(payload.get("calibration_state", "uncalibrated")),
            phase3_version=payload.get("phase3_version"),
            dermoscopic_present_override=payload.get("dermoscopic_present_override"),
            fact_mapping=(dict(fm) if fm is not None else None),
        )


@dataclass(frozen=True)
class LockedPerception:
    diagnosis_logits: Tuple[float, ...]
    diagnosis_probabilities: Dict[str, float]
    provisional_class: str
    criterion_logits: Tuple[float, ...]
    criterion_probabilities: Dict[str, float]
    criterion_values: Dict[str, str]
    model_criterion_values: Dict[str, str]
    melanoma_logit: float
    melanoma_probability: float
    abstained: bool
    model_version: str
    preprocessing_version: str
    criterion_convention: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_logits": list(self.diagnosis_logits),
            "diagnosis_probabilities": dict(self.diagnosis_probabilities),
            "provisional_class": self.provisional_class,
            "criterion_logits": list(self.criterion_logits),
            "criterion_probabilities": dict(self.criterion_probabilities),
            "criterion_values": dict(self.criterion_values),
            "model_criterion_values": dict(self.model_criterion_values),
            "melanoma_logit": self.melanoma_logit,
            "melanoma_probability": self.melanoma_probability,
            "abstained": self.abstained,
            "model_version": self.model_version,
            "preprocessing_version": self.preprocessing_version,
            "criterion_convention": dict(self.criterion_convention),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LockedPerception":
        return cls(
            diagnosis_logits=tuple(float(x) for x in payload["diagnosis_logits"]),
            diagnosis_probabilities={str(k): float(v) for k, v in payload["diagnosis_probabilities"].items()},
            provisional_class=str(payload["provisional_class"]),
            criterion_logits=tuple(float(x) for x in payload["criterion_logits"]),
            criterion_probabilities={str(k): float(v) for k, v in payload["criterion_probabilities"].items()},
            criterion_values={str(k): str(v) for k, v in payload["criterion_values"].items()},
            model_criterion_values={str(k): str(v) for k, v in payload["model_criterion_values"].items()},
            melanoma_logit=float(payload["melanoma_logit"]),
            melanoma_probability=float(payload["melanoma_probability"]),
            abstained=bool(payload["abstained"]),
            model_version=str(payload["model_version"]),
            preprocessing_version=str(payload["preprocessing_version"]),
            criterion_convention=dict(payload["criterion_convention"]),
        )


@dataclass(frozen=True)
class PerCriterionScore:
    name: str
    value: str
    points: int
    max_points: int
    determinate: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "points": self.points,
            "max_points": self.max_points,
            "determinate": self.determinate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerCriterionScore":
        return cls(
            name=str(payload["name"]),
            value=str(payload["value"]),
            points=int(payload["points"]),
            max_points=int(payload["max_points"]),
            determinate=bool(payload["determinate"]),
        )


@dataclass(frozen=True)
class ChecklistResult:
    instrument_id: str
    per_criterion: Tuple[PerCriterionScore, ...]
    score_min: int
    score_max: int
    decidable: bool
    meets_threshold: Optional[bool]
    indeterminate_criteria: Tuple[str, ...]
    management_implication: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "per_criterion": [c.to_dict() for c in self.per_criterion],
            "score_min": self.score_min,
            "score_max": self.score_max,
            "decidable": self.decidable,
            "meets_threshold": self.meets_threshold,
            "indeterminate_criteria": list(self.indeterminate_criteria),
            "management_implication": dict(self.management_implication),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChecklistResult":
        return cls(
            instrument_id=str(payload["instrument_id"]),
            per_criterion=tuple(PerCriterionScore.from_dict(c) for c in payload["per_criterion"]),
            score_min=int(payload["score_min"]),
            score_max=int(payload["score_max"]),
            decidable=bool(payload["decidable"]),
            meets_threshold=payload.get("meets_threshold"),
            indeterminate_criteria=tuple(str(x) for x in payload["indeterminate_criteria"]),
            management_implication=dict(payload["management_implication"]),
        )


@dataclass(frozen=True)
class DirectChecklistAgreement:
    present: bool
    classifier_says: str
    criteria_path_flags_melanoma: bool
    description: str
    escalate: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "present": self.present,
            "classifier_says": self.classifier_says,
            "criteria_path_flags_melanoma": self.criteria_path_flags_melanoma,
            "description": self.description,
            "escalate": self.escalate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DirectChecklistAgreement":
        return cls(
            present=bool(payload["present"]),
            classifier_says=str(payload["classifier_says"]),
            criteria_path_flags_melanoma=bool(payload["criteria_path_flags_melanoma"]),
            description=str(payload["description"]),
            escalate=bool(payload["escalate"]),
        )


@dataclass(frozen=True)
class RecommendedNextStep:
    status: str
    rule_id: Optional[str]
    action_code: Optional[str]
    display_text: str
    targeted_question: Optional[str]
    reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rule_id": self.rule_id,
            "action_code": self.action_code,
            "display_text": self.display_text,
            "targeted_question": self.targeted_question,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecommendedNextStep":
        return cls(
            status=str(payload["status"]),
            rule_id=payload.get("rule_id"),
            action_code=payload.get("action_code"),
            display_text=str(payload["display_text"]),
            targeted_question=payload.get("targeted_question"),
            reason=payload.get("reason"),
        )


@dataclass(frozen=True)
class VersionPins:
    model_version: str
    calibration_version: str
    preprocessing_version: str
    rule_catalogue_sha256: str
    derivation_policy_version: str
    engine_policy_version: str
    record_schema_version: str
    gate_policy_version: str
    template_version: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VersionPins":
        return cls(**{f.name: str(payload[f.name]) for f in dataclasses.fields(cls)})


@dataclass(frozen=True)
class ClinicianInputs:
    """The seven facts a clinician may assert. ``None`` means unasserted/unknown."""

    clinician_melanoma_suspected: Optional[bool] = None
    high_risk_status: Optional[bool] = None
    total_nevus_count: Optional[int] = None
    high_total_nevus_count: Optional[bool] = None
    personal_melanoma_history: Optional[bool] = None
    genetic_predisposition: Optional[bool] = None
    rcm_available: Optional[bool] = None

    def validate(self) -> None:
        bool_fields = (
            "clinician_melanoma_suspected", "high_risk_status", "high_total_nevus_count",
            "personal_melanoma_history", "genetic_predisposition", "rcm_available",
        )
        for name in bool_fields:
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise RecordInputError(f"ClinicianInputs.{name} must be a bool or None, got {value!r}")
        if self.total_nevus_count is not None:
            if isinstance(self.total_nevus_count, bool) or not isinstance(self.total_nevus_count, int):
                raise RecordInputError(f"ClinicianInputs.total_nevus_count must be an int or None, got {self.total_nevus_count!r}")

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ClinicianInputs":
        allowed = {f.name for f in dataclasses.fields(cls)}
        unknown = set(mapping) - allowed
        if unknown:
            raise RecordInputError(f"unknown ClinicianInputs field(s): {sorted(unknown)}")
        inst = cls(**{k: mapping[k] for k in mapping})
        inst.validate()
        return inst

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AmendmentDelta:
    """One resolved, already-validated change to apply to a record."""

    kind: str
    field: str
    before: Any
    after: Any


@dataclass(frozen=True)
class AmendmentEntry:
    amendment_id: str
    kind: str
    field: str
    before: Any
    after: Any
    actor: str
    proposed_at: str
    committed_at: str
    proposal_sha256: str
    token_sha256: str
    consequence_preview_sha256: str
    recomputed_consequences: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amendment_id": self.amendment_id,
            "kind": self.kind,
            "field": self.field,
            "before": self.before,
            "after": self.after,
            "actor": self.actor,
            "proposed_at": self.proposed_at,
            "committed_at": self.committed_at,
            "proposal_sha256": self.proposal_sha256,
            "token_sha256": self.token_sha256,
            "consequence_preview_sha256": self.consequence_preview_sha256,
            "recomputed_consequences": dict(self.recomputed_consequences),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AmendmentEntry":
        return cls(
            amendment_id=str(payload["amendment_id"]),
            kind=str(payload["kind"]),
            field=str(payload["field"]),
            before=payload.get("before"),
            after=payload.get("after"),
            actor=str(payload["actor"]),
            proposed_at=str(payload["proposed_at"]),
            committed_at=str(payload["committed_at"]),
            proposal_sha256=str(payload["proposal_sha256"]),
            token_sha256=str(payload["token_sha256"]),
            consequence_preview_sha256=str(payload["consequence_preview_sha256"]),
            recomputed_consequences=dict(payload.get("recomputed_consequences", {})),
        )




@dataclass(frozen=True)
class StructuredCaseRecord:
    identity: RecordIdentity
    inputs: InputManifest
    reliability: ReliabilityStates
    perception: LockedPerception
    checklist: ChecklistResult
    agreement: DirectChecklistAgreement
    facts: Dict[str, Any]
    applicability: Dict[str, Any]
    next_step: RecommendedNextStep
    versions: VersionPins
    amendments: Tuple[AmendmentEntry, ...]
    record_sha256: str

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "identity": self.identity.to_dict(),
            "inputs": self.inputs.to_dict(),
            "reliability": self.reliability.to_dict(),
            "perception": self.perception.to_dict(),
            "checklist": self.checklist.to_dict(),
            "agreement": self.agreement.to_dict(),
            "facts": self.facts,
            "applicability": self.applicability,
            "next_step": self.next_step.to_dict(),
            "versions": self.versions.to_dict(),
            "amendments": [a.to_dict() for a in self.amendments],
            "record_sha256": self.record_sha256,
        }
        return round_floats(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredCaseRecord":
        return cls(
            identity=RecordIdentity.from_dict(payload["identity"]),
            inputs=InputManifest.from_dict(payload["inputs"]),
            reliability=ReliabilityStates.from_dict(payload["reliability"]),
            perception=LockedPerception.from_dict(payload["perception"]),
            checklist=ChecklistResult.from_dict(payload["checklist"]),
            agreement=DirectChecklistAgreement.from_dict(payload["agreement"]),
            facts=json.loads(json.dumps(payload["facts"])),
            applicability=json.loads(json.dumps(payload["applicability"])),
            next_step=RecommendedNextStep.from_dict(payload["next_step"]),
            versions=VersionPins.from_dict(payload["versions"]),
            amendments=tuple(AmendmentEntry.from_dict(a) for a in payload.get("amendments", ())),
            record_sha256=str(payload.get("record_sha256", "")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("record_sha256", None)
        return hashlib.sha256(canonical_json(payload)).hexdigest()

    def verify_sha256(self) -> bool:
        return bool(self.record_sha256) and self.compute_sha256() == self.record_sha256

    # -- amendment primitive ----------------------------------------------

    def with_amendment(
        self, *, delta: AmendmentDelta, catalogue: RuleCatalogue, engine: ApplicabilityEngine, created_at: str,
    ) -> "StructuredCaseRecord":
        """Return a NEW record (version+1) with exactly ``delta`` applied and every derived section recomputed."""
        state = _extract_build_state(self)
        state = _apply_delta(state, delta)
        return _assemble(
            state, catalogue, engine,
            created_at=created_at,
            record_version=self.identity.record_version + 1,
            parent_record_sha256=self.record_sha256,
        )




@dataclass(frozen=True)
class _BuildState:
    case_id: str
    fold: int
    dermoscopic_sha256: str
    dermoscopic_relative_path: str
    dermoscopic_file_available: bool
    clinical_sha256: Optional[str]
    clinical_relative_path: Optional[str]
    clinical_unavailable_reason: Optional[str]
    inventory_sha256: str
    fold_manifest_sha256: str
    reliability: ReliabilityStates
    diagnosis_logits: Tuple[float, ...]
    criterion_logits: Tuple[float, ...]
    melanoma_logit: float
    criterion_values: Dict[str, str]
    model_version: str
    preprocessing_version: str
    criterion_convention: Dict[str, Any]
    sex: Optional[str]
    site: Optional[str]
    elevation: Optional[str]
    age_years: Optional[float]
    clinician_inputs: ClinicianInputs
    jurisdiction: str
    evaluation_date: date


def _apply_delta(state: _BuildState, delta: AmendmentDelta) -> _BuildState:
    fld = delta.field
    if fld in CRITERION_NAMES:
        new_values = dict(state.criterion_values)
        new_values[fld] = delta.after
        return dataclasses.replace(state, criterion_values=new_values)
    if fld in CLINICIAN_FIELD_TO_ATTR:
        attr = CLINICIAN_FIELD_TO_ATTR[fld]
        new_clinician = dataclasses.replace(state.clinician_inputs, **{attr: delta.after})
        return dataclasses.replace(state, clinician_inputs=new_clinician)
    if fld == "image.dermoscopic_present":
        new_reliability = dataclasses.replace(state.reliability, dermoscopic_present_override=delta.after)
        return dataclasses.replace(state, reliability=new_reliability)
    raise RecordInputError(f"with_amendment: unsupported field {fld!r}")


def _extract_build_state(record: StructuredCaseRecord) -> _BuildState:
    values = record.facts.get("values", {})
    provenance = record.facts.get("provenance", {})

    def val_if(name: str, source: str) -> Any:
        prov = provenance.get(name, {})
        if prov.get("source") != source:
            return None
        return values.get(name)

    clinician_inputs = ClinicianInputs(
        clinician_melanoma_suspected=_tv_to_bool(val_if("clinician.melanoma_suspected", "asserted")),
        high_risk_status=_tv_to_bool(val_if("patient.high_risk_status", "asserted")),
        total_nevus_count=val_if("patient.total_nevus_count", "asserted"),
        high_total_nevus_count=_tv_to_bool(val_if("patient.high_total_nevus_count", "asserted")),
        personal_melanoma_history=_tv_to_bool(val_if("patient.personal_melanoma_history", "asserted")),
        genetic_predisposition=_tv_to_bool(val_if("patient.genetic_predisposition", "asserted")),
        rcm_available=_tv_to_bool(val_if("site.rcm_available", "asserted")),
    )
    return _BuildState(
        case_id=record.identity.case_id,
        fold=record.inputs.fold,
        dermoscopic_sha256=record.inputs.dermoscopic_sha256,
        dermoscopic_relative_path=record.inputs.dermoscopic_relative_path,
        dermoscopic_file_available=record.inputs.input_mask.dermoscopic,
        clinical_sha256=record.inputs.clinical_sha256,
        clinical_relative_path=record.inputs.clinical_relative_path,
        clinical_unavailable_reason=record.inputs.clinical_unavailable_reason,
        inventory_sha256=record.inputs.inventory_sha256,
        fold_manifest_sha256=record.inputs.fold_manifest_sha256,
        reliability=record.reliability,
        diagnosis_logits=record.perception.diagnosis_logits,
        criterion_logits=record.perception.criterion_logits,
        melanoma_logit=record.perception.melanoma_logit,
        criterion_values=dict(record.perception.criterion_values),
        model_version=record.perception.model_version,
        preprocessing_version=record.perception.preprocessing_version,
        criterion_convention=dict(record.perception.criterion_convention),
        sex=val_if("patient.sex", "observed"),
        site=val_if("lesion.site", "observed"),
        elevation=val_if("lesion.elevation", "observed"),
        age_years=val_if("patient.age_years", "observed"),
        clinician_inputs=clinician_inputs,
        jurisdiction=str(record.facts["jurisdiction"]),
        evaluation_date=date.fromisoformat(str(record.facts["evaluation_date"])),
    )


def effective_dermoscopic_present(record: StructuredCaseRecord) -> bool:
    """True unless a ``reliability_override`` amendment asserted no usable image."""
    return record.reliability.dermoscopic_present_override is not False


def expected_effective_criterion_values(record: StructuredCaseRecord) -> Dict[str, str]:
    """The criterion_values a record OUGHT to carry: the model's own reading (``criterion_values_from_logits`` of the frozen, never-amended ``criterion_logits``) with every ``criterion_value`` amendment entry in ``record.amendments`` replayed onto it, in order."""
    values = dict(criterion_values_from_logits(record.perception.criterion_logits))
    for entry in record.amendments:
        if entry.kind == "criterion_value" and entry.field in values:
            values[entry.field] = entry.after
    return values


def current_value(record: StructuredCaseRecord, field_name: str) -> Any:
    """The record's current value for one amendable field (used by the ledger)."""
    if field_name in CRITERION_NAMES:
        return record.perception.criterion_values[field_name]
    if field_name in CLINICIAN_FIELD_TO_ATTR:
        state = _extract_build_state(record)
        return getattr(state.clinician_inputs, CLINICIAN_FIELD_TO_ATTR[field_name])
    if field_name == "image.dermoscopic_present":
        return effective_dermoscopic_present(record)
    raise RecordInputError(f"unknown amendable field {field_name!r}")


def build_agreement(provisional_class: str, scored: SevenPointScore) -> DirectChecklistAgreement:
    """Build a conflict entry for the phase-5 vocabulary."""
    classifier_says_mel = provisional_class == "melanoma"
    if not scored.decidable:
        return DirectChecklistAgreement(
            present=False,
            classifier_says=provisional_class,
            criteria_path_flags_melanoma=False,
            description=(
                f"The direct classifier reports {provisional_class}. The checklist score range "
                f"[{scored.score_min},{scored.score_max}] spans the melanoma-suspicion threshold of "
                f"{MELANOMA_SUSPICION_THRESHOLD} because of unresolved criteria "
                f"{list(scored.indeterminate)}: the checklist has not reached a verdict, so agreement "
                "cannot be assessed. Resolve the unresolved criteria to decide."
            ),
            escalate=True,
        )
    criteria_flags = scored.meets_threshold
    if criteria_flags and not classifier_says_mel:
        return DirectChecklistAgreement(
            present=True,
            classifier_says=provisional_class,
            criteria_path_flags_melanoma=True,
            description=(
                f"The direct classifier reports {provisional_class}, but the 7-point checklist scores "
                f"{scored.score_min}, at or above the threshold of {MELANOMA_SUSPICION_THRESHOLD}. The "
                "two views disagree in the direction that matters: the checklist is raising melanoma "
                "suspicion the direct classifier is not."
            ),
            escalate=True,
        )
    if classifier_says_mel and not criteria_flags:
        return DirectChecklistAgreement(
            present=True,
            classifier_says=provisional_class,
            criteria_path_flags_melanoma=False,
            description=(
                "The direct classifier reports melanoma, but the checklist scores "
                f"{scored.score_max}, below the threshold of {MELANOMA_SUSPICION_THRESHOLD}. The "
                "classifier is raising suspicion the checklist does not support."
            ),
            escalate=True,
        )
    return DirectChecklistAgreement(
        present=False,
        classifier_says=provisional_class,
        criteria_path_flags_melanoma=criteria_flags,
        description="Both views agree.",
        escalate=False,
    )


def compute_next_step(
    applicability: CaseApplicability, *, perception_usable: bool, checklist_decidable: bool,
) -> RecommendedNextStep:
    """Host-owned mapping from applicability to the one displayed next step."""
    if not perception_usable:
        return RecommendedNextStep(
            status="withheld_perception_unavailable", rule_id=None, action_code=None,
            display_text=(
                "Diagnostic next step withheld: perception outputs are unavailable for this case "
                "(no usable dermoscopic image, or image quality failed)."
            ),
            targeted_question=None, reason="perception_unavailable",
        )
    by_id = {r.rule_id: r for r in applicability.results}
    for rid in NEXT_STEP_PRIORITY:
        r = by_id.get(rid)
        if r is None:
            continue
        if r.state is ApplicabilityState.APPLICABLE:
            phrase = ACTION_PHRASES.get(r.action_code or "", r.action_code or "the recommended action")
            text = (
                f"Recommended diagnostic next step: consider {phrase} according to the applicable "
                f"guideline ({rid}) and local practice."
            )
            return RecommendedNextStep(
                status="recommended", rule_id=rid, action_code=r.action_code,
                display_text=text, targeted_question=None, reason=None,
            )
        if r.state is ApplicabilityState.UNKNOWN and r.question:
            if r.decisive_missing_fact == "lesion.melanoma_suspected" and not checklist_decidable:
                status, reason = "withheld_unknown", "checklist_undecidable"
            else:
                status, reason = "conditional", "decisive_fact_unknown"
            display = (
                f"Diagnostic next step withheld: the checklist is undecidable. {r.question}"
                if status == "withheld_unknown"
                else f"Diagnostic next step is conditional on one missing fact for {rid}: {r.question}"
            )
            return RecommendedNextStep(
                status=status, rule_id=rid, action_code=r.action_code,
                display_text=display, targeted_question=r.question, reason=reason,
            )
        # NOT_APPLICABLE / NO_COVERAGE: fall through to the next priority rule.
    if applicability.targeted_question is not None:
        rid, _fact, question = applicability.targeted_question
        return RecommendedNextStep(
            status="withheld_unknown", rule_id=rid, action_code=None,
            display_text=f"Diagnostic next step withheld: a decisive fact is unknown. {question}",
            targeted_question=question, reason="decisive_fact_unknown",
        )
    return RecommendedNextStep(
        status="withheld_no_coverage", rule_id=None, action_code=None,
        display_text=(
            "Diagnostic next step withheld: no catalogue rule is applicable, conditional, or unknown "
            "for this case."
        ),
        targeted_question=None, reason="no_catalogue_coverage",
    )


def _assemble(
    state: _BuildState, catalogue: RuleCatalogue, engine: ApplicabilityEngine, *,
    created_at: str, record_version: int, parent_record_sha256: Optional[str],
) -> StructuredCaseRecord:
    effective_derm = state.reliability.dermoscopic_present_override is not False
    fact_mapping = state.reliability.fact_mapping
    iqa_state_value = fact_mapping["image.iqa_state"]
    ood_state_value = fact_mapping["image.ood_state"]
    perception_usable = (iqa_state_value != "fail") and effective_derm

    diag_probs = _softmax(state.diagnosis_logits)
    provisional_class = GLOBAL_CLASSES[max(range(len(state.diagnosis_logits)), key=lambda i: state.diagnosis_logits[i])]
    melanoma_probability = _sigmoid(state.melanoma_logit)
    criterion_probs = {name: _sigmoid(logit) for name, logit in zip(CRITERIA, state.criterion_logits)}

    model_criterion_values = criterion_values_from_logits(state.criterion_logits)

    perception = LockedPerception(
        diagnosis_logits=tuple(state.diagnosis_logits),
        diagnosis_probabilities=dict(zip(GLOBAL_CLASSES, diag_probs)),
        provisional_class=provisional_class,
        criterion_logits=tuple(state.criterion_logits),
        criterion_probabilities=criterion_probs,
        criterion_values=dict(state.criterion_values),
        model_criterion_values=dict(model_criterion_values),
        melanoma_logit=state.melanoma_logit,
        melanoma_probability=melanoma_probability,
        abstained=(provisional_class == "abstained"),
        model_version=state.model_version,
        preprocessing_version=state.preprocessing_version,
        criterion_convention=dict(state.criterion_convention),
    )

    try:
        normalised = validate_values(state.criterion_values)
    except ValueError as exc:
        raise RecordInputError(f"invalid criterion values: {exc}") from exc
    scored = score_criteria(normalised)
    management_action, management_reason = recommend_management(scored)
    checklist = ChecklistResult(
        instrument_id=DERM7PT_INSTRUMENT,
        per_criterion=tuple(
            PerCriterionScore(c.name, c.value, c.points, c.max_points, c.determinate) for c in scored.per_criterion
        ),
        score_min=scored.score_min,
        score_max=scored.score_max,
        decidable=scored.decidable,
        meets_threshold=(scored.meets_threshold if scored.decidable else None),
        indeterminate_criteria=tuple(scored.indeterminate),
        management_implication={"action": management_action, "reason": management_reason},
    )

    agreement = build_agreement(provisional_class, scored)

    facts = derive_case_facts(
        state.case_id,
        jurisdiction=state.jurisdiction,
        evaluation_date=state.evaluation_date,
        dermoscopic_present=effective_derm,
        iqa_state=iqa_state_value,
        ood_state=ood_state_value,
        provisional_class=provisional_class,
        melanoma_probability=melanoma_probability,
        criterion_values=state.criterion_values,
        instrument_id=DERM7PT_INSTRUMENT,
        sex=state.sex, site=state.site, elevation=state.elevation, age_years=state.age_years,
        clinician_melanoma_suspected=state.clinician_inputs.clinician_melanoma_suspected,
        high_risk_status=state.clinician_inputs.high_risk_status,
        total_nevus_count=state.clinician_inputs.total_nevus_count,
        high_total_nevus_count=state.clinician_inputs.high_total_nevus_count,
        personal_melanoma_history=state.clinician_inputs.personal_melanoma_history,
        genetic_predisposition=state.clinician_inputs.genetic_predisposition,
        rcm_available=state.clinician_inputs.rcm_available,
    )

    applicability_result = engine.evaluate_case(facts, expected_case_id=state.case_id)
    applicability_dict = {
        **applicability_result.to_dict(),
        "catalogue_sha256": catalogue.catalogue_sha256,
        "engine_policy_version": ENGINE_POLICY_VERSION,
    }

    next_step = compute_next_step(
        applicability_result, perception_usable=perception_usable, checklist_decidable=scored.decidable,
    )

    inputs = InputManifest(
        dermoscopic_sha256=state.dermoscopic_sha256,
        dermoscopic_relative_path=state.dermoscopic_relative_path,
        clinical_sha256=state.clinical_sha256,
        clinical_relative_path=state.clinical_relative_path,
        clinical_unavailable_reason=state.clinical_unavailable_reason,
        input_mask=InputMask(dermoscopic=state.dermoscopic_file_available, clinical=state.clinical_sha256 is not None),
        metadata_stripping_receipt=MetadataStrippingReceipt(status="not_assessed", phase3_version=None),
        inventory_sha256=state.inventory_sha256,
        fold_manifest_sha256=state.fold_manifest_sha256,
        fold=state.fold,
    )

    versions = VersionPins(
        model_version=state.model_version,
        calibration_version=CALIBRATION_VERSION,
        preprocessing_version=state.preprocessing_version,
        rule_catalogue_sha256=catalogue.catalogue_sha256,
        derivation_policy_version=DERIVATION_POLICY_VERSION,
        engine_policy_version=ENGINE_POLICY_VERSION,
        record_schema_version=SCHEMA_VERSION,
        gate_policy_version=GATE_POLICY_VERSION,
        template_version=TEMPLATE_VERSION,
    )

    identity = RecordIdentity(
        case_id=state.case_id, record_version=record_version, parent_record_sha256=parent_record_sha256,
        created_at=created_at, schema_version=SCHEMA_VERSION,
    )

    record = StructuredCaseRecord(
        identity=identity, inputs=inputs, reliability=state.reliability, perception=perception,
        checklist=checklist, agreement=agreement, facts=facts.to_dict(), applicability=applicability_dict,
        next_step=next_step, versions=versions, amendments=(), record_sha256="",
    )
    return dataclasses.replace(record, record_sha256=record.compute_sha256())


def finalize_with_amendments(record: StructuredCaseRecord, amendments: Sequence[AmendmentEntry]) -> StructuredCaseRecord:
    """Attach a full ``amendments`` tuple and recompute the hash over it."""
    updated = dataclasses.replace(record, amendments=tuple(amendments), record_sha256="")
    return dataclasses.replace(updated, record_sha256=updated.compute_sha256())


def _diff_consequences(before: StructuredCaseRecord, after: StructuredCaseRecord) -> Dict[str, Any]:
    """The consequence preview/diff an amendment proposal or commit computes: before/after checklist, agreement, next_step and applicability states."""
    def checklist(r: StructuredCaseRecord) -> Dict[str, Any]:
        return {
            "score_min": r.checklist.score_min, "score_max": r.checklist.score_max,
            "decidable": r.checklist.decidable, "meets_threshold": r.checklist.meets_threshold,
        }

    def agreement(r: StructuredCaseRecord) -> Dict[str, Any]:
        return {"present": r.agreement.present, "escalate": r.agreement.escalate,
                "criteria_path_flags_melanoma": r.agreement.criteria_path_flags_melanoma}

    def next_step(r: StructuredCaseRecord) -> Dict[str, Any]:
        return {"status": r.next_step.status, "rule_id": r.next_step.rule_id, "action_code": r.next_step.action_code}

    def applicability(r: StructuredCaseRecord) -> Dict[str, str]:
        return {row["rule_id"]: row["state"] for row in r.applicability.get("results", [])}

    ns_before, ns_after = next_step(before), next_step(after)
    return {
        "checklist_before": checklist(before), "checklist_after": checklist(after),
        "agreement_before": agreement(before), "agreement_after": agreement(after),
        "next_step_before": ns_before, "next_step_after": ns_after,
        "applicability_before": applicability(before), "applicability_after": applicability(after),
        "next_step_changed": ns_before != ns_after,
    }


@dataclass(frozen=True)
class ChainVerdict:
    passed: bool
    failures: Tuple[str, ...]
    n_records: int


def verify_chain(
    records: Sequence[StructuredCaseRecord], *, catalogue: RuleCatalogue, engine: ApplicabilityEngine,
) -> ChainVerdict:
    """Independently reconstruct a record lineage from version 1 forward."""
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
        expected_model_values = criterion_values_from_logits(r.perception.criterion_logits)
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
        except RecordInputError:
            actual_before = None
        if entry.before != actual_before:
            failures.append(f"before_value_mismatch_at_version_{cur.identity.record_version}")
        delta = AmendmentDelta(kind=entry.kind, field=entry.field, before=entry.before, after=entry.after)
        try:
            rebuilt = prev.with_amendment(delta=delta, catalogue=catalogue, engine=engine, created_at=entry.committed_at)
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


def build_case_record(
    *,
    case: Any,  # dermpact.eval.dermpact_phase2_4b_inputs.Phase24BCaseInput
    perception_row: Mapping[str, Any],
    oof_sha256: str,
    catalogue: RuleCatalogue,
    engine: ApplicabilityEngine,
    evaluation_date: date,
    created_at: str,
    reliability: ReliabilityStates,
    clinician_inputs: ClinicianInputs,
    inventory_sha256: str,
    fold_manifest_sha256: str,
    jurisdiction: str = "EU",
    model_tag: str = OOF_MODEL_TAG,
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> StructuredCaseRecord:
    """Build the version-1 record for one locked development case."""
    if not isinstance(clinician_inputs, ClinicianInputs):
        raise RecordInputError("clinician_inputs must be a ClinicianInputs instance")
    clinician_inputs.validate()

    diag_logits = perception_row.get("diagnosis_logits")
    crit_logits = perception_row.get("criterion_logits")
    mel_logit = perception_row.get("melanoma_logit")
    if diag_logits is None or crit_logits is None or mel_logit is None:
        raise RecordInputError("perception_row must carry diagnosis_logits, criterion_logits and melanoma_logit")
    if len(diag_logits) != len(GLOBAL_CLASSES):
        raise RecordInputError(f"diagnosis_logits must have {len(GLOBAL_CLASSES)} entries, got {len(diag_logits)}")
    if len(crit_logits) != len(CRITERIA):
        raise RecordInputError(f"criterion_logits must have {len(CRITERIA)} entries, got {len(crit_logits)}")

    criterion_values = criterion_values_from_logits(crit_logits)
    criterion_convention = {
        "indeterminate_band": INDETERMINATE_BAND,
        "positive_representative": dict(POS_VALUE),
        "negative_representative": dict(NEG_VALUE),
    }
    if not model_tag or not preprocessing_version:
        raise RecordInputError("model and preprocessing provenance must be explicit")
    model_version = f"{oof_sha256}:{model_tag}"

    state = _BuildState(
        case_id=case.group_id,
        fold=case.fold,
        dermoscopic_sha256=case.dermoscopic_sha256,
        dermoscopic_relative_path=case.dermoscopic_relative_path,
        dermoscopic_file_available=True,
        clinical_sha256=case.clinical_sha256,
        clinical_relative_path=case.clinical_relative_path,
        clinical_unavailable_reason=case.clinical_unavailable_reason,
        inventory_sha256=inventory_sha256,
        fold_manifest_sha256=fold_manifest_sha256,
        reliability=reliability,
        diagnosis_logits=tuple(float(x) for x in diag_logits),
        criterion_logits=tuple(float(x) for x in crit_logits),
        melanoma_logit=float(mel_logit),
        criterion_values=criterion_values,
        model_version=model_version,
        preprocessing_version=preprocessing_version,
        criterion_convention=criterion_convention,
        sex=case.facts.sex,
        site=case.facts.site,
        elevation=case.facts.elevation,
        age_years=case.facts.age_years,
        clinician_inputs=clinician_inputs,
        jurisdiction=jurisdiction,
        evaluation_date=evaluation_date,
    )
    return _assemble(state, catalogue, engine, created_at=created_at, record_version=1, parent_record_sha256=None)


def recompute_derived(
    record: StructuredCaseRecord, *, catalogue: RuleCatalogue, engine: ApplicabilityEngine,
) -> StructuredCaseRecord:
    """Rebuild checklist/agreement/facts/applicability/next_step from the record's own stored perception, clinician-asserted facts and reliability states."""
    state = _extract_build_state(record)
    rebuilt = _assemble(
        state, catalogue, engine,
        created_at=record.identity.created_at,
        record_version=record.identity.record_version,
        parent_record_sha256=record.identity.parent_record_sha256,
    )
    if record.amendments:
        rebuilt = finalize_with_amendments(rebuilt, record.amendments)
    return rebuilt


__all__ = [
    "ACTION_PHRASES",
    "AmendmentDelta",
    "AmendmentEntry",
    "CLINICIAN_FACT_NAMES",
    "CLINICIAN_FIELD_TO_ATTR",
    "CALIBRATION_VERSION",
    "ChainVerdict",
    "ChecklistResult",
    "ClinicianInputs",
    "DirectChecklistAgreement",
    "ENGINE_POLICY_VERSION",
    "GATE_POLICY_VERSION",
    "InputManifest",
    "InputMask",
    "LockedPerception",
    "MetadataStrippingReceipt",
    "NEXT_STEP_PRIORITY",
    "OOF_MODEL_TAG",
    "PREPROCESSING_VERSION",
    "PerCriterionScore",
    "RecommendedNextStep",
    "RecordIdentity",
    "RecordInputError",
    "ReliabilityStates",
    "SCHEMA_VERSION",
    "StructuredCaseRecord",
    "TEMPLATE_VERSION",
    "VersionPins",
    "build_agreement",
    "build_case_record",
    "compute_next_step",
    "current_value",
    "effective_dermoscopic_present",
    "expected_effective_criterion_values",
    "finalize_with_amendments",
    "recompute_derived",
    "round_floats",
    "verify_chain",
]
