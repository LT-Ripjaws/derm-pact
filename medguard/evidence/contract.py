"""Typed cross-modal evidence contract for the active MedGuard thesis.

The host owns perception state and evidence provenance.  A generator may propose
claims and declare relations, but those declarations are not treated as scores.
Deterministic scoring derives image support from the frozen perception binding
and records the origin of every external-evidence relation separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import math
import re
from typing import Any, Dict, List, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = "medguard-image-evidence-contract-v1"
HAM7_CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONCEPT = re.compile(r"[a-z0-9][a-z0-9_.:-]*")


class ImageSupport(str, Enum):
    """Generator-declared relation; deterministic scorers must not trust it."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class EntailmentState(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    NOT_ASSESSED = "not_assessed"


class RelationOrigin(str, Enum):
    GENERATOR_DECLARED = "generator_declared"
    HOST_DETERMINISTIC = "host_deterministic"
    HUMAN_REVIEWED = "human_reviewed"
    NOT_ASSESSED = "not_assessed"


class ClaimType(str, Enum):
    ASSESSMENT = "assessment"
    DIAGNOSIS = "diagnosis"
    DIFFERENTIAL = "differential"
    VISUAL_FINDING = "visual_finding"
    MANAGEMENT = "management"
    REFERRAL = "referral"
    TREATMENT = "treatment"
    IMAGE_EXEMPLAR_SUPPORT = "image_exemplar_support"


class FindingType(str, Enum):
    DIAGNOSTIC_COMMITMENT = "diagnostic_commitment"
    VISUAL_OBSERVATION = "visual_observation"


class FindingState(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class FindingOrigin(str, Enum):
    FROZEN_PERCEPTION = "frozen_perception"


class UncertaintyType(str, Enum):
    IMAGE_SUPPORT = "image_support"
    EVIDENCE_SUPPORT = "evidence_support"
    DIAGNOSIS = "diagnosis"
    SOURCE_CONFLICT = "source_conflict"
    OTHER = "other"


class OutputObligation(str, Enum):
    CITE_SUPPORTED_CLAIMS = "cite_supported_claims"
    DISCLOSE_UNRESOLVED_UNCERTAINTY = "disclose_unresolved_uncertainty"
    ABSTAIN_ON_UNRESOLVED_CONFLICT = "abstain_on_unresolved_conflict"


REQUIRED_OUTPUT_OBLIGATIONS = tuple(OutputObligation)

# Claims discharged by the patient image rather than by retrieved evidence.
# Defined here, next to ClaimType, so the contract's own validation and the
# deterministic scorer cannot drift apart on which claims are image-bound.
IMAGE_BOUND_CLAIM_TYPES = frozenset({ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL})


class VerificationState(str, Enum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    NEEDS_REMEDIATION = "needs_remediation"
    FAILED_CLOSED = "failed_closed"


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256.fullmatch(str(value)):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(
            f"{context} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


@dataclass(frozen=True)
class ImageFinding:
    finding_id: str
    finding_type: FindingType
    concept_code: str
    state: FindingState
    confidence: float
    origin: FindingOrigin
    source_record_sha256: str

    def validate(self) -> None:
        if not self.finding_id.strip():
            raise ValueError("finding_id must be non-empty")
        if not isinstance(self.finding_type, FindingType):
            raise ValueError("finding_type must be closed")
        if not _CONCEPT.fullmatch(self.concept_code):
            raise ValueError(f"finding {self.finding_id!r} has a non-canonical concept_code")
        if not isinstance(self.state, FindingState):
            raise ValueError("finding state must be closed")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"finding {self.finding_id!r} confidence must lie in [0, 1]")
        if not isinstance(self.origin, FindingOrigin):
            raise ValueError("finding origin must be closed")
        _require_sha256(self.source_record_sha256, "source_record_sha256")


@dataclass(frozen=True)
class PerceptionBinding:
    artifact_id: str
    artifact_sha256: str
    record_sha256: str
    model_id: str
    class_order: Sequence[str]
    diagnosis_probabilities: Sequence[float]
    view_image_sha256s: Sequence[str]
    findings: Sequence[ImageFinding]

    def validate(self) -> None:
        if not self.artifact_id.strip() or not self.model_id.strip():
            raise ValueError("perception artifact_id and model_id must be non-empty")
        _require_sha256(self.artifact_sha256, "perception artifact_sha256")
        _require_sha256(self.record_sha256, "perception record_sha256")
        if tuple(self.class_order) != HAM7_CLASS_ORDER:
            raise ValueError("perception class_order must equal the frozen HAM7 order")
        probabilities = [float(value) for value in self.diagnosis_probabilities]
        if len(probabilities) != len(HAM7_CLASS_ORDER):
            raise ValueError("diagnosis_probabilities must match class_order")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("diagnosis probabilities must be finite values in [0, 1]")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("diagnosis probabilities must sum to one")
        if not self.view_image_sha256s:
            raise ValueError("perception binding requires at least one source-image hash")
        for digest in self.view_image_sha256s:
            _require_sha256(str(digest), "view_image_sha256")
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("image finding identifiers must be unique")
        for finding in self.findings:
            finding.validate()
            if finding.source_record_sha256 != self.record_sha256:
                raise ValueError("image finding is not bound to the perception record")
        commitments = [
            finding for finding in self.findings
            if finding.finding_type is FindingType.DIAGNOSTIC_COMMITMENT
        ]
        if len(commitments) != 1:
            raise ValueError("perception binding requires exactly one diagnostic commitment")
        commitment = commitments[0]
        if commitment.state is not FindingState.PRESENT:
            raise ValueError("diagnostic commitment must have state=present")
        if commitment.concept_code not in HAM7_CLASS_ORDER:
            raise ValueError("diagnostic commitment must use a HAM7 diagnosis code")
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        if HAM7_CLASS_ORDER[top_index] != commitment.concept_code:
            raise ValueError("diagnostic commitment does not match probability argmax")
        if abs(probabilities[top_index] - float(commitment.confidence)) > 1e-9:
            raise ValueError("commitment confidence does not match its class probability")

    @property
    def commitment(self) -> ImageFinding:
        self.validate()
        return next(
            finding for finding in self.findings
            if finding.finding_type is FindingType.DIAGNOSTIC_COMMITMENT
        )


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    text: str
    source_id: str
    source_version: str
    authority: str
    retrieval_path: Sequence[str]
    content_sha256: str
    provenance_uri: str = ""
    document_id: str = ""
    citation_id: str = ""
    eligible_claim_types: Sequence[str] = field(default_factory=tuple)
    retrieval_channel: str = ""
    record_sha256: str = ""

    def validate(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if not self.text.strip():
            raise ValueError(f"evidence {self.evidence_id!r} has no text")
        if not self.source_id.strip():
            raise ValueError(f"evidence {self.evidence_id!r} has no source_id")
        if not self.source_version.strip() or self.source_version.strip().lower() == "unknown":
            raise ValueError(f"evidence {self.evidence_id!r} has no known source_version")
        if (not self.authority.strip()
                or self.authority.strip().lower() in {"unknown", "unclassified"}):
            raise ValueError(f"evidence {self.evidence_id!r} has no known authority class")
        if not self.retrieval_path or any(not str(part).strip() for part in self.retrieval_path):
            raise ValueError(f"evidence {self.evidence_id!r} has no complete retrieval path")
        observed = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_sha256 != observed:
            raise ValueError(f"evidence {self.evidence_id!r} content hash mismatch")
        if bool(self.document_id) != bool(self.citation_id):
            raise ValueError(
                f"evidence {self.evidence_id!r} must carry document_id and citation_id together")
        if any(not str(value).strip() for value in self.eligible_claim_types):
            raise ValueError(f"evidence {self.evidence_id!r} has an empty claim-scope label")
        if self.record_sha256:
            _require_sha256(self.record_sha256, "evidence record_sha256")


@dataclass(frozen=True)
class ClinicalClaim:
    claim_id: str
    text: str
    image_support: ImageSupport
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    entailment: EntailmentState = EntailmentState.NOT_ASSESSED
    contradiction: bool = False
    claim_type: ClaimType = ClaimType.ASSESSMENT
    target_concept: str = ""
    target_state: FindingState = FindingState.UNCERTAIN
    image_finding_ids: Sequence[str] = field(default_factory=tuple)
    evidence_relation_origin: RelationOrigin = RelationOrigin.NOT_ASSESSED
    evidence_relation_assessor_id: str = ""
    evidence_relation_receipt_sha256: str = ""

    def validate(self) -> None:
        if not self.claim_id.strip() or not self.text.strip():
            raise ValueError("claim_id and claim text must be non-empty")
        if not isinstance(self.claim_type, ClaimType):
            raise ValueError("claim_type must be a closed ClaimType value")
        if not isinstance(self.image_support, ImageSupport):
            raise ValueError("image_support must be a closed declared value")
        if not isinstance(self.entailment, EntailmentState):
            raise ValueError("entailment must be a closed value")
        if not isinstance(self.evidence_relation_origin, RelationOrigin):
            raise ValueError("evidence_relation_origin must be closed")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError(f"claim {self.claim_id!r} repeats an evidence_id")
        if len(self.image_finding_ids) != len(set(self.image_finding_ids)):
            raise ValueError(f"claim {self.claim_id!r} repeats an image_finding_id")
        if self.entailment is EntailmentState.CONTRADICTED and not self.contradiction:
            raise ValueError("contradicted claim must set contradiction=true")
        if self.contradiction and self.entailment is not EntailmentState.CONTRADICTED:
            raise ValueError("contradiction=true requires entailment=contradicted")
        if self.entailment is EntailmentState.SUPPORTED and not self.evidence_ids:
            raise ValueError("supported claim requires evidence_ids")
        if (self.entailment is EntailmentState.NOT_ASSESSED
                and self.evidence_relation_origin is not RelationOrigin.NOT_ASSESSED):
            raise ValueError("not-assessed entailment must have origin=not_assessed")
        if (self.entailment is not EntailmentState.NOT_ASSESSED
                and self.evidence_relation_origin is RelationOrigin.NOT_ASSESSED):
            raise ValueError("assessed entailment requires an assessment origin")
        if self.evidence_relation_origin in {
            RelationOrigin.HOST_DETERMINISTIC, RelationOrigin.HUMAN_REVIEWED,
        }:
            if not self.evidence_relation_assessor_id.strip():
                raise ValueError("independent evidence relation requires an assessor_id")
            _require_sha256(
                self.evidence_relation_receipt_sha256,
                "evidence_relation_receipt_sha256",
            )
        elif self.evidence_relation_origin is RelationOrigin.GENERATOR_DECLARED:
            if self.evidence_relation_assessor_id != "generator_declared":
                raise ValueError("generator relation must use assessor_id=generator_declared")
            if self.evidence_relation_receipt_sha256:
                raise ValueError("generator relation cannot carry a trusted assessor receipt")
        elif self.evidence_relation_assessor_id or self.evidence_relation_receipt_sha256:
            raise ValueError("not-assessed relation cannot carry assessor lineage")
        if self.claim_type in {
            ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL, ClaimType.VISUAL_FINDING,
        }:
            if not _CONCEPT.fullmatch(self.target_concept):
                raise ValueError(f"claim {self.claim_id!r} requires a canonical target_concept")
            if not self.image_finding_ids:
                raise ValueError(f"claim {self.claim_id!r} requires image_finding_ids")
            if self.image_support is ImageSupport.NOT_APPLICABLE:
                raise ValueError(f"claim {self.claim_id!r} cannot mark image support not-applicable")
            if (self.claim_type in {ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL}
                    and self.target_concept not in HAM7_CLASS_ORDER):
                raise ValueError(f"claim {self.claim_id!r} diagnosis target is outside HAM7")
        elif self.claim_type is ClaimType.IMAGE_EXEMPLAR_SUPPORT:
            if self.target_concept not in HAM7_CLASS_ORDER:
                raise ValueError(
                    f"claim {self.claim_id!r} exemplar target is outside HAM7")
            if self.target_state is not FindingState.PRESENT:
                raise ValueError(
                    f"claim {self.claim_id!r} exemplar target must have state=present")
            if self.image_finding_ids:
                raise ValueError(
                    f"claim {self.claim_id!r} exemplar relation cannot cite patient findings")
            if self.image_support is not ImageSupport.NOT_APPLICABLE:
                raise ValueError(
                    f"claim {self.claim_id!r} exemplar relation requires not_applicable")
        else:
            if self.target_concept or self.image_finding_ids:
                raise ValueError(
                    f"claim {self.claim_id!r} type {self.claim_type.value} is not image-bound")
            if self.image_support is not ImageSupport.NOT_APPLICABLE:
                raise ValueError(
                    f"claim {self.claim_id!r} type {self.claim_type.value} requires not_applicable")


@dataclass(frozen=True)
class UnresolvedUncertainty:
    uncertainty_id: str
    uncertainty_type: UncertaintyType
    description: str
    related_claim_ids: Sequence[str] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.uncertainty_id.strip() or not self.description.strip():
            raise ValueError("uncertainty id and description must be non-empty")
        if not isinstance(self.uncertainty_type, UncertaintyType):
            raise ValueError("uncertainty_type must be closed")
        if not self.related_claim_ids:
            raise ValueError("uncertainty must bind at least one claim")


@dataclass
class EvidenceContract:
    case_id: str
    modality: str
    clinical_question: str
    perception: PerceptionBinding
    claims: List[ClinicalClaim]
    evidence: List[EvidenceItem]
    unresolved_uncertainties: List[UnresolvedUncertainty] = field(default_factory=list)
    output_obligations: Sequence[OutputObligation] = REQUIRED_OUTPUT_OBLIGATIONS
    verification_state: VerificationState = VerificationState.PENDING
    proposed_action: str = ""
    termination_reason: str = ""
    schema_version: str = CONTRACT_SCHEMA_VERSION

    @property
    def committed_diagnosis(self) -> str:
        return self.perception.commitment.concept_code

    @property
    def calibrated_confidence(self) -> float:
        return float(self.perception.commitment.confidence)

    @property
    def image_findings(self) -> List[str]:
        return [finding.concept_code for finding in self.perception.findings]

    def validate(self) -> "EvidenceContract":
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("unknown evidence-contract schema_version")
        if not self.case_id.strip() or not self.modality.strip() or not self.clinical_question.strip():
            raise ValueError("case_id, modality, and clinical_question must be non-empty")
        if not isinstance(self.perception, PerceptionBinding):
            raise ValueError("contract requires a typed perception binding")
        self.perception.validate()
        evidence_ids = [item.evidence_id for item in self.evidence]
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence identifiers must be unique")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim identifiers must be unique")
        finding_ids = {finding.finding_id for finding in self.perception.findings}
        for item in self.evidence:
            item.validate()
        known_evidence = set(evidence_ids)
        for claim in self.claims:
            claim.validate()
            dangling = set(claim.evidence_ids) - known_evidence
            if dangling:
                raise ValueError(
                    f"claim {claim.claim_id!r} references unknown evidence: {sorted(dangling)}")
            dangling_findings = set(claim.image_finding_ids) - finding_ids
            if dangling_findings:
                raise ValueError(
                    f"claim {claim.claim_id!r} references unknown findings: {sorted(dangling_findings)}")
        uncertainty_ids = [item.uncertainty_id for item in self.unresolved_uncertainties]
        if len(uncertainty_ids) != len(set(uncertainty_ids)):
            raise ValueError("uncertainty identifiers must be unique")
        known_claims = set(claim_ids)
        for uncertainty in self.unresolved_uncertainties:
            uncertainty.validate()
            dangling = set(uncertainty.related_claim_ids) - known_claims
            if dangling:
                raise ValueError(
                    f"uncertainty {uncertainty.uncertainty_id!r} references unknown claims")
        if tuple(self.output_obligations) != REQUIRED_OUTPUT_OBLIGATIONS:
            raise ValueError("contract must carry the exact required output-obligation set")
        if not isinstance(self.verification_state, VerificationState):
            raise ValueError("verification_state must be closed")
        if self.verification_state is VerificationState.SATISFIED:
            if self.unresolved_uncertainties:
                raise ValueError("satisfied contract cannot retain unresolved uncertainties")
            if not self.claims or not self.evidence:
                raise ValueError("satisfied contract requires claims and evidence")
            if any(claim.contradiction for claim in self.claims):
                raise ValueError("satisfied contract cannot retain a contradicted claim")
            for claim in self.claims:
                # The obligation follows the claim type.  An image-bound diagnosis
                # is discharged by the patient image, so demanding that retrieved
                # text entail it would force every contract closed; a claim about
                # retrieved evidence is discharged only by an assessed relation.
                if claim.claim_type in IMAGE_BOUND_CLAIM_TYPES:
                    if claim.image_support is not ImageSupport.SUPPORTS:
                        raise ValueError(
                            f"satisfied contract requires image support for image-bound "
                            f"claim {claim.claim_id!r}")
                elif claim.entailment is not EntailmentState.SUPPORTED:
                    raise ValueError(
                        f"satisfied contract requires supported evidence for claim "
                        f"{claim.claim_id!r}")
        if self.proposed_action == "PASS" and self.verification_state is not VerificationState.SATISFIED:
            raise ValueError("PASS requires a satisfied evidence contract")
        if self.proposed_action == "ABSTAIN" and not self.termination_reason.strip():
            raise ValueError("ABSTAIN requires a termination_reason")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceContract":
        required = {"schema_version", "case_id", "modality", "clinical_question", "perception",
                    "claims", "evidence", "unresolved_uncertainties", "output_obligations",
                    "verification_state", "proposed_action", "termination_reason"}
        unknown = set(value) - required
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(
                f"contract fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        raw_perception = value["perception"]
        if not isinstance(raw_perception, Mapping):
            raise ValueError("perception must be an object")
        _require_exact_fields(dict(raw_perception), {
            "artifact_id", "artifact_sha256", "record_sha256", "model_id", "class_order",
            "diagnosis_probabilities", "view_image_sha256s", "findings",
        }, "perception")
        raw_findings = raw_perception.get("findings")
        if not isinstance(raw_findings, list):
            raise ValueError("perception findings must be a list")
        findings = []
        for raw in raw_findings:
            _require_exact_fields(dict(raw), {
                "finding_id", "finding_type", "concept_code", "state", "confidence",
                "origin", "source_record_sha256",
            }, "image finding")
            findings.append(ImageFinding(
                finding_id=str(raw["finding_id"]),
                finding_type=FindingType(raw["finding_type"]),
                concept_code=str(raw["concept_code"]),
                state=FindingState(raw["state"]),
                confidence=float(raw["confidence"]),
                origin=FindingOrigin(raw["origin"]),
                source_record_sha256=str(raw["source_record_sha256"]),
            ))
        perception = PerceptionBinding(
            artifact_id=str(raw_perception["artifact_id"]),
            artifact_sha256=str(raw_perception["artifact_sha256"]),
            record_sha256=str(raw_perception["record_sha256"]),
            model_id=str(raw_perception["model_id"]),
            class_order=tuple(str(item) for item in raw_perception["class_order"]),
            diagnosis_probabilities=tuple(
                float(item) for item in raw_perception["diagnosis_probabilities"]),
            view_image_sha256s=tuple(str(item) for item in raw_perception["view_image_sha256s"]),
            findings=tuple(findings),
        )
        claims = []
        for raw in value["claims"]:
            _require_exact_fields(dict(raw), {
                "claim_id", "text", "image_support", "evidence_ids", "entailment",
                "contradiction", "claim_type", "target_concept", "target_state",
                "image_finding_ids", "evidence_relation_origin",
                "evidence_relation_assessor_id", "evidence_relation_receipt_sha256",
            }, "clinical claim")
            claims.append(ClinicalClaim(
                claim_id=str(raw["claim_id"]), text=str(raw["text"]),
                image_support=ImageSupport(raw["image_support"]),
                evidence_ids=tuple(str(item) for item in raw["evidence_ids"]),
                entailment=EntailmentState(raw["entailment"]),
                contradiction=bool(raw["contradiction"]),
                claim_type=ClaimType(raw["claim_type"]),
                target_concept=str(raw["target_concept"]),
                target_state=FindingState(raw["target_state"]),
                image_finding_ids=tuple(str(item) for item in raw["image_finding_ids"]),
                evidence_relation_origin=RelationOrigin(raw["evidence_relation_origin"]),
                evidence_relation_assessor_id=str(raw["evidence_relation_assessor_id"]),
                evidence_relation_receipt_sha256=str(
                    raw["evidence_relation_receipt_sha256"]),
            ))
        evidence = []
        for raw in value["evidence"]:
            _require_exact_fields(dict(raw), {
                "evidence_id", "text", "source_id", "source_version", "authority",
                "retrieval_path", "content_sha256", "provenance_uri", "document_id",
                "citation_id", "eligible_claim_types", "retrieval_channel", "record_sha256",
            }, "evidence item")
            evidence.append(EvidenceItem(**{
                **dict(raw),
                "retrieval_path": tuple(str(item) for item in raw["retrieval_path"]),
                "eligible_claim_types": tuple(
                    str(item) for item in raw.get("eligible_claim_types", ())),
            }))
        uncertainties = []
        for raw in value["unresolved_uncertainties"]:
            _require_exact_fields(dict(raw), {
                "uncertainty_id", "uncertainty_type", "description", "related_claim_ids",
            }, "uncertainty")
            uncertainties.append(UnresolvedUncertainty(
                uncertainty_id=str(raw["uncertainty_id"]),
                uncertainty_type=UncertaintyType(raw["uncertainty_type"]),
                description=str(raw["description"]),
                related_claim_ids=tuple(str(item) for item in raw["related_claim_ids"]),
            ))
        return cls(
            schema_version=str(value["schema_version"]),
            case_id=str(value["case_id"]), modality=str(value["modality"]),
            clinical_question=str(value["clinical_question"]), perception=perception,
            claims=claims, evidence=evidence, unresolved_uncertainties=uncertainties,
            output_obligations=tuple(OutputObligation(item) for item in value["output_obligations"]),
            verification_state=VerificationState(value["verification_state"]),
            proposed_action=str(value["proposed_action"]),
            termination_reason=str(value["termination_reason"]),
        ).validate()
