"""Closed, host-owned evidence-relation assessment for the living C1 contract.

This module deliberately does not perform lexical or model-based inference.  An
ENTAILS/CONTRADICTS decision requires either an exact curated-registry match or
an exact, hash-bound frozen image-exemplar label.  Provenance, authority,
citations and generator declarations are lineage only; none is treated as
truth.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from .atomic_claims import AtomicSourceClaim

from .contract import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    EvidenceItem,
    FindingState,
    HAM7_CLASS_ORDER,
    RelationOrigin,
)


ASSESSOR_ID = "medguard-conservative-evidence-relation"
ASSESSOR_VERSION = "1.0.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EvidenceRelation(str, Enum):
    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _enum_value(value: object) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def claim_basis_sha256(claim: ClinicalClaim) -> str:
    """Hash relation-relevant claim fields, excluding any declared relation."""
    return _canonical_sha256({
        "basis_schema": "medguard-claim-relation-basis-v1",
        "claim_id": str(claim.claim_id),
        "text": str(claim.text),
        "claim_type": _enum_value(claim.claim_type),
        "target_concept": str(claim.target_concept),
        "target_state": _enum_value(claim.target_state),
        "evidence_ids": list(claim.evidence_ids),
        "image_finding_ids": list(claim.image_finding_ids),
    })


def evidence_basis_sha256(evidence: EvidenceItem) -> str:
    """Hash evidence identity/lineage separately from its raw-content hash."""
    return _canonical_sha256({
        "basis_schema": "medguard-evidence-relation-basis-v1",
        "evidence_id": str(evidence.evidence_id),
        "source_id": str(evidence.source_id),
        "source_version": str(evidence.source_version),
        "authority": str(evidence.authority),
        "retrieval_path": list(evidence.retrieval_path),
        "content_sha256": str(evidence.content_sha256),
        "provenance_uri": str(evidence.provenance_uri),
        "document_id": str(evidence.document_id),
        "citation_id": str(evidence.citation_id),
        "eligible_claim_types": list(evidence.eligible_claim_types),
        "retrieval_channel": str(evidence.retrieval_channel),
        "record_sha256": str(evidence.record_sha256),
    })


@dataclass(frozen=True)
class CuratedRelationEntry:
    """One exact human/host-curated relation; no text matching is performed."""

    claim_sha256: str
    evidence_sha256: str
    evidence_content_sha256: str
    relation: EvidenceRelation

    def validate(self) -> None:
        for name, value in (
            ("claim_sha256", self.claim_sha256),
            ("evidence_sha256", self.evidence_sha256),
            ("evidence_content_sha256", self.evidence_content_sha256),
        ):
            if not _SHA256.fullmatch(str(value)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.relation not in {
            EvidenceRelation.ENTAILS, EvidenceRelation.CONTRADICTS,
        }:
            raise ValueError("curated entries may only establish ENTAILS/CONTRADICTS")

    @classmethod
    def from_pair(
        cls, claim: ClinicalClaim, evidence: EvidenceItem, relation: EvidenceRelation,
    ) -> "CuratedRelationEntry":
        entry = cls(
            claim_sha256=claim_basis_sha256(claim),
            evidence_sha256=evidence_basis_sha256(evidence),
            evidence_content_sha256=str(evidence.content_sha256),
            relation=relation,
        )
        entry.validate()
        return entry

    @property
    def key(self) -> tuple[str, str, str]:
        return (
            self.claim_sha256, self.evidence_sha256, self.evidence_content_sha256,
        )


@dataclass(frozen=True)
class AtomicClaimRelationEntry:
    """One relation label for a source-bound atomic claim.

    The key intentionally excludes case ID, arm, evidence rank, and retrieval
    path.  Those are runtime lineage, not source semantics.
    """

    claim_key: str
    atomic_claim_sha256: str
    source_passage_sha256: str
    relation: EvidenceRelation

    def validate(self) -> None:
        if not self.claim_key.strip():
            raise ValueError("atomic relation claim_key must be non-empty")
        for name, value in (
            ("atomic_claim_sha256", self.atomic_claim_sha256),
            ("source_passage_sha256", self.source_passage_sha256),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"atomic relation {name} must be a SHA-256 digest")
        if self.relation not in {
            EvidenceRelation.ENTAILS, EvidenceRelation.CONTRADICTS,
        }:
            raise ValueError("curated entries may only establish ENTAILS/CONTRADICTS")

    @classmethod
    def from_claim(
        cls, claim: AtomicSourceClaim, relation: EvidenceRelation,
    ) -> "AtomicClaimRelationEntry":
        claim.validate()
        entry = cls(
            claim_key=claim.claim_key,
            atomic_claim_sha256=claim.claim_sha256,
            source_passage_sha256=claim.source_passage.passage_sha256,
            relation=relation,
        )
        entry.validate()
        return entry

    @property
    def key(self) -> tuple[str, str, str]:
        return self.claim_key, self.atomic_claim_sha256, self.source_passage_sha256


class CuratedRelationRegistry:
    """Immutable-in-practice, in-memory registry with a deterministic content hash."""

    def __init__(
        self,
        entries: Iterable[CuratedRelationEntry] = (),
        *,
        atomic_entries: Iterable[AtomicClaimRelationEntry] = (),
    ) -> None:
        by_key: dict[tuple[str, str, str], CuratedRelationEntry] = {}
        for entry in entries:
            entry.validate()
            if entry.key in by_key:
                raise ValueError("duplicate curated relation key")
            by_key[entry.key] = entry
        atomic_by_key: dict[tuple[str, str, str], AtomicClaimRelationEntry] = {}
        for entry in atomic_entries:
            entry.validate()
            if entry.key in atomic_by_key:
                raise ValueError("duplicate atomic curated relation key")
            atomic_by_key[entry.key] = entry
        self._entries = tuple(sorted(
            by_key.values(), key=lambda item: (*item.key, item.relation.value),
        ))
        self._atomic_entries = tuple(sorted(
            atomic_by_key.values(), key=lambda item: (*item.key, item.relation.value),
        ))
        self._by_key = MappingProxyType(dict(by_key))
        self._atomic_by_key = MappingProxyType(dict(atomic_by_key))
        self.registry_sha256 = _canonical_sha256({
            "registry_schema": "medguard-curated-relation-registry-v2",
            "exact_entries": [{
                "claim_sha256": entry.claim_sha256,
                "evidence_sha256": entry.evidence_sha256,
                "evidence_content_sha256": entry.evidence_content_sha256,
                "relation": entry.relation.value,
            } for entry in self._entries],
            "atomic_entries": [{
                "claim_key": entry.claim_key,
                "atomic_claim_sha256": entry.atomic_claim_sha256,
                "source_passage_sha256": entry.source_passage_sha256,
                "relation": entry.relation.value,
            } for entry in self._atomic_entries],
        })

    @property
    def entries(self) -> tuple[CuratedRelationEntry, ...]:
        return self._entries

    @property
    def atomic_entries(self) -> tuple[AtomicClaimRelationEntry, ...]:
        return self._atomic_entries

    def lookup(
        self, claim_sha256: str, evidence_sha256: str, content_sha256: str,
    ) -> Optional[EvidenceRelation]:
        entry = self._by_key.get((claim_sha256, evidence_sha256, content_sha256))
        return entry.relation if entry is not None else None

    def lookup_atomic(self, claim: AtomicSourceClaim) -> Optional[EvidenceRelation]:
        claim.validate()
        entry = self._atomic_by_key.get((
            claim.claim_key, claim.claim_sha256, claim.source_passage.passage_sha256,
        ))
        return entry.relation if entry is not None else None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CuratedRelationRegistry":
        expected = {"schema_version", "exact_entries", "atomic_entries"}
        if set(value) != expected or value.get("schema_version") != (
            "medguard-curated-relation-registry-v2"
        ):
            raise ValueError("unknown curated relation registry schema")

        def relation(raw: object) -> EvidenceRelation:
            return EvidenceRelation(str(raw))

        exact_entries = []
        for raw in value["exact_entries"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "claim_sha256", "evidence_sha256", "evidence_content_sha256", "relation",
            }:
                raise ValueError("exact curated relation fields do not match the registry schema")
            exact_entries.append(CuratedRelationEntry(
                claim_sha256=str(raw["claim_sha256"]),
                evidence_sha256=str(raw["evidence_sha256"]),
                evidence_content_sha256=str(raw["evidence_content_sha256"]),
                relation=relation(raw["relation"]),
            ))
        atomic_entries = []
        for raw in value["atomic_entries"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "claim_key", "atomic_claim_sha256", "source_passage_sha256", "relation",
            }:
                raise ValueError("atomic curated relation fields do not match the registry schema")
            atomic_entries.append(AtomicClaimRelationEntry(
                claim_key=str(raw["claim_key"]),
                atomic_claim_sha256=str(raw["atomic_claim_sha256"]),
                source_passage_sha256=str(raw["source_passage_sha256"]),
                relation=relation(raw["relation"]),
            ))
        return cls(exact_entries, atomic_entries=atomic_entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "medguard-curated-relation-registry-v2",
            "exact_entries": [{
                "claim_sha256": entry.claim_sha256,
                "evidence_sha256": entry.evidence_sha256,
                "evidence_content_sha256": entry.evidence_content_sha256,
                "relation": entry.relation.value,
            } for entry in self._entries],
            "atomic_entries": [{
                "claim_key": entry.claim_key,
                "atomic_claim_sha256": entry.atomic_claim_sha256,
                "source_passage_sha256": entry.source_passage_sha256,
                "relation": entry.relation.value,
            } for entry in self._atomic_entries],
        }


@dataclass(frozen=True)
class ImageExemplarMetadata:
    """Host-supplied frozen exemplar label, bound to evidence and source record."""

    evidence_id: str
    evidence_content_sha256: str
    evidence_record_sha256: str
    canonical_label: str
    label_state: FindingState = FindingState.PRESENT

    def validate(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("image exemplar metadata requires evidence_id")
        if not _SHA256.fullmatch(self.evidence_content_sha256):
            raise ValueError("image exemplar content hash must be SHA-256")
        if not _SHA256.fullmatch(self.evidence_record_sha256):
            raise ValueError("image exemplar record hash must be SHA-256")
        if self.canonical_label not in HAM7_CLASS_ORDER:
            raise ValueError("image exemplar label must be a canonical HAM7 label")
        if self.label_state is not FindingState.PRESENT:
            raise ValueError("image exemplar label state must be present")

    @property
    def metadata_sha256(self) -> str:
        self.validate()
        return _canonical_sha256({
            "metadata_schema": "medguard-image-exemplar-metadata-v1",
            "evidence_id": self.evidence_id,
            "evidence_content_sha256": self.evidence_content_sha256,
            "evidence_record_sha256": self.evidence_record_sha256,
            "canonical_label": self.canonical_label,
            "label_state": self.label_state.value,
        })


@dataclass(frozen=True)
class RelationAssessment:
    relation: EvidenceRelation
    assessor_id: str
    assessor_version: str
    claim_id: str
    evidence_id: str
    claim_sha256: str
    evidence_sha256: str
    evidence_content_sha256: str
    observed_content_sha256: str
    curated_registry_sha256: str
    structured_metadata_sha256: str
    reason: str

    def validate(self) -> None:
        if not isinstance(self.relation, EvidenceRelation):
            raise ValueError("assessment relation must be closed")
        if self.assessor_id != ASSESSOR_ID or self.assessor_version != ASSESSOR_VERSION:
            raise ValueError("assessment identity/version does not match the host assessor")
        if not self.claim_id.strip() or not self.evidence_id.strip() or not self.reason.strip():
            raise ValueError("assessment IDs and reason must be non-empty")
        for name, value in (
            ("claim_sha256", self.claim_sha256),
            ("evidence_sha256", self.evidence_sha256),
            ("evidence_content_sha256", self.evidence_content_sha256),
            ("observed_content_sha256", self.observed_content_sha256),
        ):
            if not _SHA256.fullmatch(str(value)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name, value in (
            ("curated_registry_sha256", self.curated_registry_sha256),
            ("structured_metadata_sha256", self.structured_metadata_sha256),
        ):
            if value and not _SHA256.fullmatch(str(value)):
                raise ValueError(f"{name} must be empty or a lowercase SHA-256 digest")

    @property
    def assessment_sha256(self) -> str:
        self.validate()
        return _canonical_sha256({
            "assessment_schema": "medguard-evidence-relation-assessment-v1",
            "relation": self.relation.value,
            "assessor_id": self.assessor_id,
            "assessor_version": self.assessor_version,
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "claim_sha256": self.claim_sha256,
            "evidence_sha256": self.evidence_sha256,
            "evidence_content_sha256": self.evidence_content_sha256,
            "observed_content_sha256": self.observed_content_sha256,
            "curated_registry_sha256": self.curated_registry_sha256,
            "structured_metadata_sha256": self.structured_metadata_sha256,
            "reason": self.reason,
        })


def assess_evidence_relation(
    claim: ClinicalClaim,
    evidence: EvidenceItem,
    *,
    registry: Optional[CuratedRelationRegistry] = None,
    image_exemplar: Optional[ImageExemplarMetadata] = None,
    atomic_claim: Optional[AtomicSourceClaim] = None,
) -> RelationAssessment:
    """Assess one claim/evidence pair without deriving truth from free text."""
    claim_hash = claim_basis_sha256(claim)
    evidence_hash = evidence_basis_sha256(evidence)
    observed_hash = hashlib.sha256(str(evidence.text).encode("utf-8")).hexdigest()
    declared_content_hash = str(evidence.content_sha256)
    bound_content_hash = (
        declared_content_hash
        if _SHA256.fullmatch(declared_content_hash) else observed_hash
    )
    registry_hash = registry.registry_sha256 if registry is not None else ""
    metadata_hash = ""
    claim_id = str(claim.claim_id).strip() or "<missing-claim-id>"
    evidence_id = str(evidence.evidence_id).strip() or "<missing-evidence-id>"

    def result(relation: EvidenceRelation, reason: str) -> RelationAssessment:
        value = RelationAssessment(
            relation=relation,
            assessor_id=ASSESSOR_ID,
            assessor_version=ASSESSOR_VERSION,
            claim_id=claim_id,
            evidence_id=evidence_id,
            claim_sha256=claim_hash,
            evidence_sha256=evidence_hash,
            evidence_content_sha256=bound_content_hash,
            observed_content_sha256=observed_hash,
            curated_registry_sha256=registry_hash,
            structured_metadata_sha256=metadata_hash,
            reason=reason,
        )
        value.validate()
        return value

    if not _SHA256.fullmatch(declared_content_hash):
        # Keep the result serializable while binding the invalid declared value
        # through evidence_sha256; the explicit content field must remain a SHA.
        return result(EvidenceRelation.UNKNOWN, "invalid_declared_content_hash")
    if evidence.content_sha256 != observed_hash:
        return result(EvidenceRelation.UNKNOWN, "evidence_content_hash_mismatch")
    try:
        claim.validate()
    except (TypeError, ValueError):
        return result(EvidenceRelation.UNKNOWN, "malformed_claim")
    if evidence.evidence_id not in claim.evidence_ids:
        return result(EvidenceRelation.INSUFFICIENT, "evidence_not_cited_by_claim")
    eligible = {str(value).strip() for value in evidence.eligible_claim_types}
    if claim.claim_type.value not in eligible:
        return result(EvidenceRelation.INSUFFICIENT, "evidence_outside_claim_scope")

    candidates: list[tuple[EvidenceRelation, str]] = []
    if registry is not None:
        registered = registry.lookup(claim_hash, evidence_hash, evidence.content_sha256)
        if registered is not None:
            candidates.append((registered, "exact_curated_registry_match"))

    if atomic_claim is not None:
        try:
            atomic_claim.validate()
        except (TypeError, ValueError):
            return result(EvidenceRelation.UNKNOWN, "invalid_atomic_source_claim")
        if not atomic_claim.matches_claim(claim, evidence):
            return result(EvidenceRelation.UNKNOWN, "atomic_source_claim_mismatch")
        if registry is not None:
            registered = registry.lookup_atomic(atomic_claim)
            if registered is not None:
                candidates.append((registered, "exact_atomic_registry_match"))

    if image_exemplar is not None:
        try:
            image_exemplar.validate()
            metadata_hash = image_exemplar.metadata_sha256
        except (TypeError, ValueError):
            return result(EvidenceRelation.UNKNOWN, "invalid_image_exemplar_metadata")
        if claim.claim_type is not ClaimType.IMAGE_EXEMPLAR_SUPPORT:
            return result(EvidenceRelation.INSUFFICIENT, "exemplar_metadata_wrong_claim_type")
        if evidence.retrieval_channel != "image_exemplar":
            return result(EvidenceRelation.UNKNOWN, "exemplar_channel_mismatch")
        if (
            image_exemplar.evidence_id != evidence.evidence_id
            or image_exemplar.evidence_content_sha256 != evidence.content_sha256
            or not evidence.record_sha256
            or image_exemplar.evidence_record_sha256 != evidence.record_sha256
        ):
            return result(EvidenceRelation.UNKNOWN, "exemplar_hash_or_identity_mismatch")
        if (claim.target_concept == image_exemplar.canonical_label
                and claim.target_state is image_exemplar.label_state):
            candidates.append((
                EvidenceRelation.ENTAILS,
                "explicit_hash_bound_image_exemplar_label",
            ))
        elif not candidates:
            # A differently labelled neighbour is another lesion.  It fails to
            # support this target but is not logical counterevidence about the
            # patient.  Only an exact curated entry may establish contradiction.
            return result(
                EvidenceRelation.INSUFFICIENT,
                "exemplar_label_does_not_support_target",
            )

    if not candidates:
        return result(EvidenceRelation.UNKNOWN, "no_independent_relation_basis")
    relations = {relation for relation, _reason in candidates}
    if len(relations) != 1:
        return result(EvidenceRelation.UNKNOWN, "independent_relation_bases_conflict")
    relation = candidates[0][0]
    return result(relation, "+".join(reason for _relation, reason in candidates))


def apply_relation_assessment(
    claim: ClinicalClaim, assessment: RelationAssessment,
) -> ClinicalClaim:
    """Replace generator relation fields with one verified host assessment.

    UNKNOWN is deliberately mapped to the contract's INSUFFICIENT state: it
    remains fail-closed and cannot preserve a generator-declared SUPPORTED flag.
    """
    assessment.validate()
    if claim_basis_sha256(claim) != assessment.claim_sha256:
        raise ValueError("assessment claim hash does not match the claim")
    if assessment.evidence_id not in claim.evidence_ids:
        raise ValueError("assessment evidence is not cited by the claim")
    if assessment.relation is EvidenceRelation.ENTAILS:
        entailment, contradiction = EntailmentState.SUPPORTED, False
    elif assessment.relation is EvidenceRelation.CONTRADICTS:
        entailment, contradiction = EntailmentState.CONTRADICTED, True
    else:
        entailment, contradiction = EntailmentState.INSUFFICIENT, False
    output = replace(
        claim,
        entailment=entailment,
        contradiction=contradiction,
        evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
        evidence_relation_assessor_id=(
            f"{assessment.assessor_id}@{assessment.assessor_version}"
        ),
        evidence_relation_receipt_sha256=assessment.assessment_sha256,
    )
    output.validate()
    return output
