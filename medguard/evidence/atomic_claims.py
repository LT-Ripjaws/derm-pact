"""Host-owned, source-bound atomic claim definitions.

This module deliberately contains no clinical relation labels.  It gives a
future independently curated registry a stable target that is not affected by
the model's claim identifier, evidence rank, arm, or retrieval path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .contract import (
    ClaimType,
    ClinicalClaim,
    EvidenceItem,
    FindingState,
    ImageSupport,
)


ATOMIC_CLAIM_CATALOG_SCHEMA = "medguard-atomic-source-claim-catalog-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEY = re.compile(r"[a-z0-9][a-z0-9_.:-]*")
_SOURCE_BOUND_TYPES = {
    ClaimType.ASSESSMENT,
    ClaimType.MANAGEMENT,
    ClaimType.REFERRAL,
    ClaimType.TREATMENT,
}


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourcePassage:
    """One immutable source passage eligible for future relation curation."""

    source_id: str
    source_version: str
    document_id: str
    citation_id: str
    content_sha256: str
    record_sha256: str

    def validate(self) -> None:
        for name, value in (
            ("source_id", self.source_id),
            ("source_version", self.source_version),
            ("document_id", self.document_id),
            ("citation_id", self.citation_id),
        ):
            if not str(value).strip():
                raise ValueError(f"source passage {name} must be non-empty")
        for name, value in (
            ("content_sha256", self.content_sha256),
            ("record_sha256", self.record_sha256),
        ):
            if not _SHA256.fullmatch(str(value)):
                raise ValueError(f"source passage {name} must be a SHA-256 digest")

    @property
    def passage_sha256(self) -> str:
        self.validate()
        return _canonical_sha256({
            "schema": "medguard-source-passage-anchor-v1",
            "source_id": self.source_id,
            "source_version": self.source_version,
            "document_id": self.document_id,
            "citation_id": self.citation_id,
            "content_sha256": self.content_sha256,
            "record_sha256": self.record_sha256,
        })

    def matches(self, evidence: EvidenceItem) -> bool:
        evidence.validate()
        return (
            evidence.source_id == self.source_id
            and evidence.source_version == self.source_version
            and evidence.document_id == self.document_id
            and evidence.citation_id == self.citation_id
            and evidence.content_sha256 == self.content_sha256
            and evidence.record_sha256 == self.record_sha256
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourcePassage":
        expected = {
            "source_id", "source_version", "document_id", "citation_id",
            "content_sha256", "record_sha256",
        }
        if set(value) != expected:
            raise ValueError("source passage fields do not match the catalog schema")
        output = cls(**{key: str(value[key]) for key in expected})
        output.validate()
        return output

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "document_id": self.document_id,
            "citation_id": self.citation_id,
            "content_sha256": self.content_sha256,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class AtomicSourceClaim:
    """A non-patient-specific statement anchored to one exact source passage."""

    claim_key: str
    text: str
    claim_type: ClaimType
    source_passage: SourcePassage

    def validate(self) -> None:
        if not _KEY.fullmatch(self.claim_key):
            raise ValueError("atomic claim_key must be canonical")
        if not self.text.strip():
            raise ValueError("atomic source claim text must be non-empty")
        if self.claim_type not in _SOURCE_BOUND_TYPES:
            raise ValueError("atomic source claim type must be non-image-bound")
        self.source_passage.validate()

    @property
    def claim_sha256(self) -> str:
        self.validate()
        return _canonical_sha256({
            "schema": "medguard-atomic-source-claim-v1",
            "claim_key": self.claim_key,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "source_passage_sha256": self.source_passage.passage_sha256,
        })

    def bind(self, *, claim_id: str, evidence: EvidenceItem) -> ClinicalClaim:
        """Create the host-owned contract claim for this exact evidence item.

        This never marks the claim supported.  A curated registry entry is still
        required to establish ENTAILS or CONTRADICTS.
        """
        self.validate()
        if not self.source_passage.matches(evidence):
            raise ValueError("evidence does not match the atomic claim source passage")
        output = ClinicalClaim(
            claim_id=str(claim_id),
            text=self.text,
            image_support=ImageSupport.NOT_APPLICABLE,
            evidence_ids=(evidence.evidence_id,),
            claim_type=self.claim_type,
            target_concept="",
            target_state=FindingState.UNCERTAIN,
        )
        output.validate()
        return output

    def matches_claim(self, claim: ClinicalClaim, evidence: EvidenceItem) -> bool:
        self.validate()
        claim.validate()
        return (
            self.source_passage.matches(evidence)
            and claim.text == self.text
            and claim.claim_type is self.claim_type
            and claim.image_support is ImageSupport.NOT_APPLICABLE
            and not claim.target_concept
            and not claim.image_finding_ids
            and tuple(claim.evidence_ids) == (evidence.evidence_id,)
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AtomicSourceClaim":
        expected = {"claim_key", "text", "claim_type", "source_passage"}
        if set(value) != expected:
            raise ValueError("atomic source claim fields do not match the catalog schema")
        source = value["source_passage"]
        if not isinstance(source, Mapping):
            raise ValueError("atomic source claim requires a source passage object")
        output = cls(
            claim_key=str(value["claim_key"]),
            text=str(value["text"]),
            claim_type=ClaimType(value["claim_type"]),
            source_passage=SourcePassage.from_mapping(source),
        )
        output.validate()
        return output

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "claim_key": self.claim_key,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "source_passage": self.source_passage.to_dict(),
        }


class AtomicClaimCatalog:
    """Immutable, content-addressed collection of source-bound atomic claims."""

    def __init__(self, claims: Iterable[AtomicSourceClaim] = ()) -> None:
        by_key: dict[str, AtomicSourceClaim] = {}
        for claim in claims:
            claim.validate()
            if claim.claim_key in by_key:
                raise ValueError(f"duplicate atomic claim key: {claim.claim_key}")
            by_key[claim.claim_key] = claim
        self._claims = tuple(sorted(by_key.values(), key=lambda item: item.claim_key))
        self._by_key = dict(by_key)
        self.catalog_sha256 = _canonical_sha256({
            "schema": ATOMIC_CLAIM_CATALOG_SCHEMA,
            "claims": [claim.to_dict() for claim in self._claims],
        })

    @property
    def claims(self) -> tuple[AtomicSourceClaim, ...]:
        return self._claims

    def lookup(self, claim_key: str) -> AtomicSourceClaim | None:
        return self._by_key.get(str(claim_key))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AtomicClaimCatalog":
        expected = {"schema_version", "claims"}
        if set(value) != expected or value.get("schema_version") != ATOMIC_CLAIM_CATALOG_SCHEMA:
            raise ValueError("unknown atomic claim catalog schema")
        raw_claims = value["claims"]
        if not isinstance(raw_claims, list):
            raise ValueError("atomic claim catalog claims must be a list")
        return cls(AtomicSourceClaim.from_mapping(item) for item in raw_claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ATOMIC_CLAIM_CATALOG_SCHEMA,
            "claims": [claim.to_dict() for claim in self._claims],
        }
