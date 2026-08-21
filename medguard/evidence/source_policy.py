"""Frozen source/citation policy for MedGuard evidence admission.

The registry establishes source identity, content integrity, and claim-type
eligibility.  It never treats provenance or an authority label as evidence that a
clinical statement is true; semantic support remains a separate verifier concern.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


_HASH_FIELDS = (
    "document_id",
    "citation_id",
    "source_id",
    "source_version",
    "authority_class",
    "title",
    "text",
    "provenance_uri",
)


def canonical_evidence_record_hash(record: Mapping[str, Any]) -> str:
    """Hash the immutable identity and evidence-bearing fields of one record."""
    payload = {field: record.get(field) for field in _HASH_FIELDS}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourcePolicyDecision:
    admitted: bool
    reason: str


class CitationPolicyRegistry:
    """Immutable document registry plus authority/claim admission policy."""

    def __init__(self, policy: Mapping[str, Any], citations: Mapping[str, Mapping[str, Any]]):
        self.policy = dict(policy)
        self.citations = {str(key): dict(value) for key, value in citations.items()}
        classes = self.policy.get("authority_classes")
        if not isinstance(classes, Mapping) or not classes:
            raise ValueError("source policy has no authority classes")
        self.authority_classes = {str(key): dict(value) for key, value in classes.items()}
        if self.policy.get("provenance_semantics") != "origin_and_lineage_not_content_truth":
            raise ValueError("source policy must keep provenance separate from content truth")

    @classmethod
    def from_files(cls, policy_path: str | Path, citation_registry_path: str | Path) -> "CitationPolicyRegistry":
        policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        citations: Dict[str, Dict[str, Any]] = {}
        with Path(citation_registry_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                document_id = str(row.get("document_id") or "")
                if not document_id:
                    raise ValueError(f"citation row {line_number} has no document_id")
                if document_id in citations:
                    raise ValueError(f"duplicate citation document_id: {document_id}")
                citations[document_id] = row
        if not citations:
            raise ValueError("citation registry is empty")
        return cls(policy, citations)

    def decide(self, record: Mapping[str, Any], claim_type: str) -> SourcePolicyDecision:
        document_id = str(record.get("document_id") or "")
        registered = self.citations.get(document_id)
        if registered is None:
            return SourcePolicyDecision(False, "unknown_document")
        for field in ("citation_id", "source_id", "source_version", "authority_class"):
            if record.get(field) != registered.get(field):
                return SourcePolicyDecision(False, f"{field}_mismatch")
        text_sha256 = hashlib.sha256(str(record.get("text") or "").encode("utf-8")).hexdigest()
        if text_sha256 != registered.get("content_sha256"):
            return SourcePolicyDecision(False, "content_hash_mismatch")
        if canonical_evidence_record_hash(record) != registered.get("record_sha256"):
            return SourcePolicyDecision(False, "record_hash_mismatch")
        authority_class = str(record.get("authority_class") or "")
        authority = self.authority_classes.get(authority_class)
        if authority is None or not bool(authority.get("admitted")):
            return SourcePolicyDecision(False, "authority_not_admitted")
        registered_eligible = {str(value) for value in registered.get("eligible_claim_types", [])}
        authority_eligible = {str(value) for value in authority.get("eligible_claim_types", [])}
        normalized_claim = str(claim_type or "").strip().lower()
        if (not normalized_claim
                or normalized_claim not in registered_eligible
                or normalized_claim not in authority_eligible):
            return SourcePolicyDecision(False, "claim_type_not_eligible")
        return SourcePolicyDecision(True, "verified_identity_content_and_claim_scope")
