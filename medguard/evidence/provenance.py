"""Canonical provenance normalization for vector, graph, and guideline evidence."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .contract import EvidenceItem


def evidence_from_chunk(
    chunk: Mapping[str, Any], *, evidence_id: str, retrieval_path: Sequence[str],
) -> EvidenceItem:
    text = " ".join(str(chunk.get("text", "")).split())
    source_id = str(chunk.get("source") or chunk.get("source_id") or chunk.get("pmid") or "").strip()
    authority = str(chunk.get("authority") or chunk.get("authority_class") or "unclassified").strip()
    version = str(chunk.get("source_version") or chunk.get("version") or
                  chunk.get("publication_date") or "unknown").strip()
    uri = str(chunk.get("provenance_uri") or chunk.get("url") or chunk.get("uri") or "").strip()
    item = EvidenceItem(
        evidence_id=str(evidence_id),
        text=text,
        source_id=source_id,
        source_version=version,
        authority=authority,
        retrieval_path=tuple(str(part) for part in retrieval_path),
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        provenance_uri=uri,
        document_id=str(chunk.get("document_id") or "").strip(),
        citation_id=str(chunk.get("citation_id") or "").strip(),
        eligible_claim_types=tuple(
            str(value).strip() for value in chunk.get("eligible_claim_types", ())
            if str(value).strip()
        ),
        retrieval_channel=str(chunk.get("retrieval_channel") or "").strip(),
        record_sha256=str(chunk.get("record_sha256") or "").strip(),
    )
    item.validate()
    return item


def evidence_receipt_sha256(items: Sequence[EvidenceItem]) -> str:
    payload = [
        {
            "evidence_id": item.evidence_id,
            "source_id": item.source_id,
            "source_version": item.source_version,
            "authority": item.authority,
            "retrieval_path": list(item.retrieval_path),
            "content_sha256": item.content_sha256,
            "provenance_uri": item.provenance_uri,
            "document_id": item.document_id,
            "citation_id": item.citation_id,
            "eligible_claim_types": list(item.eligible_claim_types),
            "retrieval_channel": item.retrieval_channel,
            "record_sha256": item.record_sha256,
        }
        for item in sorted(items, key=lambda value: value.evidence_id)
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
