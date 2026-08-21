"""Receipt-bound retrieval over the frozen authoritative-guideline catalog.

This is the second text channel, deliberately kept structurally separate from the
1,243-record PubMed abstract index rather than merged into it. The separation is
load-bearing in four ways:

1.  It supplies source-exact content. Every record here is verbatim recommendation
    text projected from a byte-hashed source PDF, carrying DOI, version, section,
    page, licence and content hash. The abstract index has none of that: its three
    "guideline" records are PubMed abstracts that describe a guideline rather than
    state a recommendation.
2.  It gives INDEPENDENT_SOURCE a genuinely separate index to consult. That action
    is currently inert partly because no independent retriever exists.
3.  It gives the evidence graph two authority tiers to select across, instead of
    annotating lineage over an already-fused single tier.
4.  It matches the specified architecture, which names image-exemplar retrieval and
    authoritative guideline/document retrieval as distinct components.

Selection is scoped by the image-bound committed diagnosis, never by a label. That
is not a convenience: an unscoped catalog would surface basal-cell surgical margins
for a vascular lesion, and any citation metric computed over that would be
measuring nothing.

What this channel can and cannot support is a measured fact, not a design opinion.
See research/experiments/medguard_h2h_v1/guideline_catalog_feasibility_v1.json:
melanoma carries zero assessment-eligible and zero visual_finding-eligible records,
and even the strongest basal-cell visual record is a claim about method
("Dermatoscopy improves the accuracy of clinical diagnosis of BCC") rather than
about a patient. Nothing retrieved here establishes that a patient has a
diagnosis. The diagnosis stays image-bound; this channel carries method licensure
and conditional management only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


GUIDELINE_CATALOG_SCHEMA = "medguard-guideline-recommendation-v1"

# HAM7 commitments that map to a guideline condition. The classes absent here are
# absent from the licensed corpus, not from dermatology: actinic keratosis has an
# allow-listed AAD guideline whose text we hold only as an abstract, and both
# squamous-cell conditions are permission-gated. A commitment outside this map
# reaches nothing, and the contract must treat that as "no external claim owed"
# rather than as a failure to find evidence.
COMMITMENT_TO_CONDITION: Mapping[str, str] = {
    "bcc": "basal_cell_carcinoma",
    "melanoma": "melanoma",
    # HAM's "akiec" is actinic keratoses AND intraepithelial carcinoma, which is
    # the scope of the EADO actinic-keratosis guideline.
    "akiec": "actinic_keratosis",
}

# Fields every admitted record must carry for its lineage to be reconstructable.
_REQUIRED_FIELDS = (
    "authority_class",
    "citation_id",
    "condition",
    "content_form",
    "content_sha256",
    "document_id",
    "eligible_claim_types",
    "record_sha256",
    "source_id",
    "source_pdf_sha256",
    "source_version",
    "text",
)

# Only verbatim projections may back a claim. A guideline abstract is a
# description of a guideline, not a recommendation from it.
VERBATIM_CONTENT_FORM = "verbatim_recommendation_text_projection"


class GuidelineCatalogError(ValueError):
    """Raised when the catalog cannot be trusted to be what it claims."""


def canonical_record_hash(record: Mapping[str, Any]) -> str:
    """Hash of the record's content, independent of key order."""
    payload = {key: record[key] for key in _REQUIRED_FIELDS if key in record}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GuidelineEvidence:
    """One retrieved recommendation with its lineage intact."""

    document_id: str
    condition: str
    text: str
    citation_id: str
    source_id: str
    source_version: str
    authority_class: str
    content_sha256: str
    record_sha256: str
    source_pdf_sha256: str
    section: str
    article_page: Any
    eligible_claim_types: tuple
    retrieval_path: str

    def to_evidence_record(self) -> Dict[str, Any]:
        """Shape consumed by the contract's evidence binding."""
        return {
            "authority": self.authority_class,
            "authority_class": self.authority_class,
            "citation_id": self.citation_id,
            "condition": self.condition,
            "content_form": VERBATIM_CONTENT_FORM,
            "content_sha256": self.content_sha256,
            "document_id": self.document_id,
            "eligible_claim_types": list(self.eligible_claim_types),
            "id": self.document_id,
            "record_sha256": self.record_sha256,
            "retrieval_path": self.retrieval_path,
            "section": self.section,
            "source": self.source_id,
            "source_id": self.source_id,
            "source_pdf_sha256": self.source_pdf_sha256,
            "source_version": self.source_version,
            "text": self.text,
        }


class FrozenGuidelineRetriever:
    """Commitment-scoped retrieval over a hash-verified recommendation catalog."""

    retrieval_path = "authoritative_guideline_catalog"

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        if not records:
            raise GuidelineCatalogError("guideline catalog is empty")
        admitted: List[Dict[str, Any]] = []
        seen: set = set()
        for record in records:
            missing = [field for field in _REQUIRED_FIELDS if field not in record]
            if missing:
                raise GuidelineCatalogError(
                    f"guideline record {record.get('document_id', '?')} is missing "
                    f"lineage fields: {', '.join(missing)}"
                )
            if record.get("schema_version") not in (None, GUIDELINE_CATALOG_SCHEMA):
                raise GuidelineCatalogError(
                    f"unsupported guideline record schema: {record.get('schema_version')}"
                )
            if record["content_form"] != VERBATIM_CONTENT_FORM:
                raise GuidelineCatalogError(
                    f"guideline record {record['document_id']} is not a verbatim "
                    f"projection: {record['content_form']}"
                )
            if record["authority_class"] != "clinical_practice_guideline":
                raise GuidelineCatalogError(
                    f"guideline record {record['document_id']} has non-guideline "
                    f"authority: {record['authority_class']}"
                )
            observed = hashlib.sha256(str(record["text"]).encode("utf-8")).hexdigest()
            if observed != record["content_sha256"]:
                raise GuidelineCatalogError(
                    f"guideline record {record['document_id']} text does not match "
                    f"its content_sha256; the catalog has been altered"
                )
            if record["document_id"] in seen:
                raise GuidelineCatalogError(
                    f"duplicate guideline document_id: {record['document_id']}"
                )
            seen.add(record["document_id"])
            admitted.append(dict(record))
        self.records = admitted

    @classmethod
    def load(cls, corpus_path: str | Path, expected_sha256: str | None = None) -> "FrozenGuidelineRetriever":
        """Load the catalog, optionally pinned to a receipt hash.

        The hash check is the difference between a frozen artifact and a file that
        happens to be on disk. Callers holding a receipt must pass it.
        """
        path = Path(corpus_path)
        raw = path.read_bytes()
        if expected_sha256 is not None:
            observed = hashlib.sha256(raw).hexdigest()
            if observed != expected_sha256:
                raise GuidelineCatalogError(
                    f"guideline catalog hash mismatch: expected {expected_sha256}, "
                    f"observed {observed}"
                )
        records = []
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise GuidelineCatalogError(
                    f"invalid guideline catalog JSON on line {number}"
                ) from error
        return cls(records)

    def conditions(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record["condition"]] = counts.get(record["condition"], 0) + 1
        return counts

    def search(
        self,
        committed_diagnosis: str,
        claim_type: str | None = None,
        k: int | None = None,
    ) -> List[GuidelineEvidence]:
        """Recommendations admissible for an image-bound commitment.

        Returns [] -- never a cross-condition fallback -- when the commitment has
        no licensed condition. An empty result means no external claim is owed,
        which is a different thing from failing to find evidence, and the contract
        must be able to tell those apart.
        """
        condition = COMMITMENT_TO_CONDITION.get(committed_diagnosis)
        if condition is None:
            return []
        selected = [row for row in self.records if row["condition"] == condition]
        if claim_type is not None:
            selected = [
                row for row in selected if claim_type in row["eligible_claim_types"]
            ]
        # Deterministic order: document_id is stable and receipt-bound. No
        # relevance model is applied, because ranking 18 records with an unvalidated
        # scorer would invent a precision signal we cannot defend.
        selected.sort(key=lambda row: str(row["document_id"]))
        if k is not None:
            if k <= 0:
                return []
            selected = selected[:k]
        return [self._as_evidence(row) for row in selected]

    def _as_evidence(self, row: Mapping[str, Any]) -> GuidelineEvidence:
        return GuidelineEvidence(
            document_id=str(row["document_id"]),
            condition=str(row["condition"]),
            text=str(row["text"]),
            citation_id=str(row["citation_id"]),
            source_id=str(row["source_id"]),
            source_version=str(row["source_version"]),
            authority_class=str(row["authority_class"]),
            content_sha256=str(row["content_sha256"]),
            record_sha256=str(row["record_sha256"]),
            source_pdf_sha256=str(row["source_pdf_sha256"]),
            section=str(row.get("section", "")),
            article_page=row.get("article_page"),
            eligible_claim_types=tuple(row["eligible_claim_types"]),
            retrieval_path=self.retrieval_path,
        )

    def offered_ids(self, committed_diagnosis: str, claim_type: str | None = None) -> set:
        """The candidate set a citation must come from, for integrity scoring."""
        return {row.document_id for row in self.search(committed_diagnosis, claim_type)}

    def resolve(self, document_id: str) -> GuidelineEvidence | None:
        """Look a cited identifier up by id, for citation-integrity scoring.

        Returns None for an identifier the catalog does not contain, which is how
        a hallucinated citation is detected.
        """
        for row in self.records:
            if str(row["document_id"]) == str(document_id):
                return self._as_evidence(row)
        return None
