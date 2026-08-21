"""Sparse/dense retrieval primitives with optional provenance admission control.

The classes in this module do not generate text.  They keep retrieval ranking and
evidence admission separate so a relevance model cannot silently override a failed
content-integrity check.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Return stable lowercase word tokens for sparse retrieval."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def canonical_document_hash(record: Mapping[str, Any]) -> str:
    """Hash the evidence-bearing fields, excluding mutable ranking metadata."""
    payload = {
        "source": str(record.get("source") or record.get("pmid") or record.get("id") or ""),
        "title": str(record.get("title") or ""),
        "abstract": str(record.get("abstract") or ""),
        "text": str(record.get("text") or ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def document_id(record: Mapping[str, Any]) -> str:
    value = record.get("id") or record.get("pmid") or record.get("source")
    if value is None or not str(value).strip():
        return canonical_document_hash(record)
    return str(value)


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Mapping[str, Any]], k: int) -> List[Dict[str, Any]]:
        ...


class BM25Retriever:
    """Small exact Okapi BM25 implementation for immutable in-memory corpora."""

    def __init__(self, records: Sequence[Mapping[str, Any]], *, k1: float = 1.2, b: float = 0.75):
        if not records:
            raise ValueError("BM25Retriever requires at least one document")
        self.records = [dict(record) for record in records]
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens = [tokenize(str(record.get("text") or "")) for record in self.records]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avg_length = sum(self._lengths) / len(self._lengths)
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        n_docs = len(self.records)
        self._idf = {
            term: math.log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        if k <= 0:
            return [], []
        query_terms = tokenize(query)
        if not query_terms:
            return [], []
        scores: List[Tuple[float, int]] = []
        for index, (frequencies, length) in enumerate(zip(self._term_frequencies, self._lengths)):
            score = 0.0
            normalization = self.k1 * (1.0 - self.b + self.b * length / max(self._avg_length, 1e-12))
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / (frequency + normalization)
            if score > 0.0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], document_id(self.records[item[1]])))
        selected = scores[: min(k, len(scores))]
        chunks: List[Dict[str, Any]] = []
        values: List[float] = []
        for score, index in selected:
            row = dict(self.records[index])
            row["score"] = float(score)
            row["retrieval_channel"] = "bm25"
            chunks.append(row)
            values.append(float(score))
        return chunks, values


class FrozenBM25Retriever:
    """Load and query a receipt-bound deterministic BM25 postings artifact."""

    def __init__(self, records: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]):
        if artifact.get("schema_version") != "medguard-frozen-bm25-v1":
            raise ValueError("unsupported frozen BM25 schema")
        self.records = [dict(record) for record in records]
        self.k1 = float(artifact["k1"])
        self.b = float(artifact["b"])
        self._avg_length = float(artifact["average_document_length"])
        self._lengths = [int(value) for value in artifact["document_lengths"]]
        self._idf = {str(term): float(value) for term, value in artifact["idf"].items()}
        self._postings = {
            str(term): [(int(index), int(frequency)) for index, frequency in values]
            for term, values in artifact["postings"].items()
        }
        expected_ids = [str(record.get("document_id") or document_id(record)) for record in self.records]
        if expected_ids != [str(value) for value in artifact["document_ids"]]:
            raise ValueError("frozen BM25 document order differs from corpus")
        if len(self._lengths) != len(self.records) or not self.records:
            raise ValueError("frozen BM25 document lengths differ from corpus")
        if not math.isfinite(self._avg_length) or self._avg_length <= 0.0:
            raise ValueError("invalid frozen BM25 average document length")
        if self.k1 <= 0.0 or not 0.0 <= self.b <= 1.0:
            raise ValueError("invalid frozen BM25 parameters")
        for postings in self._postings.values():
            if any(index < 0 or index >= len(self.records) or frequency <= 0
                   for index, frequency in postings):
                raise ValueError("invalid frozen BM25 posting")

    @classmethod
    def load(cls, corpus_path: str | Path, index_path: str | Path) -> "FrozenBM25Retriever":
        corpus_file = Path(corpus_path)
        raw = corpus_file.read_bytes()
        artifact = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if hashlib.sha256(raw).hexdigest() != artifact.get("corpus_sha256"):
            raise ValueError("frozen BM25 corpus hash mismatch")
        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid corpus JSON on line {line_number}") from error
        return cls(records, artifact)

    def search(self, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        if k <= 0:
            return [], []
        query_terms = tokenize(query)
        if not query_terms:
            return [], []
        scores: Dict[int, float] = {}
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequency in self._postings.get(term, ()):
                length = self._lengths[index]
                normalization = self.k1 * (
                    1.0 - self.b + self.b * length / max(self._avg_length, 1e-12)
                )
                increment = idf * frequency * (self.k1 + 1.0) / (frequency + normalization)
                scores[index] = scores.get(index, 0.0) + increment
        ordered = sorted(
            scores,
            key=lambda index: (-scores[index], document_id(self.records[index])),
        )[: min(k, len(scores))]
        chunks: List[Dict[str, Any]] = []
        values: List[float] = []
        for index in ordered:
            row = dict(self.records[index])
            row["score"] = float(scores[index])
            row["retrieval_channel"] = "bm25"
            chunks.append(row)
            values.append(float(scores[index]))
        return chunks, values


def reciprocal_rank_fusion_records(
    ranked_lists: Sequence[Sequence[Mapping[str, Any]]],
    *,
    weights: Optional[Sequence[float]] = None,
    rank_constant: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse rankings by stable document id and retain channel-specific audit data."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must align with ranked_lists")
    scores: Dict[str, float] = {}
    records: Dict[str, Dict[str, Any]] = {}
    channel_ranks: Dict[str, List[int]] = {}
    channels: Dict[str, set[str]] = {}
    for ranking, weight in zip(ranked_lists, weights):
        for rank, record in enumerate(ranking, start=1):
            key = document_id(record)
            scores[key] = scores.get(key, 0.0) + float(weight) / (rank_constant + rank)
            records.setdefault(key, dict(record))
            channel_ranks.setdefault(key, []).append(rank)
            channel = str(record.get("retrieval_channel") or "unknown")
            channels.setdefault(key, set()).add(channel)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    output: List[Dict[str, Any]] = []
    for key in ordered:
        row = records[key]
        row["score"] = float(scores[key])
        row["rrf_score"] = float(scores[key])
        row["rrf_channel_ranks"] = channel_ranks[key]
        row["rrf_channels"] = sorted(channels[key])
        row["retrieval_channel"] = "+".join(sorted(channels[key]))
        output.append(row)
    return output


@dataclass(frozen=True)
class TrustDecision:
    admitted: bool
    reason: str


class TrustRegistry:
    """Immutable source-to-content-hash admission registry.

    A source is admitted only if both its identifier and evidence-bearing content
    match the frozen registry.  Unknown sources are quarantined, not treated as bad
    merely because of their language.
    """

    def __init__(self, source_hashes: Mapping[str, str]):
        self._source_hashes = {str(source): str(digest) for source, digest in source_hashes.items()}

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "TrustRegistry":
        source_hashes: Dict[str, str] = {}
        for record in records:
            source = str(record.get("source") or record.get("pmid") or record.get("id") or "")
            if not source:
                raise ValueError("trusted record lacks a source identifier")
            digest = canonical_document_hash(record)
            previous = source_hashes.get(source)
            if previous is not None and previous != digest:
                raise ValueError(f"trusted source {source!r} has conflicting content")
            source_hashes[source] = digest
        return cls(source_hashes)

    def decide(self, record: Mapping[str, Any]) -> TrustDecision:
        source = str(record.get("source") or record.get("pmid") or record.get("id") or "")
        expected = self._source_hashes.get(source)
        if expected is None:
            return TrustDecision(False, "unknown_source")
        if canonical_document_hash(record) != expected:
            return TrustDecision(False, "content_hash_mismatch")
        return TrustDecision(True, "verified_source_and_content")

    def partition(self, records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        admitted: List[Dict[str, Any]] = []
        quarantined: List[Dict[str, Any]] = []
        for record in records:
            row = dict(record)
            decision = self.decide(row)
            row["trust_admission_reason"] = decision.reason
            (admitted if decision.admitted else quarantined).append(row)
        return admitted, quarantined


def admit_before_indexing(
    records: Sequence[Mapping[str, Any]], registry: TrustRegistry
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split corpus updates before either sparse or dense indexing.

    This prevents untrusted documents from crowding the retriever's finite candidate
    window.  Quarantined rows remain available for audit/review but cannot be shown
    to a generator-facing retriever.
    """
    return registry.partition(records)


class HybridRetriever:
    """Dense+sparse RRF with integrity admission before optional reranking."""

    def __init__(
        self,
        dense: Any,
        sparse: BM25Retriever,
        *,
        reranker: Optional[Reranker] = None,
        trust_registry: Optional[TrustRegistry] = None,
        candidate_k: int = 20,
        rank_constant: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.trust_registry = trust_registry
        self.candidate_k = int(candidate_k)
        self.rank_constant = int(rank_constant)

    def search(self, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        if k <= 0 or not tokenize(query):
            return [], []
        width = max(k, self.candidate_k)
        dense_records, _ = self.dense.search(query, width)
        sparse_records, _ = self.sparse.search(query, width)
        candidates = reciprocal_rank_fusion_records(
            [dense_records, sparse_records], rank_constant=self.rank_constant
        )
        quarantined: List[Dict[str, Any]] = []
        if self.trust_registry is not None:
            candidates, quarantined = self.trust_registry.partition(candidates)
        if self.reranker is not None and candidates:
            candidates = self.reranker.rerank(query, candidates, k)
        else:
            candidates = candidates[:k]
        for row in candidates:
            row["quarantined_candidate_count"] = len(quarantined)
        return candidates, [float(row.get("score", 0.0)) for row in candidates]


class MedCPTReranker:
    """Pinned MedCPT cross-encoder adapter; never downloads when offline is true."""

    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Cross-Encoder",
        *,
        batch_size: int = 8,
        offline: bool = True,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=offline
        )
        self.model.eval()

    def rerank(self, query: str, candidates: Sequence[Mapping[str, Any]], k: int) -> List[Dict[str, Any]]:
        if k <= 0 or not candidates:
            return []
        import torch

        scored: List[Tuple[float, Dict[str, Any]]] = []
        with torch.inference_mode():
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start : start + self.batch_size]
                pairs = [[query, str(row.get("text") or "")] for row in batch]
                encoded = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                logits = self.model(**encoded).logits.reshape(-1).detach().cpu().tolist()
                for value, candidate in zip(logits, batch):
                    row = dict(candidate)
                    row["pre_rerank_score"] = float(row.get("score", 0.0))
                    row["score"] = float(value)
                    row["medcpt_score"] = float(value)
                    scored.append((float(value), row))
        scored.sort(key=lambda item: (-item[0], document_id(item[1])))
        return [row for _, row in scored[: min(k, len(scored))]]
