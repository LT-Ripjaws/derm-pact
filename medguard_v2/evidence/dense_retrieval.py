"""A dense top-k-over-chunked-text retrieval arm, built from the SAME guideline source text
the hash-pinned keyed corpus is built from -- and deliberately WITHOUT the structured provenance
that corpus adds.

Why this exists
---------------
The keyed path (:mod:`medguard_v2.evidence.guideline_binding`) serves whole, curated
recommendation *records*: each carries a resolvable ``record_id``, a content hash that
re-verifies, a citation id and a source version. Its citation-resolvability is 1.0 by
construction -- every served item is a verifiable record.

A *standard* medical RAG does none of that. It ingests the underlying guideline text, chops it
into fixed-size passages that do not respect the curators' record boundaries, embeds them, and
returns the top-k by cosine similarity. The passages it returns are spans of text, not records:
they carry no resolvable record id, a chunk that straddles two recommendations or truncates one
re-verifies against nothing, and nearest-neighbour search can return a passage from the wrong
condition because two treatments read alike.

This module builds exactly that arm so the comparison "verified structured retrieval vs dense
chunk retrieval" becomes a *measured control* rather than an assertion (corpus finding MG4:
standard medical RAG can DEGRADE performance). It is strictly ADDITIVE: it never touches the
hash-pinned binding, the keyed retrieval path, or the decision contract. Its output is walled
off -- see :func:`format_unverified_chunks` and the ``literature_lookup`` agent tool -- and can
never populate a structured citations array.

Two chunking modes are provided so the two costs can be isolated:

* ``mechanical`` (the realistic arm): concatenate the record texts within each source document,
  in corpus order, and slide a fixed-size window across the concatenation. Chunk boundaries fall
  wherever the window lands, so records get merged and fragmented -- the curated boundaries are
  gone, which is the whole point.
* ``record_aligned`` (dense's best case): one chunk per record, chunk text == record text. This
  is the ceiling: it shows that a dense chunk can only re-verify its provenance when its
  boundaries happen to reproduce the curated record boundaries the keyed corpus already gives you.

Embedders are injected, so nothing here forces a heavy import. Three are available:

* :class:`SentenceTransformerEmbedder` -- ``all-mpnet-base-v2`` by default, the canonical
  retrieval-tuned dense model. The primary arm, chosen so the comparison is not a strawman.
* :class:`DermLIPTextEmbedder` -- the in-domain DermLIP text tower (the same frozen model whose
  image features the whole system reads). A robustness row.
* :class:`HashingEmbedder` -- a deterministic, numpy-only hashed-token embedder with no model
  download. Used by the tests (fast and hermetic) and as an offline fallback.

Nothing here paraphrases guideline text: chunks are verbatim spans of the source ``text`` fields.
No image bytes are touched. Only chunk text (already carried in the corpus), vectors and metadata
are written.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from medguard_v2.evidence.guideline_binding import CATALOG_PATH

REPO = Path(__file__).resolve().parents[2]
DENSE_INDEX_ROOT = REPO / "research" / "evidence" / "derm7pt_v1" / "dense_index"

#: A dense chunk id is deliberately NOT a catalog record id and not even citation-shaped
#: (no colons -> cannot match verification._CITATION_TOKEN, cannot resolve via catalog.has_record).
#: This is one half of the wall: an id from this channel that a misbehaving model copied into the
#: citations array is rejected in-loop by the existing citation_array_checks and counted
#: unresolvable by score_citations -- neither of which this module modifies.
CHUNK_ID_PREFIX = "litchunk-"

_WHITESPACE = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# --------------------------------------------------------------------------- source records


@dataclass(frozen=True)
class SourceRecord:
    """The fields of one keyed record that the dense arm is allowed to see: the verbatim text,
    its condition, and the provenance the keyed path *would* attach -- kept here ONLY so the
    controlled comparison can score whether a dense chunk recovers it, never emitted as a citation.
    """

    record_id: str
    condition: str
    question_types: Tuple[str, ...]
    text: str
    content_sha256: str
    source_document_id: str


def load_source_records(path: Path = CATALOG_PATH) -> List[SourceRecord]:
    """Read the same JSONL the keyed catalog loads, projected to the fields the dense arm uses.

    Read-only. This does not verify or serve anything -- ``GuidelineCatalog`` remains the sole
    authority on what is a valid, hash-checked record.
    """
    out: List[SourceRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            out.append(
                SourceRecord(
                    record_id=str(raw["id"]),
                    condition=str(raw["condition"]),
                    question_types=tuple(str(q) for q in (raw.get("question_types") or [])),
                    text=str(raw["text"]),
                    content_sha256=str(raw.get("content_sha256", "")),
                    source_document_id=str(raw.get("source_document_id", "")),
                )
            )
    return out


# --------------------------------------------------------------------------- chunking


@dataclass(frozen=True)
class Chunk:
    """One dense passage. ``overlaps`` maps each source record_id the chunk's characters intersect
    to the number of overlapping characters -- computed exactly during construction, used only to
    SCORE provenance recovery, never surfaced as a citation.
    """

    chunk_id: str
    text: str
    source_document_id: str
    overlaps: Mapping[str, int]

    @property
    def majority_record_id(self) -> Optional[str]:
        """The record the chunk mostly came from (max character overlap). ``None`` if it overlaps
        nothing (cannot happen for these chunkers, but kept total)."""
        if not self.overlaps:
            return None
        return max(self.overlaps.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @property
    def content_sha256(self) -> str:
        """Hash of the chunk's EXACT text, the same way the catalog hashes a record's text.
        Equals some record's ``content_sha256`` only when the chunk reproduces that record's text
        verbatim -- i.e. only in record-aligned mode. In mechanical mode a windowed span matches
        nothing, which is the resolvability cost this arm is here to measure."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkConfig:
    mode: str = "mechanical"        # "mechanical" | "record_aligned"
    target_chars: int = 320
    overlap_chars: int = 64
    joiner: str = " "               # how record texts are concatenated within a source document

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "target_chars": self.target_chars,
            "overlap_chars": self.overlap_chars,
            "joiner": repr(self.joiner),
        }


def _sliding_windows(n: int, target: int, overlap: int) -> List[Tuple[int, int]]:
    """[start, end) windows over ``n`` characters. Deterministic; the last window ends at ``n``."""
    if target <= 0:
        raise ValueError("target_chars must be positive")
    if overlap < 0 or overlap >= target:
        raise ValueError("overlap_chars must satisfy 0 <= overlap < target")
    if n <= target:
        return [(0, n)]
    step = target - overlap
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + target, n)
        spans.append((start, end))
        if end == n:
            break
        start += step
    return spans


def _snap(pos: int, text: str, limit: int) -> int:
    """Nudge a cut point to the next whitespace within ``limit`` chars, so windows break on word
    boundaries rather than mid-token. Never moves past ``len(text)``; falls back to ``pos``."""
    if pos <= 0 or pos >= len(text):
        return pos
    for j in range(pos, min(pos + limit, len(text))):
        if text[j].isspace():
            return j
    return pos


def build_chunks(records: Sequence[SourceRecord], config: ChunkConfig) -> List[Chunk]:
    """Chunk the guideline source text under ``config``. Deterministic in record order."""
    if config.mode == "record_aligned":
        return [
            Chunk(
                chunk_id=f"{CHUNK_ID_PREFIX}{i:04d}",
                text=_norm_text(r.text),
                source_document_id=r.source_document_id,
                overlaps={r.record_id: len(_norm_text(r.text))},
            )
            for i, r in enumerate(records)
        ]

    if config.mode != "mechanical":
        raise ValueError(f"unknown chunk mode {config.mode!r}")

    # Group records by source document, preserving corpus order, and concatenate their texts.
    # A per-record [start, end) offset map over the concatenation lets chunk<->record overlap be
    # computed exactly, so provenance recovery is scored without guesswork.
    docs: List[Tuple[str, List[SourceRecord]]] = []
    for r in records:
        if docs and docs[-1][0] == r.source_document_id:
            docs[-1][1].append(r)
        else:
            docs.append((r.source_document_id, [r]))

    chunks: List[Chunk] = []
    counter = 0
    for doc_id, doc_records in docs:
        pieces: List[str] = []
        spans: List[Tuple[str, int, int]] = []  # (record_id, start, end) in the concatenation
        cursor = 0
        for r in doc_records:
            t = _norm_text(r.text)
            if cursor > 0:
                cursor += len(config.joiner)
                pieces.append(config.joiner)
            spans.append((r.record_id, cursor, cursor + len(t)))
            pieces.append(t)
            cursor += len(t)
        concat = "".join(pieces)

        raw_windows = _sliding_windows(len(concat), config.target_chars, config.overlap_chars)
        for (start, end) in raw_windows:
            s = _snap(start, concat, config.overlap_chars) if start > 0 else start
            e = _snap(end, concat, config.overlap_chars) if end < len(concat) else end
            chunk_text = concat[s:e].strip()
            if not chunk_text:
                continue
            overlaps: Dict[str, int] = {}
            for record_id, rstart, rend in spans:
                lo, hi = max(s, rstart), min(e, rend)
                if hi > lo:
                    overlaps[record_id] = hi - lo
            chunks.append(
                Chunk(
                    chunk_id=f"{CHUNK_ID_PREFIX}{counter:04d}",
                    text=chunk_text,
                    source_document_id=doc_id,
                    overlaps=overlaps,
                )
            )
            counter += 1
    return chunks


# --------------------------------------------------------------------------- embedders


class Embedder(Protocol):
    """Maps a list of strings to an (n, d) float32 L2-normalised array, deterministically."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def identity(self) -> Dict[str, Any]: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class HashingEmbedder:
    """A deterministic, dependency-light dense embedder: hashed word/char n-grams into a fixed
    space, L2-normalised. No model download, numpy only. NOT semantic -- it is a lexical hashing
    projection -- but it is a genuine fixed-dim cosine index, which is all the tests need, and it
    is the offline fallback when neither model checkpoint is present. Its identity records that it
    is a fallback so an artifact built with it can never be mistaken for a real dense run.
    """

    def __init__(self, dim: int = 512, ngram: Tuple[int, int] = (3, 5)):
        self.dim = int(dim)
        self.ngram = ngram

    def _features(self, text: str) -> List[str]:
        t = _norm_text(text).lower()
        feats = [tok for tok in re.split(r"\W+", t) if tok]
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            for i in range(0, max(0, len(t) - n + 1)):
                feats.append(t[i : i + n])
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int(hashlib.sha1(feat.encode("utf-8")).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        return _l2_normalise(out)

    def identity(self) -> Dict[str, Any]:
        return {
            "embedder": "hashing-fallback",
            "kind": "lexical_hashing_projection",
            "dim": self.dim,
            "ngram": list(self.ngram),
            "semantic": False,
            "note": "deterministic numpy-only fallback / test embedder, NOT a semantic model",
        }


class SentenceTransformerEmbedder:
    """The primary dense arm: a retrieval-tuned sentence-transformer (all-mpnet-base-v2 default).

    Loaded offline (local HF cache only). CPU, eval mode, fixed weights -> deterministic. Imports
    ``sentence_transformers`` lazily so this module stays importable where that package is absent.
    """

    def __init__(self, model_id: str = "sentence-transformers/all-mpnet-base-v2",
                 *, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def identity(self) -> Dict[str, Any]:
        return {
            "embedder": self.model_id,
            "kind": "sentence_transformer",
            "device": self.device,
            "normalized": True,
            "semantic": True,
        }


class DermLIPTextEmbedder:
    """The in-domain robustness arm: the DermLIP text tower, the same frozen model whose image
    features the perception layer reads. Loaded offline via open_clip; CPU, eval, fixed weights ->
    deterministic. 77-token CLIP context, which comfortably covers these passages.
    """

    def __init__(self, model_id: str = "redlessone/DermLIP_ViT-B-16"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:{self.model_id}")
            model.eval()
            self._model = model
            self._tokenizer = open_clip.get_tokenizer(f"hf-hub:{self.model_id}")
        return self._model, self._tokenizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model, tokenizer = self._load()
        with torch.inference_mode():
            tokens = tokenizer(list(texts))
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype("float32")

    def identity(self) -> Dict[str, Any]:
        return {
            "embedder": self.model_id,
            "kind": "open_clip_text_tower",
            "device": "cpu",
            "normalized": True,
            "semantic": True,
            "note": "in-domain DermLIP text tower; same frozen model the perception layer reads",
        }


# --------------------------------------------------------------------------- index


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class DenseGuidelineIndex:
    """A cosine top-k index over dense-chunked guideline text.

    Built from an :class:`Embedder` and a list of :class:`Chunk`. Vectors are L2-normalised, so a
    cosine query is a single matrix-vector product. ``version_hash`` fingerprints the embedder
    identity, the chunk config and the chunk texts together with the quantised vectors, so an
    index can be cached and a stale one detected loudly.
    """

    def __init__(self, chunks: List[Chunk], vectors: np.ndarray, *, embedder_identity: Dict[str, Any],
                 chunk_config: ChunkConfig, source_sha256: str):
        if vectors.shape[0] != len(chunks):
            raise ValueError("vector/chunk count mismatch")
        self.chunks = chunks
        self.vectors = _l2_normalise(vectors)
        self.embedder_identity = embedder_identity
        self.chunk_config = chunk_config
        self.source_sha256 = source_sha256
        self._embedder: Optional[Embedder] = None

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, embedder: Embedder, *, config: Optional[ChunkConfig] = None,
              records: Optional[Sequence[SourceRecord]] = None) -> "DenseGuidelineIndex":
        config = config or ChunkConfig()
        records = list(records if records is not None else load_source_records())
        chunks = build_chunks(records, config)
        vectors = embedder.encode([c.text for c in chunks])
        source_sha256 = _source_sha256(records)
        index = cls(
            chunks, vectors, embedder_identity=embedder.identity(),
            chunk_config=config, source_sha256=source_sha256,
        )
        index._embedder = embedder
        return index

    # -- retrieval -----------------------------------------------------------

    def set_query_embedder(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Top-k chunks for ``query`` by cosine similarity. Deterministic: ties break by ascending
        chunk index. Returns plain dicts (chunk_id, text, score, source_document_id) -- the shape
        the agent tool consumes. The list carries NO resolvable record id, by design.
        """
        results = self.retrieve_detailed(query, k)
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "text": r.chunk.text,
                "score": round(float(r.score), 6),
                "source_document_id": r.chunk.source_document_id,
            }
            for r in results
        ]

    def retrieve_detailed(self, query: str, k: int = 3) -> List[Retrieved]:
        if self._embedder is None:
            raise RuntimeError(
                "no query embedder set; build() stores one, or call set_query_embedder()"
            )
        q = self._embedder.encode([query])
        q = _l2_normalise(q)[0]
        sims = self.vectors @ q
        # Ascending chunk index breaks ties: negate sims for a stable ascending argsort by
        # (-score, index).
        order = sorted(range(len(self.chunks)), key=lambda i: (-float(sims[i]), i))
        return [Retrieved(self.chunks[i], float(sims[i])) for i in order[: max(0, k)]]

    # -- versioning / caching ------------------------------------------------

    @property
    def version_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "embedder": self.embedder_identity,
                    "chunk_config": self.chunk_config.as_dict(),
                    "source_sha256": self.source_sha256,
                    "chunk_texts": [c.text for c in self.chunks],
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        # Quantise vectors before hashing so tiny float noise across BLAS builds does not
        # gratuitously invalidate a cache, while any real change still does.
        digest.update(np.round(self.vectors, 5).astype(np.float32).tobytes())
        return digest.hexdigest()

    def manifest(self) -> Dict[str, Any]:
        return {
            "protocol": "medguard-v2-dense-guideline-index-v1",
            "embedder": self.embedder_identity,
            "chunk_config": self.chunk_config.as_dict(),
            "n_chunks": len(self.chunks),
            "dim": int(self.vectors.shape[1]),
            "source_sha256": self.source_sha256,
            "version_hash": self.version_hash,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "vectors.npz",
            vectors=self.vectors,
            chunk_ids=np.array([c.chunk_id for c in self.chunks]),
        )
        (directory / "chunks.json").write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_document_id": c.source_document_id,
                        "overlaps": dict(c.overlaps),
                    }
                    for c in self.chunks
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return directory


def _source_sha256(records: Sequence[SourceRecord]) -> str:
    digest = hashlib.sha256()
    for r in records:
        digest.update(f"{r.record_id}\t{r.content_sha256}\n".encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- walling


#: The disclaimer stamped onto every literature_lookup payload. States the channel is unverified,
#: not citable, and that its ids are not guideline record ids. This is a labelling wall on top of
#: the structural one (chunk ids do not resolve in the hash-pinned catalog, so the existing
#: citation gate rejects them and the resolvability metric excludes them).
UNVERIFIED_DISCLAIMER = (
    "UNVERIFIED background reading. These passages are dense-retrieved spans of guideline source "
    "text; they are NOT citable guideline records. Their chunk_ids are NOT guideline record IDs "
    "and must never be placed in the citations array. Citation-resolvability is not claimed for "
    "this channel. Use only to orient yourself; every clinical claim must still be grounded in a "
    "search_guidelines record."
)


def format_unverified_chunks(retrieved: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Wrap dense-retrieved chunks in the unverified-exploration envelope the agent tool returns.

    Every chunk is stamped ``unverified: true`` and ``citable: false``; the top-level payload
    carries ``channel``, ``citable: false`` and the disclaimer. Deliberately emits NO field named
    or shaped like a citable record id.
    """
    return {
        "channel": "unverified_exploration",
        "citable": False,
        "disclaimer": UNVERIFIED_DISCLAIMER,
        "n_chunks": len(retrieved),
        "chunks": [
            {
                "chunk_id": str(c["chunk_id"]),
                "text": str(c["text"]),
                "similarity": round(float(c.get("score", 0.0)), 4),
                "unverified": True,
                "citable": False,
            }
            for c in retrieved
        ],
    }
