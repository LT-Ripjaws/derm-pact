"""Fresh FAISS text retriever for the MedGuard pipeline.

Cosine retrieval over MedGuard's OWN index (built by build.py from a public corpus
such as PubMed derm abstracts). Independent of legacy baseline's src/retriever.py. faiss +
sentence-transformers are imported lazily so importing this module is cheap and the
pipeline stays fake-testable.

Contract used by MedGuardPipelineOps:
    search(query: str, k: int) -> (chunks: list[dict], scores: list[float])
where each chunk dict carries at least {"text", "id", "source"}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import config


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Read a jsonl corpus into a list of records (each must have a 'text' field)."""
    recs: List[Dict[str, Any]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # accept {"text": ...} or PubMed-style {"title","abstract"} -> synthesize text
            text = d.get("text") or " ".join(
                s for s in (d.get("title"), d.get("abstract")) if s).strip()
            if text:
                d = dict(d)
                d["text"] = text
                d.setdefault("source", d.get("pmid") or d.get("id") or "corpus")
                recs.append(d)
    return recs


class FaissTextRetriever:
    """Dense cosine retriever over a prebuilt FAISS (IndexFlatIP) + parallel metadata."""

    def __init__(self, index=None, meta: Optional[List[Dict[str, Any]]] = None,
                 embedder=None, model: Optional[str] = None):
        self._index = index
        self._meta = meta or []
        self._embedder = embedder
        self.model = model or config.EMBEDDING_MODEL

    # ---- load ----
    @classmethod
    def load(cls, index_path: Optional[str] = None, meta_path: Optional[str] = None,
             model: Optional[str] = None, embedder=None) -> "FaissTextRetriever":
        import faiss
        using_frozen_defaults = index_path is None and meta_path is None
        if using_frozen_defaults:
            from ..evidence.bundle import verify_evidence_bundle

            verify_evidence_bundle()
        index_path = index_path or config.TEXT_INDEX_PATH
        meta_path = meta_path or config.TEXT_INDEX_META
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"MedGuard text index missing at {index_path}. Build it: "
                f"python -m medguard.pipeline.build --corpus <public_corpus.jsonl>")
        index = faiss.read_index(str(index_path))
        meta = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
        if int(index.ntotal) != len(meta):
            raise ValueError("text index vector count differs from its frozen metadata")
        return cls(index=index, meta=meta, model=model, embedder=embedder)

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._embedder = SentenceTransformer(self.model)
        return self._embedder

    def embed(self, texts: List[str]):
        import numpy as np
        emb = self._ensure_embedder().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")

    # ---- search ----
    def search(self, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        if self._index is None:
            raise RuntimeError("retriever has no index; FaissTextRetriever.load(...) first")
        q = self.embed([query])
        k = max(1, min(k, len(self._meta)))
        scores, idxs = self._index.search(q, k)
        chunks: List[Dict[str, Any]] = []
        out_scores: List[float] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            rec = dict(self._meta[int(i)])
            rec["score"] = float(score)
            rec["retrieval_channel"] = "dense"
            chunks.append(rec)
            out_scores.append(float(score))
        return chunks, out_scores


def build_index(corpus_path: str, index_path: str, meta_path: str,
                model: Optional[str] = None, batch_size: int = 256) -> int:
    """Build a MedGuard FAISS index from a public jsonl corpus. Returns #vectors.

    Cosine via IndexFlatIP over L2-normalized embeddings. Writes the index and a
    parallel metadata jsonl (one record per vector, in index order)."""
    import faiss
    import numpy as np
    recs = load_corpus(corpus_path)
    if not recs:
        raise ValueError(f"empty/invalid corpus: {corpus_path}")
    retr = FaissTextRetriever(model=model)
    texts = [r["text"] for r in recs]
    vecs = np.vstack([retr.embed(texts[i:i + batch_size])
                      for i in range(0, len(texts), batch_size)])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return int(index.ntotal)
