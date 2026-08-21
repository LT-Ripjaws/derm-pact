"""Fresh visual-exemplar branch for the MedGuard pipeline.

A frozen image encoder + a kNN atlas over labelled TRAIN images (no dev/test leakage)
+ optional linear probes on the frozen features. Produces the diagnostic commitment,
confidence and neighbour evidence used by the evidence contract.

Encoder is PLUGGABLE (config.EXEMPLAR_ENCODER_MODEL): DermLIP / BiomedCLIP via
open_clip. Heavy deps (open_clip, torch, faiss,
joblib, sklearn) are lazy so this module imports cheap and the vote logic is unit-tested
with fakes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .. import config


class ImageEncoder(Protocol):
    def encode_image(self, image_path: str): ...   # -> np.ndarray (L2-normalized)


class OpenClipEncoder:
    """Frozen open_clip image encoder (DermLIP / BiomedCLIP / any hf-hub CLIP)."""

    def __init__(self, model_name: Optional[str] = None, _model=None, _preprocess=None):
        self.model_name = model_name or config.EXEMPLAR_ENCODER_MODEL
        self._model = _model
        self._preprocess = _preprocess

    def _ensure(self):
        if self._model is None:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(self.model_name)
            model.eval()
            self._model, self._preprocess = model, preprocess
        return self._model, self._preprocess

    def encode_image(self, image_path: str):
        import numpy as np
        import torch
        from PIL import Image
        model, preprocess = self._ensure()
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32")[0]


def make_encoder(model_name: Optional[str] = None) -> ImageEncoder:
    """Build a frozen dermatology-compatible open_clip encoder."""
    name = model_name or config.EXEMPLAR_ENCODER_MODEL
    low = name.lower()
    if low.startswith("hf-hub:") or "clip" in low:
        return OpenClipEncoder(name)
    raise ValueError(f"unsupported dermatology image encoder: {name!r}")


@dataclass
class ExemplarResult:
    exemplar_confidence: float          # committed-label confidence (probe softmax or vote share)
    exemplar_malignant_score: float     # P(malignant)
    exemplar_pred: str                  # committed diagnosis label
    exemplar_context: str               # neighbour-evidence string for the generator
    neighbors: List[Tuple[str, str, float]] = field(default_factory=list)  # (id, label, score)

    def as_state(self) -> Dict[str, Any]:
        return {
            "exemplar_confidence": self.exemplar_confidence,
            "exemplar_malignant_score": self.exemplar_malignant_score,
            "exemplar_pred": self.exemplar_pred,
            "exemplar_context": self.exemplar_context,
        }


def summarize_neighbors(
    neighbors: Sequence[Tuple[str, str, float]],
    labels: Sequence[str],
    malignant_labels,
    *,
    prior: Optional[Dict[str, float]] = None,
    prior_alpha: float = 0.3,
    diag_probs: Optional[Dict[str, float]] = None,
    malignant_prob: Optional[float] = None,
) -> ExemplarResult:
    """Pure vote+probe summary (unit-tested without models).

    Inverse-prevalence-corrected similarity vote: score(label) += sim / prior(label)**alpha,
    so a nevus-heavy atlas does not swamp the malignant vote. If linear-probe outputs are
    supplied (`diag_probs`, `malignant_prob`) they OVERRIDE the coarse vote (the standard
    frozen-feature protocol: probe > raw kNN vote), and the kNN still supplies evidence."""
    vote: Dict[str, float] = {lab: 0.0 for lab in labels}
    for _id, lab, sim in neighbors:
        if lab in vote:
            p = (prior or {}).get(lab, 1.0)
            denom = (p ** prior_alpha) if p > 0 else 1.0
            vote[lab] += max(0.0, sim) / (denom or 1.0)
    total = sum(vote.values()) or 1.0

    if diag_probs:
        pred = max(diag_probs.items(), key=lambda kv: kv[1])[0]
        conf = float(diag_probs[pred])
    else:
        pred = max(vote.items(), key=lambda kv: kv[1])[0]
        conf = vote[pred] / total

    if malignant_prob is not None:
        mal = float(malignant_prob)
    else:
        mal = sum(vote[l] for l in labels if l in malignant_labels) / total

    ctx = "; ".join(f"{lab} ({sim:.2f})" for _id, lab, sim in list(neighbors)[:5]) or "(no neighbours)"
    return ExemplarResult(exemplar_confidence=conf, exemplar_malignant_score=mal,
                          exemplar_pred=pred, exemplar_context=ctx, neighbors=list(neighbors))


class ExemplarAtlas:
    """kNN atlas over frozen train-image features + optional linear probes."""

    def __init__(self, index=None, meta: Optional[List[Dict[str, Any]]] = None,
                 encoder: Optional[ImageEncoder] = None, malignant_clf=None, diag_clf=None,
                 labels: Optional[Sequence[str]] = None,
                 prior_alpha: Optional[float] = None, top_k: Optional[int] = None):
        self._index = index
        self._meta = meta or []
        self.encoder = encoder
        self.malignant_clf = malignant_clf
        self.diag_clf = diag_clf
        self.labels = list(labels or config.DIAGNOSIS_LABELS)
        self.prior_alpha = config.EXEMPLAR_PRIOR_ALPHA if prior_alpha is None else prior_alpha
        self.top_k = config.EXEMPLAR_TOP_K if top_k is None else top_k
        self.prior = self._compute_prior()

    def _compute_prior(self) -> Dict[str, float]:
        c = Counter(m.get("label") for m in self._meta)
        n = sum(c.values()) or 1
        return {lab: c.get(lab, 0) / n for lab in self.labels}

    @classmethod
    def load(cls, index_path=None, meta_path=None, encoder=None,
             malignant_path=None, diag_path=None, *, load_probes: bool = True,
             strict_train_only: bool = True, **kw) -> "ExemplarAtlas":
        import faiss
        import json
        using_frozen_defaults = index_path is None and meta_path is None
        if using_frozen_defaults:
            from ..evidence.bundle import verify_evidence_bundle

            verify_evidence_bundle()
        index_path = index_path or config.EXEMPLAR_INDEX_PATH
        meta_path = meta_path or config.EXEMPLAR_META_PATH
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"exemplar atlas missing at {index_path}; build it: "
                f"python -m medguard.pipeline.build_atlas --train <train.jsonl>")
        index = faiss.read_index(str(index_path))
        meta = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
        if int(index.ntotal) != len(meta):
            raise ValueError("exemplar index vector count differs from its frozen metadata")
        if strict_train_only and any(row.get("split") != "train" for row in meta):
            raise ValueError("exemplar atlas contains a non-train record")
        if len({str(row.get("id") or "") for row in meta}) != len(meta):
            raise ValueError("exemplar atlas metadata identifiers are not unique")
        if load_probes:
            import joblib  # loads only MedGuard-owned probe artifacts

            mal = (joblib.load(malignant_path or config.EXEMPLAR_MALIGNANT_CLF)
                   if Path(malignant_path or config.EXEMPLAR_MALIGNANT_CLF).exists() else None)
            diag = (joblib.load(diag_path or config.EXEMPLAR_DIAG_CLF)
                    if Path(diag_path or config.EXEMPLAR_DIAG_CLF).exists() else None)
        else:
            mal = diag = None
        return cls(index=index, meta=meta, encoder=encoder or make_encoder(),
                   malignant_clf=mal, diag_clf=diag, **kw)

    def _neighbors(self, vec, k: int) -> List[Tuple[str, str, float]]:
        k = max(1, min(k, len(self._meta)))
        scores, idxs = self._index.search(vec[None], k)
        out: List[Tuple[str, str, float]] = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            m = self._meta[int(i)]
            out.append((str(m.get("id", i)), str(m.get("label", "?")), float(s)))
        return out

    def search_by_image(self, image_path: str, k: int) -> List[Tuple[str, float]]:
        """(id, score) list for RRF fusion in the ops layer."""
        vec = self.encoder.encode_image(image_path)
        return [(i, s) for (i, _lab, s) in self._neighbors(vec, k)]

    def search_records_by_image(self, image_path: str, k: int) -> List[Dict[str, Any]]:
        """Return generator-ready exemplar evidence without dropping its lineage."""
        if self.encoder is None or self._index is None:
            raise RuntimeError("exemplar atlas requires both a frozen index and image encoder")
        vec = self.encoder.encode_image(image_path)
        width = max(1, min(int(k), len(self._meta)))
        scores, idxs = self._index.search(vec[None], width)
        output: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], idxs[0]):
            if index < 0:
                continue
            meta = dict(self._meta[int(index)])
            if meta.get("split") != "train":
                raise ValueError("non-train exemplar reached generator-facing retrieval")
            case_id = str(meta.get("case_id") or meta.get("id") or index)
            label = str(meta.get("label") or "").strip()
            if not label:
                raise ValueError(f"exemplar {case_id!r} has no frozen diagnosis label")
            source_id = str(meta.get("source_id") or f"HAM10000:{case_id}")
            row = dict(meta)
            row.update({
                "authority": "image_exemplar",
                "authority_class": "image_exemplar",
                "citation_id": str(meta.get("citation_id") or source_id),
                "document_id": str(meta.get("document_id") or f"exemplar:{case_id}"),
                "eligible_claim_types": ["image_exemplar_support"],
                "provenance_uri": f"dataset://HAM10000/{case_id}",
                "retrieval_channel": "image_exemplar",
                "score": float(score),
                "source": source_id,
                "source_id": source_id,
                "text": f"Frozen training exemplar diagnosis label: {label}.",
            })
            output.append(row)
        return output

    def search_records_by_images(
        self, image_paths: Sequence[str], k: int, *, rank_constant: int = 60,
    ) -> List[Dict[str, Any]]:
        """Fuse per-view exemplar rankings for one lesion without dropping views."""
        paths = [str(path) for path in image_paths if str(path).strip()]
        if not paths:
            raise ValueError("multi-view exemplar retrieval requires at least one image")
        if len(set(paths)) != len(paths):
            raise ValueError("multi-view exemplar retrieval received a duplicate image path")
        if k <= 0 or rank_constant <= 0:
            raise ValueError("k and rank_constant must be positive")
        scores: Dict[str, float] = {}
        records: Dict[str, Dict[str, Any]] = {}
        ranks: Dict[str, List[Dict[str, int]]] = {}
        for view_index, image_path in enumerate(paths):
            for rank, record in enumerate(self.search_records_by_image(image_path, k), start=1):
                key = str(record["document_id"])
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
                records.setdefault(key, dict(record))
                ranks.setdefault(key, []).append({"view_index": view_index, "rank": rank})
        ordered = sorted(scores, key=lambda key: (-scores[key], key))[:k]
        output: List[Dict[str, Any]] = []
        for key in ordered:
            row = records[key]
            row["per_view_ranks"] = ranks[key]
            row["query_view_count"] = len(paths)
            row["score"] = float(scores[key])
            row["view_rrf_score"] = float(scores[key])
            output.append(row)
        return output

    def probe(self, image_path: str) -> ExemplarResult:
        """Diagnostic summary: committed label, confidence, malignant score, evidence."""
        vec = self.encoder.encode_image(image_path)
        neighbors = self._neighbors(vec, self.top_k)
        diag_probs = None
        mal_prob = None
        if self.diag_clf is not None:
            classes = list(self.diag_clf.classes_)
            probs = self.diag_clf.predict_proba(vec[None])[0]
            diag_probs = {str(c): float(p) for c, p in zip(classes, probs)}
        if self.malignant_clf is not None:
            classes = list(self.malignant_clf.classes_)
            probs = self.malignant_clf.predict_proba(vec[None])[0]
            pos = classes.index(1) if 1 in classes else (len(classes) - 1)
            mal_prob = float(probs[pos])
        return summarize_neighbors(
            neighbors, self.labels, config.MALIGNANT_LABELS,
            prior=self.prior, prior_alpha=self.prior_alpha,
            diag_probs=diag_probs, malignant_prob=mal_prob)


# ---------------------------------------------------------------------------
# Fresh build/fit code -- rebuilds MedGuard's OWN atlas + probes from public TRAIN
# images (HAM10000). No legacy baseline artifacts loaded. Train-split only => no leakage.
# ---------------------------------------------------------------------------

def _norm_path(p: str) -> str:
    """Translate WSL-style /mnt/<drive>/... -> <DRIVE>:\\... when running on Windows,
    so the public HAM image paths stored in the VQA jsonl resolve natively. No-op on
    POSIX and on already-native paths."""
    import os
    if os.name == "nt" and isinstance(p, str) and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return p[5].upper() + ":" + p[6:].replace("/", "\\")
    return p


def _read_train(train_jsonl: str, image_key: str, label_key: str) -> List[Dict[str, Any]]:
    import json
    recs: List[Dict[str, Any]] = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            img = d.get(image_key) or d.get("image") or d.get("image_path")
            lab = (d.get(label_key) or d.get("label") or d.get("class")
                   or d.get("lesion_type") or d.get("dx") or d.get("diagnosis"))
            if img and lab:
                recs.append({"id": str(d.get("id", len(recs))),
                             "image_path": _norm_path(img), "label": str(lab)})
    return recs


def _encode_all(recs, encoder):
    import numpy as np
    vecs = []
    kept = []
    for r in recs:
        try:
            vecs.append(encoder.encode_image(r["image_path"]))
            kept.append(r)
        except Exception as e:  # noqa: BLE001 -- skip unreadable images, report
            print(f"[atlas] skip {r.get('id')}: {e}")
    return np.vstack(vecs).astype("float32"), kept


def build_atlas(train_jsonl: str, index_path: Optional[str] = None,
                meta_path: Optional[str] = None, encoder: Optional[ImageEncoder] = None,
                image_key: str = "image_path", label_key: str = "label") -> int:
    """Encode TRAIN images -> FAISS IndexFlatIP atlas + parallel {id,label} meta."""
    import faiss
    import json
    index_path = index_path or config.EXEMPLAR_INDEX_PATH
    meta_path = meta_path or config.EXEMPLAR_META_PATH
    encoder = encoder or make_encoder()
    recs = _read_train(train_jsonl, image_key, label_key)
    if not recs:
        raise ValueError(f"no usable train records in {train_jsonl}")
    vecs, kept = _encode_all(recs, encoder)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "label": r["label"]}) + "\n")
    return int(index.ntotal)


def fit_probe(X, y, calibration: Optional[str] = None):
    """Fit a class-balanced LogisticRegression probe on frozen features, optionally wrapped
    in probability calibration (config.EXEMPLAR_CALIBRATION). class_weight='balanced' stays on
    the base estimator so minority (malignant) recall is preserved; calibration only reshapes
    the probabilities -> a sharper CONFIDENCE signal for the gate (better selective acc / AURC).
    method: 'isotonic' | 'sigmoid' | 'none'. cv=3 internal folds."""
    from sklearn.linear_model import LogisticRegression
    method = (calibration if calibration is not None else config.EXEMPLAR_CALIBRATION or "none").lower()
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    if method in ("isotonic", "sigmoid"):
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(base, method=method, cv=3).fit(X, y)  # positional: est name varies by sklearn ver
    return base.fit(X, y)


def fit_probes(train_jsonl: str, malignant_path: Optional[str] = None,
               diag_path: Optional[str] = None, encoder: Optional[ImageEncoder] = None,
               image_key: str = "image_path", label_key: str = "label",
               calibration: Optional[str] = None) -> Dict[str, int]:
    """Fit linear probes on frozen TRAIN features: 7-way diagnosis + binary malignant."""
    import joblib
    malignant_path = malignant_path or config.EXEMPLAR_MALIGNANT_CLF
    diag_path = diag_path or config.EXEMPLAR_DIAG_CLF
    encoder = encoder or make_encoder()
    recs = _read_train(train_jsonl, image_key, label_key)
    X, kept = _encode_all(recs, encoder)
    y_diag = [r["label"] for r in kept]
    y_mal = [1 if r["label"] in config.MALIGNANT_LABELS else 0 for r in kept]
    Path(diag_path).parent.mkdir(parents=True, exist_ok=True)
    method = (calibration if calibration is not None else config.EXEMPLAR_CALIBRATION or "none").lower()
    diag = fit_probe(X, y_diag, method)
    mal = fit_probe(X, y_mal, method)
    joblib.dump(diag, diag_path)
    joblib.dump(mal, malignant_path)
    return {"n": len(kept), "diag_classes": len(set(y_diag)),
            "malignant_pos": sum(y_mal), "calibration": method}
