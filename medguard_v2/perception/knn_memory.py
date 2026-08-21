"""Frozen, hash-versioned kNN memory bank over frozen DermLIP features.

Retrieval-augmented perception. A *training-only* memory of feature vectors, each
carrying one or more per-case labels. For a query, retrieve the ``k`` most similar
bank entries by **exact cosine similarity** and return a similarity-weighted label
distribution. No model is trained -- the backbone is frozen and the bank is fixed --
so the vote is a training-free readout of the same representation the linear probe
reads.

Grounded in RAC (Long et al., *Retrieval Augmented Classification*, CVPR 2022) and
kNN-augmented classification (Jia et al., 2021): frozen-backbone retrieval that helps
tail classes and low-data regimes specifically -- which is the SK/MISC + 413-case
Derm7pt regime here.

Two audit properties, mirroring the guideline corpus:

* **Content-hashed.** ``bank_hash`` is a SHA-256 over the feature bytes, the case ids,
  and every stored label array. A different bank -- one more case, one relabelled
  criterion, a different feature view -- is a different hash. The number is
  deterministic and quotable.
* **Train-only.** The bank is built from TRAIN cases and nothing else. TEST is never
  inserted; the confirmatory read queries the frozen bank, it does not grow it. The
  ``exclude_ids`` argument additionally supports leave-one-out so a case never
  retrieves itself when the bank and the query set overlap (e.g. train-on-train OOF).

Generalizes the cosine primitive in ``medguard_v2/agent/case_memory.py`` from
"retrieve past decision contracts" to "retrieve any labelled feature vectors and vote a
distribution over any one of their label fields", which is what the perception
ensemble's kNN diagnosis view and kNN per-criterion view both need (the criterion bank
is queried seven times, once per criterion, over one shared feature memory).

Exact cosine only -- no approximate-NN index, no randomness. The retrieval is
reproducible bit-for-bit given the same bank hash and query.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

_EPS = 1e-8


class FrozenKNNMemory:
    """Cosine-retrieval memory over a fixed feature bank with per-entry labels.

    Parameters
    ----------
    features:
        ``(N, D)`` array of bank feature vectors (e.g. the promoted 526-d
        dermlip+metadata space). Cast to float64 and L2-normalised on construction;
        the raw (un-normalised) bytes are what the hash commits to.
    case_ids:
        Length-``N`` identifiers aligned row-for-row with ``features``. Used for
        leave-one-out exclusion and recorded in the hash.
    labels:
        Mapping ``label_name -> length-N sequence`` of per-entry labels. The diagnosis
        bank stores ``{"diagnosis": [...]}``; the criterion bank stores one entry per
        criterion. Every array is committed to the hash, so the bank is auditable in
        full, not just by its features.
    name:
        Human-readable bank name, recorded in ``summary()`` and the hash.
    """

    def __init__(
        self,
        features: np.ndarray,
        case_ids: Sequence[str],
        labels: Mapping[str, Sequence],
        *,
        name: str,
    ) -> None:
        feats = np.ascontiguousarray(features, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("features must be a 2-D (N, D) array")
        n = feats.shape[0]
        if n == 0:
            raise ValueError("FrozenKNNMemory requires at least one entry")
        ids = [str(c) for c in case_ids]
        if len(ids) != n:
            raise ValueError("case_ids must align row-for-row with features")
        self._name = str(name)
        self._features = feats
        self._case_ids = ids
        self._id_to_row = {cid: i for i, cid in enumerate(ids)}
        if len(self._id_to_row) != n:
            raise ValueError("case_ids must be unique")

        self._labels: Dict[str, np.ndarray] = {}
        for key, values in labels.items():
            arr = np.asarray([str(v) for v in values], dtype=object)
            if arr.shape[0] != n:
                raise ValueError(f"label {key!r} must have length {n}")
            self._labels[key] = arr

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        self._normed = feats / np.maximum(norms, _EPS)
        self._bank_hash = self._compute_hash()

    # -- identity / audit -------------------------------------------------------

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self._name.encode("utf-8"))
        h.update(b"\x00shape\x00")
        h.update(str(self._features.shape).encode("utf-8"))
        h.update(b"\x00feat\x00")
        h.update(self._features.tobytes())
        h.update(b"\x00ids\x00")
        h.update("\n".join(self._case_ids).encode("utf-8"))
        for key in sorted(self._labels):
            h.update(b"\x00label\x00")
            h.update(key.encode("utf-8"))
            h.update(b"\x00")
            h.update("\n".join(self._labels[key].tolist()).encode("utf-8"))
        return h.hexdigest()

    @property
    def bank_hash(self) -> str:
        return self._bank_hash

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._features.shape[0]

    def label_values(self, label_name: str) -> np.ndarray:
        return self._labels[label_name]

    def observed_labels(self, label_name: str) -> List[str]:
        """Sorted unique values of a stored label field (a natural ``label_order``)."""
        return sorted(set(self._labels[label_name].tolist()))

    def summary(self) -> Dict[str, object]:
        return {
            "name": self._name,
            "n_entries": len(self),
            "feature_dim": int(self._features.shape[1]),
            "bank_hash": self._bank_hash,
            "hash_algo": "sha256",
            "label_fields": sorted(self._labels),
            "retrieval": "exact cosine (unit-normalised dot product); no ANN, no randomness",
        }

    # -- retrieval --------------------------------------------------------------

    def _similarities(self, queries: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(queries, dtype=np.float64)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._features.shape[1]:
            raise ValueError(
                f"query dim {q.shape[1]} != bank dim {self._features.shape[1]}"
            )
        qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), _EPS)
        return qn @ self._normed.T  # (M, N) cosine

    def retrieve(
        self,
        queries: np.ndarray,
        k: int,
        *,
        exclude_ids: Optional[Sequence[Optional[str]]] = None,
    ):
        """Top-``k`` neighbour rows and their cosine similarities.

        Returns ``(idx, sims)`` each of shape ``(M, k)``. ``exclude_ids[i]`` (if given)
        removes the bank row with that id from query ``i``'s candidate set -- the
        leave-one-out guard for train-on-train queries.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        sims = self._similarities(queries)
        m = sims.shape[0]
        if exclude_ids is not None:
            if len(exclude_ids) != m:
                raise ValueError("exclude_ids must align with the query rows")
            for i, cid in enumerate(exclude_ids):
                if cid is not None:
                    row = self._id_to_row.get(str(cid))
                    if row is not None:
                        sims[i, row] = -np.inf
        eff_k = min(k, self._features.shape[0])
        # argpartition for the top-eff_k, then order those by descending similarity.
        part = np.argpartition(-sims, eff_k - 1, axis=1)[:, :eff_k]
        row_idx = np.arange(m)[:, None]
        order = np.argsort(-sims[row_idx, part], axis=1)
        idx = part[row_idx, order]
        return idx, sims[row_idx, idx]

    def vote(
        self,
        queries: np.ndarray,
        k: int,
        label_name: str,
        label_order: Sequence[str],
        *,
        exclude_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> np.ndarray:
        """Similarity-weighted label distribution over ``label_order``.

        For each query, the ``k`` nearest bank entries vote for their stored
        ``label_name`` value with weight ``max(cosine, 0)``; the per-label weights are
        normalised to sum to 1. Non-negative weighting keeps the vote a proper
        distribution without a temperature hyperparameter -- ``k`` is the only knob, so
        the selection surface stays one-dimensional (a Tip-Adapter-style
        ``exp(alpha*(cos-1))`` affinity was rejected for adding an unregistered
        temperature). If all ``k`` neighbours have non-positive similarity (degenerate;
        effectively never for a top-k set), the vote falls back to a uniform count over
        those neighbours' labels.

        Returns ``(M, C)`` with ``C = len(label_order)``; rows sum to 1.
        """
        order_list = [str(v) for v in label_order]
        col = {v: j for j, v in enumerate(order_list)}
        if len(col) != len(order_list):
            raise ValueError("label_order must be unique")
        labels = self._labels[label_name]
        idx, sims = self.retrieve(queries, k, exclude_ids=exclude_ids)
        m, kk = idx.shape
        out = np.zeros((m, len(order_list)), dtype=np.float64)
        weights = np.clip(sims, 0.0, None)
        for i in range(m):
            w = weights[i]
            neigh = idx[i]
            if w.sum() <= _EPS:
                w = np.ones(kk, dtype=np.float64)  # uniform fallback
            for j in range(kk):
                lab = labels[neigh[j]]
                c = col.get(lab)
                if c is not None:
                    out[i, c] += w[j]
            total = out[i].sum()
            if total > _EPS:
                out[i] /= total
            else:
                # Every neighbour label fell outside label_order (should not happen when
                # label_order covers the observed values); leave a uniform prior.
                out[i] = 1.0 / len(order_list)
        return out


def build_memory(
    features_by_id: Mapping[str, np.ndarray],
    cases,
    labels: Mapping[str, Sequence],
    *,
    name: str,
) -> FrozenKNNMemory:
    """Assemble a :class:`FrozenKNNMemory` from a case list and a feature lookup.

    ``features_by_id`` maps ``str(case_num) -> vector``; ``cases`` fixes the row order
    (and therefore the hash). ``labels`` are aligned to ``cases``.
    """
    ids = [str(c.case_num) for c in cases]
    matrix = np.stack([features_by_id[cid] for cid in ids])
    return FrozenKNNMemory(matrix, ids, labels, name=name)
