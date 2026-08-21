"""Case-based retrieval memory over DermLIP embeddings.

Stores past system evaluations (contracts) alongside their feature vectors.
At inference time, retrieves the k most similar past cases by cosine similarity.
The decision logic is unchanged — similar cases provide context, not override.

This module also builds the FAITHFUL-BY-CONSTRUCTION case-based explanation the clinician surface
renders on demand (:func:`build_similar_cases_context`). That function retrieves over the frozen,
hash-versioned, TRAIN-ONLY :class:`~medguard_v2.perception.knn_memory.FrozenKNNMemory` -- the same
retrieval primitive the P2 ensemble uses -- rather than the older ``CaseMemory`` above, which stores
past *contracts* (system assessments), not the leak-free train bank the explanation needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from medguard_v2.evidence.contract import SimilarCase, SimilarCasesContext
from medguard_v2.perception.knn_memory import FrozenKNNMemory


@dataclass
class CaseMemoryEntry:
    case_id: str
    embedding: np.ndarray
    summary: Dict[str, Any]


class CaseMemory:
    """Cosine-similarity retrieval over a fixed set of evaluated contracts."""

    def __init__(self, entries: List[CaseMemoryEntry]):
        if not entries:
            raise ValueError("CaseMemory requires at least one entry")
        self._entries = entries
        mat = np.stack([e.embedding for e in entries])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._normed = mat / np.maximum(norms, 1e-8)

    def __len__(self) -> int:
        return len(self._entries)

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 3,
        exclude: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        q = query.ravel() / max(float(np.linalg.norm(query)), 1e-8)
        sims = self._normed @ q
        if exclude:
            for i, e in enumerate(self._entries):
                if e.case_id in exclude:
                    sims[i] = -2.0
        top = np.argsort(sims)[::-1][:k]
        return [
            {"similarity": round(float(sims[i]), 4), **self._entries[i].summary}
            for i in top
            if sims[i] > -1.0
        ]


def build_memory_from_contracts(
    contracts: list,
    features: Dict[str, np.ndarray],
    case_num_map: Optional[Dict[str, int]] = None,
) -> CaseMemory:
    """Build a CaseMemory from DecisionContract objects and their DermLIP features.

    ``features`` maps case_num (str) to embedding vector.
    ``case_num_map`` optionally maps case_id -> case_num when they differ.
    """
    from medguard_v2.evidence.router import classify_state

    entries: List[CaseMemoryEntry] = []
    for c in contracts:
        if case_num_map:
            case_num = str(case_num_map.get(c.case_id, c.case_id))
        else:
            case_num = c.case_id.replace("derm7pt-", "")

        if case_num not in features:
            continue

        state, _ = classify_state(
            classifier_diagnosis=c.classifier_diagnosis,
            classifier_confidence=c.classifier_confidence,
            score=c.score,
        )

        entries.append(CaseMemoryEntry(
            case_id=c.case_id,
            embedding=features[case_num],
            summary={
                "case_id": c.case_id,
                "diagnosis": c.classifier_diagnosis,
                "confidence": round(c.classifier_confidence, 3),
                "criteria": dict(c.criteria),
                "seven_point_score": {
                    "min": c.score.score_min,
                    "max": c.score.score_max,
                    "decidable": c.score.decidable,
                },
                "conflict": c.conflict.present,
                "conflict_description": c.conflict.description if c.conflict.present else None,
                "management_action": c.criteria_implied_action,
                "router_state": state.value,
                "guideline_covered": c.guideline.covered,
                "safe_to_display": c.safe_to_display,
            },
        ))

    return CaseMemory(entries)


def build_similar_cases_context(
    *,
    query_embedding: np.ndarray,
    reference_diagnosis: str,
    bank: FrozenKNNMemory,
    query_case_id: str,
    label_name: str = "diagnosis",
    k: int = 5,
    feature_space: str = "",
    exclude_query: bool = True,
) -> SimilarCasesContext:
    """Retrieve the faithful-by-construction case-based explanation for one query.

    Retrieves the ``k`` nearest TRAIN cases to ``query_embedding`` in ``bank`` (exact cosine) plus
    the single nearest case whose confirmed label differs from ``reference_diagnosis`` -- the
    boundary counterfactual. ``same_label`` on each neighbour is measured against
    ``reference_diagnosis``: pass the COMMITTED classifier diagnosis in production (what the clinician
    is told), or the true label when measuring retrieval sanity.

    Leak-free by two mechanisms that compose: the bank is TRAIN-only, and ``exclude_query`` removes
    the query's own id from the candidate set (the leave-one-out guard that matters when the query is
    itself a train case; a no-op, safely, when the query is a valid/test case not in the bank). The
    full ranking is scanned for the boundary case so it is found even when it falls outside the top-k.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    q = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
    exclude = [str(query_case_id)] if exclude_query else None

    # Rank the WHOLE bank once: the top-k are the neighbours, and the nearest different-label case is
    # the first row down the same ranking whose label differs -- which top-k alone could miss.
    idx, sims = bank.retrieve(q, len(bank), exclude_ids=exclude)
    # knn_memory.py is out of edit-scope here; its row-aligned id list is read directly (the existing
    # test suite already treats ``_case_ids`` as accessible, e.g. test_knn_memory.ExcludeTest).
    row_ids: Sequence[str] = bank._case_ids
    labels = bank.label_values(label_name)
    ref = str(reference_diagnosis)

    ranked = [
        (str(row_ids[int(j)]), str(labels[int(j)]), float(s))
        for j, s in zip(idx[0], sims[0])
        if np.isfinite(s)  # excluded rows were set to -inf
    ]

    neighbors = tuple(
        SimilarCase(case_id=cid, diagnosis=dx, similarity=sim, same_label=(dx == ref))
        for cid, dx, sim in ranked[:k]
    )
    nearest_different = next(
        (
            SimilarCase(case_id=cid, diagnosis=dx, similarity=sim, same_label=False)
            for cid, dx, sim in ranked
            if dx != ref
        ),
        None,
    )

    return SimilarCasesContext(
        query_case_id=str(query_case_id),
        reference_diagnosis=ref,
        neighbors=neighbors,
        nearest_different_label=nearest_different,
        feature_space=feature_space,
        bank_name=bank.name,
        bank_hash=bank.bank_hash,
        k=k,
    )
