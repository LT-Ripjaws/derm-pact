"""Matched-runner adapter for the DynaRAG GMM-ATKS component comparator.

This is deliberately a component proxy, not a DynaRAG reproduction.  The
selector sees raw continuous train-exemplar cosine similarities.  It never
sees reciprocal-rank-fusion values, ranks, truth labels, or generator output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from medguard.pipeline.adaptive_selector import (
    AdaptiveSelection,
    GMMAdaptiveTopKSelector,
)


RAW_COSINE_SEMANTICS = "raw_continuous_train_exemplar_cosine"


class AdaptiveSelector(Protocol):
    """Structural type used to inject the selector in tests and runners."""

    def select(self, scores: Sequence[float]) -> AdaptiveSelection:
        ...


@dataclass(frozen=True)
class ImageExemplarCandidate:
    exemplar_id: str
    cosine_similarity: float
    payload: Any = None

    def __post_init__(self) -> None:
        if not self.exemplar_id.strip():
            raise ValueError("exemplar_id must be non-empty")
        value = float(self.cosine_similarity)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("cosine_similarity must be finite and lie in [-1, 1]")
        object.__setattr__(self, "cosine_similarity", value)


@dataclass(frozen=True)
class OrderedViewCandidates:
    view_id: str
    candidates: Tuple[ImageExemplarCandidate, ...]
    score_semantics: str

    def __post_init__(self) -> None:
        if not self.view_id.strip():
            raise ValueError("view_id must be non-empty")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.score_semantics != RAW_COSINE_SEMANTICS:
            raise ValueError(
                "B4 accepts raw continuous train-exemplar cosine scores only; "
                "rank and RRF inputs are forbidden"
            )
        identifiers = [candidate.exemplar_id for candidate in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("an exemplar may appear at most once within a view")


@dataclass(frozen=True)
class DenseTextCandidate:
    evidence_id: str
    dense_score: float
    payload: Any = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        value = float(self.dense_score)
        if not math.isfinite(value):
            raise ValueError("dense_score must be finite")
        object.__setattr__(self, "dense_score", value)


@dataclass(frozen=True)
class AggregatedExemplar:
    exemplar_id: str
    aggregate_cosine: float
    ordered_view_cosines: Tuple[Optional[float], ...]
    payload: Any = None


@dataclass(frozen=True)
class SelectedExemplar:
    exemplar_id: str
    aggregate_cosine: float
    selector_posterior: float
    ordered_view_cosines: Tuple[Optional[float], ...]
    payload: Any = None


class EvidenceModality(str, Enum):
    IMAGE_EXEMPLAR = "image_exemplar"
    TEXT_DENSE = "text_dense"


@dataclass(frozen=True)
class MergedEvidence:
    modality: EvidenceModality
    evidence_id: str
    score: float
    modality_rank: int
    payload: Any = None


@dataclass(frozen=True)
class MergeConfig:
    """Isolated cross-modal merge policy.

    Round-robin does not pretend image and text cosine scales are calibrated.
    The default begins with image evidence because GMM-ATKS is the component
    under comparison, then alternates with dense text and fills from the
    remaining modality when one side is exhausted.
    """

    final_ceiling: int = 5
    first_modality: EvidenceModality = EvidenceModality.IMAGE_EXEMPLAR

    def __post_init__(self) -> None:
        if self.final_ceiling < 1:
            raise ValueError("final_ceiling must be positive")
        if not isinstance(self.first_modality, EvidenceModality):
            raise ValueError("first_modality must be an EvidenceModality enum")


@dataclass(frozen=True)
class B4SelectionAudit:
    ordered_view_ids: Tuple[str, ...]
    selector_input_semantics: str
    selector_input_ids: Tuple[str, ...]
    selector_input_raw_cosines: Tuple[float, ...]
    selector_input_per_view_cosines: Tuple[Tuple[Optional[float], ...], ...]
    selector_components: int
    selector_bic: float
    selector_fallback_reason: str
    fallback_applied: bool
    selected_image_count: int
    dense_candidate_count: int
    merged_count: int
    merge_rule: str


@dataclass(frozen=True)
class B4AdapterResult:
    selected_exemplars: Tuple[SelectedExemplar, ...]
    merged_evidence: Tuple[MergedEvidence, ...]
    audit: B4SelectionAudit


def aggregate_ordered_view_cosines(
    views: Sequence[OrderedViewCandidates],
) -> Tuple[Tuple[str, ...], Tuple[AggregatedExemplar, ...]]:
    """Aggregate by max raw cosine while retaining every ordered view cell."""

    ordered_views = tuple(views)
    if not ordered_views:
        raise ValueError("B4 requires at least one ordered lesion view")
    view_ids = tuple(view.view_id for view in ordered_views)
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("ordered lesion view identifiers must be unique")

    by_exemplar: dict[str, List[Optional[float]]] = {}
    payloads: dict[str, Any] = {}
    for view_index, view in enumerate(ordered_views):
        if view.score_semantics != RAW_COSINE_SEMANTICS:
            raise ValueError("selector input is not raw continuous cosine")
        for candidate in view.candidates:
            if candidate.exemplar_id not in by_exemplar:
                by_exemplar[candidate.exemplar_id] = [None] * len(ordered_views)
                payloads[candidate.exemplar_id] = candidate.payload
            by_exemplar[candidate.exemplar_id][view_index] = candidate.cosine_similarity

    aggregated: List[AggregatedExemplar] = []
    for exemplar_id in sorted(by_exemplar):
        per_view = tuple(by_exemplar[exemplar_id])
        observed = [score for score in per_view if score is not None]
        if not observed:  # defensive; construction above makes this unreachable
            continue
        aggregated.append(
            AggregatedExemplar(
                exemplar_id=exemplar_id,
                aggregate_cosine=max(observed),
                ordered_view_cosines=per_view,
                payload=payloads[exemplar_id],
            )
        )
    return view_ids, tuple(aggregated)


def merge_selected_evidence(
    selected_images: Sequence[SelectedExemplar],
    dense_text: Sequence[DenseTextCandidate],
    config: MergeConfig = MergeConfig(),
) -> Tuple[MergedEvidence, ...]:
    """Balanced deterministic merge under one cross-modal evidence ceiling."""

    images = [
        MergedEvidence(
            EvidenceModality.IMAGE_EXEMPLAR,
            item.exemplar_id,
            item.aggregate_cosine,
            rank,
            item.payload,
        )
        for rank, item in enumerate(selected_images, start=1)
    ]
    texts = [
        MergedEvidence(
            EvidenceModality.TEXT_DENSE,
            item.evidence_id,
            item.dense_score,
            rank,
            item.payload,
        )
        for rank, item in enumerate(dense_text, start=1)
    ]
    queues = {
        EvidenceModality.IMAGE_EXEMPLAR: images,
        EvidenceModality.TEXT_DENSE: texts,
    }
    order = (
        (EvidenceModality.IMAGE_EXEMPLAR, EvidenceModality.TEXT_DENSE)
        if config.first_modality is EvidenceModality.IMAGE_EXEMPLAR
        else (EvidenceModality.TEXT_DENSE, EvidenceModality.IMAGE_EXEMPLAR)
    )
    offsets = {modality: 0 for modality in order}
    merged: List[MergedEvidence] = []
    while len(merged) < config.final_ceiling:
        appended = False
        for modality in order:
            offset = offsets[modality]
            if offset < len(queues[modality]) and len(merged) < config.final_ceiling:
                merged.append(queues[modality][offset])
                offsets[modality] += 1
                appended = True
        if not appended:
            break
    return tuple(merged)


class DynaRAGGMMATKSComponentAdapter:
    """Select image exemplars with GMM-ATKS and merge with dense text."""

    def __init__(
        self,
        selector: Optional[AdaptiveSelector] = None,
        merge_config: MergeConfig = MergeConfig(),
    ) -> None:
        self.selector = selector or GMMAdaptiveTopKSelector(
            gamma=0.5,
            max_components=4,
            min_selected=1,
            max_selected=5,
            covariance_floor=1e-6,
            n_init=5,
            random_state=0,
        )
        self.merge_config = merge_config

    @staticmethod
    def _fallback_indices(
        aggregated: Sequence[AggregatedExemplar], reason: str
    ) -> Optional[List[int]]:
        ranked = sorted(
            range(len(aggregated)),
            key=lambda index: (
                -aggregated[index].aggregate_cosine,
                aggregated[index].exemplar_id,
            ),
        )
        if reason == "constant_score_distribution":
            return ranked[:5]
        if reason == "all_gmm_fits_failed":
            return ranked[:1]
        return None

    def run(
        self,
        ordered_views: Sequence[OrderedViewCandidates],
        dense_text: Sequence[DenseTextCandidate],
    ) -> B4AdapterResult:
        view_ids, aggregated = aggregate_ordered_view_cosines(ordered_views)
        dense_ids = [item.evidence_id for item in dense_text]
        if len(set(dense_ids)) != len(dense_ids):
            raise ValueError("dense evidence identifiers must be unique")
        raw_cosines = [item.aggregate_cosine for item in aggregated]
        selection = self.selector.select(raw_cosines)
        fallback_indices = self._fallback_indices(
            aggregated, selection.fallback_reason
        )
        fallback_applied = fallback_indices is not None
        selected_indices = (
            fallback_indices
            if fallback_indices is not None
            else list(selection.selected_indices)
        )
        if any(index < 0 or index >= len(aggregated) for index in selected_indices):
            raise ValueError("selector returned an out-of-range candidate index")
        if len(selection.posteriors) not in (0, len(aggregated)):
            raise ValueError("selector posterior count does not match its raw inputs")

        def posterior(index: int) -> float:
            return (
                float(selection.posteriors[index])
                if len(selection.posteriors) == len(aggregated)
                else 0.0
            )

        unique_indices = sorted(set(selected_indices))
        unique_indices.sort(
            key=lambda index: (
                -posterior(index),
                -aggregated[index].aggregate_cosine,
                aggregated[index].exemplar_id,
            )
        )
        selected = tuple(
            SelectedExemplar(
                exemplar_id=aggregated[index].exemplar_id,
                aggregate_cosine=aggregated[index].aggregate_cosine,
                selector_posterior=posterior(index),
                ordered_view_cosines=aggregated[index].ordered_view_cosines,
                payload=aggregated[index].payload,
            )
            for index in unique_indices[:5]
        )
        dense_ranked = tuple(
            sorted(dense_text, key=lambda item: (-item.dense_score, item.evidence_id))
        )
        merged = merge_selected_evidence(selected, dense_ranked, self.merge_config)
        audit = B4SelectionAudit(
            ordered_view_ids=view_ids,
            selector_input_semantics=RAW_COSINE_SEMANTICS,
            selector_input_ids=tuple(item.exemplar_id for item in aggregated),
            selector_input_raw_cosines=tuple(raw_cosines),
            selector_input_per_view_cosines=tuple(
                item.ordered_view_cosines for item in aggregated
            ),
            selector_components=int(selection.n_components),
            selector_bic=float(selection.bic),
            selector_fallback_reason=selection.fallback_reason,
            fallback_applied=fallback_applied,
            selected_image_count=len(selected),
            dense_candidate_count=len(dense_ranked),
            merged_count=len(merged),
            merge_rule=(
                "balanced_round_robin_"
                f"{self.merge_config.first_modality.value}_first_"
                f"under_shared_top_{self.merge_config.final_ceiling}"
            ),
        )
        return B4AdapterResult(selected, merged, audit)
