"""Faithful, explicitly scoped implementation of DynaRAG's GMM-ATKS node.

This implements only paper 135 equations 3--6 (per-query univariate GMM,
BIC component selection, highest-mean posterior threshold). It is not the full
DynaRAG system: QAF, AAM, L_quality, L_balance, and LLaVA-Med training are absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class AdaptiveSelection:
    selected_indices: List[int]
    posteriors: List[float]
    n_components: int
    bic: float
    fallback_reason: str = ""


class GMMAdaptiveTopKSelector:
    """DynaRAG GMM-ATKS-style context selector with declared missing details.

    The paper specifies BIC but omits the searched K range and EM stability
    settings. We predeclare K in ``1..min(max_components, floor(N/2))`` so every
    component has at least two candidate observations in principle, use sklearn's
    regularized full-covariance EM, and preserve the paper's strict ``p > gamma``.
    """

    def __init__(
        self, *, gamma: float = 0.5, max_components: int = 4,
        min_selected: int = 1, max_selected: Optional[int] = None,
        covariance_floor: float = 1e-6, n_init: int = 5, random_state: int = 0,
    ):
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")
        if max_components < 1 or min_selected < 1:
            raise ValueError("component and selection minima must be positive")
        if max_selected is not None and max_selected < min_selected:
            raise ValueError("max_selected cannot be smaller than min_selected")
        self.gamma = float(gamma)
        self.max_components = int(max_components)
        self.min_selected = int(min_selected)
        self.max_selected = int(max_selected) if max_selected is not None else None
        self.covariance_floor = float(covariance_floor)
        self.n_init = int(n_init)
        self.random_state = int(random_state)

    def select(self, scores: Sequence[float]) -> AdaptiveSelection:
        values = [float(score) for score in scores]
        if not values:
            return AdaptiveSelection([], [], 0, math.inf, "empty_candidate_set")
        if any(not math.isfinite(score) for score in values):
            raise ValueError("retrieval scores must be finite")
        if len(values) == 1:
            return AdaptiveSelection([0], [1.0], 1, 0.0, "single_candidate")
        if max(values) - min(values) <= 1e-12:
            return AdaptiveSelection(
                list(range(len(values))), [1.0] * len(values), 1, 0.0,
                "constant_score_distribution",
            )

        try:
            import numpy as np
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:  # pragma: no cover - deployment dependency failure
            raise RuntimeError("GMM-ATKS requires numpy and scikit-learn") from exc

        array = np.asarray(values, dtype=float).reshape(-1, 1)
        max_k = min(self.max_components, max(1, len(values) // 2))
        fitted = []
        for components in range(1, max_k + 1):
            try:
                model = GaussianMixture(
                    n_components=components,
                    covariance_type="full",
                    reg_covar=self.covariance_floor,
                    n_init=self.n_init,
                    random_state=self.random_state,
                ).fit(array)
                fitted.append((float(model.bic(array)), model))
            except ValueError:
                continue
        if not fitted:
            ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
            selected = ranked[: self.min_selected]
            return AdaptiveSelection(
                sorted(selected), [0.0] * len(values), 0, math.inf,
                "all_gmm_fits_failed",
            )

        bic, best = min(fitted, key=lambda pair: (pair[0], pair[1].n_components))
        high_component = int(best.means_.reshape(-1).argmax())
        posteriors = [float(value) for value in best.predict_proba(array)[:, high_component]]
        selected = [
            index for index, posterior in enumerate(posteriors)
            if posterior > self.gamma
        ]
        ranked = sorted(
            range(len(values)),
            key=lambda index: (-posteriors[index], -values[index], index),
        )
        if len(selected) < self.min_selected:
            selected = ranked[: self.min_selected]
        if self.max_selected is not None and len(selected) > self.max_selected:
            selected = [index for index in ranked if index in set(selected)][:self.max_selected]
        return AdaptiveSelection(
            sorted(selected), posteriors, int(best.n_components), bic,
        )
