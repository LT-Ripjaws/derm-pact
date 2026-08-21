"""Split-conformal prediction sets for the diagnostic classifier.

Extends the existing selective-abstention gate with proper prediction sets
that carry marginal coverage guarantees.  Given a calibration set, computes
nonconformity scores and a quantile threshold so that prediction sets on
new cases contain the true label with probability >= 1 - alpha.

Two prediction-set methods:
  1. **LAC** (Least Ambiguous set-valued Classifier): includes all classes
     whose softmax exceeds a calibrated threshold.  Tight sets.
  2. **APS** (Adaptive Prediction Sets): includes classes in descending
     softmax order until cumulative probability exceeds a threshold.
     Guaranteed marginal coverage but potentially larger sets.

Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification" (2107.07511).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PredictionSet:
    """A conformal prediction set for a single case."""
    classes: Tuple[str, ...]
    probabilities: Dict[str, float]
    set_size: int
    contains_true: Optional[bool] = None
    point_prediction: str = ""
    point_confidence: float = 0.0


@dataclass(frozen=True)
class ConformalCalibration:
    """Fitted conformal calibration from a held-out calibration set."""
    method: str
    alpha: float
    threshold: float
    n_calibration: int
    empirical_coverage: float
    mean_set_size: float
    classes: Tuple[str, ...]


class ConformalClassifier:
    """Wraps a sklearn classifier with conformal prediction sets."""

    def __init__(self, classifier, classes: Sequence[str], seed: int = 20260810):
        self._clf = classifier
        self._classes = tuple(classes)
        self._calibration: Optional[ConformalCalibration] = None
        self._rng = np.random.default_rng(seed)

    @property
    def calibrated(self) -> bool:
        return self._calibration is not None

    def calibrate(
        self,
        x_cal: np.ndarray,
        y_cal: Sequence[str],
        alpha: float = 0.10,
        method: str = "lac",
    ) -> ConformalCalibration:
        """Fit the conformal threshold on a calibration set.

        ``alpha`` is the miscoverage rate: prediction sets will contain the
        true label with probability >= 1 - alpha.
        """
        proba = self._clf.predict_proba(x_cal)
        y_arr = np.array(y_cal)
        n = len(y_arr)

        if method == "lac":
            scores = self._lac_scores(proba, y_arr)
        elif method == "aps":
            scores = self._aps_scores(proba, y_arr)
        else:
            raise ValueError(f"Unknown method {method!r}; use 'lac' or 'aps'")

        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        threshold = float(np.quantile(scores, q_level, method="higher"))

        sets = self._make_sets(proba, threshold, method, y_arr)
        coverage = sum(1 for s in sets if s.contains_true) / len(sets)
        mean_size = sum(s.set_size for s in sets) / len(sets)

        self._calibration = ConformalCalibration(
            method=method,
            alpha=alpha,
            threshold=threshold,
            n_calibration=n,
            empirical_coverage=round(coverage, 4),
            mean_set_size=round(mean_size, 3),
            classes=self._classes,
        )
        self._method = method
        return self._calibration

    def predict_sets(
        self,
        x: np.ndarray,
        y_true: Optional[Sequence[str]] = None,
    ) -> List[PredictionSet]:
        """Produce conformal prediction sets for new cases."""
        if not self.calibrated:
            raise RuntimeError("Call calibrate() first")
        proba = self._clf.predict_proba(x)
        return self._make_sets(
            proba, self._calibration.threshold, self._calibration.method,
            np.array(y_true) if y_true is not None else None,
        )

    def _lac_scores(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """LAC nonconformity: 1 - softmax(true class)."""
        class_to_idx = {c: i for i, c in enumerate(self._classes)}
        true_idx = np.array([class_to_idx[label] for label in y])
        true_proba = proba[np.arange(len(y)), true_idx]
        return 1.0 - true_proba

    def _aps_scores(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """APS nonconformity: cumulative softmax up to and including true class."""
        class_to_idx = {c: i for i, c in enumerate(self._classes)}
        scores = np.zeros(len(y))
        for i in range(len(y)):
            order = np.argsort(-proba[i])
            cumsum = 0.0
            for j in order:
                cumsum += proba[i, j]
                if j == class_to_idx[y[i]]:
                    u = self._rng.uniform()
                    scores[i] = cumsum - u * proba[i, j]
                    break
        return scores

    def _make_sets(
        self,
        proba: np.ndarray,
        threshold: float,
        method: str,
        y_true: Optional[np.ndarray] = None,
    ) -> List[PredictionSet]:
        sets = []
        for i in range(len(proba)):
            point_idx = np.argmax(proba[i])
            point_class = self._classes[point_idx]
            point_conf = float(proba[i, point_idx])

            if method == "lac":
                included = [
                    self._classes[j]
                    for j in range(len(self._classes))
                    if 1.0 - proba[i, j] <= threshold
                ]
            else:
                order = np.argsort(-proba[i])
                included = []
                cumsum = 0.0
                for j in order:
                    included.append(self._classes[j])
                    cumsum += proba[i, j]
                    if cumsum >= threshold:
                        break

            if not included:
                included = [point_class]

            contains = None
            if y_true is not None:
                contains = y_true[i] in included

            sets.append(PredictionSet(
                classes=tuple(included),
                probabilities={
                    self._classes[j]: round(float(proba[i, j]), 4)
                    for j in range(len(self._classes))
                },
                set_size=len(included),
                contains_true=contains,
                point_prediction=point_class,
                point_confidence=round(point_conf, 4),
            ))
        return sets

    def to_dict(self) -> Dict:
        if not self.calibrated:
            return {"calibrated": False}
        cal = self._calibration
        return {
            "calibrated": True,
            "method": cal.method,
            "alpha": cal.alpha,
            "threshold": cal.threshold,
            "n_calibration": cal.n_calibration,
            "empirical_coverage": cal.empirical_coverage,
            "mean_set_size": cal.mean_set_size,
            "classes": list(cal.classes),
        }
