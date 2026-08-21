"""Give INDETERMINATE a producer, so criterion-level abstention actually fires.

`medguard_v2/criteria/aggregate.py` has handled INDETERMINATE from the start -- an unassessed
criterion yields a score *range*, and a case stays decidable when the unknowns cannot change the
threshold verdict. The machinery is implemented and tested. Nothing has ever emitted it: the
criterion heads return an argmax on every case, so the observed undecidable rate is 0.0000.

G1 killed the obvious producer. A VLM's self-reported confidence separated correct from incorrect
answers by **0.0281**, which is no signal at all, and it declined on 0.98% of judgements. So
self-report is out.

What remains is the head's own posterior. This module abstains on **margin** -- the gap between
the top and second-choice class -- rather than on the top probability, because a head that is
torn between `typical` and `atypical` at 0.45/0.44 is uninformative in a way that a head at 0.45
across four spread-out classes is not.

The threshold is chosen on **Derm7pt train by out-of-fold selective prediction**: the smallest
margin whose retained predictions reach a target accuracy. It is not tuned on valid or test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE

REPO = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO / "research" / "evidence" / "derm7pt_v1" / "criterion_abstention_v1.json"

SEED = 20260811
N_FOLDS = 5
MARGIN_GRID = tuple(round(x, 3) for x in np.arange(0.0, 0.55, 0.05))


@dataclass(frozen=True)
class AbstentionPolicy:
    """Per-criterion margin thresholds. A margin of 0.0 means the criterion never abstains."""

    margins: Mapping[str, float]
    target_retained_accuracy: float
    fitted_on: str

    def apply(self, head, x: np.ndarray, criterion: str) -> Tuple[List[str], np.ndarray]:
        proba = head.predict_proba(x)
        classes = head.classes_
        order = np.argsort(-proba, axis=1)
        top = proba[np.arange(len(proba)), order[:, 0]]
        second = (
            proba[np.arange(len(proba)), order[:, 1]] if proba.shape[1] > 1 else np.zeros(len(proba))
        )
        margin = top - second
        cutoff = float(self.margins.get(criterion, 0.0))
        values = [
            INDETERMINATE if m < cutoff else str(classes[order[i, 0]])
            for i, m in enumerate(margin)
        ]
        return values, margin

    def to_dict(self) -> Dict:
        return {
            "protocol": "derm7pt-criterion-margin-abstention-v1",
            "rule": "abstain when (top posterior - second posterior) < margin",
            "fitted_on": self.fitted_on,
            "target_retained_accuracy": self.target_retained_accuracy,
            "margins": dict(self.margins),
            "seed": SEED,
        }


def _out_of_fold_margins(x: np.ndarray, y: Sequence[str], c: float = 10.0):
    """Out-of-fold posteriors on train, so the threshold is not chosen on fitted predictions."""
    y_arr = np.asarray(y)
    from collections import Counter

    n_splits = min(N_FOLDS, min(Counter(y_arr).values()))
    margins = np.zeros(len(y_arr))
    correct = np.zeros(len(y_arr), dtype=bool)
    if n_splits < 2:
        return margins, correct

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, va in folds.split(x, y_arr):
        model = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
        model.fit(x[tr], y_arr[tr])
        proba = model.predict_proba(x[va])
        order = np.argsort(-proba, axis=1)
        top = proba[np.arange(len(proba)), order[:, 0]]
        second = (
            proba[np.arange(len(proba)), order[:, 1]]
            if proba.shape[1] > 1
            else np.zeros(len(proba))
        )
        margins[va] = top - second
        correct[va] = np.asarray([str(model.classes_[i]) for i in order[:, 0]]) == y_arr[va]
    return margins, correct


def fit_policy(
    x_train: np.ndarray,
    y_train: Mapping[str, Sequence[str]],
    *,
    target_retained_accuracy: float = 0.80,
    min_coverage: float = 0.60,
) -> AbstentionPolicy:
    """Smallest margin reaching the target accuracy on retained predictions.

    ``min_coverage`` stops a criterion from buying accuracy by abstaining on almost everything --
    the degenerate solution that made v1's A1 ablation meaningless.
    """
    margins: Dict[str, float] = {}
    for criterion in CRITERION_NAMES:
        margin, correct = _out_of_fold_margins(x_train, list(y_train[criterion]))
        chosen = 0.0
        for candidate in MARGIN_GRID:
            retained = margin >= candidate
            coverage = float(retained.mean())
            if coverage < min_coverage:
                break
            accuracy = float(correct[retained].mean()) if retained.any() else 0.0
            if accuracy >= target_retained_accuracy:
                chosen = float(candidate)
                break
            chosen = float(candidate)   # keep the largest margin still within coverage
        margins[criterion] = chosen
    return AbstentionPolicy(
        margins=margins,
        target_retained_accuracy=target_retained_accuracy,
        fitted_on="derm7pt train, out-of-fold",
    )


def save_policy(policy: AbstentionPolicy, path: Path = THRESHOLDS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_policy(path: Path = THRESHOLDS_PATH) -> AbstentionPolicy:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AbstentionPolicy(
        margins=data["margins"],
        target_retained_accuracy=data["target_retained_accuracy"],
        fitted_on=data["fitted_on"],
    )
