"""Frozen metric definitions for the selected-configuration final evaluation."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


def _ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    total = len(y)
    value = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            value += float(mask.sum() / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _macro_ovr_auc(y: np.ndarray, probabilities: np.ndarray) -> float | None:
    values = []
    for index in range(probabilities.shape[1]):
        binary = (y == index).astype(int)
        if np.unique(binary).size == 2:
            values.append(float(roc_auc_score(binary, probabilities[:, index])))
    return float(np.mean(values)) if values else None


def sensitivity_at_specificity(y_binary: np.ndarray, scores: np.ndarray, specificity: float) -> dict[str, float | None]:
    negatives = scores[y_binary == 0]
    positives = scores[y_binary == 1]
    if not len(negatives) or not len(positives):
        return {"target_specificity": specificity, "threshold": None, "sensitivity": None, "observed_specificity": None}
    candidates = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    eligible = []
    for threshold in candidates:
        observed = float((negatives < threshold).mean())
        if observed >= specificity:
            eligible.append((float((positives >= threshold).mean()), observed, float(threshold)))
    sensitivity, observed, threshold = max(eligible, key=lambda row: (row[0], -row[2]))
    return {"target_specificity": specificity, "threshold": threshold, "sensitivity": sensitivity,
            "observed_specificity": observed}


def multiclass_metrics(y: Sequence[int], probabilities: np.ndarray, *, melanoma_index: int) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(y) != len(p) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("invalid multiclass inputs")
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    pred = p.argmax(axis=1)
    labels = sorted(set(y.tolist()))
    one_hot = np.eye(p.shape[1], dtype=np.float64)[y]
    mel = (y == melanoma_index).astype(int)
    result = {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "macro_ovr_auroc": _macro_ovr_auc(y, p),
        "melanoma_auroc": float(roc_auc_score(mel, p[:, melanoma_index])) if np.unique(mel).size == 2 else None,
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "multiclass_brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "ece_10bin": _ece(y, p),
        "class_support": {str(label): int((y == label).sum()) for label in range(p.shape[1])},
        "confusion_matrix": [[int(((y == truth) & (pred == guess)).sum()) for guess in range(p.shape[1])]
                             for truth in range(p.shape[1])],
        "melanoma_sensitivity_at_90_specificity": sensitivity_at_specificity(mel, p[:, melanoma_index], 0.90),
        "melanoma_sensitivity_at_95_specificity": sensitivity_at_specificity(mel, p[:, melanoma_index], 0.95),
    }
    return result


def binary_metrics(y: Sequence[int], scores: Sequence[float], *, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError("invalid binary inputs")
    pred = (s >= threshold).astype(int)
    return {
        "n": len(y), "positive_support": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, s)) if np.unique(y).size == 2 else None,
        "nll": float(-(y * np.log(np.clip(s, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - s, 1e-12, 1))).mean()),
        "brier": float(np.square(s - y).mean()),
        "sensitivity": float(pred[y == 1].mean()) if np.any(y == 1) else None,
        "specificity": float((1 - pred[y == 0]).mean()) if np.any(y == 0) else None,
    }


def bootstrap_metrics(y: Sequence[int], probabilities: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
                      *, keys: Sequence[str], seed: int, n_boot: int = 2000) -> dict[str, dict[str, float | None]]:
    y = np.asarray(y)
    p = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in keys}
    for _ in range(n_boot):
        index = rng.integers(0, len(y), size=len(y))
        row = metric(y[index], p[index])
        for key in keys:
            value = row.get(key)
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    return {
        key: ({"ci_low": float(np.quantile(rows, 0.025)), "ci_high": float(np.quantile(rows, 0.975)), "n_boot": n_boot}
              if rows else {"ci_low": None, "ci_high": None, "n_boot": n_boot})
        for key, rows in values.items()
    }


def criterion_metrics(truth_states: Sequence[str], state_probabilities: np.ndarray, state_labels: Sequence[str],
                      positive_values: set[str], binary_probability: Sequence[float]) -> dict[str, Any]:
    labels = list(state_labels)
    index = {label: i for i, label in enumerate(labels)}
    truth = np.asarray([index[value] for value in truth_states], dtype=np.int64)
    state_p = np.asarray(state_probabilities, dtype=np.float64)
    predicted = state_p.argmax(axis=1)
    truth_positive = np.asarray([value in positive_values for value in truth_states], dtype=np.int64)
    binary = binary_metrics(truth_positive, binary_probability)
    return {
        "n": len(truth),
        "raw_state_accuracy": float(accuracy_score(truth, predicted)),
        "raw_state_macro_f1": float(f1_score(truth, predicted, labels=range(len(labels)), average="macro", zero_division=0)),
        "presence_accuracy": binary["accuracy"],
        "presence_f1": binary["f1"],
        "melanoma_indicative_auroc": binary["auroc"],
        "positive_support": binary["positive_support"],
        "indeterminate_rate": 0.0,
        "state_labels": labels,
        "state_support": {label: int(sum(value == label for value in truth_states)) for label in labels},
    }
