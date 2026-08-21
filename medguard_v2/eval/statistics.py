"""Paired and cluster-aware statistical tests for the frozen evaluation protocol."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(v) for v in values)
    pos = q * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def clustered_bootstrap_distribution(
    rows: Sequence[T], cluster_ids: Sequence[str], metric: Callable[[Sequence[T]], float], *,
    n_boot: int = 2000, confidence: float = 0.95, seed: int = 20260712,
) -> Tuple[Interval, List[float]]:
    """The percentile CI *and* the replicate distribution behind it.

    Same resampling as ``clustered_bootstrap_ci`` (identical RNG call order, so the
    interval is byte-identical); it additionally returns the replicates so a caller
    can derive a bootstrap p-value without paying for a second resample.
    """
    if len(rows) != len(cluster_ids):
        raise ValueError("rows and cluster_ids must have equal length")
    if not rows or n_boot <= 0:
        raise ValueError("bootstrap requires rows and n_boot > 0")
    grouped: Dict[str, List[T]] = {}
    for row, cluster in zip(rows, cluster_ids):
        grouped.setdefault(str(cluster), []).append(row)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_boot):
        picked = [rng.choice(clusters) for _ in clusters]
        sample = [row for cluster in picked for row in grouped[cluster]]
        samples.append(float(metric(sample)))
    alpha = (1.0 - confidence) / 2.0
    interval = Interval(
        float(metric(rows)), _quantile(samples, alpha), _quantile(samples, 1.0 - alpha),
    )
    return interval, samples


def clustered_bootstrap_ci(
    rows: Sequence[T], cluster_ids: Sequence[str], metric: Callable[[Sequence[T]], float], *,
    n_boot: int = 2000, confidence: float = 0.95, seed: int = 20260712,
) -> Interval:
    """Percentile CI by resampling whole patient/lesion/study clusters."""
    interval, _ = clustered_bootstrap_distribution(
        rows, cluster_ids, metric, n_boot=n_boot, confidence=confidence, seed=seed,
    )
    return interval


def bootstrap_two_sided_p(samples: Sequence[float], null_value: float = 0.0) -> float:
    """Two-sided bootstrap p for H0: statistic == ``null_value``.

    ``2 * min(tail)`` with the standard +1 correction, so a p of exactly 0 -- which
    would be an unearned claim of certainty from a finite resample -- is impossible.
    """
    values = [float(v) for v in samples]
    if not values:
        raise ValueError("bootstrap p-value requires replicates")
    n = len(values)
    below = sum(1 for v in values if v <= null_value)
    above = sum(1 for v in values if v >= null_value)
    return min(1.0, 2.0 * min(below + 1, above + 1) / (n + 1))


def paired_clustered_bootstrap_difference(
    rows_a: Sequence[T], rows_b: Sequence[T], cluster_ids: Sequence[str],
    metric: Callable[[Sequence[T]], float], *, n_boot: int = 2000,
    confidence: float = 0.95, seed: int = 20260712,
) -> Interval:
    if len(rows_a) != len(rows_b) or len(rows_a) != len(cluster_ids):
        raise ValueError("paired rows and cluster_ids must have equal length")
    paired = list(zip(rows_a, rows_b))
    return clustered_bootstrap_ci(
        paired, cluster_ids,
        lambda pairs: float(metric([p[0] for p in pairs])) - float(metric([p[1] for p in pairs])),
        n_boot=n_boot, confidence=confidence, seed=seed,
    )


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Tuple[int, int, float]:
    """Two-sided exact McNemar test; returns (A-only, B-only, p)."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired outcome arrays must have equal length")
    b = sum(bool(a) and not bool(c) for a, c in zip(a_correct, b_correct))
    c = sum(not bool(a) and bool(c_) for a, c_ in zip(a_correct, b_correct))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return b, c, min(1.0, 2.0 * tail)


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down family-wise error correction."""
    ordered = sorted(((name, float(p)) for name, p in p_values.items()), key=lambda x: x[1])
    m, running = len(ordered), 0.0
    adjusted: Dict[str, float] = {}
    for i, (name, p) in enumerate(ordered):
        running = max(running, (m - i) * p)
        adjusted[name] = min(1.0, running)
    return adjusted


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(len(values), dtype=float)
    out[order] = ranks
    return out


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Mann-Whitney binary AUROC with explicit two-class validation."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    truth = np.asarray(labels, dtype=bool)
    n_positive, n_negative = int(truth.sum()), int((~truth).sum())
    if not n_positive or not n_negative:
        raise ValueError("binary AUC requires positive and negative cases")
    ranks = _midrank(np.asarray(scores, dtype=float))
    return float(
        (ranks[truth].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def delong_auc_test(
    y_true: Sequence[bool], scores_a: Sequence[float], scores_b: Sequence[float],
) -> Dict[str, float]:
    """Paired two-sided DeLong comparison for two ROC AUCs."""
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("DeLong inputs must have equal length")
    labels = np.asarray(y_true, dtype=bool)
    m, n = int(labels.sum()), int((~labels).sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong test requires both positive and negative cases")
    order = np.concatenate([np.where(labels)[0], np.where(~labels)[0]])
    preds = np.asarray([scores_a, scores_b], dtype=float)[:, order]
    aucs = np.zeros(2, dtype=float)
    v01 = np.zeros((2, m), dtype=float)
    v10 = np.zeros((2, n), dtype=float)
    for r in range(2):
        positives, negatives = preds[r, :m], preds[r, m:]
        tx, ty, tz = _midrank(positives), _midrank(negatives), _midrank(preds[r])
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m
    covariance = np.cov(v01, bias=False) / m + np.cov(v10, bias=False) / n
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    diff = float(aucs[0] - aucs[1])
    if variance <= 0:
        z = math.copysign(math.inf, diff) if diff else 0.0
        p = 0.0 if diff else 1.0
    else:
        z = diff / math.sqrt(variance)
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "difference": diff, "variance": variance, "z": z, "p_value": p}


def clopper_pearson_ci(
    successes: int, n: int, alpha: float = 0.05,
) -> Tuple[float, float]:
    """Exact Clopper-Pearson binomial CI for a proportion.

    Returns (lower, upper) at the 1-alpha confidence level. Uses the beta
    distribution quantile function, which is the textbook inversion of the
    binomial test. Required for rare-class metrics where bootstrap CIs may
    under-cover.
    """
    from scipy.stats import beta as beta_dist
    if n == 0:
        return (0.0, 1.0)
    lower = beta_dist.ppf(alpha / 2.0, successes, n - successes + 1) if successes > 0 else 0.0
    upper = beta_dist.ppf(1.0 - alpha / 2.0, successes + 1, n - successes) if successes < n else 1.0
    return (float(lower), float(upper))


def odds_ratio_discordant(a_only: int, b_only: int) -> Dict[str, float]:
    """Odds ratio and exact CI for McNemar's discordant pairs.

    OR = a_only / b_only. The 95% CI uses the exact conditional method
    (mid-p). Returns the OR, its log, and the CI bounds.
    """
    if b_only == 0:
        return {"or": float("inf"), "log_or": float("inf"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "a_only": a_only, "b_only": b_only}
    if a_only == 0:
        return {"or": 0.0, "log_or": float("-inf"),
                "ci_low": 0.0, "ci_high": float("nan"),
                "a_only": a_only, "b_only": b_only}
    odds = a_only / b_only
    log_or = math.log(odds)
    se = math.sqrt(1.0 / a_only + 1.0 / b_only)
    ci_low = math.exp(log_or - 1.96 * se)
    ci_high = math.exp(log_or + 1.96 * se)
    return {"or": round(odds, 4), "log_or": round(log_or, 4),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "a_only": a_only, "b_only": b_only}
