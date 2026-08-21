"""Dependency-light metrics for the matched architecture study.

Diagnostic confusion matrices, selective prediction, evidence grounding, and
clinical triage are kept separate so no composite score can hide a regression.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import MALIGNANT_LABELS


DERM_CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc")
_CITATION = re.compile(r"\[(\d+)\]")


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", (value or "").lower()))


def grounding_score(answer: str, chunks: Sequence[dict]) -> float:
    """Legacy lexical grounding proxy; RAGAS is reported separately."""
    if not answer:
        return 0.0
    cited = bool(_CITATION.search(answer))
    evidence = _tokens(" ".join(str(chunk.get("text", "")) for chunk in chunks or ()))
    answer_tokens = _tokens(answer)
    overlap = len(answer_tokens & evidence) / len(answer_tokens) if answer_tokens else 0.0
    return round((0.5 if cited else 0.0) + 0.5 * min(1.0, overlap / 0.3), 4)


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = DERM_CLASS_ORDER,
) -> List[List[int]]:
    """Raw count matrix with rows=true and columns=predicted.

    Abstentions must be reported separately and filtered by the caller; unknown
    labels are rejected instead of silently disappearing from the denominator.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    ordered = tuple(str(label).lower() for label in labels)
    if len(ordered) != len(set(ordered)) or not ordered:
        raise ValueError("labels must be a non-empty unique sequence")
    index = {label: i for i, label in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]
    for truth, pred in zip(y_true, y_pred):
        t, p = str(truth).lower(), str(pred).lower()
        if t not in index or p not in index:
            raise ValueError(f"unknown diagnostic label truth={truth!r}, pred={pred!r}")
        matrix[index[t]][index[p]] += 1
    return matrix


def row_normalize(matrix: Sequence[Sequence[int]]) -> List[List[float]]:
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("confusion matrix must be square")
    return [[float(value) / sum(row) if sum(row) else 0.0 for value in row] for row in matrix]


def classification_summary(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    labels = tuple(labels or DERM_CLASS_ORDER)
    matrix = confusion_matrix(y_true, y_pred, labels)
    n = sum(sum(row) for row in matrix)
    if n == 0:
        raise ValueError("classification summary requires at least one case")
    per_class: Dict[str, Dict[str, float | int]] = {}
    f1_values: List[float] = []
    recalls: List[float] = []
    weighted_f1 = 0.0
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))]
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = col_sums[i] - tp
        fn = row_sums[i] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": precision, "recall": recall, "f1": f1, "support": row_sums[i],
        }
        f1_values.append(f1)
        recalls.append(recall)
        weighted_f1 += f1 * row_sums[i] / n
    accuracy = correct / n
    chance = sum(row_sums[i] * col_sums[i] for i in range(len(matrix))) / (n * n)
    kappa = (accuracy - chance) / (1.0 - chance) if chance < 1.0 else 1.0
    numerator = correct * n - sum(row_sums[i] * col_sums[i] for i in range(len(matrix)))
    denominator = ((n * n - sum(value * value for value in row_sums)) *
                   (n * n - sum(value * value for value in col_sums))) ** 0.5
    return {
        "n": n,
        "labels": list(labels),
        "confusion_matrix": matrix,
        "row_normalized_confusion_matrix": row_normalize(matrix),
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1,
        "mcc": numerator / denominator if denominator else 0.0,
        "cohen_kappa": kappa,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Axis D: calibration / selective prediction
# ---------------------------------------------------------------------------
def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error: sum_b (|b|/n) * |acc_b - conf_b|."""
    n = len(conf)
    if n == 0:
        return 0.0
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for c, y in zip(conf, correct):
        b = min(n_bins - 1, max(0, int(float(c) * n_bins)))
        bins[b].append((float(c), 1.0 if y else 0.0))
    e = 0.0
    for group in bins:
        if not group:
            continue
        avg_conf = sum(c for c, _ in group) / len(group)
        acc = sum(y for _, y in group) / len(group)
        e += (len(group) / n) * abs(acc - avg_conf)
    return round(e, 4)


def risk_coverage_curve(conf: Sequence[float], correct: Sequence[bool]) -> List[Tuple[float, float]]:
    """Answer most-confident-first; (coverage, selective_risk=error-rate-among-answered)."""
    n = len(conf)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -float(conf[i]))
    curve, err = [], 0
    for k, i in enumerate(order, 1):
        if not correct[i]:
            err += 1
        curve.append((round(k / n, 4), round(err / k, 4)))
    return curve


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area Under the Risk-Coverage curve (lower = better selective prediction)."""
    curve = risk_coverage_curve(conf, correct)
    return round(sum(r for _, r in curve) / len(curve), 4) if curve else 0.0


def harm_coverage_curve(conf: Sequence[float], harm: Sequence[float]) -> List[Tuple[float, float]]:
    """Confidence-ranked clinical harm frontier.

    Harm must be a nonnegative per-case severity (zero for a correct or safely
    deferred case).  At each prefix, risk is mean harm among answered cases.  This
    makes abstention visible through coverage while retaining acuity information.
    """
    if len(conf) != len(harm):
        raise ValueError("confidence and harm arrays must have equal length")
    if any(float(h) < 0 for h in harm):
        raise ValueError("harm values must be nonnegative")
    n = len(conf)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (-float(conf[i]), i))
    curve: List[Tuple[float, float]] = []
    cumulative = 0.0
    for k, i in enumerate(order, 1):
        cumulative += float(harm[i])
        curve.append((k / n, cumulative / k))
    return curve


def harm_coverage_auc(conf: Sequence[float], harm: Sequence[float]) -> float:
    """Discrete area under the harm-coverage curve; lower is safer."""
    curve = harm_coverage_curve(conf, harm)
    return sum(risk for _, risk in curve) / len(curve) if curve else 0.0


def matched_harm_coverage_auc(
    conf: Sequence[float], harm: Sequence[float], answered: Sequence[bool],
    *, total_cases: int, max_coverage: float,
) -> Optional[float]:
    """Partial harm-coverage AUC at a common, externally frozen coverage.

    Only observed answered outputs are eligible for the confidence ranking.
    Abstentions are not assigned zero harm. The same ``max_coverage`` must be
    frozen on calibration data and applied to every method. Returns ``None``
    when a method did not achieve enough coverage for a fair comparison.
    """
    if not (len(conf) == len(harm) == len(answered)):
        raise ValueError("confidence, harm, and answered arrays must have equal length")
    if total_cases < len(conf) or total_cases <= 0:
        raise ValueError("total_cases must be positive and cover every supplied record")
    if not 0.0 < float(max_coverage) <= 1.0:
        raise ValueError("max_coverage must lie in (0, 1]")
    if any(float(value) < 0 for value in harm):
        raise ValueError("harm values must be nonnegative")
    k_max = int(round(float(max_coverage) * total_cases))
    if k_max <= 0:
        raise ValueError("max_coverage selects zero cases")
    eligible = [i for i, value in enumerate(answered) if bool(value)]
    if len(eligible) < k_max:
        return None
    order = sorted(eligible, key=lambda i: (-float(conf[i]), i))[:k_max]
    cumulative = 0.0
    risks: List[float] = []
    for k, index in enumerate(order, start=1):
        cumulative += float(harm[index])
        risks.append(cumulative / k)
    return sum(risks) / len(risks)


def selective_accuracy_at_coverage(conf: Sequence[float], correct: Sequence[bool],
                                   coverage: float = 0.65) -> float:
    """Accuracy among the top-`coverage` fraction ranked by confidence."""
    n = len(conf)
    if n == 0:
        return 0.0
    k = max(1, int(round(coverage * n)))
    order = sorted(range(n), key=lambda i: -float(conf[i]))[:k]
    return round(sum(1 for i in order if correct[i]) / k, 4)


# ---------------------------------------------------------------------------
# Axis B: faithfulness / hallucination
# ---------------------------------------------------------------------------
def faithfulness(answers: Sequence[str], chunks_list: Sequence[Sequence[dict]]) -> float:
    """Mean grounding (citation x lexical-overlap-with-evidence) over answered cases."""
    scored = [grounding_score(a, ch) for a, ch in zip(answers, chunks_list) if (a or "").strip()]
    return round(sum(scored) / len(scored), 4) if scored else 0.0


def over_reliance(confident: Sequence[bool], correct: Sequence[bool]) -> float:
    """Fraction of cases that answer CONFIDENTLY but WRONG (over-relying on bad evidence)."""
    n = len(confident)
    if n == 0:
        return 0.0
    return round(sum(1 for c, y in zip(confident, correct) if c and not y) / n, 4)


def copy_reference_ratio(answer: str, chunks: Sequence[dict], ngram: int = 5) -> float:
    """Fraction of the answer's n-grams copied verbatim from the evidence (lower = less parroting)."""
    a = (answer or "").lower().split()
    if len(a) < ngram:
        return 0.0
    ev = " ".join(str(c.get("text", "")) for c in (chunks or [])).lower()
    grams = [" ".join(a[i:i + ngram]) for i in range(len(a) - ngram + 1)]
    hit = sum(1 for g in grams if g in ev)
    return round(hit / len(grams), 4) if grams else 0.0


def completeness(answer: str, chunks: Sequence[dict]) -> float:
    """Recall of evidence content tokens in the answer (how much of the evidence it covers)."""
    ev = _tokens(" ".join(str(c.get("text", "")) for c in (chunks or [])))
    if not ev:
        return 0.0
    return round(len(_tokens(answer) & ev) / len(ev), 4)


# ---------------------------------------------------------------------------
# Axis H: clinical evaluation rigor (QWK + triage), acuity-ordinal on the taxonomy
# ---------------------------------------------------------------------------
def acuity_ordinal(label: str) -> int:
    """Diagnosis -> 3-tier acuity ordinal: 0 benign, 1 pre-malignant (akiec), 2 malignant."""
    lab = (label or "").lower()
    if lab in ("akiec",):
        return 1
    return 2 if lab in MALIGNANT_LABELS else 0


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3) -> float:
    """Quadratic Weighted Kappa on ordinal ratings (Axis-H eval rigor, cf. CDSS QWK 0.902)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    O = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        O[int(t)][int(p)] += 1
    W = [[((i - j) ** 2) / ((n_classes - 1) ** 2) for j in range(n_classes)] for i in range(n_classes)]
    trow = [sum(O[i]) for i in range(n_classes)]
    pcol = [sum(O[i][j] for i in range(n_classes)) for j in range(n_classes)]
    E = [[trow[i] * pcol[j] / n for j in range(n_classes)] for i in range(n_classes)]
    num = sum(W[i][j] * O[i][j] for i in range(n_classes) for j in range(n_classes))
    den = sum(W[i][j] * E[i][j] for i in range(n_classes) for j in range(n_classes))
    return round(1.0 - num / den, 4) if den else 1.0


def triage_rates(pred_labels: Sequence[str], true_labels: Sequence[str]) -> Dict[str, float]:
    """Over/under-triage + high-risk (malignant) detection recall on the malignancy tiers."""
    over = under = mal_true = mal_detected = benign_true = 0
    for p, t in zip(pred_labels, true_labels):
        p_mal = (p or "").lower() in MALIGNANT_LABELS
        t_mal = (t or "").lower() in MALIGNANT_LABELS
        benign_true += (not t_mal); mal_true += t_mal
        if t_mal and p_mal:
            mal_detected += 1
        if (not t_mal) and p_mal:
            over += 1                     # benign called malignant
        if t_mal and (not p_mal):
            under += 1                    # MISSED cancer (the dangerous error)
    return {
        "over_triage": round(over / benign_true, 4) if benign_true else 0.0,
        "under_triage": round(under / mal_true, 4) if mal_true else 0.0,
        "high_risk_detection": round(mal_detected / mal_true, 4) if mal_true else 0.0,
    }


# ---------------------------------------------------------------------------
# Malignant / actionable binary confusion matrix (protocol-mandated 2x2)
# ---------------------------------------------------------------------------
def _actionable_positive_set(positive_labels: Optional[Sequence[str]]) -> frozenset[str]:
    labels = MALIGNANT_LABELS if positive_labels is None else positive_labels
    resolved = frozenset(str(label).lower() for label in labels)
    if not resolved:
        raise ValueError("the actionable positive-label set must be non-empty")
    return resolved


def is_actionable(label: str, positive_labels: Optional[Sequence[str]] = None) -> bool:
    """True if a diagnosis is in the actionable (malignant/pre-malignant) set.

    Default set is ``MALIGNANT_LABELS`` (akiec, bcc, melanoma) for HAM7. BCN adds
    ``scc``; pass ``positive_labels=MALIGNANT_LABELS | {"scc"}`` at that call site.
    """
    return (label or "").lower() in _actionable_positive_set(positive_labels)


def actionable_confusion(
    pred_labels: Sequence[str],
    true_labels: Sequence[str],
    *,
    positive_labels: Optional[Sequence[str]] = None,
    scores: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Binary actionable 2x2 with the derived clinical rates one table row needs.

    Positive = actionable (a lesion that warrants biopsy/excision/referral). The
    matrix rows are truth, columns are prediction, ordered (actionable, benign)::

                       pred_actionable   pred_benign
        true_actionable       TP              FN
        true_benign           FP              TN

    A rate whose denominator is empty is reported as ``None`` (undefined), never
    coerced to 0.0, so a stratum with no positives cannot masquerade as perfect
    specificity or zero sensitivity. ``auroc`` is filled only when ``scores`` are
    supplied and both classes are present; it ranks by the actionable score and
    reuses the DeLong-consistent estimator in ``eval.statistics.binary_auc``.
    """
    if len(pred_labels) != len(true_labels):
        raise ValueError("pred_labels and true_labels must have equal length")
    positive = _actionable_positive_set(positive_labels)
    tp = fn = fp = tn = 0
    truth_bool: List[bool] = []
    for pred, truth in zip(pred_labels, true_labels):
        p_pos = (pred or "").lower() in positive
        t_pos = (truth or "").lower() in positive
        truth_bool.append(t_pos)
        if t_pos and p_pos:
            tp += 1
        elif t_pos and not p_pos:
            fn += 1
        elif (not t_pos) and p_pos:
            fp += 1
        else:
            tn += 1
    n = tp + fn + fp + tn
    positives, negatives = tp + fn, tn + fp
    predicted_pos = tp + fp
    predicted_neg = tn + fn

    def _rate(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 4) if denominator else None

    sensitivity = _rate(tp, positives)
    auroc: Optional[float] = None
    if scores is not None:
        if len(scores) != len(true_labels):
            raise ValueError("scores must align with the label sequences")
        if positives and negatives:
            from .statistics import binary_auc
            auroc = round(binary_auc(truth_bool, [float(s) for s in scores]), 4)
    return {
        "n": n,
        "positive_labels": sorted(positive),
        "confusion_matrix": [[tp, fn], [fp, tn]],
        "matrix_axes": {"rows": ["true_actionable", "true_benign"],
                        "cols": ["pred_actionable", "pred_benign"]},
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "prevalence": _rate(positives, n),
        "sensitivity": sensitivity,
        "specificity": _rate(tn, negatives),
        "ppv": _rate(tp, predicted_pos),
        "npv": _rate(tn, predicted_neg),
        "false_negative_rate": _rate(fn, positives),
        "accuracy": _rate(tp + tn, n),
        "auroc": auroc,
    }


def sensitivity_at_fixed_specificity(
    actionable_true: Sequence[bool],
    scores: Sequence[float],
    target_specificity: float,
) -> Optional[Dict[str, float]]:
    """Sensitivity at the operating point that just meets a target specificity.

    A case is predicted actionable when ``score >= threshold``. Among all
    thresholds whose specificity is at least ``target_specificity`` this returns
    the one with the greatest sensitivity (the lowest such threshold, since
    sensitivity is non-increasing and specificity non-decreasing in the cut). The
    ``target_specificity`` LEVEL is a preregistration parameter and must be frozen
    on calibration data before outcomes are viewed -- this function takes it as an
    argument and does not choose it. Returns ``None`` when no threshold reaches the
    target, or when either class is absent (both are required to define the point).
    """
    if len(actionable_true) != len(scores):
        raise ValueError("actionable_true and scores must have equal length")
    if not 0.0 <= float(target_specificity) <= 1.0:
        raise ValueError("target_specificity must lie in [0, 1]")
    truth = [bool(value) for value in actionable_true]
    positives = sum(truth)
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    values = [float(s) for s in scores]
    # A threshold above every score predicts all-negative (specificity 1.0); the
    # distinct scores enumerate every other operating point.
    thresholds = sorted(set(values) | {max(values) + 1.0})
    best: Optional[Dict[str, float]] = None
    for threshold in thresholds:
        tp = sum(1 for t, s in zip(truth, values) if t and s >= threshold)
        fp = sum(1 for t, s in zip(truth, values) if not t and s >= threshold)
        specificity = (negatives - fp) / negatives
        if specificity + 1e-12 < float(target_specificity):
            continue
        sensitivity = tp / positives
        if best is None or sensitivity > best["sensitivity"]:
            best = {
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "threshold": threshold,
                "target_specificity": float(target_specificity),
            }
    return best
