"""Small, serializable calibrators with identity fallback and no weight training."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp, softmax


def logits_array(values):
    z = np.asarray(values, dtype=np.float64)
    if z.ndim not in (1, 2) or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError('logits must be a nonempty finite vector or matrix')
    if z.ndim==2 and z.shape[1]<2:
        raise ValueError('multiclass logits require at least two classes')
    return z


@dataclass(frozen=True)
class Temperature:
    value: float = 1.0
    status: str = 'identity'

    def __post_init__(self):
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError('temperature must be positive and finite')

    def probabilities(self, logits):
        z = logits_array(logits) / self.value
        return expit(z) if z.ndim == 1 else softmax(z, axis=1)

    @classmethod
    def fit(cls, logits, labels):
        z = logits_array(logits)
        y = np.asarray(labels)
        if y.shape != (len(z),) or not np.isfinite(y).all() or not np.equal(y, y.astype(int)).all():
            raise ValueError('integer labels must align to logits')
        y = y.astype(int)
        classes = 2 if z.ndim == 1 else z.shape[1]
        if np.any(y < 0) or np.any(y >= classes):
            raise ValueError('label outside model vocabulary')
        if len(np.unique(y)) < 2:
            return cls(1.0, 'identity_insufficient_class_support')
        def objective(log_t):
            scaled = z / np.exp(log_t)
            if z.ndim == 1:
                return float(np.mean(np.logaddexp(0, scaled) - y * scaled))
            return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))
        fit = minimize_scalar(objective, bounds=(np.log(.05), np.log(20.)), method='bounded')
        if not fit.success or not np.isfinite(fit.fun):
            return cls(1., 'identity_optimizer_failed')
        if fit.fun >= objective(0):
            return cls(1., 'identity_no_fit_improvement')
        return cls(float(np.exp(fit.x)), 'fitted')


def calibration_metrics(labels, probabilities, bins=10):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.ndim == 1:
        p = np.stack([1 - p, p], axis=1)
    if len(p) != len(y) or not len(y) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('invalid probabilities or labels')
    if not np.allclose(p.sum(1), 1) or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError('probability rows must sum to one; labels must be in range')
    confidence = p.max(1)
    correct = p.argmax(1) == y
    groups = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = sum(np.mean(groups == b) * abs(float(correct[groups == b].mean()) - float(confidence[groups == b].mean()))
              for b in range(bins) if np.any(groups == b))
    target = np.eye(p.shape[1])[y]
    return {'n': len(y), 'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            'brier_multiclass_sum': float(np.square(p - target).sum(1).mean()),
            'ece_top_label_10_equal_width_bins': float(ece), 'accuracy': float(correct.mean())}


def risk_score(diagnosis_probability, binary_probabilities):
    p = np.asarray(diagnosis_probability, dtype=float)
    b = np.asarray(binary_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1]<2 or b.shape != (len(p), 8) or not len(p):
        raise ValueError('expected diagnosis [N,C] and binary [N,8] probabilities')
    if not np.isfinite(p).all() or not np.isfinite(b).all() or np.any(b < 0) or np.any(b > 1):
        raise ValueError('nonfinite or out-of-range probabilities')
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1):
        raise ValueError('invalid diagnosis probabilities')
    q = np.sort(p, axis=1)
    margin_risk = 1 - (q[:, -1] - q[:, -2])
    clipped = np.clip(b, 1e-12, 1 - 1e-12)
    entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
    # Fixed equal weighting; no tuning against evaluation outcomes.
    return .5 * margin_risk + .5 * entropy.max(1)


def fit_acceptance_threshold(scores, coverage=.90):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < coverage <= 1:
        raise ValueError('invalid scores or coverage')
    return float(np.quantile(values, coverage, method='higher'))


def risk_coverage(errors, scores):
    e, s = np.asarray(errors, dtype=float), np.asarray(scores, dtype=float)
    if e.shape != s.shape or e.ndim != 1 or not len(e) or not np.isfinite(e).all() or not np.isfinite(s).all():
        raise ValueError('aligned finite error/risk vectors required')
    if np.any(e < 0) or np.any(e > 1):
        raise ValueError('errors outside [0,1]')
    order = np.argsort(s, kind='stable')
    risk = np.cumsum(e[order]) / np.arange(1, len(e) + 1)
    return {'aurc_discrete': float(risk.mean()), 'coverage': (np.arange(1, len(e) + 1) / len(e)).tolist(),
            'risk': risk.tolist(), 'tie_policy': 'stable_input_order'}


def conformal_quantile(scores, alpha=.1):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not 0 < alpha < 1 or not np.isfinite(scores).all():
        raise ValueError('invalid conformal scores')
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return 1. if rank > len(scores) else float(np.sort(scores)[rank - 1])
