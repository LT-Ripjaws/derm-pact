"""Calibrated selective-abstention gate, fit label-free on dev score quantiles.

Answered-accuracy is bounded by the FROZEN DermLIP perception probe (which is never
retrained here). The one honest lever left is selective prediction: abstain on the
lesions the probe is least confident about -- the ones most likely to be wrong --
so the accuracy *conditional on answering* rises while coverage falls in a
controlled way (the standard risk-coverage trade-off the AURC endpoint already
measures).

Following the CALIBER split-conformal recipe, the threshold is fit on the dev
confidence QUANTILES only. No per-case label is read, so there is no label leakage:
the gate cannot know which lesions are wrong, only which are low-confidence. To
answer a target fraction ``c`` of lesions, the diagnostic-abstention tau is the
``(1 - c)`` quantile of the calibrated-confidence distribution; a lesion whose
confidence falls below tau abstains. The fitted tau is serialisable so a sealed run
reuses the exact dev-frozen threshold rather than refitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


SCHEMA_VERSION = "medguard-selective-gate-v1"
DEFAULT_TARGET_ANSWER_COVERAGE = 0.90
# Until fitted, a tau of 0.0 abstains on nothing -- the gate is inert rather than
# silently abstaining on cases it was never calibrated for.
FALLBACK_DIAGNOSTIC_TAU = 0.0


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an ascending sequence, q in [0, 1]."""
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    q = min(1.0, max(0.0, float(q)))
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low]) * (1.0 - fraction) + float(sorted_values[high]) * fraction


@dataclass(frozen=True)
class CalibratedAbstentionGate:
    """A frozen, label-free diagnostic-abstention threshold on probe confidence."""

    diagnostic_confidence_tau: float
    target_answer_coverage: float = DEFAULT_TARGET_ANSWER_COVERAGE
    fitted: bool = False
    n_calibration: int = 0

    def should_abstain(self, calibrated_confidence: Optional[float]) -> bool:
        """Abstain when the probe's calibrated confidence is below the fitted tau."""
        if calibrated_confidence is None:
            return False
        return float(calibrated_confidence) < self.diagnostic_confidence_tau

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "diagnostic_confidence_tau": self.diagnostic_confidence_tau,
            "target_answer_coverage": self.target_answer_coverage,
            "fitted": self.fitted,
            "n_calibration": self.n_calibration,
            "label_free": True,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CalibratedAbstentionGate":
        return cls(
            diagnostic_confidence_tau=float(value["diagnostic_confidence_tau"]),
            target_answer_coverage=float(
                value.get("target_answer_coverage", DEFAULT_TARGET_ANSWER_COVERAGE)),
            fitted=bool(value.get("fitted", False)),
            n_calibration=int(value.get("n_calibration", 0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedAbstentionGate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_diagnostic_abstention_gate(
    calibrated_confidences: Sequence[Optional[float]],
    *,
    target_answer_coverage: float = DEFAULT_TARGET_ANSWER_COVERAGE,
    fallback_tau: float = FALLBACK_DIAGNOSTIC_TAU,
) -> CalibratedAbstentionGate:
    """Fit the diagnostic-abstention tau from dev confidence quantiles (label-free).

    ``target_answer_coverage`` is the fraction of lesions to answer; the tau is the
    ``(1 - coverage)`` quantile of the confidence distribution, so the lowest-
    confidence ``(1 - coverage)`` fraction abstains. With no usable confidences the
    gate falls back to an inert tau rather than guessing.
    """
    values = sorted(float(c) for c in calibrated_confidences if c is not None)
    if not values:
        return CalibratedAbstentionGate(
            diagnostic_confidence_tau=float(fallback_tau),
            target_answer_coverage=float(target_answer_coverage),
            fitted=False, n_calibration=0,
        )
    tau = _quantile(values, 1.0 - float(target_answer_coverage))
    return CalibratedAbstentionGate(
        diagnostic_confidence_tau=tau,
        target_answer_coverage=float(target_answer_coverage),
        fitted=True, n_calibration=len(values),
    )
