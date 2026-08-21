"""Matched-case evaluation records for the active MedGuard thesis.

This module only aggregates already produced case records.  It deliberately
does not choose a controller, call a model, or encode a clinical harm oracle.
Accuracy/confusion metrics, RAGAS fields, coverage, cost and trajectory trust
remain separate outputs so one favorable scalar cannot hide a weak axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..agents.process_trust import ProcessTrustProfile
from ..evidence import EvidenceContract
from .metrics import classification_summary


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    cluster_id: str
    truth: str
    prediction: str
    answered: bool
    final_action: str
    contract: EvidenceContract
    process_trust: ProcessTrustProfile
    calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    ragas: Mapping[str, Optional[float]] = field(default_factory=dict)

    def validate(self) -> "CaseResult":
        if not self.case_id or not self.cluster_id or not self.truth:
            raise ValueError("case_id, cluster_id, and truth are required")
        if self.answered and not self.prediction:
            raise ValueError("answered records require a prediction")
        if not self.answered and self.prediction:
            raise ValueError("abstained records must not carry a prediction")
        if min(self.calls, self.tokens, self.latency_ms) < 0:
            raise ValueError("cost fields must be non-negative")
        self.contract.validate()
        return self


@dataclass(frozen=True)
class RunReport:
    arm: str
    n: int
    coverage: float
    classification: Dict[str, Any]
    action_counts: Dict[str, int]
    process_trust_dimensions: Dict[str, Optional[float]]
    ragas_means: Dict[str, Optional[float]]
    mean_calls: float
    mean_tokens: float
    mean_latency_ms: float


def _means(rows: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    names = sorted({name for row in rows for name in row})
    result: Dict[str, Optional[float]] = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


def aggregate_run(
    records: Sequence[CaseResult], *, arm: str, labels: Optional[Sequence[str]] = None,
) -> RunReport:
    if not arm.strip():
        raise ValueError("arm must be non-empty")
    checked: List[CaseResult] = [record.validate() for record in records]
    n = len(checked)
    answered = [record for record in checked if record.answered]
    actions: Dict[str, int] = {}
    for record in checked:
        actions[record.final_action] = actions.get(record.final_action, 0) + 1
    classification = classification_summary(
        [record.truth for record in answered],
        [record.prediction for record in answered],
        labels=labels,
    ) if answered else {
        "n": 0, "labels": list(labels or ()), "confusion_matrix": [],
        "accuracy": None, "balanced_accuracy": None, "macro_f1": None,
        "weighted_f1": None, "mcc": None, "cohen_kappa": None,
        "per_class": {},
    }
    trust = _means([record.process_trust.dimensions for record in checked])
    ragas = _means([record.ragas for record in checked])
    denom = n or 1
    return RunReport(
        arm=arm,
        n=n,
        coverage=len(answered) / n if n else 0.0,
        classification=classification,
        action_counts=actions,
        process_trust_dimensions=trust,
        ragas_means=ragas,
        mean_calls=sum(record.calls for record in checked) / denom,
        mean_tokens=sum(record.tokens for record in checked) / denom,
        mean_latency_ms=sum(record.latency_ms for record in checked) / denom,
    )
