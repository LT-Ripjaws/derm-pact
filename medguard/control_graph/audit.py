"""Clinician sign-off records for graph edges before final-test use."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .model import ClinicalControlGraph


ROLE_BY_MODALITY = {"dermoscopy": "dermatologist", "cxr": "radiologist"}


@dataclass(frozen=True)
class GraphAuditRecord:
    edge_id: str
    modality: str
    reviewer_role: str
    reviewer_code: str
    decision: str
    reviewed_at: str
    notes: str = ""

    def validate(self) -> None:
        if self.modality not in ROLE_BY_MODALITY:
            raise ValueError(f"unknown graph audit modality {self.modality!r}")
        if self.reviewer_role != ROLE_BY_MODALITY[self.modality]:
            raise ValueError(f"{self.modality} edges require a {ROLE_BY_MODALITY[self.modality]}")
        if self.decision not in {"accept", "reject", "revise"}:
            raise ValueError(f"unknown graph audit decision {self.decision!r}")
        if not self.reviewer_code or not self.reviewed_at:
            raise ValueError("graph audit requires coded reviewer and timestamp")


def audit_status(graph: ClinicalControlGraph,
                 records: Iterable[GraphAuditRecord]) -> Dict[str, List[str]]:
    by_edge: Dict[str, List[GraphAuditRecord]] = {}
    errors: List[str] = []
    for record in records:
        try:
            record.validate()
        except ValueError as exc:
            errors.append(str(exc))
        by_edge.setdefault(record.edge_id, []).append(record)
    missing = sorted(set(graph.edges) - set(by_edge))
    unknown = sorted(set(by_edge) - set(graph.edges))
    unresolved = sorted(edge_id for edge_id, group in by_edge.items()
                        if edge_id in graph.edges and not any(r.decision == "accept" for r in group))
    return {k: v for k, v in {"missing": missing, "unknown": unknown,
                              "unresolved": unresolved, "invalid": errors}.items() if v}


def require_clinician_audit(graph: ClinicalControlGraph,
                            records: Iterable[GraphAuditRecord]) -> None:
    issues = audit_status(graph, records)
    if issues:
        raise ValueError(f"clinical graph audit incomplete: {issues}")
