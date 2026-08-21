"""External integrity pins for Clinical Control Graph deployment artifacts.

An integrity pin must live outside the graph's write boundary.  Hashing provenance
inside the same mutable artifact does not protect against an attacker who can edit
both content and metadata.  This module detects post-approval mutation; it does
not claim that an approved graph is clinically true.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .audit import GraphAuditRecord, audit_status, require_clinician_audit
from .model import ClinicalControlGraph
from .runtime import GraphQuery, GraphSignals, evaluate_graph


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode("utf-8")


def canonical_graph_payload(graph: ClinicalControlGraph) -> dict[str, object]:
    """Return order-independent graph content for hashing."""
    value = graph.to_dict()
    return {
        "nodes": sorted(value["nodes"], key=lambda row: str(row["node_id"])),
        "edges": sorted(value["edges"], key=lambda row: str(row["edge_id"])),
    }


def graph_sha256(graph: ClinicalControlGraph) -> str:
    return hashlib.sha256(_canonical_bytes(canonical_graph_payload(graph))).hexdigest()


def audit_sha256(records: Iterable[GraphAuditRecord]) -> str:
    payload = sorted((asdict(record) for record in records),
                     key=lambda row: (str(row["edge_id"]), str(row["reviewed_at"]),
                                      str(row["reviewer_code"]), str(row["decision"])))
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class GraphPin:
    artifact_id: str
    graph_sha256: str
    audit_sha256: str
    policy_version: str = "clinical-control-graph-pin-v1"

    def validate(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("graph pin requires an artifact id")
        for name, value in (("graph_sha256", self.graph_sha256),
                            ("audit_sha256", self.audit_sha256)):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class GraphIntegrityStatus:
    valid: bool
    reasons: tuple[str, ...]
    observed_graph_sha256: str
    observed_audit_sha256: str


def build_graph_pin(graph: ClinicalControlGraph, records: Sequence[GraphAuditRecord],
                    *, artifact_id: str) -> GraphPin:
    """Pin only a graph with complete modality-appropriate clinical acceptance."""
    require_clinician_audit(graph, records)
    pin = GraphPin(artifact_id, graph_sha256(graph), audit_sha256(records))
    pin.validate()
    return pin


def verify_graph_pin(graph: ClinicalControlGraph, records: Sequence[GraphAuditRecord],
                     pin: GraphPin) -> GraphIntegrityStatus:
    pin.validate()
    observed_graph = graph_sha256(graph)
    observed_audit = audit_sha256(records)
    reasons: list[str] = []
    if observed_graph != pin.graph_sha256:
        reasons.append("graph_hash_mismatch")
    if observed_audit != pin.audit_sha256:
        reasons.append("audit_hash_mismatch")
    if audit_status(graph, records):
        reasons.append("clinical_audit_incomplete")
    return GraphIntegrityStatus(not reasons, tuple(reasons), observed_graph, observed_audit)


def evaluate_pinned_graph(graph: ClinicalControlGraph, query: GraphQuery,
                          records: Sequence[GraphAuditRecord], pin: GraphPin) -> GraphSignals:
    """Fail closed before reading semantic signals from a mutated graph."""
    status = verify_graph_pin(graph, records, pin)
    if not status.valid:
        return GraphSignals(
            out_of_graph_distance=1.0,
            integrity_valid=False,
            integrity_reasons=list(status.reasons),
            graph_sha256=status.observed_graph_sha256,
        )
    signals = evaluate_graph(graph, query)
    signals.integrity_valid = True
    signals.graph_sha256 = status.observed_graph_sha256
    return signals
