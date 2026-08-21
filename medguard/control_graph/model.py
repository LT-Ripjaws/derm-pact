"""Provenance-first data model for the Clinical Control Graph.

The graph is a control-plane artifact, not generator context.  Every clinical edge
must carry enough provenance to be independently audited and invalidated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


class NodeType(str, Enum):
    MODALITY = "modality"
    ANATOMY = "anatomy"
    VISUAL_FINDING = "visual_finding"
    DIAGNOSIS = "diagnosis"
    RISK_STATE = "risk_state"
    MANAGEMENT_ACTION = "management_action"
    SOURCE = "source"


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUGGESTIVE_OF = "suggestive_of"
    LOCATED_AT = "located_at"
    REQUIRES_ACTION = "requires_action"
    CONTRAINDICATES = "contraindicates"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: NodeType
    label: str
    ontology_id: str = ""

    def validate(self) -> None:
        if not self.node_id.strip() or not self.label.strip():
            raise ValueError("graph nodes require node_id and label")


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    evidence_span: str
    source: str
    version: str
    authority: float
    extraction_method: str
    confidence: float
    independent_source_group: str
    valid_from: str = ""
    valid_until: str = ""

    def validate(self) -> None:
        required = {
            "edge_id": self.edge_id, "source_id": self.source_id,
            "target_id": self.target_id, "evidence_span": self.evidence_span,
            "source": self.source, "version": self.version,
            "extraction_method": self.extraction_method,
            "independent_source_group": self.independent_source_group,
        }
        missing = [k for k, v in required.items() if not str(v).strip()]
        if missing:
            raise ValueError(f"graph edge missing required provenance: {missing}")
        if not 0.0 <= self.authority <= 1.0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("edge authority and confidence must lie in [0, 1]")
        for value in (self.valid_from, self.valid_until):
            if value:
                date.fromisoformat(value)

    def is_current(self, as_of: Optional[date] = None) -> bool:
        when = as_of or date.today()
        return ((not self.valid_from or date.fromisoformat(self.valid_from) <= when)
                and (not self.valid_until or when <= date.fromisoformat(self.valid_until)))

    def trust(self, as_of: Optional[date] = None) -> float:
        return self.authority * self.confidence if self.is_current(as_of) else 0.0


class ClinicalControlGraph:
    def __init__(self, nodes: Iterable[Node] = (), edges: Iterable[Edge] = ()):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: Node) -> None:
        node.validate()
        if node.node_id in self.nodes and self.nodes[node.node_id] != node:
            raise ValueError(f"conflicting node definition: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        edge.validate()
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"edge {edge.edge_id} references an unknown node")
        if edge.edge_id in self.edges and self.edges[edge.edge_id] != edge:
            raise ValueError(f"conflicting edge definition: {edge.edge_id}")
        self.edges[edge.edge_id] = edge

    def outgoing(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges.values() if e.source_id == node_id]

    def between(self, source_ids: Iterable[str], target_ids: Iterable[str]) -> List[Edge]:
        sources, targets = set(source_ids), set(target_ids)
        return [e for e in self.edges.values()
                if e.source_id in sources and e.target_id in targets]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [{**asdict(n), "node_type": n.node_type.value} for n in self.nodes.values()],
            "edges": [{**asdict(e), "edge_type": e.edge_type.value} for e in self.edges.values()],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClinicalControlGraph":
        nodes = [Node(node_id=str(n["node_id"]), node_type=NodeType(n["node_type"]),
                      label=str(n["label"]), ontology_id=str(n.get("ontology_id", "")))
                 for n in value.get("nodes", [])]
        edges = [Edge(
            edge_id=str(e["edge_id"]), source_id=str(e["source_id"]),
            target_id=str(e["target_id"]), edge_type=EdgeType(e["edge_type"]),
            evidence_span=str(e["evidence_span"]), source=str(e["source"]),
            version=str(e["version"]), authority=float(e["authority"]),
            extraction_method=str(e["extraction_method"]), confidence=float(e["confidence"]),
            independent_source_group=str(e["independent_source_group"]),
            valid_from=str(e.get("valid_from", "")), valid_until=str(e.get("valid_until", "")),
        ) for e in value.get("edges", [])]
        return cls(nodes, edges)
