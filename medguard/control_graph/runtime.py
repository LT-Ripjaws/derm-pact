"""Independent controller features computed from the Clinical Control Graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Sequence

from .model import ClinicalControlGraph, Edge, EdgeType


SUPPORT_TYPES = {EdgeType.SUPPORTS, EdgeType.SUGGESTIVE_OF, EdgeType.REQUIRES_ACTION}
CONTRADICTION_TYPES = {EdgeType.CONTRADICTS, EdgeType.CONTRAINDICATES}


@dataclass(frozen=True)
class GraphQuery:
    observed_node_ids: Sequence[str]
    claim_node_ids: Sequence[str]
    as_of: str = ""


@dataclass
class GraphSignals:
    support_score: float = 0.0
    contradiction_score: float = 0.0
    path_coverage: float = 0.0
    source_quorum: int = 0
    minimum_edge_trust: float = 0.0
    out_of_graph_distance: float = 1.0
    supporting_paths: List[List[str]] = field(default_factory=list)
    contradicting_paths: List[List[str]] = field(default_factory=list)
    episode_action_prior: float = 0.0
    integrity_valid: bool = True
    integrity_reasons: List[str] = field(default_factory=list)
    graph_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _path(edge: Edge) -> List[str]:
    return [edge.source_id, edge.edge_type.value, edge.target_id, edge.edge_id]


def evaluate_graph(graph: ClinicalControlGraph, query: GraphQuery) -> GraphSignals:
    """Score direct observed->claim relations without using generator text.

    `out_of_graph_distance` is a bounded coverage deficit, not a shortest-path
    distance: 0 means every requested node exists, 1 means none do.  This explicit
    definition avoids implying a metric property the heterogeneous graph lacks.
    """
    as_of = date.fromisoformat(query.as_of) if query.as_of else date.today()
    requested = list(dict.fromkeys([*query.observed_node_ids, *query.claim_node_ids]))
    known = [n for n in requested if n in graph.nodes]
    distance = 1.0 - (len(known) / len(requested)) if requested else 1.0
    edges = graph.between(query.observed_node_ids, query.claim_node_ids)
    support = [e for e in edges if e.edge_type in SUPPORT_TYPES and e.is_current(as_of)]
    contradict = [e for e in edges if e.edge_type in CONTRADICTION_TYPES and e.is_current(as_of)]
    support_trust = [e.trust(as_of) for e in support]
    contradiction_trust = [e.trust(as_of) for e in contradict]
    covered_claims = {e.target_id for e in [*support, *contradict]}
    claim_count = len(set(query.claim_node_ids))
    all_used = [*support, *contradict]
    source_groups = {e.independent_source_group for e in support if e.trust(as_of) > 0}
    return GraphSignals(
        support_score=round(sum(support_trust) / len(support_trust), 6) if support_trust else 0.0,
        contradiction_score=(round(max(contradiction_trust), 6)
                             if contradiction_trust else 0.0),
        path_coverage=round(len(covered_claims) / claim_count, 6) if claim_count else 0.0,
        source_quorum=len(source_groups),
        minimum_edge_trust=round(min((e.trust(as_of) for e in all_used), default=0.0), 6),
        out_of_graph_distance=round(distance, 6),
        supporting_paths=[_path(e) for e in support],
        contradicting_paths=[_path(e) for e in contradict],
    )
