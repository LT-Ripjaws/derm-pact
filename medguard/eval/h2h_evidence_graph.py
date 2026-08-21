"""Frozen evidence-lineage graph over the shared dermatology text corpus.

The graph is a MedGraphRAG-inspired *compatible adaptation*, never a
reproduction.  It realizes the source paper's triple-linked idea
``[RAG entity, source, definition]`` as four explicit tiers over the frozen
corpus:

    concept (controlled vocabulary + definition)
        -> evidence (one frozen corpus record)
            -> source (publication identity)
                -> version (frozen source version)

Two structural deviations are declared rather than hidden.  The frozen corpus
stores exactly one abstract record per publication, so the paper's tier-1
(private RAG document) and tier-2 (credible source repository) coincide here;
there is no separate private-document tier.  The controlled-concept tier holds
the study's seven frozen HAM10000 lesion classes, not UMLS.

Membership in this graph is an *index and lineage* relation.  A concept edge is
an embedding-similarity link at the source paper's disclosed threshold; it never
establishes that a record's content is true, and no entailment may be derived
from it.  Semantic support remains the relation assessor's exclusive concern.

This module performs no generation and holds no model dependency: vectors are
supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple


# Paper 133 discloses "unless otherwise noted, all thresholds are set as 0.5".
DELTA_R = 0.5
# Paper 133 builds entity content as Ce = Text[name: na; type: ty; context: cx].
# UMLS semantic types are unavailable offline, so the type slot carries the
# study's own frozen vocabulary label instead of an asserted clinical category.
CONCEPT_TYPE = "ham10000_lesion_class"
CONCEPT_CONTENT_TEMPLATE = "name: {name}; type: {type}; context: {context}"
GRAPH_SCHEMA_VERSION = "medguard-h2h-evidence-lineage-graph-v1"


class GraphBuildError(ValueError):
    """Raised when the frozen graph cannot be built from receipt-bound inputs."""


class NodeType(str, Enum):
    CONCEPT = "concept"
    EVIDENCE = "evidence"
    SOURCE = "source"
    VERSION = "version"


class EdgeType(str, Enum):
    # Paper 133 annotates the tier-1 -> tier-2 link "TheReferenceOf" and the
    # tier-2 -> tier-3 link "the definition of".
    THE_DEFINITION_OF = "the_definition_of"
    THE_REFERENCE_OF = "the_reference_of"
    HAS_VERSION = "has_version"


_EDGE_TIERS = {
    EdgeType.THE_DEFINITION_OF: (NodeType.EVIDENCE, NodeType.CONCEPT),
    EdgeType.THE_REFERENCE_OF: (NodeType.EVIDENCE, NodeType.SOURCE),
    EdgeType.HAS_VERSION: (NodeType.SOURCE, NodeType.VERSION),
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise GraphBuildError(f"{name} must be non-empty")
    return text


@dataclass(frozen=True)
class GraphNode:
    """One typed graph node.

    ``display_name`` deliberately avoids the name ``label``: the H2H protocol
    rejects truth-bearing keys, and ``label`` is one of them.
    """

    node_id: str
    node_type: NodeType
    display_name: str
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text("node_id", self.node_id)
        _require_text("display_name", self.display_name)
        if not isinstance(self.node_type, NodeType):
            raise GraphBuildError("node_type must be a closed NodeType")
        if not isinstance(self.attributes, Mapping):
            raise GraphBuildError(f"node {self.node_id} attributes must be a mapping")
        object.__setattr__(self, "attributes", dict(self.attributes))

    @property
    def populated_attribute_count(self) -> int:
        return sum(
            1 for value in self.attributes.values()
            if value is not None and str(value).strip() != ""
        )


@dataclass(frozen=True)
class GraphEdge:
    """One typed, auditable edge carrying the basis that created it."""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    basis: str
    basis_value: float | None = None

    def __post_init__(self) -> None:
        _require_text("edge_id", self.edge_id)
        _require_text("source_id", self.source_id)
        _require_text("target_id", self.target_id)
        _require_text("basis", self.basis)
        if not isinstance(self.edge_type, EdgeType):
            raise GraphBuildError("edge_type must be a closed EdgeType")
        if self.basis_value is not None:
            value = float(self.basis_value)
            if not math.isfinite(value):
                raise GraphBuildError(f"edge {self.edge_id} basis_value must be finite")
            object.__setattr__(self, "basis_value", value)


@dataclass(frozen=True)
class GraphPath:
    """One auditable returned path from the seed concept to frozen lineage."""

    evidence_id: str
    node_ids: Tuple[str, ...]
    edge_ids: Tuple[str, ...]
    hops: int
    source_id: str
    source_version: str
    content_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if self.hops != len(self.edge_ids):
            raise GraphBuildError("graph path hop count must equal its edge count")
        if len(self.node_ids) != self.hops + 1:
            raise GraphBuildError("graph path nodes must form one edge-connected chain")
        if self.evidence_id not in self.node_ids:
            raise GraphBuildError("graph path must contain its evidence node")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "hops": self.hops,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "content_sha256": self.content_sha256,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class GraphRetrieval:
    """Result of one bounded traversal; carries every audit field the report needs.

    ``candidate_evidence_ids`` is the bounded retrieved subgraph, and every
    sufficiency dimension is measured over it.  ``returned_evidence_ids`` is the
    prefix that would reach a generator under the shared final-evidence ceiling.
    """

    seed_concept_ids: Tuple[str, ...]
    depth: int
    node_limit: int
    candidate_evidence_ids: Tuple[str, ...]
    returned_evidence_ids: Tuple[str, ...]
    paths: Tuple[GraphPath, ...]
    subgraph_node_ids: Tuple[str, ...]
    subgraph_edge_ids: Tuple[str, ...]
    reached_evidence_count: int
    frontier_sizes: Tuple[int, ...]
    mean_populated_attributes: float

    @property
    def subgraph_node_count(self) -> int:
        return len(self.subgraph_node_ids)

    @property
    def paths_by_evidence(self) -> Dict[str, GraphPath]:
        return {path.evidence_id: path for path in self.paths}


class EvidenceLineageGraph:
    """Immutable four-tier graph with bounded, deterministic traversal."""

    def __init__(self, nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        for node in nodes:
            if node.node_id in self._nodes:
                raise GraphBuildError(f"duplicate graph node: {node.node_id}")
            self._nodes[node.node_id] = node
        self._edges: Dict[str, GraphEdge] = {}
        self._out: Dict[str, List[str]] = {}
        self._in: Dict[str, List[str]] = {}
        for edge in edges:
            if edge.edge_id in self._edges:
                raise GraphBuildError(f"duplicate graph edge: {edge.edge_id}")
            for endpoint in (edge.source_id, edge.target_id):
                if endpoint not in self._nodes:
                    raise GraphBuildError(
                        f"edge {edge.edge_id} references unknown node {endpoint}"
                    )
            expected = _EDGE_TIERS[edge.edge_type]
            observed = (
                self._nodes[edge.source_id].node_type,
                self._nodes[edge.target_id].node_type,
            )
            if observed != expected:
                raise GraphBuildError(
                    f"edge {edge.edge_id} violates its tier contract: "
                    f"{observed[0].value}->{observed[1].value}"
                )
            self._edges[edge.edge_id] = edge
            self._out.setdefault(edge.source_id, []).append(edge.edge_id)
            self._in.setdefault(edge.target_id, []).append(edge.edge_id)
        for adjacency in (self._out, self._in):
            for edge_ids in adjacency.values():
                edge_ids.sort()

    @property
    def nodes(self) -> Mapping[str, GraphNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> Mapping[str, GraphEdge]:
        return dict(self._edges)

    def node_ids_of_type(self, node_type: NodeType) -> Tuple[str, ...]:
        return tuple(sorted(
            node_id for node_id, node in self._nodes.items()
            if node.node_type is node_type
        ))

    def counts(self) -> Dict[str, int]:
        counts = {node_type.value: 0 for node_type in NodeType}
        for node in self._nodes.values():
            counts[node.node_type.value] += 1
        edge_counts = {edge_type.value: 0 for edge_type in EdgeType}
        for edge in self._edges.values():
            edge_counts[edge.edge_type.value] += 1
        return {
            **{f"nodes_{key}": value for key, value in counts.items()},
            **{f"edges_{key}": value for key, value in edge_counts.items()},
            "nodes_total": len(self._nodes),
            "edges_total": len(self._edges),
        }

    @property
    def graph_sha256(self) -> str:
        """Content hash over the whole frozen graph; this is lineage, not truth."""
        return _canonical_sha256({
            "schema": GRAPH_SCHEMA_VERSION,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "display_name": node.display_name,
                    "attributes": {
                        key: node.attributes[key] for key in sorted(node.attributes)
                    },
                }
                for node in sorted(self._nodes.values(), key=lambda item: item.node_id)
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type.value,
                    "basis": edge.basis,
                    "basis_value": edge.basis_value,
                }
                for edge in sorted(self._edges.values(), key=lambda item: item.edge_id)
            ],
        })

    def _neighbours(self, node_id: str, edge_type: EdgeType, *, reverse: bool) -> List[Tuple[str, str]]:
        adjacency = self._in if reverse else self._out
        result: List[Tuple[str, str]] = []
        for edge_id in adjacency.get(node_id, ()):
            edge = self._edges[edge_id]
            if edge.edge_type is not edge_type:
                continue
            result.append((edge.source_id if reverse else edge.target_id, edge_id))
        return result

    def lineage_path(self, evidence_id: str, seed_concept_id: str) -> GraphPath:
        """Return the concept -> evidence -> source -> version triple path."""
        node = self._nodes.get(evidence_id)
        if node is None or node.node_type is not NodeType.EVIDENCE:
            raise GraphBuildError(f"unknown evidence node: {evidence_id}")
        definition = [
            (target, edge_id)
            for target, edge_id in self._neighbours(
                evidence_id, EdgeType.THE_DEFINITION_OF, reverse=False,
            )
            if target == seed_concept_id
        ]
        sources = self._neighbours(evidence_id, EdgeType.THE_REFERENCE_OF, reverse=False)
        if len(sources) != 1:
            raise GraphBuildError(
                f"evidence {evidence_id} must reference exactly one source"
            )
        source_id, source_edge = sources[0]
        versions = self._neighbours(source_id, EdgeType.HAS_VERSION, reverse=False)
        if len(versions) != 1:
            raise GraphBuildError(f"source {source_id} must bind exactly one version")
        version_id, version_edge = versions[0]
        # A bridged evidence node is reached through another concept, so the seed
        # definition edge may be absent.  Record the path honestly either way.
        head_nodes: Tuple[str, ...] = (seed_concept_id,) if definition else ()
        head_edges: Tuple[str, ...] = (definition[0][1],) if definition else ()
        return GraphPath(
            evidence_id=evidence_id,
            node_ids=(*head_nodes, evidence_id, source_id, version_id),
            edge_ids=(*head_edges, source_edge, version_edge),
            hops=len(head_edges) + 2,
            source_id=str(self._nodes[source_id].attributes.get("source_id") or ""),
            source_version=str(self._nodes[version_id].attributes.get("source_version") or ""),
            content_sha256=str(node.attributes.get("content_sha256") or ""),
            record_sha256=str(node.attributes.get("record_sha256") or ""),
        )

    def _routed_lineage_path(
        self,
        evidence_id: str,
        predecessors: Mapping[str, Tuple[str, str]],
    ) -> GraphPath:
        """Bind the actual BFS route to the evidence node's frozen lineage.

        Definition edges are stored evidence->concept, while traversal crosses
        them in either direction. ``node_ids`` records traversal order and
        ``edge_ids`` records the real typed edge used at each hop. This keeps
        multi-hop paths connected without fabricating a direct seed edge.
        """
        node = self._nodes.get(evidence_id)
        if node is None or node.node_type is not NodeType.EVIDENCE:
            raise GraphBuildError(f"unknown evidence node: {evidence_id}")

        route_nodes = [evidence_id]
        route_edges: List[str] = []
        cursor = evidence_id
        visited = {cursor}
        while cursor in predecessors:
            previous, edge_id = predecessors[cursor]
            if previous in visited:
                raise GraphBuildError("graph traversal predecessor cycle")
            route_edges.append(edge_id)
            route_nodes.append(previous)
            visited.add(previous)
            cursor = previous
        route_nodes.reverse()
        route_edges.reverse()
        if not route_nodes or self._nodes[route_nodes[0]].node_type is not NodeType.CONCEPT:
            raise GraphBuildError(
                f"evidence {evidence_id} has no connected route from a seed concept"
            )

        sources = self._neighbours(evidence_id, EdgeType.THE_REFERENCE_OF, reverse=False)
        if len(sources) != 1:
            raise GraphBuildError(
                f"evidence {evidence_id} must reference exactly one source"
            )
        source_id, source_edge = sources[0]
        versions = self._neighbours(source_id, EdgeType.HAS_VERSION, reverse=False)
        if len(versions) != 1:
            raise GraphBuildError(f"source {source_id} must bind exactly one version")
        version_id, version_edge = versions[0]
        node_ids = (*route_nodes, source_id, version_id)
        edge_ids = (*route_edges, source_edge, version_edge)
        return GraphPath(
            evidence_id=evidence_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            hops=len(edge_ids),
            source_id=str(self._nodes[source_id].attributes.get("source_id") or ""),
            source_version=str(self._nodes[version_id].attributes.get("source_version") or ""),
            content_sha256=str(node.attributes.get("content_sha256") or ""),
            record_sha256=str(node.attributes.get("record_sha256") or ""),
        )

    def traverse(
        self,
        *,
        seed_concept_ids: Sequence[str],
        similarity: Mapping[str, float],
        depth: int = 1,
        node_limit: int = 20,
        final_k: int = 5,
    ) -> GraphRetrieval:
        """Breadth-first bounded traversal of the concept<->evidence bipartite tier.

        ``depth`` counts bipartite hops from the seed concept: hop 1 reaches the
        seed concept's evidence, hop 2 reaches concepts co-linked to that
        evidence, hop 3 reaches their evidence, and so on.  Only odd hops land on
        evidence.  Every returned evidence node is then completed with its
        mandatory ``evidence -> source -> version`` lineage, which is not a
        traversal knob.
        """
        if depth < 1:
            raise GraphBuildError("traversal depth must be at least one hop")
        if node_limit < 1 or final_k < 1 or final_k > node_limit:
            raise GraphBuildError("require 0 < final_k <= node_limit")
        seeds = tuple(dict.fromkeys(str(value) for value in seed_concept_ids))
        if not seeds:
            raise GraphBuildError("graph traversal requires at least one seed concept")
        for seed in seeds:
            node = self._nodes.get(seed)
            if node is None or node.node_type is not NodeType.CONCEPT:
                raise GraphBuildError(f"unknown seed concept node: {seed}")

        reached_evidence: Dict[str, int] = {}
        reached_concepts: Dict[str, int] = {seed: 0 for seed in seeds}
        predecessors: Dict[str, Tuple[str, str]] = {}
        frontier: List[str] = list(seeds)
        frontier_sizes: List[int] = []
        for hop in range(1, depth + 1):
            on_concepts = hop % 2 == 1  # odd hop expands concept -> evidence
            next_frontier: List[str] = []
            for node_id in frontier:
                if on_concepts:
                    neighbours = self._neighbours(
                        node_id, EdgeType.THE_DEFINITION_OF, reverse=True,
                    )
                    for evidence_id, _edge_id in neighbours:
                        if evidence_id not in reached_evidence:
                            reached_evidence[evidence_id] = hop
                            predecessors[evidence_id] = (node_id, _edge_id)
                            next_frontier.append(evidence_id)
                else:
                    neighbours = self._neighbours(
                        node_id, EdgeType.THE_DEFINITION_OF, reverse=False,
                    )
                    for concept_id, _edge_id in neighbours:
                        if concept_id not in reached_concepts:
                            reached_concepts[concept_id] = hop
                            predecessors[concept_id] = (node_id, _edge_id)
                            next_frontier.append(concept_id)
            next_frontier.sort()
            frontier_sizes.append(len(next_frontier))
            frontier = next_frontier
            if not frontier:
                break

        missing = [item for item in reached_evidence if item not in similarity]
        if missing:
            raise GraphBuildError(
                f"traversal reached {len(missing)} evidence node(s) without a similarity score"
            )
        for value in similarity.values():
            if not math.isfinite(float(value)):
                raise GraphBuildError("graph ranking received a non-finite similarity")

        # Paper 133 ranks the retrieved entities by query/entity content
        # similarity.  Ties resolve on the stable node id so the traversal is
        # reproducible byte-for-byte.
        ranked = sorted(
            reached_evidence,
            key=lambda evidence_id: (-float(similarity[evidence_id]), evidence_id),
        )
        candidates = tuple(ranked[:node_limit])
        returned = tuple(candidates[:final_k])
        # Paths cover the whole bounded subgraph, not just the returned prefix:
        # the sufficiency dimensions are defined over the retrieved subgraph.
        paths = tuple(
            self._routed_lineage_path(evidence_id, predecessors)
            for evidence_id in candidates
        )

        subgraph_nodes: List[str] = list(seeds)
        subgraph_edges: List[str] = []
        for path in paths:
            subgraph_nodes.extend(path.node_ids)
            subgraph_edges.extend(path.edge_ids)
        node_ids = tuple(sorted(set(subgraph_nodes)))
        attribute_counts = [
            self._nodes[node_id].populated_attribute_count for node_id in node_ids
        ]
        return GraphRetrieval(
            seed_concept_ids=seeds,
            depth=depth,
            node_limit=node_limit,
            candidate_evidence_ids=candidates,
            returned_evidence_ids=returned,
            paths=paths,
            subgraph_node_ids=node_ids,
            subgraph_edge_ids=tuple(sorted(set(subgraph_edges))),
            reached_evidence_count=len(reached_evidence),
            frontier_sizes=tuple(frontier_sizes),
            mean_populated_attributes=(
                sum(attribute_counts) / len(attribute_counts) if attribute_counts else 0.0
            ),
        )


def is_concept_connected(
    record: Mapping[str, Any],
    committed_diagnosis: str,
    concept_links: Mapping[str, Mapping[str, float]],
) -> bool:
    """Whether a fused record is graph-linked to the committed-diagnosis concept.

    A record is connected when its content embeds within delta_r of the committed
    concept, i.e. it carries a ``the_definition_of`` edge to that concept in the
    frozen graph.  Image-exemplar records have no text-corpus embedding and no
    ``document_id`` in ``concept_links``, so they are never connected -- which is
    correct: the concept tier is a text-content relation.
    """
    document_id = str(record.get("document_id") or record.get("id") or "")
    if not document_id:
        return False
    return str(committed_diagnosis) in concept_links.get(document_id, {})


def select_evidence_by_graph_connectivity(
    records: Sequence[Mapping[str, Any]],
    *,
    committed_diagnosis: str,
    concept_links: Mapping[str, Mapping[str, float]],
    final_k: int,
) -> List[Mapping[str, Any]]:
    """Graph-connectivity evidence selection for the MedGuard method arm (A2 fix).

    This is what makes the evidence graph *selection-bearing* rather than pure
    lineage annotation.  Records graph-connected to the committed-diagnosis concept
    are promoted ahead of unconnected ones, preserving the fusion order within each
    group (a stable partition), and the top ``final_k`` are returned.  Because a
    connected record ranked below ``final_k`` by fusion can now displace an
    unconnected record ranked above it, this can change *which* records enter the
    answer set, not merely their order.  Disabling the graph (the A2 ablation) skips
    this and returns the raw fusion prefix, so the two evidence sets genuinely
    differ whenever the connected/unconnected boundary crosses ``final_k``.

    The only signal used is the frozen delta_r=0.5 concept edge -- a disclosed,
    embedding-based index relation, never tuned on development outcomes and never a
    truth claim about a record's content.
    """
    if final_k < 0:
        raise GraphBuildError("final_k must be non-negative")
    order = sorted(
        range(len(records)),
        key=lambda index: (
            0 if is_concept_connected(
                records[index], committed_diagnosis, concept_links,
            ) else 1,
            index,
        ),
    )
    return [records[index] for index in order[:final_k]]


def aggregate_context_text(
    graph: "EvidenceLineageGraph", retrieval: GraphRetrieval,
) -> str:
    """Compile the retrieved subgraph into one structured textual context.

    Paper 104's Context Integrator "compiles the retrieved nodes, relations, and
    path information into a structured textual context", and its semantic
    relevance is the cosine between the query and that aggregated context.  The
    rendering is therefore over node display names and typed relations -- the
    same compact graph serialization the source frameworks hand to a generator --
    and never over the raw record bodies.

    Mean-pooling the retrieved records' vectors would be a different and
    materially easier measure: averaging vectors that were themselves selected by
    query similarity cancels their orthogonal components and inflates the cosine
    above any individual record's.
    """
    lines: List[str] = []
    for concept_id in retrieval.seed_concept_ids:
        node = graph.nodes[concept_id]
        lines.append(f"{node.node_type.value}: {node.display_name}")
    for path in retrieval.paths:
        for edge_id in path.edge_ids:
            edge = graph.edges[edge_id]
            source = graph.nodes[edge.source_id].display_name
            target = graph.nodes[edge.target_id].display_name
            lines.append(f"{source} {edge.edge_type.value} {target}")
    return "\n".join(lines)


def concept_content(code: str, canonical_term: str) -> str:
    """Build one concept's entity content in paper 133's Ce format."""
    return CONCEPT_CONTENT_TEMPLATE.format(
        name=_require_text("concept code", code),
        type=CONCEPT_TYPE,
        context=_require_text("canonical term", canonical_term),
    )


def _evidence_attributes(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": str(record.get("document_id") or ""),
        "citation_id": str(record.get("citation_id") or ""),
        "source_id": str(record.get("source_id") or ""),
        "source_version": str(record.get("source_version") or ""),
        "authority": str(record.get("authority_class") or record.get("authority") or ""),
        "provenance_uri": str(record.get("provenance_uri") or ""),
        "content_sha256": str(record.get("content_sha256") or ""),
        "record_sha256": str(record.get("record_sha256") or ""),
    }


def link_concepts_by_embedding(
    *,
    records: Sequence[Mapping[str, Any]],
    concept_terms: Mapping[str, str],
    concept_vectors: Sequence[Sequence[float]],
    evidence_vectors: Sequence[Sequence[float]],
    delta_r: float = DELTA_R,
) -> Dict[str, Dict[str, float]]:
    """Compute the frozen concept<->evidence edge set at paper 133's threshold.

    Paper 133 links tiers when the cosine similarity of their content embeddings
    is at or above a pre-defined threshold delta_r, and discloses 0.5 as the
    default.  This is an *index* relation over frozen vectors; it establishes no
    clinical entailment and no arm may read truth from it.

    ``concept_vectors`` and ``evidence_vectors`` must be unit-norm embeddings of
    the concept contents and the corpus record texts under the same frozen text
    model, ordered like ``concept_terms`` and ``records``.
    """
    import numpy as np

    codes = list(concept_terms)
    if not codes:
        raise GraphBuildError("the controlled vocabulary must not be empty")
    if not records:
        raise GraphBuildError("the frozen corpus must not be empty")
    if not math.isfinite(float(delta_r)) or not -1.0 <= float(delta_r) <= 1.0:
        raise GraphBuildError("delta_r must be a finite cosine threshold in [-1, 1]")
    concepts = np.asarray(concept_vectors, dtype="float64")
    evidence = np.asarray(evidence_vectors, dtype="float64")
    if concepts.ndim != 2 or evidence.ndim != 2:
        raise GraphBuildError("graph vectors must be two-dimensional")
    if concepts.shape[0] != len(codes) or evidence.shape[0] != len(records):
        raise GraphBuildError("graph vectors do not align with their entities")
    if concepts.shape[1] != evidence.shape[1]:
        raise GraphBuildError("concept and evidence vectors have different dimensions")
    if not np.isfinite(concepts).all() or not np.isfinite(evidence).all():
        raise GraphBuildError("graph vectors must be finite")
    for name, matrix in (("concept", concepts), ("evidence", evidence)):
        norms = np.linalg.norm(matrix, axis=1)
        if float(np.abs(norms - 1.0).max()) > 1e-3:
            raise GraphBuildError(f"{name} vectors must be unit-norm for cosine linking")

    similarity = evidence @ concepts.T
    links: Dict[str, Dict[str, float]] = {}
    for evidence_index, record in enumerate(records):
        document_id = _require_text("document_id", record.get("document_id"))
        if document_id in links:
            raise GraphBuildError(f"duplicate corpus document_id: {document_id}")
        matched = {
            code: round(float(similarity[evidence_index, concept_index]), 9)
            for concept_index, code in enumerate(codes)
            if float(similarity[evidence_index, concept_index]) >= float(delta_r)
        }
        links[document_id] = matched
    return links


def build_evidence_lineage_graph(
    *,
    records: Sequence[Mapping[str, Any]],
    concept_terms: Mapping[str, str],
    concept_links: Mapping[str, Mapping[str, float]],
    delta_r: float = DELTA_R,
) -> EvidenceLineageGraph:
    """Assemble the four-tier graph from receipt-bound records and frozen links.

    ``concept_links`` maps a record's ``document_id`` to the concept codes it is
    linked to and the cosine that created each link.  Assembly is deliberately
    separated from the similarity rule so the frozen corpus graph and a per-case
    lineage view are built by exactly one code path.  A record with no concept
    link (an image exemplar, which carries no text-corpus embedding) still
    receives its full evidence -> source -> version lineage.
    """
    codes = list(concept_terms)
    if not codes:
        raise GraphBuildError("the controlled vocabulary must not be empty")
    if not records:
        raise GraphBuildError("the record set must not be empty")
    unknown = {
        code
        for links in concept_links.values()
        for code in links
        if code not in concept_terms
    }
    if unknown:
        raise GraphBuildError(f"concept links reference unknown concepts: {sorted(unknown)}")

    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    for index, code in enumerate(codes):
        term = _require_text(f"concept term for {code}", concept_terms[code])
        nodes.append(GraphNode(
            node_id=f"concept:{code}",
            node_type=NodeType.CONCEPT,
            display_name=term,
            attributes={
                "concept_code": code,
                "concept_type": CONCEPT_TYPE,
                "canonical_term": term,
                "content": concept_content(code, term),
            },
        ))

    seen_sources: Dict[str, str] = {}
    seen_versions: set[str] = set()
    evidence_ids: List[str] = []
    document_ids: List[str] = []
    for record in records:
        attributes = _evidence_attributes(record)
        document_id = _require_text("document_id", attributes["document_id"])
        source_id = _require_text("source_id", attributes["source_id"])
        source_version = _require_text("source_version", attributes["source_version"])
        for field in ("citation_id", "authority", "provenance_uri", "content_sha256", "record_sha256"):
            _require_text(f"{document_id}.{field}", attributes[field])
        # Duplicate document_ids surface as a duplicate node in the constructor.
        evidence_id = f"evidence:{document_id}"
        evidence_ids.append(evidence_id)
        document_ids.append(document_id)
        nodes.append(GraphNode(
            node_id=evidence_id,
            node_type=NodeType.EVIDENCE,
            display_name=str(record.get("title") or document_id),
            attributes=attributes,
        ))
        source_node = f"source:{source_id}"
        if source_node not in seen_sources:
            seen_sources[source_node] = source_id
            nodes.append(GraphNode(
                node_id=source_node,
                node_type=NodeType.SOURCE,
                display_name=source_id,
                attributes={
                    "source_id": source_id,
                    "citation_id": attributes["citation_id"],
                    "authority": attributes["authority"],
                    "provenance_uri": attributes["provenance_uri"],
                },
            ))
        version_node = f"version:{source_id}@{source_version}"
        if version_node not in seen_versions:
            seen_versions.add(version_node)
            nodes.append(GraphNode(
                node_id=version_node,
                node_type=NodeType.VERSION,
                display_name=source_version,
                attributes={"source_id": source_id, "source_version": source_version},
            ))
            edges.append(GraphEdge(
                edge_id=f"{EdgeType.HAS_VERSION.value}:{source_node}->{version_node}",
                source_id=source_node,
                target_id=version_node,
                edge_type=EdgeType.HAS_VERSION,
                basis="frozen_corpus_source_version_field",
            ))
        edges.append(GraphEdge(
            edge_id=f"{EdgeType.THE_REFERENCE_OF.value}:{evidence_id}->{source_node}",
            source_id=evidence_id,
            target_id=source_node,
            edge_type=EdgeType.THE_REFERENCE_OF,
            basis="frozen_corpus_source_identity_field",
        ))

    for document_id, evidence_id in zip(document_ids, evidence_ids):
        for code in sorted(concept_links.get(document_id, {})):
            concept_node = f"concept:{code}"
            edges.append(GraphEdge(
                edge_id=f"{EdgeType.THE_DEFINITION_OF.value}:{evidence_id}->{concept_node}",
                source_id=evidence_id,
                target_id=concept_node,
                edge_type=EdgeType.THE_DEFINITION_OF,
                basis=f"frozen_text_embedding_cosine_at_or_above_delta_r_{delta_r}",
                basis_value=float(concept_links[document_id][code]),
            ))
    return EvidenceLineageGraph(nodes, edges)


__all__ = [
    "CONCEPT_CONTENT_TEMPLATE",
    "CONCEPT_TYPE",
    "DELTA_R",
    "GRAPH_SCHEMA_VERSION",
    "EdgeType",
    "EvidenceLineageGraph",
    "GraphBuildError",
    "GraphEdge",
    "GraphNode",
    "GraphPath",
    "GraphRetrieval",
    "NodeType",
    "aggregate_context_text",
    "build_evidence_lineage_graph",
    "concept_content",
    "is_concept_connected",
    "link_concepts_by_embedding",
    "select_evidence_by_graph_connectivity",
]
