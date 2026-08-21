"""Focused tests for the B5 evidence-lineage graph.

These use small injected vectors so the frozen model and index are not required.
"""
from __future__ import annotations

import math

import pytest

from medguard.eval.h2h_evidence_graph import (
    DELTA_R,
    EdgeType,
    EvidenceLineageGraph,
    GraphBuildError,
    GraphEdge,
    GraphNode,
    NodeType,
    aggregate_context_text,
    build_evidence_lineage_graph,
    concept_content,
    is_concept_connected,
    link_concepts_by_embedding,
    select_evidence_by_graph_connectivity,
)


CONCEPTS = {"melanoma": "cutaneous melanoma", "nevus": "melanocytic nevus"}


def _record(index: int, *, source: str | None = None, **overrides):
    source_id = source or f"PMID:{index}"
    record = {
        "document_id": f"pubmed:{index}:abstract",
        "citation_id": source_id,
        "source_id": source_id,
        "source_version": "pubmed-snapshot-2026-07-17",
        "authority_class": "peer_reviewed_biomedical_literature",
        "provenance_uri": f"https://example.invalid/{index}",
        "content_sha256": f"{index:064d}",
        "record_sha256": f"{index + 500:064d}",
        "title": f"record {index} title",
    }
    record.update(overrides)
    return record


def _unit(*values):
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _graph(links=None):
    records = [_record(1), _record(2), _record(3)]
    links = links if links is not None else {
        "pubmed:1:abstract": {"melanoma": 0.9},
        "pubmed:2:abstract": {"melanoma": 0.7, "nevus": 0.6},
        "pubmed:3:abstract": {"nevus": 0.8},
    }
    return records, build_evidence_lineage_graph(
        records=records, concept_terms=CONCEPTS, concept_links=links,
    )


def test_graph_has_four_real_tiers_and_is_not_a_flat_provenance_list():
    _records, graph = _graph()
    counts = graph.counts()
    assert counts["nodes_concept"] == 2
    assert counts["nodes_evidence"] == 3
    assert counts["nodes_source"] == 3
    assert counts["nodes_version"] == 3
    # Four populated node tiers and three typed relations are what separates this
    # from a flat provenance list.
    assert counts[f"edges_{EdgeType.THE_DEFINITION_OF.value}"] == 4
    assert counts[f"edges_{EdgeType.THE_REFERENCE_OF.value}"] == 3
    assert counts[f"edges_{EdgeType.HAS_VERSION.value}"] == 3
    assert len({node.node_type for node in graph.nodes.values()}) == 4


def test_delta_r_linking_uses_the_source_paper_threshold_and_content_format():
    assert DELTA_R == 0.5
    assert concept_content("melanoma", "cutaneous melanoma") == (
        "name: melanoma; type: ham10000_lesion_class; context: cutaneous melanoma"
    )
    records = [_record(1), _record(2)]
    # Concept 0 matches evidence 0 exactly; evidence 1 is orthogonal to it.
    concept_vectors = [_unit(1.0, 0.0), _unit(0.0, 1.0)]
    evidence_vectors = [_unit(1.0, 0.0), _unit(0.6, 0.8)]
    links = link_concepts_by_embedding(
        records=records, concept_terms=CONCEPTS,
        concept_vectors=concept_vectors, evidence_vectors=evidence_vectors,
    )
    assert links["pubmed:1:abstract"] == {"melanoma": 1.0}
    # 0.6 is above 0.5 for melanoma and 0.8 is above 0.5 for nevus.
    assert sorted(links["pubmed:2:abstract"]) == ["melanoma", "nevus"]
    assert links["pubmed:2:abstract"]["melanoma"] == pytest.approx(0.6)


def test_link_threshold_is_exclusive_of_values_below_delta_r():
    records = [_record(1)]
    links = link_concepts_by_embedding(
        records=records, concept_terms={"melanoma": "cutaneous melanoma"},
        concept_vectors=[_unit(1.0, 0.0)], evidence_vectors=[_unit(0.49, 1.0)],
        delta_r=0.5,
    )
    assert links["pubmed:1:abstract"] == {}


def test_non_unit_or_misaligned_vectors_fail_closed():
    with pytest.raises(GraphBuildError, match="unit-norm"):
        link_concepts_by_embedding(
            records=[_record(1)], concept_terms={"melanoma": "cutaneous melanoma"},
            concept_vectors=[[3.0, 0.0]], evidence_vectors=[_unit(1.0, 0.0)],
        )
    with pytest.raises(GraphBuildError, match="do not align"):
        link_concepts_by_embedding(
            records=[_record(1), _record(2)],
            concept_terms={"melanoma": "cutaneous melanoma"},
            concept_vectors=[_unit(1.0, 0.0)], evidence_vectors=[_unit(1.0, 0.0)],
        )


def test_evidence_without_complete_lineage_cannot_enter_the_graph():
    for field in ("source_id", "source_version", "provenance_uri", "record_sha256"):
        broken = _record(1)
        broken[field] = ""
        with pytest.raises(GraphBuildError):
            build_evidence_lineage_graph(
                records=[broken], concept_terms=CONCEPTS,
                concept_links={"pubmed:1:abstract": {}},
            )


def test_edges_must_respect_their_tier_contract():
    nodes = [
        GraphNode("concept:melanoma", NodeType.CONCEPT, "cutaneous melanoma", {}),
        GraphNode("source:PMID:1", NodeType.SOURCE, "PMID:1", {}),
    ]
    bad = GraphEdge(
        edge_id="e1", source_id="source:PMID:1", target_id="concept:melanoma",
        edge_type=EdgeType.THE_DEFINITION_OF, basis="test",
    )
    with pytest.raises(GraphBuildError, match="tier contract"):
        EvidenceLineageGraph(nodes, [bad])


def test_unknown_concept_links_and_unknown_seeds_fail_closed():
    with pytest.raises(GraphBuildError, match="unknown concepts"):
        build_evidence_lineage_graph(
            records=[_record(1)], concept_terms=CONCEPTS,
            concept_links={"pubmed:1:abstract": {"not_a_concept": 0.9}},
        )
    _records, graph = _graph()
    with pytest.raises(GraphBuildError, match="unknown seed concept"):
        graph.traverse(
            seed_concept_ids=["concept:missing"], similarity={}, depth=1, node_limit=2,
            final_k=1,
        )


def test_traversal_is_bounded_ranked_and_returns_auditable_triple_paths():
    _records, graph = _graph()
    similarity = {
        "evidence:pubmed:1:abstract": 0.4,
        "evidence:pubmed:2:abstract": 0.9,
        "evidence:pubmed:3:abstract": 0.8,
    }
    result = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=1, node_limit=2, final_k=1,
    )
    # Depth 1 reaches only the seed concept's own evidence, ranked by similarity.
    assert result.reached_evidence_count == 2
    assert result.candidate_evidence_ids == (
        "evidence:pubmed:2:abstract", "evidence:pubmed:1:abstract",
    )
    assert result.returned_evidence_ids == ("evidence:pubmed:2:abstract",)
    path = result.paths_by_evidence["evidence:pubmed:2:abstract"]
    assert path.node_ids == (
        "concept:melanoma", "evidence:pubmed:2:abstract",
        "source:PMID:2", "version:PMID:2@pubmed-snapshot-2026-07-17",
    )
    assert path.hops == 3
    assert path.source_id == "PMID:2"
    assert path.source_version == "pubmed-snapshot-2026-07-17"
    assert path.content_sha256 and path.record_sha256


def test_depth_bridges_across_the_concept_tier():
    _records, graph = _graph()
    similarity = {
        "evidence:pubmed:1:abstract": 0.4,
        "evidence:pubmed:2:abstract": 0.9,
        "evidence:pubmed:3:abstract": 0.8,
    }
    shallow = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=1, node_limit=5, final_k=1,
    )
    deep = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=3, node_limit=5, final_k=1,
    )
    # record 3 is reachable only by bridging melanoma -> record 2 -> nevus.
    assert shallow.reached_evidence_count == 2
    assert deep.reached_evidence_count == 3
    assert "evidence:pubmed:3:abstract" in deep.candidate_evidence_ids
    assert "evidence:pubmed:3:abstract" not in shallow.candidate_evidence_ids


def test_bridged_evidence_records_the_connected_route_without_fabricating_an_edge():
    _records, graph = _graph()
    similarity = {
        "evidence:pubmed:1:abstract": 0.1,
        "evidence:pubmed:2:abstract": 0.2,
        "evidence:pubmed:3:abstract": 0.9,
    }
    deep = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=3, node_limit=5, final_k=1,
    )
    path = deep.paths_by_evidence["evidence:pubmed:3:abstract"]
    assert path.node_ids == (
        "concept:melanoma",
        "evidence:pubmed:2:abstract",
        "concept:nevus",
        "evidence:pubmed:3:abstract",
        "source:PMID:3",
        "version:PMID:3@pubmed-snapshot-2026-07-17",
    )
    assert path.hops == 5
    assert len(path.edge_ids) == path.hops
    assert (
        "the_definition_of:evidence:pubmed:3:abstract->concept:melanoma"
        not in path.edge_ids
    )
    assert path.source_id == "PMID:3"
    assert path.source_version == "pubmed-snapshot-2026-07-17"


def test_every_traversal_path_is_edge_connected_in_node_order():
    _records, graph = _graph()
    similarity = {
        "evidence:pubmed:1:abstract": 0.1,
        "evidence:pubmed:2:abstract": 0.2,
        "evidence:pubmed:3:abstract": 0.9,
    }
    result = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=3, node_limit=5, final_k=1,
    )
    for path in result.paths:
        assert path.node_ids[0] in result.seed_concept_ids
        assert len(path.node_ids) == len(path.edge_ids) + 1
        for left, right, edge_id in zip(
            path.node_ids, path.node_ids[1:], path.edge_ids,
        ):
            edge = graph.edges[edge_id]
            assert {left, right} == {edge.source_id, edge.target_id}


def test_missing_similarity_or_bad_bounds_fail_closed():
    _records, graph = _graph()
    with pytest.raises(GraphBuildError, match="without a similarity score"):
        graph.traverse(
            seed_concept_ids=["concept:melanoma"], similarity={}, depth=1,
            node_limit=2, final_k=1,
        )
    similarity = {f"evidence:pubmed:{i}:abstract": 0.5 for i in (1, 2, 3)}
    with pytest.raises(GraphBuildError, match="final_k <= node_limit"):
        graph.traverse(
            seed_concept_ids=["concept:melanoma"], similarity=similarity, depth=1,
            node_limit=1, final_k=5,
        )
    with pytest.raises(GraphBuildError, match="depth must be at least one hop"):
        graph.traverse(
            seed_concept_ids=["concept:melanoma"], similarity=similarity, depth=0,
            node_limit=2, final_k=1,
        )
    with pytest.raises(GraphBuildError, match="non-finite"):
        graph.traverse(
            seed_concept_ids=["concept:melanoma"],
            similarity={**similarity, "evidence:pubmed:1:abstract": float("nan")},
            depth=1, node_limit=2, final_k=1,
        )


def test_graph_hash_is_deterministic_and_content_sensitive():
    _records, first = _graph()
    _records, second = _graph()
    assert first.graph_sha256 == second.graph_sha256
    _records, changed = _graph(links={
        "pubmed:1:abstract": {"melanoma": 0.9},
        "pubmed:2:abstract": {"melanoma": 0.7},
        "pubmed:3:abstract": {"nevus": 0.8},
    })
    assert changed.graph_sha256 != first.graph_sha256


def test_aggregate_context_renders_nodes_and_typed_relations():
    _records, graph = _graph()
    similarity = {f"evidence:pubmed:{i}:abstract": 1.0 / i for i in (1, 2, 3)}
    result = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=1, node_limit=2, final_k=1,
    )
    context = aggregate_context_text(graph, result)
    assert "concept: cutaneous melanoma" in context
    assert f"{EdgeType.THE_REFERENCE_OF.value} PMID:1" in context
    assert f"{EdgeType.HAS_VERSION.value} pubmed-snapshot-2026-07-17" in context
    # The context renders the whole bounded subgraph, not just the returned prefix.
    assert "record 2 title" in context
    assert aggregate_context_text(graph, result) == context


# --- A2: the graph is selection-bearing, not just lineage annotation -------------
_A2_LINKS = {
    "pubmed:connected": {"melanoma": 0.8},   # graph-linked to the commitment
    "pubmed:offconcept": {"nevus": 0.9},     # linked to a different concept only
}


def test_a2_is_concept_connected_reads_the_delta_r_edge():
    connected = {"document_id": "pubmed:connected"}
    off = {"document_id": "pubmed:offconcept"}
    exemplar = {"document_id": "", "id": "isic:1"}  # image exemplar, no text link
    assert is_concept_connected(connected, "melanoma", _A2_LINKS) is True
    assert is_concept_connected(off, "melanoma", _A2_LINKS) is False
    assert is_concept_connected(exemplar, "melanoma", _A2_LINKS) is False


def test_a2_graph_selection_changes_which_records_enter_the_answer_set():
    # Fusion order puts two off-concept records first and the concept-connected
    # record third. With final_k=2 the raw prefix (the A2 ablation) drops the
    # connected record; graph selection (the method arm) promotes it in.
    fused = [
        {"document_id": "pubmed:offconcept", "rank": 1},
        {"document_id": "pubmed:other", "rank": 2},
        {"document_id": "pubmed:connected", "rank": 3},
    ]
    method = select_evidence_by_graph_connectivity(
        fused, committed_diagnosis="melanoma", concept_links=_A2_LINKS, final_k=2,
    )
    ablation = fused[:2]  # A2: raw fusion prefix, no graph selection

    method_ids = [record["document_id"] for record in method]
    ablation_ids = [record["document_id"] for record in ablation]
    assert method_ids == ["pubmed:connected", "pubmed:offconcept"]
    assert ablation_ids == ["pubmed:offconcept", "pubmed:other"]
    # The evidence *set* differs, so disabling the graph (A2) measures something.
    assert set(method_ids) != set(ablation_ids)


def test_a2_selection_is_stable_within_each_connectivity_group():
    # All three records connected: order is preserved (a no-op) so selection never
    # reshuffles evidence it has no basis to reorder.
    links = {f"pubmed:{i}": {"melanoma": 0.7} for i in range(3)}
    fused = [{"document_id": f"pubmed:{i}"} for i in range(3)]
    selected = select_evidence_by_graph_connectivity(
        fused, committed_diagnosis="melanoma", concept_links=links, final_k=3,
    )
    assert [r["document_id"] for r in selected] == ["pubmed:0", "pubmed:1", "pubmed:2"]


def test_graph_never_exposes_a_truth_bearing_key():
    from medguard.eval.h2h_protocol import reject_truth_keys

    _records, graph = _graph()
    similarity = {f"evidence:pubmed:{i}:abstract": 1.0 / i for i in (1, 2, 3)}
    result = graph.traverse(
        seed_concept_ids=["concept:melanoma"], similarity=similarity,
        depth=1, node_limit=3, final_k=2,
    )
    reject_truth_keys(
        [path.to_dict() for path in result.paths], path="graph_paths",
    )
    reject_truth_keys(
        {node_id: dict(node.attributes) for node_id, node in graph.nodes.items()},
        path="graph_nodes",
    )
