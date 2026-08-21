from medguard.control_graph import (
    ClinicalControlGraph, Edge, EdgeType, GraphQuery, Node, NodeType,
    build_graph_pin, evaluate_graph, evaluate_pinned_graph, graph_sha256,
)
from medguard.control_graph.audit import GraphAuditRecord


def _graph():
    nodes = [Node("f1", NodeType.VISUAL_FINDING, "irregular border"),
             Node("d1", NodeType.DIAGNOSIS, "melanoma")]
    edges = [Edge("e1", "f1", "d1", EdgeType.SUGGESTIVE_OF,
                  "irregular border supports melanoma", "AAD guideline", "2026",
                  0.9, "clinician", 0.8, "aad")]
    return ClinicalControlGraph(nodes, edges)


def test_control_graph_emits_auditable_support_features():
    signals = evaluate_graph(_graph(), GraphQuery(["f1"], ["d1"]))
    assert signals.support_score == 0.72
    assert signals.path_coverage == 1.0 and signals.source_quorum == 1
    assert signals.supporting_paths[0][-1] == "e1"


def test_graph_hash_is_independent_of_insertion_order():
    graph = _graph()
    reordered = ClinicalControlGraph(reversed(list(graph.nodes.values())),
                                     reversed(list(graph.edges.values())))
    assert graph_sha256(graph) == graph_sha256(reordered)


def test_pinned_graph_fails_closed_after_mutation():
    graph = _graph()
    audits = [GraphAuditRecord("e1", "dermoscopy", "dermatologist", "D1",
                               "accept", "2026-07-13")]
    pin = build_graph_pin(graph, audits, artifact_id="fixture")
    graph.edges.pop("e1")
    signals = evaluate_pinned_graph(graph, GraphQuery(["f1"], ["d1"]), audits, pin)
    assert not signals.integrity_valid
    assert "graph_hash_mismatch" in signals.integrity_reasons
