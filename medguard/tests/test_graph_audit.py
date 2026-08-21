from medguard.control_graph.audit import GraphAuditRecord, audit_status, require_clinician_audit
from medguard.tests.test_control_graph import _graph


def test_graph_requires_modality_appropriate_clinician_acceptance():
    graph = _graph()
    accepted = [GraphAuditRecord("e1", "dermoscopy", "dermatologist", "D1",
                                 "accept", "2026-07-12")]
    require_clinician_audit(graph, accepted)
    assert not audit_status(graph, accepted)


def test_graph_audit_flags_missing_edges():
    assert audit_status(_graph(), [])["missing"] == ["e1"]
