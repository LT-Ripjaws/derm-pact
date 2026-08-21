import importlib.util
import tempfile
from pathlib import Path
from unittest import SkipTest

from medguard.eval.dynarag_gmm_component_audit import audit, run


def _require_sklearn():
    if importlib.util.find_spec("sklearn") is None:
        raise SkipTest("scikit-learn is absent from the bundled runtime")


def test_component_audit_exposes_score_only_security_failure():
    _require_sklearn()
    result = audit()
    assert result["implementation_checks_passed"]
    assert not result["robustness_gate_passed"]
    assert result["decision"] == (
        "retain_only_as_clean_retrieval_ablation_not_security_controller"
    )
    rows = {row["id"]: row for row in result["scenarios"]}
    assert rows["single_high_score_poison"]["adaptive"]["selected_indices"] == [0]
    assert rows["high_score_false_consensus"]["adaptive"]["selected_indices"] == [0, 1, 2]


def test_component_audit_freezes_manifest_and_result():
    _require_sklearn()
    with tempfile.TemporaryDirectory() as tmp:
        result = run(Path(tmp))
        assert len(result["protocol_manifest_sha256"]) == 64
        assert Path(tmp, "protocol_manifest.json").exists()
        assert Path(tmp, "result.json").exists()
