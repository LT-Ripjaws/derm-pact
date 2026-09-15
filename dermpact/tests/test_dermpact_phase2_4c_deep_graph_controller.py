from __future__ import annotations

import importlib.util
from pathlib import Path

from dermpact.eval import dermpact_phase2_4c_deep_graph as deep
from dermpact.eval import dermpact_phase2_4c_deep_graph_experiment as experiment


CONTROLLER = (
    deep.ROOT / "research/scripts/dermpact_phase2_4c_deep_graph_controller.py"
)


def _controller_module():
    spec = importlib.util.spec_from_file_location("phase24c_controller_test", CONTROLLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_is_exactly_92_ordered_stages() -> None:
    module = _controller_module()
    jobs = module.matrix()
    assert len(jobs) == 92
    assert jobs[0]["stage"] == "protocol"
    assert [item["stage"] for item in jobs[1:46]] == ["train"] * 45
    assert [item["stage"] for item in jobs[46:91]] == ["canonical"] * 45
    assert jobs[-1]["stage"] == "finalize"
    assert len({item["identity"] for item in jobs}) == 92


def test_protocol_binds_deep_sources_and_promoted_evidence() -> None:
    protocol = experiment.generate_protocol(controller_source=CONTROLLER)
    assert protocol["status"] == "FROZEN"
    assert protocol["orchestration"]["stage_count"] == 92
    assert protocol["evidence"]["oof_only"] is True
    assert protocol["evidence"]["official_or_external_rows_loaded"] == 0
    assert protocol["model"]["module"] == "SemanticDirectedEvidenceGraph"
    assert protocol["selection"]["primary"] == "paired mean AUROC over melanoma plus seven criteria"
    assert set(protocol["source_files"]) == {
        "experiment", "training_core", "evaluation", "evidence_cache",
        "deep_graph", "fold_local_prior", "ranking_core", "bootstrap_core",
        "rule_evaluation", "label_definitions", "controller",
    }


def test_checkpoint_and_receipt_paths_are_arm_and_fold_scoped(tmp_path: Path) -> None:
    assert experiment.checkpoint_path("standard_k2", 3, tmp_path) != experiment.checkpoint_path("standard_k2", 4, tmp_path)
    assert experiment.training_receipt_path("standard_k2", 3, tmp_path).parent == experiment.checkpoint_path("standard_k2", 3, tmp_path).parent
    assert experiment.canonical_raw_path("standard_k2", 3, tmp_path).suffix == ".npz"
