"""Focused contracts for the Phase 2.4B materializer and CPU controller."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dermpact.eval import dermpact_phase2_4b_experiment as experiment


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = (
    ROOT / "research/scripts/dermpact_phase2_4b_full_run_controller.py"
)


@pytest.fixture(scope="module")
def controller():
    return runpy.run_path(str(CONTROLLER_PATH))


def test_fixed_matrix_is_the_exact_12_stage_cpu_contract(controller):
    matrix = controller["fixed_matrix"]()
    assert len(matrix) == 12
    assert matrix[0] == {
        "identity": "protocol", "stage": "protocol", "fold": None,
        "seed": 20260827,
    }
    assert [(row["stage"], row["fold"]) for row in matrix[1:6]] == [
        ("train", fold) for fold in range(5)
    ]
    assert [(row["stage"], row["fold"]) for row in matrix[6:11]] == [
        ("canonical_evaluate", fold) for fold in range(5)
    ]
    assert (matrix[-1]["stage"], matrix[-1]["fold"]) == ("finalize", None)
    serialized = json.dumps(matrix).lower()
    assert "official" not in serialized
    assert "final_dataset" not in serialized


def test_commands_use_only_the_frozen_cpu_python_and_bind_controller_source(
    controller, tmp_path
):
    matrix = controller["fixed_matrix"]()
    commands = [controller["_command"](row, tmp_path) for row in matrix]
    assert all(command[0] == str(controller["CPU_PYTHON"]) for command in commands)
    assert commands[0][4:6] == ["--protocol", "--controller-source"]
    assert commands[1][4:8] == ["--train-fold", "--fold", "0", "--output-dir"]
    assert commands[6][4:8] == [
        "--canonical-evaluate-fold", "--fold", "0", "--output-dir",
    ]
    assert "--finalize" in commands[-1]


def test_generated_protocol_freezes_science_conditions_and_zero_reads(monkeypatch):
    base = {
        "schema_version": experiment.training.PROTOCOL_SCHEMA,
        "status": "SPECIFIED_NOT_FROZEN",
        "identity": {"seed": 20260827, "folds": list(range(5)), "baseline_arm": "sharp"},
        "training": {"epochs": 20, "effective_batch_size": 64},
        "model": {
            "trainable_scope": "clinical_tail.adapters and cross_modal_fusion only"
        },
        "artifacts": {},
        "hard_scope": {
            "derm7pt_development_only": True,
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
        "protocol_sha256": "discarded",
        "signature_payload": {},
    }
    monkeypatch.setattr(
        experiment.training, "generate_protocol_spec", lambda **kwargs: dict(base)
    )
    payload = experiment.generate_protocol(controller_source=CONTROLLER_PATH)
    assert payload["status"] == "FROZEN"
    assert payload["identity"] == {
        "seed": 20260827, "folds": list(range(5)), "baseline_arm": "sharp",
    }
    assert payload["canonical_evaluation"]["conditions"] == list(
        experiment.evaluation.CONDITIONS
    )
    assert payload["canonical_evaluation"]["bootstrap_seed"] == 20260827
    assert payload["canonical_evaluation"]["wrong_pair"] == {
        "algorithm": "cyclic_next_available_within_validation_fold",
        "seed": 20260827,
        "random_draws": 0,
    }
    assert len(payload["orchestration"]["matrix"]) == 12
    assert payload["hard_scope"]["derm7pt_official_test_rows_loaded"] == 0
    sources = payload["artifacts"]["controller_sources"]
    assert {
        "materializer", "cli", "training", "evaluation", "paired_model",
        "cross_modal_fusion", "clinical_tail_adapter", "controller",
    }.issubset(sources)
    assert sources["controller"]["sha256"] == hashlib.sha256(
        CONTROLLER_PATH.read_bytes()
    ).hexdigest()


def test_atomic_json_requires_matching_sha_sidecar(tmp_path):
    path = tmp_path / "artifact.json"
    digest = experiment._atomic_json(path, {"status": "PASS"})
    assert experiment._verified_json(path) == ({"status": "PASS"}, digest)
    path.write_text('{"status":"CHANGED"}\n', encoding="utf-8")
    assert experiment._verified_json(path) is None


def test_status_is_atomic_and_status_never_authorizes_reuse(controller, tmp_path):
    state = controller["_new_state"](controller["fixed_matrix"]())
    controller["_write_status"](state, output_dir=tmp_path)
    compact = controller["read_status"](output_dir=tmp_path)
    assert compact["overall_status"] == "RUNNING"
    assert compact["completed_stage_identities"] == []
    status = tmp_path / "full_run_status.json"
    assert status.with_suffix(".json.sha256").read_text().strip() == hashlib.sha256(
        status.read_bytes()
    ).hexdigest()
    assert state["resume_contract"]["status_file"].startswith("informational only")


def test_all_verified_stages_are_reused_without_launching_subprocess(
    controller, tmp_path
):
    globals_ = controller["run"].__globals__
    original_core = globals_["_core"]
    original_subprocess = globals_["_run_subprocess"]
    globals_["_core"] = lambda: SimpleNamespace(
        verify_stage=lambda stage, fold, output_dir: [
            {"path": f"{stage}-{fold}", "sha256": "a" * 64}
        ]
    )
    globals_["_run_subprocess"] = lambda *args, **kwargs: pytest.fail(
        "verified stages must not execute"
    )
    try:
        state = controller["run"](output_dir=tmp_path)
    finally:
        globals_["_core"] = original_core
        globals_["_run_subprocess"] = original_subprocess
    assert state["overall_status"] == "PASS"
    assert state["completed_stage_identities"] == [
        row["identity"] for row in controller["fixed_matrix"]()
    ]
    assert {row["disposition"] for row in state["completed_stages"]} == {
        "REUSED_VERIFIED"
    }
    assert state["official_test_rows_loaded"] == 0
    assert state["final_dataset_rows_loaded"] == 0


def test_live_lock_is_refused_and_demonstrably_stale_lock_is_recovered(
    controller, tmp_path
):
    lock = controller["_controller_lock"]
    with lock(output_dir=tmp_path):
        with pytest.raises(RuntimeError, match="another Phase 2.4B controller"):
            with lock(output_dir=tmp_path):
                pass
    lock_path = tmp_path / ".controller.lock"
    lock_path.write_text(
        json.dumps({"pid": 2_147_483_647, "owner_token": "stale"}),
        encoding="utf-8",
    )
    with lock(output_dir=tmp_path):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert not lock_path.exists()
