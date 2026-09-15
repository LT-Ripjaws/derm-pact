"""Focused production-wrapper tests for the Phase 2.4A loss factor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dermpact.eval import dermpact_phase2_4a_loss_factor as factor
from dermpact.eval import dermpact_phase2_4a_loss_factor_core as core


CONTROLLER_PATH = (
    factor.ROOT
    / "research/scripts/dermpact_phase2_4a_loss_factor_full_run_controller.py"
)


def _load_controller():
    spec = importlib.util.spec_from_file_location("loss_factor_controller_test", CONTROLLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LossFactorWrapperTest(unittest.TestCase):
    def test_matrix_is_cpu_only_fixed_and_has_no_directml_stage(self):
        matrix = factor.controller_matrix()
        self.assertEqual(len(matrix), 12)
        self.assertEqual(matrix[0]["stage"], "protocol")
        self.assertEqual([row["fold"] for row in matrix if row["stage"] == "train"], list(core.FOLDS))
        self.assertEqual([row["fold"] for row in matrix if row["stage"] == "reevaluate"], list(core.FOLDS))
        self.assertTrue(all(row["environment"] == "venv_p2_cpu" for row in matrix))
        self.assertNotIn("directml", json.dumps(matrix).lower())

    def test_dry_run_inspection_performs_no_write(self):
        controller = _load_controller()
        with mock.patch.object(controller, "_atomic_status") as write:
            payload = controller.inspect_matrix()
        write.assert_not_called()
        self.assertFalse(payload["write_performed"])
        self.assertTrue(payload["cpu_only"])
        self.assertFalse(payload["directml_path"])
        self.assertTrue(all("venv_p2" in command[0] for command in payload["commands"]))

    def test_live_duplicate_controller_is_rejected_without_replacing_lock(self):
        controller = _load_controller()
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "controller.lock"
            original = {"pid": 43210, "started_at": "2026-08-30T01:02:03Z", "owner_token": "other"}
            lock.write_text(json.dumps(original), encoding="utf-8")
            with (
                mock.patch.object(controller, "LOCK_PATH", lock),
                mock.patch.object(controller, "_pid_is_live", return_value=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    controller._acquire_controller_lock()
            self.assertEqual(json.loads(lock.read_text()), original)
            self.assertEqual(list(Path(temporary).glob("*.stale.*")), [])

    def test_stale_lock_is_preserved_and_only_owned_lock_is_released(self):
        controller = _load_controller()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "controller.lock"
            stale = {"pid": 76543, "started_at": "2026-08-29T01:02:03Z", "owner_token": "dead"}
            lock.write_text(json.dumps(stale), encoding="utf-8")
            with (
                mock.patch.object(controller, "LOCK_PATH", lock),
                mock.patch.object(controller.factor, "ROOT", root),
                mock.patch.object(controller, "_pid_is_live", return_value=False),
            ):
                owner = controller._acquire_controller_lock()
                preserved = list(root.glob("controller.lock.stale.*"))
                self.assertEqual(len(preserved), 1)
                self.assertEqual(json.loads(preserved[0].read_text()), stale)
                self.assertTrue(lock.exists())
                controller._release_controller_lock({"owner_token": "not-owner"})
                self.assertTrue(lock.exists())
                controller._release_controller_lock(owner)
                self.assertFalse(lock.exists())
                self.assertTrue(preserved[0].exists())

    def test_stage_logs_are_signed_and_tamper_is_rejected(self):
        controller = _load_controller()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            stdout = logs / "probe.stdout.log"
            stderr = logs / "probe.stderr.log"
            with (
                mock.patch.object(controller, "ROOT", root),
                mock.patch.object(controller, "LOG_DIR", logs),
                mock.patch.object(controller.factor, "ROOT", root),
            ):
                execution = controller._run_subprocess(
                    [sys.executable, "-c", "print('ok')"], stdout, stderr
                )
                self.assertEqual(execution["return_code"], 0)
                self.assertEqual(controller._log_evidence("probe")["status"], "SIGNED_VERIFIED")
                stdout.write_bytes(stdout.read_bytes() + b"tampered")
                self.assertEqual(
                    controller._log_evidence("probe")["status"],
                    "INVALID_NOT_AUTHORIZING_REUSE",
                )

    def test_failed_stage_logs_are_also_signed(self):
        controller = _load_controller()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            with (
                mock.patch.object(controller, "ROOT", root),
                mock.patch.object(controller, "LOG_DIR", logs),
                mock.patch.object(controller.factor, "ROOT", root),
            ):
                execution = controller._run_subprocess(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    logs / "failed.stdout.log",
                    logs / "failed.stderr.log",
                )
                self.assertEqual(execution["return_code"], 7)
                self.assertEqual(
                    controller._log_evidence("failed")["status"],
                    "SIGNED_VERIFIED",
                )

    def test_direct_script_help_imports_without_package_path_setup(self):
        completed = subprocess.run(
            [sys.executable, str(CONTROLLER_PATH), "--help"],
            cwd=factor.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--dry-run", completed.stdout)

    def test_promotion_requires_positive_ci_and_original_safety_gates(self):
        paired = {
            name: {"estimate": 0.001, "ci_low": 0.0001, "ci_high": 0.002}
            for name in factor.METRICS
        }
        checks, decision = factor.promotion_decision(paired)
        self.assertTrue(all(checks.values()))
        self.assertEqual(decision, "PROMOTE_RANKING_LOSS")
        paired["criterion_mean_auc"]["ci_low"] = -0.0001
        self.assertEqual(factor.promotion_decision(paired)[1], "RETAIN_C10_EPOCH20")
        paired["criterion_mean_auc"]["ci_low"] = 0.0001
        paired["primary"]["estimate"] = -0.020001
        self.assertEqual(factor.promotion_decision(paired)[1], "RETAIN_C10_EPOCH20")
        paired["primary"]["estimate"] = 0.0
        paired["direct_rule_disagreement"]["estimate"] = 0.020001
        self.assertEqual(factor.promotion_decision(paired)[1], "RETAIN_C10_EPOCH20")

    def test_signed_json_rejects_tamper_and_unsigned_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            with mock.patch.object(factor, "ROOT", Path(temporary)):
                factor._atomic_json(path, {"status": "PASS"})
                self.assertEqual(factor._artifact(path)["sha256"], factor._sha256(path))
                path.write_text('{"status":"FAIL"}\n', encoding="utf-8")
                with self.assertRaises(AssertionError):
                    factor._artifact(path)
                path.with_suffix(path.suffix + ".sha256").unlink()
                with self.assertRaises(AssertionError):
                    factor._artifact(path)

    def test_quarantine_preserves_prior_artifact_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text("old", encoding="utf-8")
            sidecar = path.with_suffix(path.suffix + ".sha256")
            sidecar.write_text("stale", encoding="ascii")
            factor._quarantine_existing(path)
            self.assertFalse(path.exists())
            self.assertFalse(sidecar.exists())
            preserved = list(Path(temporary).glob("*.rejected.*"))
            self.assertEqual(len(preserved), 2)
            self.assertEqual(sorted(item.read_text() for item in preserved), ["old", "stale"])

    def test_protocol_generation_is_read_only_and_binds_full_inputs(self):
        digest = "a" * 64
        retained = [{"fold": fold, "checkpoint": {"path": f"r{fold}.pt", "sha256": digest}} for fold in core.FOLDS]
        comparator = {
            "candidate_id": core.COMPARATOR_ID, "immutable": True, "retrained": False,
            "final_receipt": {"path": "final.json", "sha256": digest},
            "comparison": {"path": "comparison.json", "sha256": digest},
            "canonical_folds": [
                {"fold": fold, "receipt": {"path": f"c{fold}.json", "sha256": digest}, "raw_oof": {"path": f"o{fold}.npz", "sha256": digest}}
                for fold in core.FOLDS
            ],
        }
        ledger = {
            "schema_version": factor.SELECTION_SCHEMA, "status": "BOUND", "seed": core.SEED,
            "candidate_id": core.CANDIDATE_ID, "candidate_config_sha256": digest,
            "source_configuration": {}, "retained_folds": retained,
            "immutable_comparator": comparator, "upstream": {},
        }
        upstream = {"resources": {"fold_manifest": {}, "semantic_embeddings": {}, "semantic_manifest": {}, "venv_p2_lock": {}}}
        with (
            mock.patch.object(factor, "generate_selection_ledger", return_value=ledger),
            mock.patch.object(factor, "_verified_upstream", return_value=upstream),
            mock.patch.object(factor, "_source_artifacts", return_value={"wrapper": {"path": "wrapper.py", "sha256": digest}}),
            mock.patch.object(factor.core, "canonical_environment", return_value=dict(core.CANONICAL_ENVIRONMENT)),
            mock.patch.object(factor, "_atomic_json") as write,
        ):
            protocol = factor.generate_protocol()
        write.assert_not_called()
        self.assertEqual(protocol["status"], "FROZEN")
        self.assertEqual(protocol["artifacts"]["retained_folds"], retained)
        self.assertEqual(protocol["artifacts"]["immutable_comparator"], comparator)
        self.assertIn("sources", protocol["artifacts"])
        self.assertIn("resources", protocol["artifacts"])
        self.assertEqual(protocol["environment_identity"]["expected"], core.CANONICAL_ENVIRONMENT)


if __name__ == "__main__":
    unittest.main()
