"""Focused public behavior tests for the fixed Phase 2.4A execution API."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import runpy
import tempfile

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_4a_training as phase24a


class Phase24ATrainingContractTest(unittest.TestCase):
    @staticmethod
    def _reload_outputs(value: float) -> dict[str, torch.Tensor]:
        return {
            name: torch.tensor([value], dtype=torch.float32)
            for name in (
                "diagnosis", "melanoma", "criteria",
                "criterion_residual", "criterion_attention",
            )
        }

    def test_output_reload_accepts_exact_equality(self):
        values = self._reload_outputs(1.0)
        maximums = phase24a.validate_output_reload(values, values)
        self.assertEqual(set(maximums.values()), {0.0})

    def test_output_reload_accepts_just_inside_absolute_tolerance(self):
        before = self._reload_outputs(0.0)
        after = self._reload_outputs(0.0000009)
        maximums = phase24a.validate_output_reload(before, after)
        self.assertTrue(all(0.0 < value <= 1e-6 for value in maximums.values()))

    def test_output_reload_rejects_over_tolerance(self):
        with self.assertRaisesRegex(AssertionError, "exceeds absolute tolerance"):
            phase24a.validate_output_reload(
                self._reload_outputs(0.0), self._reload_outputs(0.0000011)
            )

    def test_output_reload_rejects_nonfinite(self):
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            phase24a.validate_output_reload(
                self._reload_outputs(0.0), self._reload_outputs(float("nan"))
            )

    def test_protocol_source_inventory_includes_the_public_facade(self):
        artifacts = phase24a._implementation._source_artifacts()
        self.assertEqual(
            artifacts["public_training_api"]["path"],
            "dermpact/eval/dermpact_phase2_4a_training.py",
        )
        self.assertEqual(
            artifacts["controller"]["path"],
            "research/scripts/dermpact_phase2_4a_controller.py",
        )
        self.assertEqual(
            artifacts["full_run_controller"]["path"],
            "research/scripts/dermpact_phase2_4a_full_run_controller.py",
        )

    def test_final_residual_checkpoint_name_is_epoch_independent(self):
        self.assertEqual(
            phase24a._implementation._residual_path(4).name,
            "final_residual_state.pt",
        )

    def test_public_train_fold_reads_microbatch_from_protocol(self):
        protocol = {"training": {"directml_microbatch_size": 13}}
        with (
            patch.object(phase24a._implementation, "load_protocol", return_value=(protocol, "hash")),
            patch.object(phase24a._implementation, "train_fold", return_value={"status": "PASS"}) as execute,
        ):
            phase24a.train_fold(2, "cpu")
        self.assertEqual(execute.call_args.kwargs["microbatch_size"], 13)

    def test_npz_and_sidecar_commit_without_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            digest = phase24a._implementation._save_npz(
                path, row_indices=np.asarray([1, 2], dtype="int64")
            )
            sidecar = path.with_suffix(".npz.sha256")
            self.assertEqual(sidecar.read_text(encoding="ascii").strip(), digest)
            self.assertFalse(path.with_suffix(".partial.npz").exists())
            self.assertFalse(sidecar.with_suffix(".sha256.partial").exists())

    def test_public_receipt_paths_and_controller_matrix_are_fixed(self):
        self.assertEqual(
            phase24a.training_receipt_path(3).name,
            "seed_20260827_fold_3_training.json",
        )
        self.assertEqual(
            phase24a.canonical_receipt_path(3).name,
            "seed_20260827_fold_3_canonical.json",
        )
        matrix = phase24a.controller_matrix(
            {"identity": {"folds": [0, 1, 2, 3, 4]}}
        )
        self.assertEqual(len(matrix), 11)
        self.assertEqual(matrix[:5], [
            {"stage": "train", "fold": fold, "environment": "venv_dml"}
            for fold in range(5)
        ])
        self.assertEqual(matrix[-1]["stage"], "finalize")

    def test_resume_identity_rejects_another_protocol(self):
        payload = {
            "format": "dermpact-phase2-4a-epoch-resume-v1",
            "protocol_sha256": "wrong",
            "source_checkpoint_sha256": "source",
            "seed": 20260827,
            "fold": 2,
        }
        with self.assertRaisesRegex(AssertionError, "identity mismatch"):
            phase24a.validate_resume_identity(
                payload,
                protocol_sha256="expected",
                fold=2,
                source_checkpoint_sha256="source",
            )

    def test_controller_rejects_wrong_final_receipt_schema(self):
        controller = runpy.run_path(
            str(
                phase24a.ROOT
                / "research/scripts/dermpact_phase2_4a_full_run_controller.py"
            )
        )
        validate = controller["_final_receipt_identity_valid"]
        receipt = {
            "schema_version":
                "dermpact-phase2-4a-localized-criterion-final-receipt-v1",
            "status": "PASS",
            "protocol_sha256": "protocol",
            "scientific_decision": "PROMOTE",
        }
        self.assertTrue(validate(receipt, "protocol"))
        receipt["schema_version"] = "wrong-final-receipt-schema"
        self.assertFalse(validate(receipt, "protocol"))


if __name__ == "__main__":
    unittest.main()
