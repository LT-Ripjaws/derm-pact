"""Tests for the execution-only loss-factor canonical compatibility path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_4a_loss_factor as factor
from dermpact.eval import dermpact_phase2_4a_loss_factor_canonical_compat as compat


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion_residual = torch.nn.Linear(2, 2, bias=False)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = self.criterion_residual(value)
        return {
            "diagnosis": value,
            "melanoma": value[:, :1],
            "criteria": residual,
            "criterion_residual": residual,
            "criterion_attention": torch.ones_like(residual),
        }


def _raw(indices: np.ndarray) -> dict[str, np.ndarray]:
    rows = len(indices)
    return {
        "row_indices": indices.copy(),
        "diagnosis_logits": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        "melanoma_logits": np.arange(rows, dtype=np.float32).reshape(rows, 1),
        "base_criterion_logits": np.ones((rows, 2), dtype=np.float32),
        "final_criterion_logits": np.full((rows, 2), 2.0, dtype=np.float32),
        "residual_logits": np.ones((rows, 2), dtype=np.float32),
        "criterion_attention": np.full((rows, 2), 0.5, dtype=np.float32),
    }


class LossFactorCanonicalCompatTest(unittest.TestCase):
    def _run_fixture(self, temporary: str):
        root = Path(temporary)
        receipt_path = root / "canonical.json"
        raw_path = root / "canonical.npz"
        comparator_path = root / "comparator.npz"
        indices = np.asarray([0, 2], dtype=np.int64)
        arrays = _raw(indices)
        np.savez_compressed(comparator_path, **arrays)
        protocol_sha = "1" * 64
        ledger_sha = "2" * 64
        config_sha = "3" * 64
        checkpoint_sha = "4" * 64
        source_sha = "5" * 64
        protocol = {"hard_scope_checks": {"cpu_only": True}}
        ledger = {
            "candidate_config_sha256": config_sha,
            "immutable_comparator": {
                "canonical_folds": [
                    {
                        "fold": 0,
                        "raw_oof": {
                            "path": "comparator.npz",
                            "sha256": "6" * 64,
                        },
                    }
                ]
            },
        }
        training = {
            "source_checkpoint_sha256": source_sha,
            "residual_checkpoint": {"sha256": checkpoint_sha},
        }
        data = {
            "derm_folds": np.asarray([0, 1, 0]),
            "derm_tokens": np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        }
        model = _TinyModel()
        load = mock.Mock(return_value=(model, training))
        captured_arrays = {}
        captured_receipt = {}

        def write_npz(path, values):
            self.assertEqual(path, raw_path)
            captured_arrays.update({name: value.copy() for name, value in values.items()})
            return "7" * 64

        def write_json(path, value):
            self.assertEqual(path, receipt_path)
            captured_receipt.update(value)
            return "8" * 64

        patches = (
            mock.patch.object(compat, "_output_paths", return_value=(receipt_path, raw_path)),
            mock.patch.object(factor, "ROOT", root),
            mock.patch.object(factor, "load_protocol", return_value=(protocol, protocol_sha)),
            mock.patch.object(factor, "_ledger", return_value=(ledger, ledger_sha)),
            mock.patch.object(factor, "_load_trained_model", load),
            mock.patch.object(factor.stage1, "_derm_data", return_value=data),
            mock.patch.object(factor.core, "canonical_raw_oof", return_value=arrays),
            mock.patch.object(factor.core, "validate_paired_oof", return_value=indices),
            mock.patch.object(factor, "_artifact", return_value={"path": "comparator.npz", "sha256": "6" * 64}),
            mock.patch.object(factor, "_atomic_npz", side_effect=write_npz),
            mock.patch.object(factor, "_atomic_json", side_effect=write_json),
            mock.patch.object(factor.core, "canonical_environment", return_value={"device": "cpu"}),
            mock.patch.object(factor.core, "require_canonical_cpu"),
        )
        return patches, load, captured_arrays, captured_receipt, arrays, receipt_path, raw_path

    def test_one_fresh_loader_exact_arrays_and_original_receipt_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                patches,
                load,
                captured_arrays,
                captured_receipt,
                expected_arrays,
                _,
                _,
            ) = self._run_fixture(temporary)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12]:
                result = compat.reevaluate_fold(0)

        self.assertEqual(load.call_count, 1)
        self.assertEqual(set(captured_arrays), set(expected_arrays))
        for name in expected_arrays:
            np.testing.assert_array_equal(captured_arrays[name], expected_arrays[name])
        self.assertEqual(captured_receipt["schema_version"], factor.CANONICAL_SCHEMA)
        self.assertEqual(result["receipt_sha256"], "8" * 64)
        self.assertEqual(
            set(captured_receipt),
            {
                "schema_version", "status", "protocol_sha256",
                "selection_ledger_sha256", "candidate_config_sha256",
                "candidate_id", "seed", "fold", "source_checkpoint_sha256",
                "residual_checkpoint_sha256", "raw_oof", "comparator_raw_oof",
                "protected_logits_equal_comparator", "identical_oof_indices",
                "residual_state_reload_exact", "output_reload_within_tolerance",
                "output_reload_tolerance", "output_reload_max_abs", "environment",
                "hard_scope_checks",
            },
        )
        self.assertNotIn("compat", json.dumps(captured_receipt).lower())

    def test_existing_tamper_is_quarantined_and_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "canonical.json"
            raw = root / "canonical.npz"
            receipt.write_text("tampered", encoding="utf-8")
            factor._sidecar(receipt).write_text("bad", encoding="ascii")
            raw.write_bytes(b"tampered")
            factor._sidecar(raw).write_text("bad", encoding="ascii")
            with (
                mock.patch.object(compat, "_output_paths", return_value=(receipt, raw)),
                mock.patch.object(factor, "load_protocol", return_value=({}, "1" * 64)),
                mock.patch.object(
                    factor,
                    "_ledger",
                    return_value=({"candidate_config_sha256": "3" * 64}, "2" * 64),
                ),
                mock.patch.object(factor, "_verified_canonical", side_effect=AssertionError("tamper")),
                mock.patch.object(factor, "_load_trained_model") as load,
                mock.patch.object(factor.core, "require_canonical_cpu"),
            ):
                with self.assertRaisesRegex(AssertionError, "made no replacement"):
                    compat.reevaluate_fold(0)
            load.assert_not_called()
            self.assertFalse(receipt.exists())
            self.assertFalse(raw.exists())
            self.assertEqual(len(list(root.glob("*.rejected.*"))), 4)

    def test_no_training_entrypoint_or_training_artifact_is_mutated(self):
        with tempfile.TemporaryDirectory() as temporary:
            training_artifact = Path(temporary) / "training.json"
            training_artifact.write_bytes(b"immutable-training")
            before = (
                training_artifact.read_bytes(),
                training_artifact.stat().st_mtime_ns,
            )
            patches, _, _, _, _, _, _ = self._run_fixture(temporary)
            with (
                patches[0], patches[1], patches[2], patches[3], patches[4],
                patches[5], patches[6], patches[7], patches[8], patches[9],
                patches[10], patches[11], patches[12],
                mock.patch.object(factor, "train_fold") as train,
            ):
                compat.reevaluate_fold(0)
            train.assert_not_called()
            self.assertEqual(
                (training_artifact.read_bytes(), training_artifact.stat().st_mtime_ns),
                before,
            )

    def test_amendment_payload_is_deterministic_hash_bound_and_never_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.py"
            original.write_text("original", encoding="utf-8")
            original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
            protocol = {
                "artifacts": {
                    "sources": {
                        "wrapper": {"path": "original.py", "sha256": original_sha}
                    }
                }
            }
            with (
                mock.patch.object(factor, "ROOT", root),
                mock.patch.object(factor, "PROTOCOL_PATH", root / "protocol.json"),
                mock.patch.object(factor, "load_protocol", return_value=(protocol, "a" * 64)),
                mock.patch.object(factor, "_relative", side_effect=lambda path: str(path)),
                mock.patch.object(factor, "_plain_artifact", return_value={"path": "compat.py", "sha256": "b" * 64}),
                mock.patch.object(factor, "_atomic_json") as write,
            ):
                first = compat.generate_execution_amendment_payload(
                    error_evidence={"exception": "fresh-process guard"},
                    log_evidence={"stderr_sha256": "c" * 64},
                )
                second = compat.generate_execution_amendment_payload(
                    error_evidence={"exception": "fresh-process guard"},
                    log_evidence={"stderr_sha256": "c" * 64},
                )
            write.assert_not_called()
            self.assertEqual(first, second)
            binding = first.pop("binding_sha256")
            self.assertEqual(binding, factor._canonical_digest(first))
            self.assertTrue(first["scope"]["execution_only"])
            self.assertFalse(first["scope"]["scientific_change"])
            self.assertEqual(first["frozen_protocol"]["sha256"], "a" * 64)
            self.assertEqual(
                first["original_protocol_bound_sources"]["wrapper"]["sha256"],
                original_sha,
            )


if __name__ == "__main__":
    unittest.main()
