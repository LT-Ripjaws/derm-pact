"""Focused contract tests for the isolated attention-factor core."""
from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from dermpact.eval.dermpact_phase2_backbone_adaptation import AdaptedHybridModel
from dermpact.eval.dermpact_phase2_4_model import Phase24LocalizedCriterionModel
from dermpact.eval import dermpact_phase2_4a_attention_factor as core


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, values, *, rel_pos_bias=None):
        del rel_pos_bias
        return self.linear(values)


class _Visual(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(width) for _ in range(12))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 512)
        self.rel_pos_bias = None


def _confirmed() -> Phase24LocalizedCriterionModel:
    model = Phase24LocalizedCriterionModel(
        AdaptedHybridModel(_Visual(), "last_block").eval(),
        prototype_embeddings=torch.randn(7, 2, 16),
        attention_dim=8,
    ).eval()
    with torch.no_grad():
        model.criterion_residual.residual_scale.copy_(torch.linspace(-0.2, 0.2, 7))
    return model


def _protocol() -> dict:
    return {
        "training": {
            "optimizer": {
                "projection_group": {"learning_rate": 1e-3, "weight_decay": 1e-3},
                "scale_group": {"learning_rate": 3e-3, "weight_decay": 0.0},
                "sharp_temperature_group": {"learning_rate": 1e-3, "weight_decay": 0.0},
            }
        }
    }


class AttentionFactorCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = core.generate_protocol_spec()

    def test_protocol_is_valid_but_rejects_optimizer_tuning(self):
        digest = core.validate_protocol_spec(self.protocol)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            self.protocol["directml_probe"]["initial_cpu_device_max_abs_tolerance"],
            1e-5,
        )
        self.assertIn(
            "direct_rule_disagreement estimate <= +0.02",
            self.protocol["comparison"]["safe_arm_rule"],
        )
        changed = copy.deepcopy(self.protocol)
        changed["training"]["optimizer"]["projection_group"]["learning_rate"] *= 2
        with self.assertRaises(AssertionError):
            core.validate_protocol_spec(changed)

    def test_both_arms_are_exact_c10_at_initialization_and_freeze_c10(self):
        torch.manual_seed(20260830)
        tokens = torch.randn(2, 197, 8)
        for arm in core.ARMS:
            model = core.AttentionFactorAdapter(arm, copy.deepcopy(_confirmed()))
            core.assert_exact_c10_initialization(model, tokens)
            trainable = [name for name, value in model.named_parameters() if value.requires_grad]
            self.assertTrue(trainable)
            self.assertTrue(all("confirmed_model" not in name for name in trainable))

    def test_initialization_check_never_creates_inference_mode_tensors(self):
        model = core.AttentionFactorAdapter("sharp", _confirmed())
        tokens = torch.randn(2, 197, 8)
        with mock.patch("torch.inference_mode", side_effect=AssertionError("must not be used")):
            evidence = core.assert_exact_c10_initialization(model, tokens)
        self.assertTrue(evidence["diagnosis_exact"])
        self.assertTrue(evidence["melanoma_exact"])
        self.assertTrue(evidence["new_residual_exact_zero"])
        self.assertLessEqual(evidence["criteria_max_abs"], 1e-6)

    def test_safe_arm_ranking_uses_direct_pair_when_decisive(self):
        comparisons = {
            "sharp": {"safety_decision": "SAFE", "paired": {"criterion_mean_auc": {"estimate": 0.003, "ci_low": 0.001, "ci_high": 0.005}}},
            "multidepth": {"safety_decision": "SAFE", "paired": {"criterion_mean_auc": {"estimate": 0.004, "ci_low": 0.001, "ci_high": 0.006}}},
        }
        direct = {
            "left_arm": "sharp", "right_arm": "multidepth", "row_identity_exact": True,
            "paired": {"criterion_mean_auc": {"estimate": 0.002, "ci_low": 0.0001, "ci_high": 0.004}},
        }
        ranking = core.rank_safe_arms(comparisons, direct)
        self.assertEqual(ranking["winner"], "sharp")
        self.assertEqual(ranking["decision_reason"], "DECISIVE_ARM_TO_ARM_95CI")

    def test_safe_arm_ranking_retains_c10_when_no_arm_passes_gate(self):
        comparisons = {
            arm: {"safety_decision": "SAFE", "paired": {"criterion_mean_auc": {"estimate": 0.001, "ci_low": -0.001, "ci_high": 0.003}}}
            for arm in core.ARMS
        }
        direct = {
            "left_arm": "sharp", "right_arm": "multidepth", "row_identity_exact": True,
            "paired": {"criterion_mean_auc": {"estimate": 0.0, "ci_low": -0.002, "ci_high": 0.002}},
        }
        self.assertEqual(core.rank_safe_arms(comparisons, direct)["winner"], "c10@20")

    def test_positive_auroc_arm_marked_unsafe_cannot_promote(self):
        comparisons = {
            "sharp": {"safety_decision": "UNSAFE", "paired": {"criterion_mean_auc": {"estimate": 0.01, "ci_low": 0.005, "ci_high": 0.015}}},
            "multidepth": {"safety_decision": "SAFE", "paired": {"criterion_mean_auc": {"estimate": 0.001, "ci_low": -0.001, "ci_high": 0.003}}},
        }
        direct = {
            "left_arm": "sharp", "right_arm": "multidepth", "row_identity_exact": True,
            "paired": {"criterion_mean_auc": {"estimate": 0.009, "ci_low": 0.004, "ci_high": 0.014}},
        }
        ranking = core.rank_safe_arms(comparisons, direct)
        self.assertEqual(ranking["winner"], "c10@20")
        self.assertNotIn("sharp", ranking["promoted_arms"])

    def test_optimizer_groups_match_c10_and_temperature_has_zero_decay(self):
        sharp = core.AttentionFactorAdapter("sharp", _confirmed())
        optimizer = core._optimizer(sharp, _protocol())
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-3, 3e-3, 1e-3])
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [1e-3, 0.0, 0.0])
        temperature_ids = {id(value) for value in optimizer.param_groups[2]["params"]}
        self.assertEqual(temperature_ids, {id(sharp.new_residual.log_temperature)})

    def test_attention_evidence_is_finite_bounded_and_shaped(self):
        attention = torch.softmax(torch.randn(3, 7, 2, 196), dim=-1)
        evidence = core.attention_evidence(attention, top_k=10)
        for values in evidence.values():
            self.assertEqual(values.shape, (3, 7, 2))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values >= 0.0).all())
            self.assertTrue((values <= 1.0 + 1e-6).all())

    def test_resume_round_trip_restores_full_identity_and_rng(self):
        torch.manual_seed(7)
        model = core.AttentionFactorAdapter("sharp", _confirmed())
        optimizer = core._optimizer(model, _protocol())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        expected_python = random.getstate()
        expected_numpy = np.random.get_state()
        expected_torch = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            core, "EXPERIMENT_DIR", Path(directory)
        ):
            core._write_resume(
                "sharp", 0, protocol_sha="a" * 64, retained_sha="b" * 64,
                c10_sha="c" * 64, completed_epochs=1, model=model,
                optimizer=optimizer, scheduler=scheduler, history=[{"epoch": 1}],
            )
            random.random(); np.random.random(); torch.rand(1)
            loaded = core._load_resume(
                "sharp", 0, protocol_sha="a" * 64, retained_sha="b" * 64,
                c10_sha="c" * 64, model=model, optimizer=optimizer,
                scheduler=scheduler, device="cpu", epochs=20,
            )
        self.assertEqual(loaded, (1, [{"epoch": 1}]))
        self.assertEqual(random.getstate(), expected_python)
        self.assertEqual(np.random.get_state()[1].tolist(), expected_numpy[1].tolist())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))

    def test_completed_training_and_canonical_reuse_fail_closed_on_tamper(self):
        digest = "a" * 64
        protocol = {
            "hard_scope_checks": {
                "derm7pt_official_test_rows_loaded": 0,
                "ham_calibration_or_final_rows_loaded": 0,
                "bcn_validation_or_final_rows_loaded": 0,
                "ph2_rows_loaded": 0,
            },
            "artifacts": {"fold_inputs": [{
                "retained_core": {"sha256": "b" * 64},
                "c10_residual": {"sha256": "c" * 64},
            }]},
        }
        with tempfile.TemporaryDirectory(dir=core.ROOT) as directory:
            root = Path(directory)
            with (
                mock.patch.object(core, "MANIFEST_DIR", root / "manifests"),
                mock.patch.object(core, "EXPERIMENT_DIR", root / "experiments"),
                mock.patch.object(core, "validate_protocol_spec", return_value=digest),
                mock.patch.object(core, "FOLDS", (0,)),
            ):
                checkpoint = core._residual_path("sharp", 0)
                checkpoint_sha = core._atomic_torch(checkpoint, {"x": torch.ones(1)})
                training = {
                    "schema_version": core.TRAINING_SCHEMA, "status": "PASS",
                    "protocol_sha256": digest, "arm": "sharp", "seed": core.SEED,
                    "fold": 0, "completed_epochs": 20,
                    "retained_core_sha256": "b" * 64, "c10_residual_sha256": "c" * 64,
                    "effective_batch_size": 64, "microbatch_size": 8,
                    "history": [{"epoch": value} for value in range(1, 21)],
                    "new_residual_checkpoint": {"path": core._relative(checkpoint), "sha256": checkpoint_sha},
                    "hard_scope_checks": protocol["hard_scope_checks"],
                }
                core._atomic_json(core.training_receipt_path("sharp", 0), training)
                self.assertIsNotNone(core.verified_training_receipt(protocol, "sharp", 0))

                raw = core._fold_dir("sharp", 0) / "canonical_derm7pt_oof_raw.npz"
                raw_sha = core._save_npz(
                    raw, row_indices=np.arange(2), diagnosis_logits=np.zeros((2, 3)),
                    melanoma_logits=np.zeros((2,)), c10_criterion_logits=np.zeros((2, 7)),
                    final_criterion_logits=np.zeros((2, 7)), new_residual_logits=np.zeros((2, 7)),
                    attention_weights=np.zeros((2, 7, 2, 196)),
                    attention_normalized_entropy=np.zeros((2, 7, 2)),
                    attention_top_k_mass=np.zeros((2, 7, 2)),
                )
                canonical = {
                    "schema_version": core.CANONICAL_SCHEMA, "status": "PASS",
                    "protocol_sha256": digest, "arm": "sharp", "seed": core.SEED,
                    "fold": 0, "protected_logits_equal_c10": True,
                    "new_residual_state_reload_exact": True,
                    "raw_oof": {"path": core._relative(raw), "sha256": raw_sha, "rows": 2},
                    "hard_scope_checks": protocol["hard_scope_checks"],
                }
                core._atomic_json(core.canonical_receipt_path("sharp", 0), canonical)
                self.assertIsNotNone(core.verified_canonical_receipt(protocol, "sharp", 0))
                raw.write_bytes(b"tampered")
                with self.assertRaises(AssertionError):
                    core.verified_canonical_receipt(protocol, "sharp", 0)

    def test_matrix_is_closed_ordered_and_never_launches(self):
        matrix = core.controller_matrix()
        self.assertEqual(len(matrix), 24)
        self.assertEqual({row["arm"] for row in matrix}, set(core.ARMS))
        self.assertTrue(all(row["seed"] == core.SEED for row in matrix))
        self.assertEqual([row["stage"] for row in matrix].count("probe"), 2)
        self.assertEqual([row["stage"] for row in matrix].count("train"), 10)
        self.assertEqual([row["stage"] for row in matrix].count("reevaluate"), 10)
        self.assertEqual([row["stage"] for row in matrix].count("compare"), 2)


if __name__ == "__main__":
    unittest.main()
