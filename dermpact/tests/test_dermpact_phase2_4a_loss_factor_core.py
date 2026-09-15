"""Focused tests for the isolated Phase 2.4A ranking-loss factor core."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dermpact.eval import dermpact_phase2_4a_loss_factor_core as core
from dermpact.eval import dermpact_phase2_architecture_ladder as ladder
from dermpact.perception.criterion_losses import CriterionLoss


class _Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


class _ResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion_residual = nn.Linear(3, 2)


class _Base(nn.Module):
    def __init__(self, changed: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.changed = changed

    def forward(self, tokens):
        diagnosis = torch.stack((tokens[:, 0], -tokens[:, 0]), dim=1)
        melanoma = tokens[:, 1]
        criteria = tokens[:, :1].repeat(1, 7)
        return {
            "diagnosis": diagnosis,
            "melanoma": melanoma,
            "criteria": criteria,
        }


class _OOFModel(nn.Module):
    def __init__(self, changed: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.base_model = _Base(changed=changed)

    def forward(self, tokens):
        base = self.base_model(tokens)
        diagnosis = base["diagnosis"] + (1.0 if self.base_model.changed else 0.0)
        residual = torch.zeros_like(base["criteria"])
        return {
            "diagnosis": diagnosis,
            "melanoma": base["melanoma"],
            "base_criteria": base["criteria"],
            "criteria": base["criteria"] + residual,
            "criterion_residual": residual,
            "criterion_attention": torch.ones(len(tokens), 7, 2, 3) / 3.0,
        }


class _TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Linear(3, 3)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.criterion_residual = nn.Linear(3, 7)
        self.forward_calls = 0

    def forward(self, tokens):
        self.forward_calls += 1
        criteria = self.criterion_residual(tokens)
        return {
            "diagnosis": torch.zeros(len(tokens), 5),
            "melanoma": torch.zeros(len(tokens)),
            "criteria": criteria,
        }


def _objective_inputs():
    torch.manual_seed(core.SEED)
    outputs = {
        "diagnosis": torch.randn(64, 5, requires_grad=True),
        "melanoma": torch.randn(64, requires_grad=True),
        "criteria": torch.randn(64, 7, requires_grad=True),
    }
    diagnosis = torch.arange(64) % 5
    melanoma = (torch.arange(64) % 2).float()
    criteria = torch.stack(
        [((torch.arange(64) + index) % (index + 2) == 0).float() for index in range(7)],
        dim=1,
    )
    diagnosis_weights = torch.rand(64) + 0.2
    melanoma_weights = torch.rand(64) + 0.2
    criterion_weights = torch.rand(64, 7) + 0.2
    return (
        outputs,
        diagnosis,
        melanoma,
        criteria,
        diagnosis_weights,
        melanoma_weights,
        criterion_weights,
    )


class LossFactorCoreTest(unittest.TestCase):
    def test_protocol_spec_is_pure_fixed_and_fail_closed(self) -> None:
        spec = core.generate_protocol_spec()
        core.validate_protocol_spec(spec)

        self.assertEqual(spec["training"]["effective_batch_size"], 64)
        self.assertIsNone(spec["training"]["microbatch_size"])
        self.assertEqual(spec["training"]["ranking"]["weight"], 0.10)
        self.assertEqual(
            spec["training"]["ranking"]["effective_total_objective_weight"],
            "0.10/3",
        )
        self.assertTrue(spec["identity"]["comparator_immutable"])
        changed = copy.deepcopy(spec)
        changed["training"]["microbatch_size"] = 8
        with self.assertRaisesRegex(AssertionError, "training"):
            core.validate_protocol_spec(changed)

    def test_full_batch_objective_preserves_old_terms_and_adds_rank010(self) -> None:
        (
            outputs,
            diagnosis,
            melanoma,
            criteria,
            diagnosis_weights,
            melanoma_weights,
            criterion_weights,
        ) = _objective_inputs()

        result = core.full_batch_objective(
            outputs,
            diagnosis=diagnosis,
            melanoma=melanoma,
            criteria=criteria,
            diagnosis_weights=diagnosis_weights,
            melanoma_weights=melanoma_weights,
            criterion_weights=criterion_weights,
        )

        diagnosis_loss = (
            F.cross_entropy(outputs["diagnosis"], diagnosis, reduction="none")
            * diagnosis_weights
        ).sum() / diagnosis_weights.sum()
        melanoma_loss = (
            F.binary_cross_entropy_with_logits(
                outputs["melanoma"], melanoma, reduction="none"
            )
            * melanoma_weights
        ).sum() / melanoma_weights.sum()
        criteria_result = CriterionLoss(ranking_weight=0.10)(
            outputs["criteria"], criteria, criterion_weights
        )
        consistency, _ = ladder.truth_anchored_consistency_loss(
            outputs, melanoma=melanoma, criteria=criteria, diagnosis=diagnosis
        )
        expected = (
            diagnosis_loss
            + melanoma_loss
            + criteria_result.bce
            + 0.10 * criteria_result.ranking
        ) / 3.0 + 0.10 * consistency

        self.assertTrue(torch.equal(result.total, expected))
        self.assertTrue(torch.equal(result.criteria_bce, criteria_result.bce))
        self.assertTrue(
            torch.equal(
                result.ranking_contribution,
                (0.10 * criteria_result.ranking) / 3.0,
            )
        )
        self.assertTrue(
            torch.equal(
                result.criterion.ranking_contribution,
                0.10 * criteria_result.ranking,
            )
        )

    def test_ranking_uses_cross_half_pairs_from_the_full_effective_batch(self) -> None:
        outputs = {
            "diagnosis": torch.zeros(64, 5),
            "melanoma": torch.zeros(64),
            "criteria": torch.cat((torch.ones(32, 7), -torch.ones(32, 7))),
        }
        diagnosis = torch.zeros(64, dtype=torch.long)
        melanoma = torch.zeros(64)
        criteria = torch.cat((torch.ones(32, 7), torch.zeros(32, 7)))

        result = core.full_batch_objective(
            outputs,
            diagnosis=diagnosis,
            melanoma=melanoma,
            criteria=criteria,
            diagnosis_weights=torch.ones(64),
            melanoma_weights=torch.ones(64),
            criterion_weights=torch.ones(64, 7),
        )

        self.assertTrue(torch.equal(result.criterion.eligible_pair_counts, torch.full((7,), 1024)))
        self.assertGreater(result.ranking.item(), 0.0)
        left = CriterionLoss(ranking_weight=0.10)(
            outputs["criteria"][:32], criteria[:32], torch.ones(32, 7)
        )
        right = CriterionLoss(ranking_weight=0.10)(
            outputs["criteria"][32:], criteria[32:], torch.ones(32, 7)
        )
        self.assertEqual((left.ranking + right.ranking).item(), 0.0)

    def test_objective_rejects_any_non_64_batch(self) -> None:
        values = _objective_inputs()
        with self.assertRaisesRegex(ValueError, "full effective batch"):
            core.full_batch_objective(
                {name: value[:63] for name, value in values[0].items()},
                diagnosis=values[1][:63],
                melanoma=values[2][:63],
                criteria=values[3][:63],
                diagnosis_weights=values[4][:63],
                melanoma_weights=values[5][:63],
                criterion_weights=values[6][:63],
            )

    def test_training_step_performs_one_unsplit_forward_and_one_scheduler_step(self) -> None:
        model = _TrainModel()
        optimizer = torch.optim.AdamW(model.criterion_residual.parameters(), lr=0.01)
        scheduler = _Scheduler()
        criteria = torch.stack(
            [((torch.arange(64) + index) % 2).float() for index in range(7)], dim=1
        )

        result = core.full_batch_training_step(
            model,
            optimizer,
            scheduler,
            tokens=torch.randn(64, 3),
            diagnosis=torch.zeros(64, dtype=torch.long),
            melanoma=torch.zeros(64),
            criteria=criteria,
            diagnosis_weights=torch.ones(64),
            melanoma_weights=torch.ones(64),
            criterion_weights=torch.ones(64, 7),
        )

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(scheduler.steps, 1)
        self.assertTrue(torch.isfinite(result.total))
        self.assertTrue(
            torch.equal(result.criterion.eligible_pair_counts, torch.full((7,), 1024))
        )

    def test_atomic_resume_round_trip_and_identity_rejection(self) -> None:
        digest = "a" * 64
        identity = core.ResumeIdentity(
            protocol_sha256=digest,
            source_checkpoint_sha256="b" * 64,
            candidate_config_sha256="c" * 64,
            fold=2,
        )
        model = _ResumeModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = _Scheduler()
        original = {
            name: value.detach().clone()
            for name, value in model.criterion_residual.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.pt"
            written_sha = core.write_epoch_resume(
                path,
                identity=identity,
                completed_epochs=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=[{"epoch": 1}],
            )
            self.assertEqual(len(written_sha), 64)
            self.assertTrue(path.with_suffix(".pt.sha256").is_file())
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            loaded = core.load_epoch_resume(
                path,
                identity=identity,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self.assertEqual(loaded, (1, [{"epoch": 1}]))
            for name, value in model.criterion_residual.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]))
            wrong = core.ResumeIdentity(
                protocol_sha256="d" * 64,
                source_checkpoint_sha256="b" * 64,
                candidate_config_sha256="c" * 64,
                fold=2,
            )
            with self.assertRaisesRegex(AssertionError, "identity mismatch"):
                core.load_epoch_resume(
                    path,
                    identity=wrong,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

    def test_canonical_raw_oof_is_pure_and_checks_protected_logits(self) -> None:
        tokens = np.arange(30, dtype="float32").reshape(10, 3)
        rows = np.asarray([1, 4, 7, 8], dtype="int64")

        raw = core.canonical_raw_oof(
            _OOFModel(), tokens=tokens, row_indices=rows, prediction_batch_size=2
        )

        self.assertEqual(set(raw), set(core._RAW_KEYS))
        self.assertTrue(np.array_equal(raw["row_indices"], rows))
        self.assertEqual(raw["criterion_attention"].shape, (4, 7, 2, 3))
        with self.assertRaisesRegex(AssertionError, "diagnosis logits changed"):
            core.canonical_raw_oof(
                _OOFModel(changed=True),
                tokens=tokens,
                row_indices=rows,
                prediction_batch_size=2,
            )

    def test_paired_hooks_require_exact_indices_and_are_directional(self) -> None:
        base = {
            "row_indices": np.asarray([2, 5], dtype="int64"),
            "diagnosis_logits": np.zeros((2, 5), dtype="float32"),
            "melanoma_logits": np.zeros(2, dtype="float32"),
            "base_criterion_logits": np.zeros((2, 7), dtype="float32"),
            "final_criterion_logits": np.zeros((2, 7), dtype="float32"),
            "residual_logits": np.zeros((2, 7), dtype="float32"),
            "criterion_attention": np.zeros((2, 7, 2, 3), dtype="float32"),
        }
        candidate = copy.deepcopy(base)
        candidate["final_criterion_logits"][:] = 2.0
        difference = core.paired_metric_difference(
            candidate,
            base,
            lambda raw: float(np.asarray(raw["final_criterion_logits"]).mean()),
        )
        self.assertEqual(difference, 2.0)
        mismatched = copy.deepcopy(base)
        mismatched["row_indices"] = np.asarray([2, 6], dtype="int64")
        with self.assertRaisesRegex(AssertionError, "indices differ"):
            core.validate_paired_oof(candidate, mismatched)


if __name__ == "__main__":
    unittest.main()
