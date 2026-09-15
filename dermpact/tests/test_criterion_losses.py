"""Focused behavior tests for auditable criterion losses."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from dermpact.perception.criterion_losses import CriterionLoss


class CriterionLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260830)

    def test_zero_ranking_weight_exactly_reproduces_weighted_bce(self) -> None:
        logits = torch.randn(9, 7, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, 2, (9, 7), dtype=torch.float64)
        weights = torch.rand(9, 7, dtype=torch.float64) + 0.1
        expected_per_criterion = torch.stack(
            [
                (
                    F.binary_cross_entropy_with_logits(
                        logits[:, index], targets[:, index], reduction="none"
                    )
                    * weights[:, index]
                ).sum()
                / weights[:, index].sum().clamp_min(1e-12)
                for index in range(7)
            ]
        )
        expected = expected_per_criterion.mean()

        result = CriterionLoss(ranking_weight=0.0)(logits, targets, weights)

        self.assertTrue(torch.equal(result.total, expected))
        self.assertTrue(torch.equal(result.bce, expected))
        self.assertTrue(
            torch.equal(result.per_criterion_bce, expected_per_criterion)
        )
        self.assertEqual(result.ranking_contribution.item(), 0.0)

    def test_pairwise_logistic_value_weights_and_counts_are_transparent(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        weights = torch.tensor([[2.0, 1.0], [3.0, 1.0], [5.0, 1.0]])

        result = CriterionLoss(ranking_weight=0.25)(logits, targets, weights)

        pair_weights = torch.tensor([6.0, 10.0])
        pair_losses = F.softplus(torch.tensor([0.0, -1.0]) - 2.0)
        expected_numerator = (pair_weights * pair_losses).sum()
        expected_ranking = expected_numerator / pair_weights.sum()
        self.assertTrue(torch.allclose(result.ranking, expected_ranking))
        self.assertTrue(
            torch.allclose(result.ranking_numerators, torch.tensor([expected_numerator, 0.0]))
        )
        self.assertTrue(
            torch.equal(result.ranking_weight_sums, torch.tensor([16.0, 0.0]))
        )
        self.assertTrue(torch.equal(result.eligible_pair_counts, torch.tensor([2, 0])))
        self.assertTrue(
            torch.equal(result.eligible_criteria, torch.tensor([True, False]))
        )
        self.assertTrue(
            torch.allclose(result.total, result.bce + 0.25 * expected_ranking)
        )

    def test_one_class_and_zero_weight_criteria_are_finite_and_differentiable(self) -> None:
        logits = torch.randn(5, 3, requires_grad=True)
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
             [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
        )
        weights = torch.ones_like(logits)
        weights[:, 2] = 0.0

        result = CriterionLoss(ranking_weight=1.0)(logits, targets, weights)
        result.total.backward()

        self.assertTrue(torch.isfinite(result.total))
        self.assertEqual(result.ranking.item(), 0.0)
        self.assertTrue(torch.equal(result.eligible_pair_counts, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_duplicate_samples_do_not_change_normalized_loss_scale(self) -> None:
        logits = torch.tensor([[1.5, -0.2], [-0.5, 0.8], [0.2, -1.1], [2.0, 1.3]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        weights = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [4.0, 3.0]])
        loss = CriterionLoss(ranking_weight=0.4)

        original = loss(logits, targets, weights)
        duplicated = loss(
            logits.repeat((2, 1)), targets.repeat((2, 1)), weights.repeat((2, 1))
        )

        self.assertTrue(torch.allclose(original.bce, duplicated.bce, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(original.ranking, duplicated.ranking, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(original.total, duplicated.total, atol=1e-7, rtol=0.0))

    def test_chunking_is_deterministic_and_preserves_gradients(self) -> None:
        targets = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        weights = torch.rand(4, 2, dtype=torch.float64) + 0.2
        first_logits = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        second_logits = first_logits.detach().clone().requires_grad_(True)

        small = CriterionLoss(ranking_weight=0.7, pair_chunk_size=1)(
            first_logits, targets, weights
        )
        large = CriterionLoss(ranking_weight=0.7, pair_chunk_size=10_000)(
            second_logits, targets, weights
        )
        small.total.backward()
        large.total.backward()

        self.assertTrue(torch.allclose(small.total, large.total, atol=1e-12, rtol=0.0))
        self.assertTrue(
            torch.allclose(first_logits.grad, second_logits.grad, atol=1e-12, rtol=0.0)
        )

    def test_bce_statistics_reconstruct_full_batch_from_microbatches(self) -> None:
        logits = torch.randn(8, 4, dtype=torch.float64)
        targets = torch.randint(0, 2, (8, 4), dtype=torch.float64)
        weights = torch.rand(8, 4, dtype=torch.float64) + 0.1
        loss = CriterionLoss(ranking_weight=0.0)

        full = loss(logits, targets, weights)
        left = loss(logits[:3], targets[:3], weights[:3])
        right = loss(logits[3:], targets[3:], weights[3:])
        reconstructed = (
            (left.bce_numerators + right.bce_numerators)
            / (left.bce_weight_sums + right.bce_weight_sums).clamp_min(1e-12)
        ).mean()

        self.assertTrue(torch.allclose(full.bce, reconstructed, atol=1e-15, rtol=0.0))

    def test_empty_batch_is_safe_and_has_no_eligible_pairs(self) -> None:
        logits = torch.empty(0, 3, requires_grad=True)
        targets = torch.empty(0, 3)
        weights = torch.empty(0)

        result = CriterionLoss(ranking_weight=1.0)(logits, targets, weights)
        result.total.backward()

        self.assertEqual(result.total.item(), 0.0)
        self.assertTrue(torch.equal(result.eligible_pair_counts, torch.zeros(3, dtype=torch.long)))

    def test_rejects_nonbinary_targets_and_negative_weights(self) -> None:
        logits = torch.zeros(2, 1)
        with self.assertRaisesRegex(ValueError, "binary"):
            CriterionLoss()(logits, torch.tensor([[0.0], [0.5]]), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            CriterionLoss()(logits, torch.tensor([[0.0], [1.0]]), torch.tensor([1.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
