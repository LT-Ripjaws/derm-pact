from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_hybrid_screen as screen
from dermpact.eval import dermpact_phase2_structured_facts as facts
from dermpact.eval import dermpact_phase2_structured_fact_audit as fact_audit


class DermPactPhase2StructuredFactsTest(unittest.TestCase):
    def test_fact_contract_updates_diagnosis_only(self) -> None:
        torch.manual_seed(11)
        model = facts.BoundedStructuredFactHead().eval()
        image = torch.randn(4, 512)
        values = torch.randn(4, facts.FACT_VECTOR_DIM)
        available = torch.ones(4, len(facts.FACT_FIELDS), dtype=torch.bool)
        with torch.inference_mode():
            absent = model(image)
            present = model(image, values, available)
        self.assertTrue(torch.equal(absent["criteria"], present["criteria"]))
        self.assertFalse(torch.equal(absent["diagnosis"], present["diagnosis"]))
        self.assertEqual(present["diagnosis"].shape, (4, len(screen.GLOBAL_CLASSES)))
        self.assertEqual(present["melanoma"].shape, (4,))

    def test_all_missing_is_bit_exact_for_arbitrary_values(self) -> None:
        torch.manual_seed(12)
        model = facts.BoundedStructuredFactHead().eval()
        image = torch.randn(7, 512)
        arbitrary = torch.randn(7, facts.FACT_VECTOR_DIM) * 1_000
        missing = torch.zeros(7, len(facts.FACT_FIELDS), dtype=torch.bool)
        with torch.inference_mode():
            absent = model(image)
            masked = model(image, arbitrary, missing)
        for name in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(absent[name], masked[name]))
        self.assertEqual(float(masked["diagnosis_fact_delta"].abs().max()), 0.0)
        self.assertEqual(float(masked["melanoma_fact_delta"].abs().max()), 0.0)

    def test_mixed_batch_masks_each_missing_field(self) -> None:
        torch.manual_seed(13)
        model = facts.BoundedStructuredFactHead().eval()
        image = torch.randn(3, 512)
        values = torch.randn(3, facts.FACT_VECTOR_DIM)
        available = torch.tensor(
            [[True, False, False, False], [False, False, False, False], [False, True, True, False]]
        )
        changed = values.clone()
        for field_index, field in enumerate(facts.FACT_FIELDS):
            missing_rows = ~available[:, field_index]
            changed[missing_rows, facts.FIELD_SLICES[field]] += 10_000
        with torch.inference_mode():
            first = model(image, values, available)
            second = model(image, changed, available)
        for name in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(first[name], second[name]))

    def test_updates_are_bounded(self) -> None:
        model = facts.BoundedStructuredFactHead().eval()
        with torch.inference_mode():
            output = model(
                torch.randn(20, 512),
                torch.randn(20, facts.FACT_VECTOR_DIM) * 100,
                torch.ones(20, len(facts.FACT_FIELDS), dtype=torch.bool),
            )
        self.assertLessEqual(
            float(output["diagnosis_fact_delta"].abs().max()), facts.MAX_LOGIT_DELTA
        )
        self.assertLessEqual(
            float(output["melanoma_fact_delta"].abs().max()), facts.MAX_LOGIT_DELTA
        )

    def test_source_specific_encoding_and_missing_sentinels(self) -> None:
        ham = [
            {"metadata": {"age": -1, "sex": "unknown", "localization": "back"}},
            {"metadata": {"age": 0, "sex": "female", "localization": "hand"}},
        ]
        values, available, labels = facts.encode_structured_facts("ham", ham)
        self.assertEqual(values.shape, (2, facts.FACT_VECTOR_DIM))
        self.assertEqual(available[0].tolist(), [False, False, True, False])
        self.assertEqual(available[1].tolist(), [True, True, True, False])
        self.assertEqual(labels[0]["site"], "posterior_torso")
        self.assertEqual(labels[1]["site"], "acral")

        derm = [{"structured_facts": {"sex": "male", "location": "chest", "elevation": "flat"}}]
        _, derm_available, derm_labels = facts.encode_structured_facts("derm", derm)
        self.assertEqual(derm_available[0].tolist(), [False, True, True, True])
        self.assertEqual(derm_labels[0]["site"], "anterior_torso")

    def test_unknown_nonmissing_categories_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            facts.encode_structured_facts(
                "bcn",
                [{"metadata": {"age_approx": "40", "sex": "other", "anatom_site_general": "head/neck"}}],
            )

    def test_unavailable_encoded_positions_are_exact_zero(self) -> None:
        rows = [{"structured_facts": {"sex": "female", "location": "back", "elevation": "nodular"}}]
        values, available, _ = facts.encode_structured_facts("derm", rows)
        age = values[0, facts.FIELD_SLICES["age"]]
        self.assertTrue(np.array_equal(age, np.zeros_like(age)))
        self.assertFalse(bool(available[0, facts.FACT_FIELDS.index("age")]))

    def test_real_development_fact_matrices_have_the_frozen_shapes(self) -> None:
        data = facts._data()
        self.assertEqual(data["derm_fact_values"].shape, (616, facts.FACT_VECTOR_DIM))
        self.assertEqual(data["ham_fact_values"].shape, (7008, facts.FACT_VECTOR_DIM))
        self.assertEqual(data["bcn_fact_values"].shape, (7405, facts.FACT_VECTOR_DIM))
        self.assertEqual(int(data["derm_fact_available"].sum()), 1848)
        self.assertEqual(int(data["ham_fact_available"].sum()), 20784)
        self.assertEqual(int(data["bcn_fact_available"].sum()), 22037)

    def test_protocol_fixes_the_diagnosis_only_contract_and_closed_scope(self) -> None:
        with (
            mock.patch.object(facts.screen, "_write_json", return_value="hash"),
            mock.patch.object(facts, "sha256_file", return_value="sha"),
        ):
            protocol = facts.build_protocol()
        self.assertEqual(
            protocol["fixed_model"]["max_abs_logit_delta_per_output"], 0.25
        )
        self.assertEqual(protocol["fixed_model"]["checklist_fact_path"].split(";")[0], "none")
        self.assertIn("forced-missing", protocol["promotion_rule"]["declared_endpoint"])
        self.assertEqual(protocol["hard_scope"]["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(protocol["hard_scope"]["ph2_rows_loaded"], 0)

    def test_counterfactuals_change_only_the_requested_field(self) -> None:
        values = np.zeros((2, facts.FACT_VECTOR_DIM), dtype="float32")
        values[:, facts.FIELD_SLICES["age"]] = np.asarray([[-0.2], [0.8]])
        values[:, facts.FIELD_SLICES["sex"]] = np.asarray([[1, 0], [0, 1]])
        values[:, facts.FIELD_SLICES["site"]][:, 0] = 1
        values[:, facts.FIELD_SLICES["elevation"]][:, 1] = 1
        for field in facts.FACT_FIELDS:
            changed = fact_audit._counterfactual(values, field)
            self.assertFalse(np.array_equal(changed[:, facts.FIELD_SLICES[field]], values[:, facts.FIELD_SLICES[field]]))
            for other in set(facts.FACT_FIELDS) - {field}:
                self.assertTrue(
                    np.array_equal(
                        changed[:, facts.FIELD_SLICES[other]],
                        values[:, facts.FIELD_SLICES[other]],
                    )
                )

    def test_availability_signature_audit_exposes_derm_pattern(self) -> None:
        data = {
            "derm_fact_available": np.asarray([[False, True, True, True]]),
            "ham_fact_available": np.asarray([[True, True, True, False]]),
            "bcn_fact_available": np.asarray([[True, True, True, False]]),
        }
        audit = fact_audit._availability_signature_audit(data)
        self.assertEqual(audit["signature_counts"]["derm"], {"0111": 1})
        self.assertEqual(audit["derm_vs_non_derm_signature_overlap"], [])


if __name__ == "__main__":
    unittest.main()
