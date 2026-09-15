from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_clinical_photo as clinical
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


class DermPactPhase2ClinicalPhotoTest(unittest.TestCase):
    def test_output_shapes_match_the_selected_perception_contract(self) -> None:
        model = clinical.GatedClinicalPhotoHead()
        outputs = model(torch.randn(5, 512), torch.randn(5, 512), torch.ones(5, dtype=torch.bool))
        self.assertEqual(outputs["diagnosis"].shape, (5, len(screen.GLOBAL_CLASSES)))
        self.assertEqual(outputs["melanoma"].shape, (5,))
        self.assertEqual(outputs["criteria"].shape, (5, len(screen.CRITERIA)))

    def test_missing_mask_makes_clinical_contribution_exactly_zero(self) -> None:
        torch.manual_seed(5)
        model = clinical.GatedClinicalPhotoHead().eval()
        derm = torch.randn(8, 512)
        arbitrary_clinical = torch.randn(8, 512) * 1000
        with torch.inference_mode():
            absent = model(derm)
            masked = model(derm, arbitrary_clinical, torch.zeros(8, dtype=torch.bool))
        for key in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(absent[key], masked[key]))
        self.assertEqual(float(masked["clinical_gate"].abs().max()), 0.0)
        self.assertEqual(float(masked["clinical_residual_norm"].max()), 0.0)

    def test_masked_rows_are_unchanged_in_a_mixed_batch(self) -> None:
        torch.manual_seed(6)
        model = clinical.GatedClinicalPhotoHead().eval()
        derm = torch.randn(4, 512)
        photo = torch.randn(4, 512)
        mask = torch.tensor([True, False, True, False])
        with torch.inference_mode():
            absent = model(derm)
            mixed = model(derm, photo, mask)
        for key in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(absent[key][~mask], mixed[key][~mask]))

    def test_protocol_fixes_dropout_and_keeps_protected_sets_closed(self) -> None:
        with (
            mock.patch.object(clinical.screen, "_write_json", return_value="hash"),
            mock.patch.object(clinical, "sha256_file", return_value="sha"),
        ):
            protocol = clinical.build_protocol()
        self.assertEqual(protocol["fixed_model"]["clinical_modality_dropout"], 0.5)
        self.assertEqual(protocol["hard_scope"]["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(protocol["hard_scope"]["ph2_rows_loaded"], 0)
        self.assertIn("forced-missing", protocol["promotion_rule"]["declared_endpoint"])

    def test_batch_loss_masks_consistency_to_derm7pt(self) -> None:
        model = clinical.GatedClinicalPhotoHead()
        rng = np.random.default_rng(7)
        data = {
            "ham_features": rng.normal(size=(4, 512)).astype("float32"),
            "ham_global": np.asarray([0, 1, 0, 1], dtype="int64"),
            "ham_melanoma": np.asarray([1, 0, 1, 0], dtype="float32"),
        }
        weights = {
            "ham_diagnosis": np.ones(4, dtype="float32"),
            "ham_melanoma": np.ones(4, dtype="float32"),
        }
        loss, components = clinical._batch_loss(
            model,
            dataset="ham",
            indices=np.arange(4),
            data=data,
            weights=weights,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(components["consistency"].detach()), 0.0)
        self.assertEqual(float(components["clinical_gate_mean"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
