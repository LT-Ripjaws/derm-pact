"""WP-X measured XAI: region geometry, perturbation, full-recompute proof, receipts, and clinical-photo contribution."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from dermpact.eval.dermpact_phase2_5_window_cache import (
    GRID_SIZE, TARGET_SIZE, WINDOW_SIZE, aspect_preserve_pad,
)
from dermpact.core import xai


def _random_image(size, seed=0):
    w, h = size
    arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class RegionGeometryTest(unittest.TestCase):
    def test_sixteen_regions_tile_padded_square(self):
        regions = xai.compute_region_geometry((900, 600))
        self.assertEqual(len(regions), 16)
        self.assertEqual({r["index"] for r in regions}, set(range(16)))
        # padded boxes exactly tile the 896x896 square with no gaps/overlaps
        covered = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
        for r in regions:
            y0, y1, x0, x1 = r["padded_box"]
            self.assertFalse(covered[y0:y1, x0:x1].any(), "regions overlap")
            covered[y0:y1, x0:x1] = True
        self.assertTrue(covered.all())

    def test_mapping_matches_real_padding_landscape_and_portrait(self):
        for size in [(1024, 768), (600, 900)]:
            w, h = size
            padded, padding = aspect_preserve_pad(Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)))
            expected = padding  # (pad_top, pad_bottom, pad_left, pad_right)
            params = xai._resize_and_pad_params(size)
            self.assertEqual(
                (params["pad_top"], params["pad_bottom"], params["pad_left"], params["pad_right"]),
                expected,
            )

    def test_padding_only_regions_reported_and_skipped(self):
        # Extremely wide image forces rows entirely inside the vertical padding.
        regions = xai.compute_region_geometry((2000, 50))
        statuses = [r["status"] for r in regions]
        self.assertIn("padding_only", statuses)
        for r in regions:
            if r["status"] == "padding_only":
                self.assertIsNone(r["original_box"])
                self.assertEqual(r["area_original_px"], 0.0)
            else:
                self.assertIsNotNone(r["original_box"])
                self.assertGreater(r["area_original_px"], 0.0)

    def test_area_recorded_in_original_pixels(self):
        regions = xai.compute_region_geometry((896, 896))  # no padding, exact 224x224 windows
        for r in regions:
            self.assertEqual(r["status"], "active")
            x0, y0, x1, y1 = r["original_box"]
            self.assertAlmostEqual((x1 - x0) * (y1 - y0), r["area_original_px"])
            self.assertAlmostEqual(x1 - x0, WINDOW_SIZE)
            self.assertAlmostEqual(y1 - y0, WINDOW_SIZE)

    def test_equal_area_controls_are_equal_area_and_distinct_per_region(self):
        controls = xai.equal_area_control_geometry((896, 896))
        self.assertEqual(len(controls), 16)
        boxes = set()
        for c in controls:
            x0, y0, x1, y1 = c["original_box"]
            self.assertAlmostEqual((x1 - x0), WINDOW_SIZE)
            self.assertAlmostEqual((y1 - y0), WINDOW_SIZE)
            boxes.add(c["original_box"])
        self.assertGreater(len(boxes), 1, "controls should not all collapse to one fixed region")


class PerturbationTest(unittest.TestCase):
    def test_local_mean_flattens_region_to_its_own_mean(self):
        image = _random_image((300, 300), seed=1)
        box = (50.0, 50.0, 150.0, 150.0)
        out = xai.apply_local_mean(image, box)
        arr = np.array(out)
        region = arr[50:150, 50:150]
        self.assertTrue(np.allclose(region, region[0, 0]))
        # outside the box unchanged
        self.assertTrue(np.array_equal(arr[:50], np.array(image)[:50]))

    def test_gaussian_blur_changes_region_only(self):
        image = _random_image((300, 300), seed=2)
        box = (50.0, 50.0, 150.0, 150.0)
        out = xai.apply_gaussian_blur(image, box)
        arr_in, arr_out = np.array(image), np.array(out)
        self.assertFalse(np.array_equal(arr_in[50:150, 50:150], arr_out[50:150, 50:150]))
        self.assertTrue(np.array_equal(arr_in[:50], arr_out[:50]))

    def test_both_methods_use_identical_coordinates(self):
        image = _random_image((300, 300), seed=3)
        box = (10.0, 10.0, 90.0, 90.0)
        a = xai.apply_local_mean(image, box)
        b = xai.apply_gaussian_blur(image, box)
        # both perturbed exactly the same footprint (rest of image untouched identically)
        arr_a, arr_b, arr_in = np.array(a), np.array(b), np.array(image)
        self.assertTrue(np.array_equal(arr_a[90:], arr_in[90:]))
        self.assertTrue(np.array_equal(arr_b[90:], arr_in[90:]))


class FullRecomputeTest(unittest.TestCase):
    """Proves every perturbed image reaches global + all 16 windows + the frozen baseline (never a reused original-image prefix)."""

    def test_every_perturbation_recomputes_all_three_streams(self):
        image = _random_image((896, 896), seed=4)
        runtime = xai.FakePerceptionRuntime()
        result = xai.run_region_sensitivity(runtime, image, None)
        self.assertEqual(result["status"], "complete")
        # 1 no_op + 16 regions * 2 methods * (primary + control) = 65
        self.assertEqual(len(runtime.prefix_calls), 65)
        for call in runtime.prefix_calls:
            self.assertEqual(call["streams_recomputed"], ("global", "windows", "frozen_baseline"))

    def test_perturbed_images_are_actually_distinct_pixels(self):
        image = _random_image((896, 896), seed=5)
        runtime = xai.FakePerceptionRuntime()
        xai.run_region_sensitivity(runtime, image, None)
        hashes = {c["image_sha256"] for c in runtime.prefix_calls}
        # every perturbation produced genuinely different pixels (no accidental reuse)
        self.assertEqual(len(hashes), len(runtime.prefix_calls))

    def test_clinical_tokens_unchanged_for_dermoscopic_perturbations(self):
        image = _random_image((896, 896), seed=6)
        clinical = _random_image((256, 256), seed=7)
        runtime = xai.FakePerceptionRuntime()
        xai.run_region_sensitivity(runtime, image, clinical)
        clinical_hashes = {c["clinical_sha256"] for c in runtime.prefix_calls}
        self.assertEqual(len(clinical_hashes), 1, "clinical image must be identical across all perturbations")


class RegionSensitivitySweepTest(unittest.TestCase):
    def test_complete_sweep_ranked_by_local_mean_reports_both_methods(self):
        image = _random_image((896, 896), seed=8)
        result = xai.run_region_sensitivity(xai.FakePerceptionRuntime(), image, None)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["members_measured"], 5)
        self.assertEqual(len(result["results"]), 16)
        ranks = [r["rank_key_local_mean_melanoma_head_abs_delta"] for r in result["results"]]
        self.assertEqual(ranks, sorted(ranks, reverse=True))
        for region in result["results"]:
            self.assertEqual(set(region["methods"]), {"local_mean", "gaussian_blur"})
            self.assertEqual(set(region["equal_area_controls"]), {"local_mean", "gaussian_blur"})
            self.assertIn("sign_agreement_across_methods", region)
            self.assertIn("std_across_methods_melanoma_head", region)
        self.assertEqual(set(result["heatmap"]), {"local_mean", "gaussian_blur"})
        for method_grid in result["heatmap"].values():
            self.assertEqual(set(method_grid), {"melanoma_head", "diagnosis_head_melanoma"})
            grid = method_grid["melanoma_head"]
            self.assertEqual(len(grid), GRID_SIZE)
            self.assertEqual(len(grid[0]), GRID_SIZE)

    def test_fold_failure_yields_failed_status_no_synthetic_heatmap(self):
        image = _random_image((896, 896), seed=9)
        runtime = xai.FakePerceptionRuntime(fail_folds=[3])
        result = xai.run_region_sensitivity(runtime, image, None)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["members_measured"], 4)
        self.assertIn("3", result["failures"])
        self.assertIsNone(result["results"])
        self.assertIsNone(result["heatmap"])

    def test_determinism_same_setup_same_results(self):
        image = _random_image((896, 896), seed=10)
        r1 = xai.run_region_sensitivity(xai.FakePerceptionRuntime(), image, None)
        r2 = xai.run_region_sensitivity(xai.FakePerceptionRuntime(), image, None)
        self.assertEqual(r1["results"], r2["results"])
        self.assertEqual(r1["heatmap"], r2["heatmap"])

    def test_cancel_check_raises_cancelled(self):
        image = _random_image((896, 896), seed=11)
        calls = {"n": 0}

        def cancel_after_a_few():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(xai.Cancelled):
            xai.run_region_sensitivity(xai.FakePerceptionRuntime(), image, None, cancel_check=cancel_after_a_few)


class NoOpParityTest(unittest.TestCase):
    def test_matching_raw_outputs_have_zero_parity_diff(self):
        calibrators = xai._identity_calibrators()
        raw_fold_outputs = []
        no_op_member_raw = {}
        rng = np.random.default_rng(0)
        for fold in range(5):
            diag, mel, crit = rng.normal(size=9), float(rng.normal()), rng.normal(size=7)
            raw_fold_outputs.append({"diagnosis": diag.tolist(), "melanoma": mel, "criteria": crit.tolist()})
            member = xai._calibrated_member(
                {"diagnosis": np.array([diag]), "melanoma": np.array([mel]), "criteria": np.array([crit])},
                fold, calibrators)
            no_op_member_raw[str(fold)] = member
        prediction = {"raw_fold_outputs": raw_fold_outputs}
        parity = xai.no_op_parity(prediction, calibrators, no_op_member_raw)
        self.assertLess(parity["max_abs_diff"], 1e-9)

    def test_mismatched_raw_outputs_detected(self):
        calibrators = xai._identity_calibrators()
        raw_fold_outputs = [{"diagnosis": [0.0] * 9, "melanoma": 0.0, "criteria": [0.0] * 7} for _ in range(5)]
        no_op_member_raw = {
            str(fold): {"diagnosis_probabilities": [1.0] + [0.0] * 8, "binary_probabilities": [0.9] + [0.5] * 7}
            for fold in range(5)
        }
        prediction = {"raw_fold_outputs": raw_fold_outputs}
        parity = xai.no_op_parity(prediction, calibrators, no_op_member_raw)
        self.assertGreater(parity["max_abs_diff"], 0.1)


class ClinicalContributionTest(unittest.TestCase):
    def test_present_reports_real_deltas(self):
        image = _random_image((896, 896), seed=12)
        clinical = _random_image((256, 256), seed=13)
        result = xai.run_clinical_contribution(xai.FakePerceptionRuntime(), image, clinical, "present")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["members_measured"], 5)
        self.assertEqual(set(result["targets"]), set(xai.TARGETS))
        for target in xai.TARGETS:
            self.assertEqual(len(result["targets"][target]["per_member_delta"]), 5)

    def test_absent_and_duplicate_and_not_supplied_are_not_applicable(self):
        image = _random_image((896, 896), seed=14)
        for availability in ("absent", "duplicate_dermoscopy_excluded", "not_supplied"):
            result = xai.run_clinical_contribution(xai.FakePerceptionRuntime(), image, None, availability)
            self.assertEqual(result["status"], "not_applicable")
            self.assertNotIn("targets", result)

    def test_fold_failure_reported_not_fabricated(self):
        image = _random_image((896, 896), seed=15)
        clinical = _random_image((256, 256), seed=16)
        runtime = xai.FakePerceptionRuntime(fail_folds=[1])
        result = xai.run_clinical_contribution(runtime, image, clinical, "present")
        self.assertEqual(result["status"], "failed")
        self.assertIn("1", result["failures"])


class CacheKeyAndReceiptTest(unittest.TestCase):
    def _record_like(self, **overrides):
        base = {
            "inputs": {"dermoscopic": {"sanitized_sha256": "a" * 64},
                       "clinical": {"sanitized_sha256": None}},
            "perception": {"checkpoint_bindings": [{"sha256": f"c{i}" * 8} for i in range(5)],
                           "calibration_sha256": "cal" * 8},
            "identity": {"case_id": "derm7pt-case-22", "record_version": 1},
            "record_sha256": "b" * 64,
        }
        base.update(overrides)
        return base

    def _cache_key(self):
        return xai.build_cache_key(
            dermoscopic_sha256="a" * 64, clinical_sha256=None, case_id="derm7pt-case-22",
            record_version=1, record_sha256="b" * 64,
            checkpoint_shas=[f"c{i}" * 8 for i in range(5)],
            preprocessing_version="p1", calibration_sha256="cal" * 8,
            target_heads=xai.TARGETS, methods=xai.METHODS, sigma=xai.GAUSSIAN_SIGMA_PX, seed=0,
        )

    def test_cache_key_contains_every_required_element(self):
        key = self._cache_key()
        for field in ["dermoscopic_sanitized_sha256", "clinical_sanitized_sha256", "case_id",
                      "record_version", "record_sha256", "checkpoint_shas", "preprocessing_version",
                      "calibration_sha256", "target_heads", "grid_geometry_signature", "methods",
                      "sigma", "seed", "protocol_version"]:
            self.assertIn(field, key)

    def test_missing_or_bad_no_op_parity_downgrades_complete_sweep(self):
        sweep = {"status": "complete", "n_members": 5, "members_measured": 5,
                 "results": [{"region_index": 0}], "heatmap": {"local_mean": [[0.0]]}}
        missing = xai.build_region_receipt(
            cache_key=self._cache_key(), sweep_result=sweep, started_at="t0", finished_at="t1",
            runtime_seconds=1.0, peak_rss_bytes=1, no_op_parity_result=None,
        )
        self.assertEqual(missing["status"], "failed")
        bad = xai.build_region_receipt(
            cache_key=self._cache_key(), sweep_result=sweep, started_at="t0", finished_at="t1",
            runtime_seconds=1.0, peak_rss_bytes=1,
            no_op_parity_result={"max_abs_diff": 1e-3, "atol_target": 1e-6},
        )
        self.assertEqual(bad["status"], "failed")

    def test_sealed_complete_receipt_rejects_content_tamper(self):
        sweep = {"status": "complete", "n_members": 5, "members_measured": 5, "failures": {},
                 "results": [{"region_index": 0}], "heatmap": {"local_mean": [[0.0]]}}
        receipt = xai.build_region_receipt(
            cache_key=self._cache_key(), sweep_result=sweep, started_at="t0", finished_at="t1",
            runtime_seconds=1.0, peak_rss_bytes=1,
            no_op_parity_result={"max_abs_diff": 0.0, "atol_target": 1e-6},
        )
        self.assertEqual(receipt["status"], "complete")
        self.assertTrue(xai.verify_receipt_content(receipt))
        receipt["runtime_seconds"] = 999.0
        with self.assertRaises(xai.ReceiptMismatch):
            xai.verify_receipt_content(receipt)

    def test_verify_accepts_matching_record(self):
        key = self._cache_key()
        receipt = xai.build_region_receipt(cache_key=key, sweep_result={"status": "complete", "n_members": 5, "members_measured": 5},
                                            started_at="t0", finished_at="t1", runtime_seconds=1.0,
                                            peak_rss_bytes=1, no_op_parity_result=None)
        self.assertTrue(xai.verify_receipt(receipt, self._record_like()))

    def test_verify_rejects_wrong_image_sha(self):
        key = self._cache_key()
        receipt = xai.build_region_receipt(cache_key=key, sweep_result={"status": "complete", "n_members": 5, "members_measured": 5},
                                            started_at="t0", finished_at="t1", runtime_seconds=1.0,
                                            peak_rss_bytes=1, no_op_parity_result=None)
        bad = self._record_like(inputs={"dermoscopic": {"sanitized_sha256": "z" * 64}, "clinical": {"sanitized_sha256": None}})
        with self.assertRaises(xai.ReceiptMismatch):
            xai.verify_receipt(receipt, bad)

    def test_verify_accepts_amended_record_same_image_and_reports_provenance(self):
        """A clinician amendment (new record sha/version, same image, weights and calibration) must not invalidate a measured-attribution receipt (brief 02); the provenance helper reports that the measurement predates the amendment."""
        key = self._cache_key()
        receipt = xai.build_region_receipt(cache_key=key, sweep_result={"status": "complete", "n_members": 5, "members_measured": 5},
                                            started_at="t0", finished_at="t1", runtime_seconds=1.0,
                                            peak_rss_bytes=1, no_op_parity_result=None)
        amended = self._record_like(record_sha256="z" * 64)
        amended["identity"] = {**amended["identity"], "record_version": 3}
        self.assertTrue(xai.verify_receipt(receipt, amended))
        prov = xai.receipt_provenance(receipt, amended)
        self.assertEqual(prov["measured_at_record_version"], 1)
        self.assertEqual(prov["current_record_version"], 3)
        self.assertFalse(prov["record_unchanged_since_measurement"])
        self.assertTrue(xai.receipt_provenance(receipt, self._record_like())["record_unchanged_since_measurement"])

    def test_verify_rejects_wrong_case_id(self):
        key = self._cache_key()
        receipt = xai.build_region_receipt(cache_key=key, sweep_result={"status": "complete", "n_members": 5, "members_measured": 5},
                                            started_at="t0", finished_at="t1", runtime_seconds=1.0,
                                            peak_rss_bytes=1, no_op_parity_result=None)
        bad = self._record_like()
        bad["identity"] = {**bad["identity"], "case_id": "derm7pt-case-999"}
        with self.assertRaises(xai.ReceiptMismatch):
            xai.verify_receipt(receipt, bad)

    def test_verify_rejects_wrong_checkpoint_set(self):
        key = self._cache_key()
        receipt = xai.build_region_receipt(cache_key=key, sweep_result={"status": "complete", "n_members": 5, "members_measured": 5},
                                            started_at="t0", finished_at="t1", runtime_seconds=1.0,
                                            peak_rss_bytes=1, no_op_parity_result=None)
        bad = self._record_like(perception={"checkpoint_bindings": [{"sha256": "different"}] * 5, "calibration_sha256": "cal" * 8})
        with self.assertRaises(xai.ReceiptMismatch):
            xai.verify_receipt(receipt, bad)

    def test_verify_rejects_wrong_calibration_sha(self):
        key = self._cache_key()
        receipt = xai.build_region_receipt(cache_key=key, sweep_result={"status": "complete", "n_members": 5, "members_measured": 5},
                                            started_at="t0", finished_at="t1", runtime_seconds=1.0,
                                            peak_rss_bytes=1, no_op_parity_result=None)
        bad = self._record_like(perception={"checkpoint_bindings": [{"sha256": f"c{i}" * 8} for i in range(5)],
                                             "calibration_sha256": "different"})
        with self.assertRaises(xai.ReceiptMismatch):
            xai.verify_receipt(receipt, bad)


if __name__ == "__main__":
    unittest.main()
