"""The domain gate must be deterministic, pass in-domain inputs, gate far-OOD inputs, and
downgrade predict_one to a disclosed single-view report -- never a silent dual-view one.

N3 (abstention_under_shift.json) showed criterion margins get MORE confident out-of-domain,
so they cannot gate; the bank-similarity gate (medguard_v2/perception/domain_check.py) is the
mechanism that can. These tests pin its contract:

1. **Frozen identity.** Two independently built gates agree on bank hash and threshold, and
   ``summary()`` quotes bank hash + k + threshold + the valid-only fit provenance.
2. **In-domain passes.** The valid split (the fitting population) passes at ~99% by
   construction, and its median-score case is never flagged.
3. **Far-OOD is gated.** A synthetic vector pointing AWAY from the bank (negative mean) falls
   far below threshold; a wrong-dimension query raises instead of scoring garbage.
4. **The downgrade path is honest.** A gated embedding yields a SingleViewContract whose
   render carries the mandated disclosure (bank similarity s < t, classifier-also-suspect
   wording), an EMPTY criteria dict (the checklist is absent, not imputed), and the domain
   block in prediction + receipt. ``force_dual_view=True`` bypasses -- recorded, not silent.
5. **The artifact matches the code.** domain_gate_validation.json's registered gate carries
   the same bank hash / k / threshold as ``load_domain_gate()``, and its threshold provenance
   says valid-only (BCN never fit on).

All tests run on the frozen feature caches -- the encoder is never loaded here (that identity
is scripts/dev/smoke_predict_one.py's job, from raw JPGs).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from medguard_v2.evidence.guideline_binding import CATALOG_PATH

REPO = Path(__file__).resolve().parents[2]
FEATURES = REPO / "research" / "evidence" / "derm7pt_v1" / "features" / "dermlip" / "features.npz"
FEATURES_TTA = (
    REPO / "research" / "evidence" / "derm7pt_v1" / "features_phase1" / "dermlip_tta"
    / "features.npz"
)
DATASET = REPO / "release_v0" / "meta" / "meta.csv"
VALIDATION_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "domain_gate_validation.json"

_READY = FEATURES.is_file() and DATASET.is_file()


@unittest.skipUnless(_READY, "dermlip feature cache or release_v0 dataset absent")
class DomainGateCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from medguard_v2.data.derm7pt import load_cases
        from medguard_v2.perception.domain_check import (
            _load_dermlip_features,
            load_domain_gate,
        )

        cls.gate = load_domain_gate()
        cls.features = _load_dermlip_features()
        cls.cases = load_cases()
        cls.train_matrix = np.stack(
            [cls.features[str(c.case_num)] for c in cls.cases if c.split == "train"]
        )
        cls.valid_matrix = np.stack(
            [cls.features[str(c.case_num)] for c in cls.cases if c.split == "valid"]
        )
        cls.far_ood = -cls.train_matrix.mean(axis=0)  # points away from the whole bank

    # -- identity / determinism ---------------------------------------------------------------

    def test_two_fresh_builds_agree_on_hash_and_threshold(self):
        from medguard_v2.perception.domain_check import (
            DOMAIN_GATE_K,
            build_train_bank,
            fit_valid_threshold,
        )

        first = build_train_bank(self.features, self.cases)
        second = build_train_bank(self.features, self.cases)
        self.assertEqual(first.bank_hash, second.bank_hash)
        self.assertEqual(first.bank_hash, self.gate.bank_hash)
        fit_a = fit_valid_threshold(first, self.valid_matrix, k=DOMAIN_GATE_K)
        fit_b = fit_valid_threshold(second, self.valid_matrix, k=DOMAIN_GATE_K)
        self.assertEqual(fit_a["threshold"], fit_b["threshold"])
        self.assertEqual(fit_a["threshold"], self.gate.threshold)

    def test_summary_quotes_bank_hash_k_threshold_and_valid_only_provenance(self):
        summary = self.gate.summary()
        self.assertEqual(summary["k"], self.gate.k)
        self.assertAlmostEqual(summary["threshold"], self.gate.threshold, places=6)
        self.assertEqual(summary["bank"]["bank_hash"], self.gate.bank_hash)
        self.assertEqual(summary["bank"]["n_entries"], self.train_matrix.shape[0])
        provenance = summary["threshold_provenance"]
        self.assertEqual(provenance["fit_population"], "derm7pt-valid")
        self.assertFalse(provenance["bcn_used_for_threshold"])
        self.assertEqual(provenance["n_fit"], self.valid_matrix.shape[0])

    # -- in-domain passes, far-OOD does not ---------------------------------------------------

    def test_valid_split_passes_at_the_constructed_rate(self):
        scores = self.gate.score(self.valid_matrix)
        pass_fraction = float((scores >= self.gate.threshold).mean())
        self.assertGreaterEqual(pass_fraction, 0.97)  # p1 threshold => ~99% by construction

    def test_median_valid_case_is_not_flagged(self):
        scores = self.gate.score(self.valid_matrix)
        median_case = self.valid_matrix[int(np.argsort(scores)[len(scores) // 2])]
        evaluation = self.gate.evaluate(median_case)
        self.assertFalse(evaluation["below_threshold"])
        self.assertEqual(evaluation["bank_hash"], self.gate.bank_hash)

    def test_synthetic_far_ood_vector_is_gated(self):
        evaluation = self.gate.evaluate(self.far_ood)
        self.assertTrue(evaluation["below_threshold"])
        self.assertLess(evaluation["score"], 0.0)  # anti-parallel to the bank's cone

    def test_wrong_dimension_query_raises_instead_of_scoring(self):
        with self.assertRaises(ValueError):
            self.gate.evaluate(np.ones(526))  # a promoted 526-d vector fed by mistake


@unittest.skipUnless(
    _READY and FEATURES_TTA.is_file() and CATALOG_PATH.exists(),
    "feature caches, dataset or guideline catalog absent",
)
class PredictOneDomainDowngradeTest(unittest.TestCase):
    """The embedding-level core: gated -> disclosed single view; forced -> recorded dual view."""

    @classmethod
    def setUpClass(cls):
        from medguard_v2.data.derm7pt import load_cases
        from medguard_v2.perception.domain_check import _load_dermlip_features, load_domain_gate

        cls.gate = load_domain_gate()
        cls.features = _load_dermlip_features()
        tta = np.load(FEATURES_TTA)
        cls.tta_features = {str(c): v for c, v in zip(tta["case_num"], tta["features"])}
        cases = load_cases()
        train = np.stack([cls.features[str(c.case_num)] for c in cases if c.split == "train"])
        cls.far_ood = -train.mean(axis=0)
        valid = [c for c in cases if c.split == "valid"]
        scores = cls.gate.score(
            np.stack([cls.features[str(c.case_num)] for c in valid])
        )
        cls.valid_case = valid[int(np.argsort(scores)[len(scores) // 2])]  # comfortably in-domain

    def test_gated_embedding_downgrades_to_a_disclosed_single_view(self):
        from medguard_v2.evidence.single_view import SingleViewContract
        from medguard_v2.pipeline.predict_one import predict_from_embeddings

        pred = predict_from_embeddings(self.far_ood, case_id="ood-1")
        self.assertIsInstance(pred.contract, SingleViewContract)
        self.assertTrue(pred.domain["gated"])
        self.assertTrue(pred.domain["below_threshold"])
        self.assertEqual(pred.criteria, {})  # the checklist is absent, never imputed
        self.assertIn("outside the checklist's validated domain", pred.report_markdown)
        self.assertIn("weighted accordingly", pred.report_markdown)
        self.assertIn("bank similarity", pred.contract.checklist_unavailable_reason)
        self.assertTrue(pred.contract.safe_to_display)
        self.assertEqual(pred.contract.receipt["domain_gate"]["bank_hash"], self.gate.bank_hash)

    def test_disclosure_quotes_the_actual_score_and_threshold(self):
        from medguard_v2.pipeline.predict_one import predict_from_embeddings

        pred = predict_from_embeddings(self.far_ood, case_id="ood-2")
        expected = (
            f"bank similarity {pred.domain['score']:.2f} < {pred.domain['threshold']:.2f}"
        )
        self.assertIn(expected, pred.contract.checklist_unavailable_reason)

    def test_force_dual_view_bypasses_the_gate_and_is_recorded(self):
        from medguard_v2.criteria.schema import CRITERION_NAMES
        from medguard_v2.evidence.contract import DecisionContract
        from medguard_v2.pipeline.predict_one import predict_from_embeddings

        pred = predict_from_embeddings(
            self.far_ood, self.far_ood, case_id="ood-forced", force_dual_view=True
        )
        self.assertIsInstance(pred.contract, DecisionContract)
        self.assertFalse(pred.domain["gated"])
        self.assertTrue(pred.domain["below_threshold"])  # the truth is still recorded
        self.assertTrue(pred.domain["forced_dual_view"])
        self.assertEqual(set(pred.criteria), set(CRITERION_NAMES))

    def test_in_domain_embedding_stays_dual_view_with_domain_block(self):
        from medguard_v2.evidence.contract import DecisionContract
        from medguard_v2.pipeline.predict_one import predict_from_embeddings

        key = str(self.valid_case.case_num)
        pred = predict_from_embeddings(
            self.features[key],
            self.tta_features[key],
            sex=self.valid_case.sex,
            location=self.valid_case.location,
            elevation=self.valid_case.elevation,
            case_id=f"derm7pt-{key}",
        )
        self.assertIsInstance(pred.contract, DecisionContract)
        self.assertFalse(pred.domain["gated"])
        self.assertFalse(pred.domain["forced_dual_view"])
        self.assertEqual(len(pred.criteria), 7)
        self.assertEqual(pred.contract.receipt["domain_gate"]["gated"], False)

    def test_dual_view_without_tta_embedding_is_refused(self):
        from medguard_v2.pipeline.predict_one import predict_from_embeddings

        key = str(self.valid_case.case_num)
        with self.assertRaises(ValueError):
            predict_from_embeddings(self.features[key], None, case_id=f"derm7pt-{key}")


@unittest.skipUnless(
    _READY and VALIDATION_JSON.is_file(), "domain_gate_validation.json not generated"
)
class ValidationArtifactProvenanceTest(unittest.TestCase):
    """The frozen artifact must agree with the code and record the valid-only threshold fit."""

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))

    def test_registered_gate_matches_the_loaded_gate(self):
        from medguard_v2.perception.domain_check import DOMAIN_GATE_K, load_domain_gate

        gate = load_domain_gate()
        registered = self.report["registered_gate"]
        self.assertEqual(registered["k"], DOMAIN_GATE_K)
        self.assertEqual(registered["bank"]["bank_hash"], gate.bank_hash)
        self.assertAlmostEqual(registered["threshold"], gate.threshold, places=6)
        self.assertTrue(self.report["k_selection"]["registered_matches_selected"])

    def test_threshold_provenance_is_valid_only_and_bcn_test_never_read(self):
        provenance = self.report["registered_gate"]["threshold_provenance"]
        self.assertEqual(provenance["fit_population"], "derm7pt-valid")
        self.assertFalse(provenance["bcn_used_for_threshold"])
        self.assertEqual(provenance["percentile"], 1.0)
        self.assertFalse(self.report["boundaries"]["bcn_test_read"])
        # k selection DOES read BCN (pre-registered 3-point grid) -- disclosed, not hidden.
        self.assertIn("k_selection", self.report["boundaries"])
        self.assertIn("BCN", self.report["boundaries"]["k_selection"])


if __name__ == "__main__":
    unittest.main()
