from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dermpact.eval import dermpact_phase2_4b_inputs as inputs


class DermPactPhase24BInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = inputs.resolve_development_inputs()

    def test_real_metadata_resolves_exact_development_scope(self) -> None:
        resolved = self.resolved
        self.assertEqual(len(resolved.cases), 616)
        self.assertEqual(len({case.group_id for case in resolved.cases}), 616)
        self.assertEqual(sum(resolved.clinical_available_mask), 609)
        self.assertEqual(resolved.clinical_available_mask.count(False), 7)
        self.assertTrue(all(case.source_split in {"train", "valid"} for case in resolved.cases))
        self.assertTrue(all(case.dermoscopic_path.is_file() for case in resolved.cases))
        self.assertTrue(
            all(
                case.clinical_path is not None and case.clinical_path.is_file()
                if case.clinical_available
                else case.clinical_path is None
                for case in resolved.cases
            )
        )

    def test_real_metadata_matches_locked_fold_ledger(self) -> None:
        ledger = json.loads(inputs.DEVELOPMENT_FOLDS.read_text(encoding="utf-8"))
        locked = {
            row["group_id"]: row["fold"] for row in ledger["derm7pt"]["folds"]
        }
        observed = {case.group_id: case.fold for case in self.resolved.cases}
        self.assertEqual(observed, locked)
        self.assertEqual(
            {fold: sum(case.fold == fold for case in self.resolved.cases) for fold in range(5)},
            inputs.FOLD_COUNTS,
        )

    def test_official_and_external_scope_counters_remain_zero(self) -> None:
        scope = self.resolved.hard_scope
        self.assertEqual(scope["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(scope["ham_calibration_or_final_rows_loaded"], 0)
        self.assertEqual(scope["bcn_validation_or_final_rows_loaded"], 0)
        self.assertEqual(scope["ph2_rows_loaded"], 0)
        self.assertEqual(scope["clinical_images_opened"], 0)
        self.assertEqual(scope["prefix_tokens_extracted"], 0)
        self.assertEqual(scope["training_steps"], 0)

    def test_facts_are_typed_and_missing_is_not_imputed(self) -> None:
        facts = [case.facts for case in self.resolved.cases]
        self.assertTrue(all(fact.age_years is None for fact in facts))
        self.assertTrue(all(not fact.available.age for fact in facts))
        self.assertTrue(all(fact.available.sex for fact in facts))
        self.assertTrue(all(fact.available.site for fact in facts))
        self.assertTrue(all(fact.available.elevation for fact in facts))
        missing = inputs.parse_prediagnostic_facts(
            {"structured_facts": {"sex": "unknown", "location": "", "elevation": "not-reported"}}
        )
        self.assertIsNone(missing.sex)
        self.assertIsNone(missing.site)
        self.assertIsNone(missing.elevation)
        self.assertEqual(missing.available.as_tuple(), (False, False, False, False))

    def test_reused_and_missing_clinical_paths_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "derm.jpg").write_bytes(b"derm")
            base = {
                "case_index": 0,
                "case_num": 1,
                "group_id": "derm7pt-case-1",
                "phase1_role": "development",
                "source_split": "train",
                "development_fold": 0,
                "structured_facts": {
                    "sex": "female",
                    "location": "back",
                    "elevation": "flat",
                },
                "modalities": {
                    "dermoscopic": {
                        "available": True,
                        "path": "derm.jpg",
                        "sha256": "1" * 64,
                    },
                    "clinical": {
                        "available": True,
                        "path": "derm.jpg",
                        "sha256": "1" * 64,
                        "distinct_from_dermoscopic": False,
                        "duplicate_of": "dermoscopic",
                    },
                },
            }
            reused = inputs.resolve_case_input(base, fold=0, images_root=root)
            self.assertFalse(reused.clinical_available)
            self.assertIsNone(reused.clinical_path)
            self.assertEqual(reused.clinical_unavailable_reason, "dermoscopy_reused")

            base["modalities"]["clinical"] = {
                "available": True,
                "path": "missing.jpg",
                "sha256": "2" * 64,
                "distinct_from_dermoscopic": True,
                "duplicate_of": None,
            }
            missing = inputs.resolve_case_input(base, fold=0, images_root=root)
            self.assertFalse(missing.clinical_available)
            self.assertIsNone(missing.clinical_path)
            self.assertEqual(missing.clinical_unavailable_reason, "missing")

    def test_block10_manifest_identity_is_deterministic_and_complete(self) -> None:
        config = inputs.Block10CacheConfig(
            model_id="pinned/model",
            model_revision="0123456789abcdef",
            checkpoint_sha256="a" * 64,
            preprocessing_sha256="b" * 64,
            hidden_size=768,
            tokens_per_image=197,
        )
        first = inputs.build_block10_cache_manifest_spec(self.resolved, config)
        second = inputs.build_block10_cache_manifest_spec(self.resolved, config)
        self.assertEqual(first, second)
        self.assertEqual(first["encoder_identity"]["shared_prefix_blocks"], 10)
        self.assertEqual(first["row_contract"]["n_cases"], 616)
        self.assertEqual(first["row_contract"]["n_available_clinical"], 609)
        self.assertIn("sha256", first["source_artifacts"]["phase1_inventory"])
        self.assertIn("sha256", first["source_artifacts"]["phase2_1_fold_ledger"])
        self.assertIn("sha256", first["source_artifacts"]["phase2_3_clinical_photo_audit"])
        self.assertIn("sha256", first["source_artifacts"]["phase2_3_structured_fact_support"])
        self.assertIn("fold_identity_sha256", first["row_contract"])
        self.assertIn("paired_paths_sha256", first["path_contract"])
        self.assertEqual(first["hard_scope"]["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(first["hard_scope"]["prefix_tokens_extracted"], 0)
        self.assertEqual(
            first["cache_identity_sha256"],
            inputs.canonical_hash(
                {key: value for key, value in first.items() if key != "cache_identity_sha256"}
            ),
        )
        changed_preprocess = inputs.Block10CacheConfig(
            model_id=config.model_id,
            model_revision=config.model_revision,
            checkpoint_sha256=config.checkpoint_sha256,
            preprocessing_sha256="c" * 64,
            hidden_size=config.hidden_size,
            tokens_per_image=config.tokens_per_image,
        )
        changed = inputs.build_block10_cache_manifest_spec(
            self.resolved, changed_preprocess
        )
        self.assertNotEqual(
            first["cache_identity_sha256"], changed["cache_identity_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
