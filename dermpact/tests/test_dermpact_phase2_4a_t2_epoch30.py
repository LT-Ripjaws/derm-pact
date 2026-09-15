"""Public contract tests for Phase 2.4A-T2-30 finalist refits."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dermpact.eval import dermpact_phase2_4a_t2_epoch30 as epoch30


ROOT = Path(__file__).resolve().parents[2]


class Phase24AT2Epoch30ContractTest(unittest.TestCase):
    def test_generated_spec_is_an_isolated_from_scratch_epoch30_design(self):
        spec = epoch30.generate_protocol_spec()

        self.assertEqual(epoch30.candidate_ids(), ("c10", "c05"))
        self.assertEqual(spec["seed"], 20260827)
        self.assertEqual(spec["folds"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["training"]["epochs"], 30)
        self.assertEqual(spec["training"]["initialization"], "from_scratch")
        self.assertEqual(spec["training"]["scheduler"]["horizon_epochs"], 30)
        self.assertFalse(spec["training"]["scheduler"]["restart_from_earlier_epochs"])
        self.assertEqual(spec["training"]["effective_batch_size"], 64)
        self.assertEqual(spec["training"]["directml_microbatch_size"], 8)
        self.assertFalse(spec["scope"]["official_or_external_test_loading"])
        self.assertEqual(spec["ranking"]["reference_candidate_id"], "c00@20")
        self.assertEqual(spec["ranking"]["head_to_head"]["pair"], ["c10", "c05"])
        self.assertFalse(spec["ranking"]["lora_factor_launched"])
        self.assertIn(
            "localized_criterion_tuning_t2_epoch30_v1", epoch30.PROTOCOL_PATH.as_posix()
        )
        serialized = json.dumps(spec).lower()
        self.assertNotIn("official_test_loading\": true", serialized)

    def test_finalist_preference_follows_the_prespecified_tie_rules(self):
        c10 = {
            "candidate_id": "c10", "safety_decision": "SAFE",
            "attention_dim": 64, "disagreement_point": 0.10,
        }
        c05 = {
            "candidate_id": "c05", "safety_decision": "SAFE",
            "attention_dim": 256, "disagreement_point": 0.08,
        }

        def paired(estimate, ci_low, ci_high):
            return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}

        # CI excludes zero: the head-to-head sign decides.
        self.assertEqual(
            epoch30.finalist_preference(c10, c05, paired(0.004, 0.001, 0.007)), "c10"
        )
        self.assertEqual(
            epoch30.finalist_preference(c10, c05, paired(-0.004, -0.007, -0.001)), "c05"
        )
        # CI includes zero: smaller attention dimension wins.
        self.assertEqual(
            epoch30.finalist_preference(c10, c05, paired(0.002, -0.001, 0.005)), "c10"
        )
        # Equal attention dimensions: lower pooled disagreement wins.
        same_dim = {**c05, "attention_dim": 64}
        self.assertEqual(
            epoch30.finalist_preference(c10, same_dim, paired(0.002, -0.001, 0.005)),
            "c05",
        )
        # Only one SAFE finalist: it is preferred regardless of the interval.
        unsafe_c10 = {**c10, "safety_decision": "UNSAFE"}
        self.assertEqual(
            epoch30.finalist_preference(unsafe_c10, c05, paired(0.02, 0.01, 0.03)),
            "c05",
        )
        # No SAFE finalist: nobody is preferred (retain c00@20 downstream).
        unsafe_c05 = {**c05, "safety_decision": "UNSAFE"}
        self.assertIsNone(
            epoch30.finalist_preference(unsafe_c10, unsafe_c05, paired(0.0, -0.1, 0.1))
        )

    def test_augmented_fold_receipts_carry_the_resume_format_field(self):
        """Regression for the T2-20 defect: receipts omitted the resume format."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(
                json.dumps({
                    "status": "PASS", "seed": 20260827, "fold": 2,
                    "source_checkpoint_sha256": "s" * 64,
                }),
                encoding="utf-8",
            )
            receipt = epoch30._augment_fold_receipt(
                path,
                candidate_id="c10",
                protocol_sha="p" * 64,
                selection_sha="l" * 64,
                config_sha="c" * 64,
                schema=epoch30.TRAINING_SCHEMA,
            )
            self.assertEqual(receipt["format"], epoch30.RESUME_SCHEMA)
            reread = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reread["format"], epoch30.RESUME_SCHEMA)
            # The augmented receipt satisfies the resume-identity contract.
            epoch30.validate_resume_identity(
                reread,
                protocol_sha256="p" * 64,
                fold=2,
                source_checkpoint_sha256="s" * 64,
                t2_epoch20_selection_sha256=epoch30.T2_EPOCH20_SELECTION_SHA256,
                t2_epoch20_ranking_sha256=epoch30.T2_EPOCH20_RANKING_SHA256,
                candidate_id="c10",
                candidate_config_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
