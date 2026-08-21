"""Enforce the v2 preregistration, so it is a constraint rather than a document.

v1's clearest lesson was that a governance test caught an error the auditor introduced while
auditing.  These tests exist for the same reason: the registration claims specific things about
what has been read and what has been frozen, and each of those claims is checkable.

The most important one is the **exposure boundary**.  The registration discloses that the
classifier baseline was scored on test before the ensemble was designed, and asserts that nothing
else was.  If a future run leaks a stacked-ensemble or criterion-head test result into the
evidence directory, the confirmatory claim is gone -- and it would be invisible without a test
that looks.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "research" / "PREREGISTRATION_V2.json"
EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"

#: The single artifact permitted to contain test-split model outputs, per the disclosed exposure.
ALLOWED_TEST_OUTPUT_ARTIFACTS = {"perception_baseline.json"}
#: manifest.json carries split labels and metadata, not model outputs.
METADATA_ONLY_ARTIFACTS = {"manifest.json"}


@unittest.skipUnless(PREREG.exists(), "v2 preregistration not present")
class PreregistrationShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    def test_it_is_valid_json_with_the_required_sections(self):
        for key in (
            "protocol_id",
            "thesis",
            "scope",
            "frozen_system",
            "endpoints",
            "statistics",
            "test_split_exposure",
            "stop_conditions",
            "invalid_post_test_actions",
        ):
            self.assertIn(key, self.prereg, f"missing required section {key!r}")

    def test_exactly_three_primary_endpoints_with_multiplicity_declared(self):
        endpoints = self.prereg["endpoints"]
        self.assertEqual(len(endpoints["primary"]), 3)
        self.assertIn("Holm", endpoints["multiplicity"])
        for endpoint in endpoints["primary"]:
            self.assertEqual(endpoint["class"], "inferential")
            self.assertIn("success_condition", endpoint)

    def test_difficulty_is_not_a_model_feature(self):
        meta = self.prereg["frozen_system"]["meta_layer"]
        self.assertEqual(meta["excluded_feature"], "level_of_diagnostic_difficulty")
        self.assertNotIn("difficulty", json.dumps(meta["composition"]).lower())

    def test_the_operating_threshold_is_fixed_and_train_derived(self):
        point = self.prereg["frozen_system"]["operating_point"]
        self.assertIsInstance(point["primary_threshold"], float)
        self.assertIn("train", point["policy"])

    def test_failed_hypotheses_are_recorded_not_quietly_dropped(self):
        failed = {h["id"] for h in self.prereg["failed_hypotheses_recorded_before_the_confirmatory_run"]}
        self.assertIn("H_vlm_reads_criteria", failed)
        self.assertIn("H_vlm_confidence_supports_abstention", failed)


@unittest.skipUnless(EVIDENCE.exists(), "derm7pt evidence directory not present")
class ExposureBoundaryTest(unittest.TestCase):
    """The disclosed exposure is a ceiling, not a floor."""

    def _artifacts_with_test_outputs(self):
        offenders = {}
        for path in sorted(EVIDENCE.glob("*.json")):
            if path.name in METADATA_ONLY_ARTIFACTS:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            reasons = []
            cases = data.get("cases")
            if isinstance(cases, list):
                n_test = sum(
                    1 for row in cases if isinstance(row, dict) and row.get("split") == "test"
                )
                if n_test:
                    reasons.append(f"{n_test} per-case test rows")
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                for target, block in metrics.items():
                    if isinstance(block, dict) and "test" in block:
                        reasons.append(f"metrics.{target}.test")
            if reasons:
                offenders[path.name] = reasons
        return offenders

    def test_only_the_disclosed_artifact_holds_test_split_outputs(self):
        offenders = self._artifacts_with_test_outputs()
        unexpected = set(offenders) - ALLOWED_TEST_OUTPUT_ARTIFACTS
        self.assertEqual(
            unexpected,
            set(),
            "test-split model outputs appeared in artifacts the preregistration does not "
            f"disclose: { {k: offenders[k] for k in sorted(unexpected)} }. Either this is the "
            "registered confirmatory run (update the preregistration and the ledger) or the "
            "exposure boundary has been breached and the confirmatory claim is void.",
        )

    def test_the_disclosed_exposure_is_actually_present(self):
        # Guards against the opposite failure: quietly deleting the evidence of exposure and
        # then claiming a pristine test split.
        offenders = self._artifacts_with_test_outputs()
        self.assertIn(
            "perception_baseline.json",
            offenders,
            "the preregistration discloses that the classifier baseline was scored on test; "
            "that artifact no longer shows it, so either it was regenerated or the disclosure "
            "is now inaccurate",
        )

    def test_no_stacked_or_criterion_result_exists_on_test(self):
        for name in ("g6_multiview_stacking.json", "g1b_trained_criterion_heads.json"):
            path = EVIDENCE / name
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            blob = json.dumps(data).lower()
            self.assertNotIn(
                '"evaluated_on": "test"',
                blob,
                f"{name} reports an evaluation on the test split; the registered system must not "
                "have been run on test before the confirmatory run",
            )


@unittest.skipUnless(PREREG.exists(), "v2 preregistration not present")
class FrozenArtifactTest(unittest.TestCase):
    """Every artifact the registration pins must still hash to what it pinned."""

    def test_pinned_artifacts_still_match_their_recorded_digests(self):
        prereg = json.loads(PREREG.read_text(encoding="utf-8"))
        drifted = []
        for record in prereg["frozen_artifacts"]["development_results"]:
            path = REPO / record["path"]
            if not path.exists():
                drifted.append((record["path"], "missing"))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not digest.startswith(record["sha256_prefix"]):
                drifted.append((record["path"], f"{digest[:24]} != {record['sha256_prefix']}"))
        self.assertEqual(
            drifted,
            [],
            f"pinned development artifacts changed after registration: {drifted}",
        )

    def test_dataset_manifest_body_hash_is_unchanged(self):
        prereg = json.loads(PREREG.read_text(encoding="utf-8"))
        pinned = prereg["frozen_artifacts"]["dataset_manifest"]["body_sha256"]
        manifest = json.loads((REPO / "research/evidence/derm7pt_v1/manifest.json").read_text("utf-8"))
        self.assertEqual(manifest["body_sha256"], pinned)


if __name__ == "__main__":
    unittest.main()
