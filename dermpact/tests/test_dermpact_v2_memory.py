"""WP-M: comparison-oriented Verified Case Memory v2 (dermpact.core.memory)."""
from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dermpact.core import memory as vcm2
from dermpact.core.versions import MEMORY_EMBEDDING_MODEL, MEMORY_PROTOCOL_VERSION

FORBIDDEN_KEY_PATTERN = re.compile(r"distribution|vote|assisted|alpha|blend", re.IGNORECASE)


def _unit(angle_degrees: float) -> tuple[float, float]:
    radians = math.radians(angle_degrees)
    return (math.cos(radians), math.sin(radians))


def _episode(
    *,
    case_id: str,
    angle: float = 0.0,
    diagnosis: str = "NEV",
    lesion_id: str | None = None,
    patient_id: str | None = None,
    data_role: str = vcm2.DATA_ROLE_DEVELOPMENT,
    image_sha256: str | None = None,
    eligibility: str = "eligible",
    evidence_grade: str = "dataset_reference_label",
    documented_criterion_values: dict | None = None,
    site: str | None = "head_neck",
    sex: str | None = "female",
    elevation: str | None = "flat",
    embedding_model: str = MEMORY_EMBEDDING_MODEL,
    supersedes: str | None = None,
    event_time: str = "2026-09-06T00:00:00Z",
) -> vcm2.MemoryEpisode:
    return vcm2.MemoryEpisode.create(
        case_id=case_id,
        lesion_id=lesion_id,
        patient_id=patient_id,
        data_role=data_role,
        image_sha256=image_sha256 or f"sha-{case_id}",
        embedding_model=embedding_model,
        embedding=_unit(angle),
        site=site,
        sex=sex,
        elevation=elevation,
        documented_criterion_values=documented_criterion_values or {},
        original_model_record={"provisional_class": diagnosis, "grade": vcm2.MODEL_RECORD_GRADE},
        outcome={
            "diagnosis": diagnosis,
            "evidence_grade": evidence_grade,
            "source": "synthetic test fixture",
            "date": None,
        },
        field_provenance={"outcome.diagnosis": {"grade": evidence_grade, "source": "synthetic test fixture"}},
        event_time=event_time,
        author="test-fixture",
        eligibility=eligibility,
        supersedes=supersedes,
    )


def _query(
    *,
    case_id: str = "query-case",
    angle: float = 0.0,
    provisional_class: str = "NEV",
    data_role: str = vcm2.DATA_ROLE_DEVELOPMENT,
    embedding_model: str = MEMORY_EMBEDDING_MODEL,
    patient_id: str | None = None,
    lesion_id: str | None = None,
    image_sha256: str | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "embedding": list(_unit(angle)),
        "embedding_model": embedding_model,
        "data_role": data_role,
        "image_sha256": image_sha256,
        "site": "head_neck",
        "sex": "female",
        "elevation": "flat",
        "criterion_values": {},
        "provisional_class": provisional_class,
        "identity": {"patient_id": patient_id, "lesion_id": lesion_id},
    }


def _assert_no_forbidden_keys(obj) -> list[str]:
    hits: list[str] = []

    def walk(value):
        if isinstance(value, dict):
            for key, sub in value.items():
                if FORBIDDEN_KEY_PATTERN.search(str(key)):
                    hits.append(str(key))
                walk(sub)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(obj)
    return hits


class MemoryEpisodeTest(unittest.TestCase):
    def test_create_round_trip_and_hash(self):
        episode = _episode(case_id="derm7pt-case-1")
        payload = episode.to_dict()
        reloaded = vcm2.MemoryEpisode.from_dict(payload)
        self.assertEqual(reloaded.episode_id, episode.episode_id)
        self.assertEqual(reloaded.to_dict(), payload)

    def test_tampered_hash_rejected(self):
        payload = _episode(case_id="derm7pt-case-2").to_dict()
        payload["site"] = "acral"
        with self.assertRaises(ValueError):
            vcm2.MemoryEpisode.from_dict(payload)

    def test_tampered_episode_id_rejected_even_with_matching_hash(self):
        episode = _episode(case_id="derm7pt-case-3")
        payload = dict(episode.unsigned_dict())
        payload["episode_id"] = "vcm2-forged00000000000"
        payload["record_sha256"] = vcm2._hash(payload)
        with self.assertRaises(ValueError):
            vcm2.MemoryEpisode.from_dict(payload)

    def test_invalid_evidence_grade_rejected(self):
        with self.assertRaises(ValueError):
            _episode(case_id="derm7pt-case-4", evidence_grade="not_a_real_grade")


class MemoryBankTest(unittest.TestCase):
    def test_write_load_roundtrip_verifies_every_hash(self):
        episodes = [_episode(case_id=f"derm7pt-case-{i}", angle=i * 5.0) for i in range(5)]
        bank = vcm2.MemoryBank.from_episodes(episodes, embedding_model=MEMORY_EMBEDDING_MODEL)
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bank"
            bank.write(out_dir, source={"synthetic": True}, n_embedded=5, n_development_cases=5)
            loaded = vcm2.MemoryBank.load(out_dir)
        self.assertEqual(loaded.bank_sha256, bank.bank_sha256)
        self.assertEqual(len(loaded), 5)

    def test_write_load_rejects_corrupted_line(self):
        episodes = [_episode(case_id="derm7pt-case-x")]
        bank = vcm2.MemoryBank.from_episodes(episodes, embedding_model=MEMORY_EMBEDDING_MODEL)
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bank"
            bank.write(out_dir, source={}, n_embedded=1, n_development_cases=1)
            bank_path = out_dir / "bank.jsonl"
            lines = bank_path.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            row["site"] = "tampered"
            bank_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                vcm2.MemoryBank.load(out_dir)

    def test_bank_hash_changes_when_episode_changes(self):
        e1 = _episode(case_id="derm7pt-case-a", site="head_neck")
        e2 = _episode(case_id="derm7pt-case-a", site="acral")
        bank1 = vcm2.MemoryBank.from_episodes([e1], embedding_model=MEMORY_EMBEDDING_MODEL)
        bank2 = vcm2.MemoryBank.from_episodes([e2], embedding_model=MEMORY_EMBEDDING_MODEL)
        self.assertNotEqual(bank1.bank_sha256, bank2.bank_sha256)

    def test_effective_drops_superseded_episode(self):
        old = _episode(case_id="derm7pt-case-old")
        new = _episode(case_id="derm7pt-case-old", angle=1.0, supersedes=old.episode_id)
        bank = vcm2.MemoryBank.from_episodes([old, new], embedding_model=MEMORY_EMBEDDING_MODEL)
        effective_ids = {e.episode_id for e in bank.effective()}
        self.assertNotIn(old.episode_id, effective_ids)
        self.assertIn(new.episode_id, effective_ids)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.policy = vcm2.ComparisonPolicy()

    def test_empty_bank(self):
        bank = vcm2.MemoryBank.from_episodes([], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(), bank, self.policy)
        self.assertEqual(result.status, vcm2.STATUS_EMPTY_BANK)
        self.assertEqual(result.matches, ())
        self.assertIsNone(result.counterexample)

    def test_no_match_all_below_threshold(self):
        far = _episode(case_id="derm7pt-case-far", angle=90.0)
        bank = vcm2.MemoryBank.from_episodes([far], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, self.policy)
        self.assertEqual(result.status, vcm2.STATUS_NO_MATCH)
        self.assertEqual(result.matches, ())
        self.assertIsNone(result.counterexample)
        self.assertEqual(result.counterexample_status, vcm2.COUNTEREXAMPLE_NONE_ABOVE_THRESHOLD)

    def test_declined_prohibited_data_role_on_query(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(data_role="official_test"), bank, self.policy)
        self.assertEqual(result.status, vcm2.STATUS_DECLINED_DATA_ROLE)

    def test_query_only_roles_may_query_but_never_enter_the_bank(self):
        episode = _episode(case_id="derm7pt-case-1", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([episode], embedding_model=MEMORY_EMBEDDING_MODEL)
        for role in (vcm2.DATA_ROLE_EVALUATION_QUERY, vcm2.DATA_ROLE_DEPLOYMENT_QUERY):
            result = vcm2.compare(_query(data_role=role), bank, self.policy)
            self.assertEqual(result.status, vcm2.STATUS_OK)

    def test_excluded_prohibited_data_role_on_episode(self):
        good = _episode(case_id="derm7pt-case-good", angle=1.0)
        bad_role = _episode(case_id="derm7pt-case-bad-role", angle=0.5, data_role="official_test")
        bank = vcm2.MemoryBank.from_episodes([good, bad_role], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, self.policy)
        excluded_ids = {(e["episode_id"], e["reason"]) for e in result.exclusions}
        self.assertIn((bad_role.episode_id, vcm2.EXCLUDED_DATA_ROLE), excluded_ids)
        self.assertNotIn(bad_role.episode_id, {m["episode_id"] for m in result.matches})

    def test_declined_embedding_version_mismatch(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(embedding_model="some-other-model"), bank, self.policy)
        self.assertEqual(result.status, vcm2.STATUS_DECLINED_EMBEDDING)

    def test_declined_identity_unverifiable_under_require_patient_identity(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0, patient_id=None)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        policy = vcm2.ComparisonPolicy(identity_policy="require_patient_identity")
        result = vcm2.compare(_query(patient_id=None), bank, policy)
        self.assertEqual(result.status, vcm2.STATUS_DECLINED_IDENTITY)
        self.assertIsNotNone(result.reason)

    def test_case_level_research_only_proceeds_with_identity_stamp(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0, patient_id=None)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        policy = vcm2.ComparisonPolicy(identity_policy="case_level_research_only")
        result = vcm2.compare(_query(patient_id=None), bank, policy)
        self.assertNotEqual(result.status, vcm2.STATUS_DECLINED_IDENTITY)
        self.assertIn("Derm7pt", result.receipt["identity_exclusion"])

    def test_excluded_same_case_and_lesion(self):
        ep = _episode(case_id="shared-case", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(case_id="shared-case", angle=1.0), bank, self.policy)
        self.assertEqual(result.status, vcm2.STATUS_NO_MATCH)
        reasons = {e["reason"] for e in result.exclusions}
        self.assertIn(vcm2.EXCLUDED_SELF, reasons)

    def test_excluded_same_patient_when_both_known(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0, patient_id="patient-42")
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(patient_id="patient-42"), bank, self.policy)
        reasons = {e["reason"] for e in result.exclusions}
        self.assertIn(vcm2.EXCLUDED_SAME_PATIENT, reasons)

    def test_excluded_insufficient_evidence_not_eligible(self):
        pending = _episode(case_id="derm7pt-case-pending", angle=1.0, eligibility="pending")
        bank = vcm2.MemoryBank.from_episodes([pending], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(), bank, self.policy)
        reasons = {e["reason"] for e in result.exclusions}
        self.assertIn(vcm2.EXCLUDED_INSUFFICIENT_EVIDENCE, reasons)

    def test_excluded_insufficient_evidence_unknown_outcome_grade(self):
        unknown_outcome = _episode(case_id="derm7pt-case-unknown", angle=1.0, evidence_grade="unknown")
        bank = vcm2.MemoryBank.from_episodes([unknown_outcome], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(), bank, self.policy)
        reasons = {e["reason"] for e in result.exclusions}
        self.assertIn(vcm2.EXCLUDED_INSUFFICIENT_EVIDENCE, reasons)

    def test_excluded_duplicate_image_between_candidates(self):
        best = _episode(case_id="derm7pt-case-best", angle=1.0, image_sha256="dup-sha")
        worse = _episode(case_id="derm7pt-case-worse", angle=5.0, image_sha256="dup-sha")
        bank = vcm2.MemoryBank.from_episodes([best, worse], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, self.policy)
        matched_ids = {m["episode_id"] for m in result.matches}
        self.assertIn(best.episode_id, matched_ids)
        self.assertNotIn(worse.episode_id, matched_ids)
        excluded = {(e["episode_id"], e["reason"]) for e in result.exclusions}
        self.assertIn((worse.episode_id, vcm2.EXCLUDED_DUPLICATE_IMAGE), excluded)

    def test_excluded_duplicate_image_against_query(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0, image_sha256="query-image-sha")
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(image_sha256="query-image-sha"), bank, self.policy)
        reasons = {e["reason"] for e in result.exclusions}
        self.assertIn(vcm2.EXCLUDED_DUPLICATE_IMAGE, reasons)

    def test_superseded_episode_never_returned(self):
        old = _episode(case_id="history-case", angle=1.0)
        new = _episode(case_id="history-case", angle=1.0, supersedes=old.episode_id)
        bank = vcm2.MemoryBank.from_episodes([old, new], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(case_id="other-query-case", angle=1.0), bank, self.policy)
        matched_ids = {m["episode_id"] for m in result.matches}
        self.assertIn(new.episode_id, matched_ids)
        self.assertNotIn(old.episode_id, matched_ids)
        self.assertNotIn(old.episode_id, {e["episode_id"] for e in result.exclusions})

    def test_k_is_respected(self):
        episodes = [
            _episode(case_id=f"derm7pt-case-k{i}", angle=float(i))
            for i in range(1, 6)
        ]
        bank = vcm2.MemoryBank.from_episodes(episodes, embedding_model=MEMORY_EMBEDDING_MODEL)
        policy = vcm2.ComparisonPolicy(k=2)
        result = vcm2.compare(_query(angle=0.0), bank, policy)
        self.assertEqual(len(result.matches), 2)

    def test_deterministic_ordering_similarity_desc_then_episode_id(self):
        e1 = _episode(case_id="derm7pt-case-tie-b", angle=1.0)
        e2 = _episode(case_id="derm7pt-case-tie-a", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([e1, e2], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, self.policy)
        ordered_ids = [m["episode_id"] for m in result.matches]
        self.assertEqual(ordered_ids, sorted(ordered_ids))

    def test_counterexample_available(self):
        top = _episode(case_id="derm7pt-case-top", angle=5.0, diagnosis="NEV")
        second = _episode(case_id="derm7pt-case-second", angle=20.0, diagnosis="NEV")
        discordant = _episode(case_id="derm7pt-case-discordant", angle=40.0, diagnosis="MEL")
        bank = vcm2.MemoryBank.from_episodes(
            [top, second, discordant], embedding_model=MEMORY_EMBEDDING_MODEL
        )
        policy = vcm2.ComparisonPolicy(k=1)
        result = vcm2.compare(_query(angle=0.0, provisional_class="NEV"), bank, policy)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0]["episode_id"], top.episode_id)
        self.assertEqual(result.counterexample_status, vcm2.COUNTEREXAMPLE_AVAILABLE)
        self.assertEqual(result.counterexample["episode_id"], discordant.episode_id)

    def test_counterexample_all_candidates_one_class(self):
        top = _episode(case_id="derm7pt-case-top2", angle=5.0, diagnosis="NEV")
        second = _episode(case_id="derm7pt-case-second2", angle=20.0, diagnosis="NEV")
        bank = vcm2.MemoryBank.from_episodes([top, second], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0, provisional_class="NEV"), bank, self.policy)
        self.assertEqual(result.counterexample_status, vcm2.COUNTEREXAMPLE_ALL_ONE_CLASS)
        self.assertIsNone(result.counterexample)

    def test_unknown_fields_never_matched(self):
        ep = _episode(case_id="derm7pt-case-unknown-field", angle=1.0, site=None)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        query = _query(angle=0.0)
        query["site"] = None
        result = vcm2.compare(query, bank, self.policy)
        match = result.matches[0]
        self.assertIn("site", match["unknown_fields"])
        self.assertNotIn("site", match["matched_fields"])
        self.assertNotIn("site", match["different_fields"])

    def test_result_has_no_forbidden_keys(self):
        ep = _episode(case_id="derm7pt-case-clean", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, self.policy)
        hits = _assert_no_forbidden_keys(result.to_dict())
        self.assertEqual(hits, [])

    def test_comparison_result_top_level_keys(self):
        ep = _episode(case_id="derm7pt-case-keys", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, self.policy).to_dict()
        self.assertEqual(
            set(result.keys()),
            {
                "status",
                "matches",
                "n_candidates",
                "counterexample",
                "counterexample_status",
                "exclusions",
                "reason",
                "dropped_after_revocation",
                "receipt",
            },
        )


class RefreshTest(unittest.TestCase):
    def test_refresh_drops_episode_superseded_in_new_bank(self):
        old = _episode(case_id="history-case", angle=1.0)
        original_bank = vcm2.MemoryBank.from_episodes([old], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(case_id="query-case", angle=1.0), original_bank, vcm2.ComparisonPolicy())
        self.assertEqual([m["episode_id"] for m in result.matches], [old.episode_id])
        replacement = _episode(case_id="history-case", angle=2.0, supersedes=old.episode_id)
        refreshed = vcm2.refresh(
            result, vcm2.MemoryBank.from_episodes([old, replacement], embedding_model=MEMORY_EMBEDDING_MODEL)
        )
        self.assertNotIn(old.episode_id, {m["episode_id"] for m in refreshed.matches})
        self.assertIn(old.episode_id, refreshed.dropped_after_revocation)

    def test_refresh_drops_revoked_match(self):
        ep = _episode(case_id="derm7pt-case-1", angle=1.0)
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(_query(angle=0.0), bank, vcm2.ComparisonPolicy())
        self.assertEqual(len(result.matches), 1)

        revoked = vcm2.MemoryEpisode.create(
            case_id=ep.case_id,
            data_role=ep.data_role,
            image_sha256=ep.image_sha256,
            embedding_model=ep.embedding_model,
            embedding=ep.embedding,
            outcome=dict(ep.outcome),
            author=ep.author,
            eligibility="revoked",
            revoked_reason="corrected diagnosis",
            event_time=ep.event_time,
        )
        new_bank = vcm2.MemoryBank.from_episodes([revoked], embedding_model=MEMORY_EMBEDDING_MODEL)
        refreshed = vcm2.refresh(result, new_bank)
        self.assertEqual(refreshed.matches, ())
        self.assertEqual(refreshed.status, vcm2.STATUS_NO_MATCH)
        self.assertIn(ep.episode_id, refreshed.dropped_after_revocation)


class ReminderAllowedTest(unittest.TestCase):
    def test_true_with_no_events(self):
        self.assertTrue(vcm2.reminder_allowed([]))

    def test_true_with_unrelated_events(self):
        self.assertTrue(vcm2.reminder_allowed([{"kind": "recomputed"}]))

    def test_false_after_one_unsolicited_reminder(self):
        events = [{"kind": vcm2.UNSOLICITED_REMINDER_KIND}]
        self.assertFalse(vcm2.reminder_allowed(events))

    def test_false_for_persisted_episode_event_shape(self):
        events = [{"event_type": vcm2.UNSOLICITED_REMINDER_KIND}]
        self.assertFalse(vcm2.reminder_allowed(events))


class ThresholdDerivationTest(unittest.TestCase):
    def test_floor_applies_when_natural_percentile_is_low(self):
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        derivation = vcm2.derive_similarity_thresholds(vectors, date="2026-09-06T00:00:00Z")
        self.assertEqual(derivation["n"], 3)
        for key in ("p10", "p25", "p50", "p90"):
            self.assertEqual(derivation[key], 0.0)
        self.assertEqual(derivation["min_similarity"], vcm2.MIN_SIMILARITY_FLOOR)
        self.assertEqual(derivation["counterexample_min_similarity"], vcm2.MIN_SIMILARITY_FLOOR)
        self.assertEqual(derivation["date"], "2026-09-06T00:00:00Z")

    def test_derivation_above_floor_uses_the_25th_percentile(self):
        vectors = [_unit(0.0), _unit(0.0), _unit(60.0), _unit(60.0)]
        derivation = vcm2.derive_similarity_thresholds(vectors)
        self.assertEqual(derivation["n"], 4)
        for key in ("p10", "p25", "p50", "p90"):
            self.assertAlmostEqual(derivation[key], 1.0, places=6)
        self.assertEqual(derivation["min_similarity"], 1.0)
        self.assertEqual(derivation["counterexample_min_similarity"], 0.95)

    def test_requires_at_least_two_embeddings(self):
        with self.assertRaises(ValueError):
            vcm2.derive_similarity_thresholds([[1.0, 0.0]])


class ComparisonPolicyFromManifestTest(unittest.TestCase):
    def test_from_manifest_roundtrip(self):
        episodes = [_episode(case_id=f"derm7pt-case-fm{i}", angle=i * 7.0) for i in range(6)]
        bank = vcm2.MemoryBank.from_episodes(episodes, embedding_model=MEMORY_EMBEDDING_MODEL)
        derivation = vcm2.derive_similarity_thresholds([e.embedding for e in bank.effective()])
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bank"
            bank.write(
                out_dir,
                source={"synthetic": True},
                n_embedded=6,
                n_development_cases=6,
                threshold_derivation=derivation,
            )
            policy = vcm2.ComparisonPolicy.from_manifest(out_dir)
            policy_from_file = vcm2.ComparisonPolicy.from_manifest(out_dir / "manifest.json")
        self.assertEqual(policy.bank_sha256, bank.bank_sha256)
        self.assertEqual(policy.min_similarity, derivation["min_similarity"])
        self.assertEqual(policy.counterexample_min_similarity, derivation["counterexample_min_similarity"])
        self.assertEqual(policy.threshold_derivation, derivation)
        self.assertEqual(policy_from_file.bank_sha256, policy.bank_sha256)

    def test_from_manifest_refuses_without_derivation(self):
        episodes = [_episode(case_id="derm7pt-case-fm-solo")]
        bank = vcm2.MemoryBank.from_episodes(episodes, embedding_model=MEMORY_EMBEDDING_MODEL)
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bank"
            bank.write(out_dir, source={}, n_embedded=1, n_development_cases=1)
            with self.assertRaises(ValueError):
                vcm2.ComparisonPolicy.from_manifest(out_dir)

    def test_policy_requires_derivation_when_bank_sha256_set(self):
        with self.assertRaises(ValueError):
            vcm2.ComparisonPolicy(bank_sha256="deadbeef")

    def test_compare_declines_policy_bound_to_a_different_bank(self):
        ep_a = _episode(case_id="derm7pt-case-fm-a", angle=1.0)
        ep_b = _episode(case_id="derm7pt-case-fm-b", angle=2.0)
        bank_a = vcm2.MemoryBank.from_episodes([ep_a], embedding_model=MEMORY_EMBEDDING_MODEL)
        bank_b = vcm2.MemoryBank.from_episodes([ep_b], embedding_model=MEMORY_EMBEDDING_MODEL)
        frozen_policy = vcm2.ComparisonPolicy(
            bank_sha256=bank_a.bank_sha256,
            threshold_derivation={
                "method": "test", "n": 1, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p90": 0.0,
                "min_similarity": 0.5, "counterexample_min_similarity": 0.5,
                "date": "2026-09-06T00:00:00Z",
            },
        )
        result = vcm2.compare(_query(angle=0.0), bank_b, frozen_policy)
        self.assertEqual(result.status, vcm2.STATUS_DECLINED_POLICY_BANK_MISMATCH)

        matching_policy = vcm2.ComparisonPolicy(
            bank_sha256=bank_b.bank_sha256, threshold_derivation=frozen_policy.threshold_derivation
        )
        ok_result = vcm2.compare(_query(angle=0.0), bank_b, matching_policy)
        self.assertNotEqual(ok_result.status, vcm2.STATUS_DECLINED_POLICY_BANK_MISMATCH)


class DiagnosisHarmonisationTest(unittest.TestCase):
    def test_derm7pt_and_global_class_labels_all_covered(self):
        from dermpact.eval import dermpact_phase2_baselines as baselines
        from dermpact.eval.dermpact_phase4_benchmark import GLOBAL_CLASSES

        records = baselines._derm7pt_records()
        raw_labels = {row["diagnosis"]["harmonized5"] for row in records}
        self.assertTrue(raw_labels)
        for label in raw_labels:
            self.assertIn(label, vcm2.DIAGNOSIS_HARMONISATION_TABLE)
            self.assertIn(vcm2.harmonise_diagnosis(label), vcm2.COMPARISON_DIAGNOSES)
        for label in GLOBAL_CLASSES:
            self.assertIn(label, vcm2.DIAGNOSIS_HARMONISATION_TABLE)

    def test_unmapped_label_is_unknown(self):
        self.assertEqual(vcm2.harmonise_diagnosis("some_unrecognised_label"), vcm2.COMPARISON_UNMAPPED)
        self.assertEqual(vcm2.harmonise_diagnosis(None), vcm2.COMPARISON_UNMAPPED)

    def test_match_view_carries_raw_and_harmonised_diagnosis(self):
        ep = _episode(case_id="derm7pt-case-diag", angle=1.0, diagnosis="MEL")
        bank = vcm2.MemoryBank.from_episodes([ep], embedding_model=MEMORY_EMBEDDING_MODEL)
        result = vcm2.compare(
            _query(angle=0.0, provisional_class="melanoma"), bank, vcm2.ComparisonPolicy()
        )
        diagnosis = result.matches[0]["diagnosis"]
        self.assertEqual(diagnosis["episode_raw"], "MEL")
        self.assertEqual(diagnosis["episode_harmonised"], vcm2.COMPARISON_MELANOMA)
        self.assertEqual(diagnosis["query_raw"], "melanoma")
        self.assertEqual(diagnosis["query_harmonised"], vcm2.COMPARISON_MELANOMA)
        self.assertEqual(diagnosis["status"], "match")

    def test_unmapped_diagnosis_never_becomes_the_counterexample(self):
        top = _episode(case_id="derm7pt-case-um-top", angle=5.0, diagnosis="NEV")
        unmapped = _episode(case_id="derm7pt-case-um-unmapped", angle=40.0, diagnosis="some_bogus_label")
        bank = vcm2.MemoryBank.from_episodes([top, unmapped], embedding_model=MEMORY_EMBEDDING_MODEL)
        policy = vcm2.ComparisonPolicy(k=1)
        result = vcm2.compare(_query(angle=0.0, provisional_class="NEV"), bank, policy)
        self.assertIsNone(result.counterexample)
        self.assertEqual(result.counterexample_status, vcm2.COUNTEREXAMPLE_ALL_ONE_CLASS)

    def test_unmapped_query_diagnosis_never_yields_a_counterexample(self):
        top = _episode(case_id="derm7pt-case-uq-top", angle=5.0, diagnosis="NEV")
        discordant = _episode(case_id="derm7pt-case-uq-discordant", angle=40.0, diagnosis="MEL")
        bank = vcm2.MemoryBank.from_episodes([top, discordant], embedding_model=MEMORY_EMBEDDING_MODEL)
        policy = vcm2.ComparisonPolicy(k=1)
        result = vcm2.compare(_query(angle=0.0, provisional_class="other"), bank, policy)
        self.assertIsNone(result.counterexample)
        self.assertEqual(result.counterexample_status, vcm2.COUNTEREXAMPLE_ALL_ONE_CLASS)


class DevelopmentBankBuildTest(unittest.TestCase):
    """Exercises the real 616-case builder end to end (no images opened)."""

    def test_build_and_self_check_smoke(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "memory_bank_dev_v1"
            manifest = vcm2.build_development_bank(out_dir=out_dir)
            self.assertEqual(manifest["n_development_cases"], 616)
            self.assertGreater(manifest["n_embedded"], 0)
            self.assertLessEqual(manifest["n_embedded"], 616)
            self.assertEqual(manifest["embedding_model"], MEMORY_EMBEDDING_MODEL)

            bank = vcm2.MemoryBank.load(out_dir)
            self.assertEqual(bank.bank_sha256, manifest["bank_sha256"])
            for episode in bank.episodes:
                self.assertIsNone(episode.patient_id)
                self.assertEqual(episode.outcome["evidence_grade"], "dataset_reference_label")
                self.assertIn(episode.outcome["diagnosis"], vcm2._HARMONIZED5_TO_COMPARISON)

            derivation = manifest["threshold_derivation"]
            self.assertEqual(derivation["n"], manifest["n_embedded"])
            self.assertGreaterEqual(derivation["min_similarity"], vcm2.MIN_SIMILARITY_FLOOR)
            self.assertGreaterEqual(derivation["counterexample_min_similarity"], vcm2.MIN_SIMILARITY_FLOOR)
            self.assertLessEqual(derivation["counterexample_min_similarity"], derivation["min_similarity"])

            policy = vcm2.ComparisonPolicy.from_manifest(out_dir)
            self.assertEqual(policy.bank_sha256, manifest["bank_sha256"])
            self.assertEqual(policy.min_similarity, derivation["min_similarity"])

            summary = vcm2.self_check(bank_dir=out_dir, case_id=bank.episodes[0].case_id)
            self.assertIn(summary["status"], {vcm2.STATUS_OK, vcm2.STATUS_NO_MATCH})
            self.assertEqual(summary["bank_sha256"], manifest["bank_sha256"])
            self.assertEqual(summary["min_similarity"], derivation["min_similarity"])


if __name__ == "__main__":
    unittest.main()
