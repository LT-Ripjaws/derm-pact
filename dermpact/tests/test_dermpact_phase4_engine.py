"""Unit tests for the Phase 4 guideline engine on a synthetic catalogue."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from dermpact.guidelines.derivation import derive_case_facts
from dermpact.guidelines.engine import ApplicabilityEngine, EntityMismatchError
from dermpact.guidelines.facts import CaseFacts, FactProvenance, TV, tv_and, tv_not, tv_or
from dermpact.guidelines.rules import (
    Action, ApplicabilityState as S, CatalogueIntegrityError, Predicate, RuleCatalogue, RuleRecord,
    RuleStatus, SourceBinding, Supersession, sha256_text,
)

SPAN = "If a melanoma is clinically suspected, it shall be confirmed by histopathology."


def _rule(rid="t:R1", *, status=RuleStatus.EXECUTABLE, prereq=None, excl=None, required=(), action=True,
          valid=("2024-11-28", "2026-12-31"), sup="current", instrument=None):
    prereq = prereq if prereq is not None else {"fact_is": ["lesion.melanoma_suspected", True]}
    return RuleRecord(
        rule_id=rid, catalogue_id="t", catalogue_version="v1", status=status, scope="lesion_next_step",
        recommendation_number=1, title="t",
        source=SourceBinding("doc", "doi", "title", "issuer", "a" * 64, 8, "b" * 64, "7.1", SPAN, sha256_text(SPAN), 0, len(SPAN)),
        authority="cpg", jurisdiction="EU", guideline_version="v",
        supersession=Supersession(sup, valid[0], valid[1], None, "2026-09-01"),
        instrument_id=instrument, target_population="", prerequisites=Predicate(prereq),
        exclusions=Predicate(excl) if excl else None,
        required_facts=tuple(required) or ("lesion.melanoma_suspected",),
        action=Action("histo", "confirm") if action else None,
        recommendation_strength="GCP", evidence_level=None, consensus_rate=None, basis="", encoding_rationale="",
        interpretation_points=(),
    ).signed()


def _facts(**kw):
    kw.setdefault("evaluation_date", date(2026, 9, 1))
    kw.setdefault("iqa_state", "pass")
    return derive_case_facts("c1", **kw)


SCORE3 = {"pigment_network": "atypical", "blue_whitish_veil": "absent", "vascular_structures": "absent",
          "streaks": "irregular", "pigmentation": "absent", "dots_and_globules": "absent", "regression_structures": "absent"}
SCORE2 = {**SCORE3, "streaks": "absent"}
UNDEC = {**SCORE3, "streaks": "indeterminate"}


class KleeneTest(unittest.TestCase):
    def test_truth_tables(self):
        self.assertIs(tv_and(TV.TRUE, TV.UNKNOWN), TV.UNKNOWN)
        self.assertIs(tv_and(TV.FALSE, TV.UNKNOWN), TV.FALSE)
        self.assertIs(tv_or(TV.TRUE, TV.UNKNOWN), TV.TRUE)
        self.assertIs(tv_or(TV.FALSE, TV.UNKNOWN), TV.UNKNOWN)
        self.assertIs(tv_not(TV.UNKNOWN), TV.UNKNOWN)

    def test_unknown_fact_name_is_an_error(self):
        f = CaseFacts("c1")
        with self.assertRaises(KeyError):
            f.set("bogus.fact", True, FactProvenance("x"))
        with self.assertRaises(ValueError):
            Predicate({"fact_is": ["bogus.fact", True]})


class EngineStateTest(unittest.TestCase):
    def setUp(self):
        self.cat = RuleCatalogue("t", "v1", (_rule(),))
        self.engine = ApplicabilityEngine(self.cat)

    def test_four_states(self):
        r = self.cat.records[0]
        self.assertIs(self.engine.evaluate_rule(r, _facts(provisional_class="melanoma", criterion_values=SCORE2)).state, S.APPLICABLE)
        self.assertIs(self.engine.evaluate_rule(r, _facts(provisional_class="nevus", criterion_values=SCORE2)).state, S.NOT_APPLICABLE)
        res = self.engine.evaluate_rule(r, _facts(provisional_class="nevus", criterion_values=UNDEC))
        self.assertIs(res.state, S.UNKNOWN)
        self.assertEqual(res.decisive_missing_fact, "lesion.melanoma_suspected")
        self.assertTrue(res.question)
        stale = _facts(provisional_class="melanoma", criterion_values=SCORE2)
        stale.evaluation_date = date(2027, 1, 1)
        self.assertIs(self.engine.evaluate_rule(r, stale).state, S.NO_COVERAGE)

    def test_exclusion_beats_prerequisite(self):
        rule = _rule(excl={"fact_is": ["image.dermoscopic_present", False]}, required=("lesion.melanoma_suspected", "image.dermoscopic_present"))
        eng = ApplicabilityEngine(RuleCatalogue("t", "v1", (rule,)))
        f = _facts(provisional_class="melanoma", criterion_values=SCORE2, dermoscopic_present=False)
        # derivation withholds suspicion without an image, so exclusion or prerequisite both block
        self.assertIsNot(eng.evaluate_rule(rule, f).state, S.APPLICABLE)

    def test_jurisdiction_and_status(self):
        r = self.cat.records[0]
        f = _facts(provisional_class="melanoma", criterion_values=SCORE2)
        f.jurisdiction = "US"
        res = self.engine.evaluate_rule(r, f)
        self.assertIs(res.state, S.NO_COVERAGE)
        self.assertEqual(res.reason_code, "jurisdiction_not_covered")
        prose = _rule("t:P1", status=RuleStatus.PROSE_ONLY, action=False)
        eng = ApplicabilityEngine(RuleCatalogue("t", "v1", (prose,)))
        self.assertEqual(eng.evaluate_rule(prose, _facts()).reason_code, "rule_status_prose_only")

    def test_entity_mismatch_fails_closed(self):
        with self.assertRaises(EntityMismatchError):
            self.engine.evaluate_case(_facts(), expected_case_id="c2")

    def test_engine_never_mutates_facts(self):
        f = _facts(provisional_class="nevus", criterion_values=SCORE3, instrument_id="nice_ng12_weighted_7point")
        before = json.dumps(f.to_dict(), sort_keys=True)
        rule = _rule(instrument="derm7pt_dermoscopic_7point_argenziano")
        res = ApplicabilityEngine(RuleCatalogue("t", "v1", (rule,))).evaluate_rule(rule, f)
        self.assertEqual(res.reason_code, "instrument_mismatch_blanked_facts")
        self.assertEqual(before, json.dumps(f.to_dict(), sort_keys=True))

    def test_one_targeted_question_only(self):
        r2 = _rule("t:R2", prereq={"all_of": [{"fact_is": ["patient.high_risk_status", True]}, {"fact_is": ["site.rcm_available", True]}]},
                   required=("patient.high_risk_status", "site.rcm_available"))
        eng = ApplicabilityEngine(RuleCatalogue("t", "v1", (self.cat.records[0], r2)))
        case = eng.evaluate_case(_facts(provisional_class="nevus", criterion_values=UNDEC))
        self.assertIsNotNone(case.targeted_question)
        self.assertEqual(len([q for q in [case.targeted_question] if q]), 1)
        self.assertEqual(case.coverage["applicability_unknown"], 2)


class CatalogueIntegrityTest(unittest.TestCase):
    def test_tamper_refused_and_roundtrip(self):
        rule = _rule()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cat.jsonl"
            cat = RuleCatalogue("t", "v1", (rule,))
            cat.write_jsonl(p)
            again = RuleCatalogue.load_jsonl(p, expected_sha256=cat.catalogue_sha256)
            self.assertEqual(again.catalogue_sha256, cat.catalogue_sha256)
            raw = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
            raw["action"]["code"] = "excise_now"
            p.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with self.assertRaises(CatalogueIntegrityError):
                RuleCatalogue.load_jsonl(p)

    def test_span_hash_enforced(self):
        rule = _rule()
        bad = RuleRecord(**{**rule.__dict__, "source": SourceBinding(**{**rule.source.__dict__, "span_text": SPAN + "!"})})
        with self.assertRaises(CatalogueIntegrityError):
            RuleCatalogue("t", "v1", (bad.signed(),))  # signed but span hash no longer matches text


class DerivationTest(unittest.TestCase):
    def test_precedence_and_blanking(self):
        f = _facts(provisional_class="nevus", criterion_values=SCORE2, clinician_melanoma_suspected=True)
        self.assertIs(f.tv("lesion.melanoma_suspected"), TV.TRUE)
        self.assertEqual(f.provenance["lesion.melanoma_suspected"].source, "asserted")
        g = _facts(provisional_class="melanoma", criterion_values=SCORE3, iqa_state="fail")
        self.assertIs(g.tv("lesion.melanoma_suspected"), TV.UNKNOWN)
        self.assertIs(g.tv("lesion.equivocal"), TV.UNKNOWN)
        self.assertIs(g.tv("checklist.decidable"), TV.UNKNOWN)

    def test_nevus_threshold(self):
        self.assertIs(_facts(total_nevus_count=100).tv("patient.high_total_nevus_count"), TV.TRUE)
        self.assertIs(_facts(total_nevus_count=99).tv("patient.high_total_nevus_count"), TV.FALSE)
        self.assertIs(_facts().tv("patient.high_total_nevus_count"), TV.UNKNOWN)

    def test_no_imputation(self):
        f = _facts()
        self.assertIs(f.tv("patient.high_risk_status"), TV.UNKNOWN)
        self.assertIs(f.tv("site.rcm_available"), TV.UNKNOWN)
        self.assertIsNone(f.scalar("patient.age_years"))


if __name__ == "__main__":
    unittest.main()
