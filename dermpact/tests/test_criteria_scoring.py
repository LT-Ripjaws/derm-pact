"""The scoring rule must reproduce Derm7pt's expert score on every case."""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

from dermpact.criteria.aggregate import (
    MANAGEMENT_LADDER,
    leave_one_criterion_out,
    recommend_management,
    score_criteria,
    verdict_hinges,
)
from dermpact.criteria.schema import (
    CRITERION_NAMES,
    INDETERMINATE,
    MELANOMA_SUSPICION_THRESHOLD,
    validate_values,
)

REPO = Path(__file__).resolve().parents[2]
META = REPO / "release_v0" / "meta" / "meta.csv"


def _load_rows():
    with META.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(META.exists(), "Derm7pt not present")
class TestScoreReproducesGold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _load_rows()

    def test_dataset_shape_is_what_the_rule_was_solved_against(self):
        self.assertEqual(len(self.rows), 1011)

    def test_score_matches_expert_score_on_every_case(self):
        mismatches = []
        for index, row in enumerate(self.rows):
            values = {name: row[name] for name in CRITERION_NAMES}
            scored = score_criteria(values)
            expected = int(row["seven_point_score"])
            if scored.score != expected:
                mismatches.append((index, row["case_num"], scored.score, expected))
        self.assertEqual(
            mismatches,
            [],
            f"{len(mismatches)} of {len(self.rows)} cases disagree with the gold score; "
            f"first few: {mismatches[:5]}",
        )

    def test_every_gold_value_is_in_vocabulary(self):
        for row in self.rows:
            validate_values({name: row[name] for name in CRITERION_NAMES})

    def test_gold_management_values_are_the_declared_ladder(self):
        observed = {row["management"] for row in self.rows}
        self.assertEqual(observed, set(MANAGEMENT_LADDER))


@unittest.skipUnless(META.exists(), "Derm7pt not present")
class TestPartialInformation(unittest.TestCase):
    """A criterion may be unknown without forcing an abstention."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _load_rows()

    def _values(self, index):
        return {name: self.rows[index][name] for name in CRITERION_NAMES}

    def test_indeterminate_yields_a_range_not_a_point(self):
        values = self._values(0)
        values["streaks"] = INDETERMINATE
        scored = score_criteria(values)
        self.assertFalse(scored.determinate)
        self.assertEqual(scored.indeterminate, ("streaks",))
        self.assertLess(scored.score_min, scored.score_max)
        with self.assertRaises(ValueError):
            _ = scored.score

    def test_a_case_far_above_threshold_stays_decidable_with_one_unknown(self):
        target = None
        for index, row in enumerate(self.rows):
            if int(row["seven_point_score"]) >= MELANOMA_SUSPICION_THRESHOLD + 2:
                target = index
                break
        self.assertIsNotNone(target, "expected at least one high-scoring case")

        values = self._values(target)
        values["regression_structures"] = INDETERMINATE
        scored = score_criteria(values)
        self.assertFalse(scored.determinate)
        self.assertTrue(scored.decidable)
        self.assertTrue(scored.meets_threshold)

    def test_a_borderline_case_becomes_undecidable(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["dots_and_globules"] = "irregular"      # 1 point
        values["streaks"] = "irregular"                # 1 point -> total 2, below threshold
        values["blue_whitish_veil"] = INDETERMINATE    # worth 2 -> range [2, 4]
        scored = score_criteria(values)
        self.assertFalse(scored.decidable)
        self.assertEqual((scored.score_min, scored.score_max), (2, 4))
        management, reason = recommend_management(scored)
        self.assertEqual(management, "clinical follow up")
        self.assertIn("undecidable", reason)
        with self.assertRaises(ValueError):
            _ = scored.meets_threshold

    def test_management_recommendation_follows_the_threshold(self):
        high = {name: "absent" for name in CRITERION_NAMES}
        high["pigment_network"] = "atypical"        # 2
        high["blue_whitish_veil"] = "present"       # 2 -> 4
        self.assertEqual(recommend_management(score_criteria(high))[0], "excision")

        low = {name: "absent" for name in CRITERION_NAMES}
        self.assertEqual(recommend_management(score_criteria(low))[0], "clinical follow up")


@unittest.skipUnless(META.exists(), "Derm7pt not present")
class TestLeaveOneCriterionOut(unittest.TestCase):
    """Influence testing must need no regeneration."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _load_rows()

    def test_reports_every_criterion(self):
        values = {name: self.rows[0][name] for name in CRITERION_NAMES}
        out = leave_one_criterion_out(values)
        self.assertEqual(set(out), set(CRITERION_NAMES))

    def test_a_criterion_carrying_the_verdict_is_flagged(self):
        # Exactly at threshold via one major criterion: removing it must change the verdict.
        values = {name: "absent" for name in CRITERION_NAMES}
        values["pigment_network"] = "atypical"     # 2
        values["streaks"] = "irregular"            # 1 -> total 3, exactly at threshold
        base = score_criteria(values)
        self.assertTrue(base.meets_threshold)

        out = leave_one_criterion_out(values)
        self.assertTrue(out["pigment_network"]["verdict_changed"])
        self.assertTrue(out["streaks"]["verdict_changed"])
        self.assertFalse(out["dots_and_globules"]["verdict_changed"])

    def test_zero_scoring_criteria_carry_no_influence(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["pigment_network"] = "atypical"
        values["blue_whitish_veil"] = "present"
        values["streaks"] = "irregular"            # total 5, comfortably above
        out = leave_one_criterion_out(values)
        self.assertEqual(out["dots_and_globules"]["points_contributed"], 0)
        self.assertFalse(out["dots_and_globules"]["verdict_changed"])

    def test_undecidable_base_reports_no_verdict_change_for_any_criterion(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["dots_and_globules"] = "irregular"       # 1
        values["streaks"] = "irregular"                 # 1 -> 2
        values["blue_whitish_veil"] = INDETERMINATE     # worth 2 -> range [2, 4] spans 3
        base = score_criteria(values)
        self.assertFalse(base.decidable)

        out = leave_one_criterion_out(values)
        self.assertEqual(len(out), 7)
        self.assertTrue(
            all(not row["verdict_changed"] for row in out.values()),
            f"expected every criterion False, got {out}",
        )

    def test_decidable_base_becomes_undecidable_when_a_major_criterion_is_removed(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["pigment_network"] = "atypical"      # 2
        values["blue_whitish_veil"] = "present"      # 2 -> total 4, comfortably decidable
        base = score_criteria(values)
        self.assertTrue(base.decidable)
        self.assertTrue(base.meets_threshold)

        out = leave_one_criterion_out(values)
        self.assertFalse(out["pigment_network"]["still_decidable"])
        self.assertTrue(out["pigment_network"]["verdict_changed"])


@unittest.skipUnless(META.exists(), "Derm7pt not present")
class TestVerdictHinges(unittest.TestCase):
    """verdict_hinges (evidence/render.py's R2 LOCO surface) wraps leave_one_criterion_out."""

    def test_undecidable_base_has_no_hinges(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["dots_and_globules"] = "irregular"
        values["streaks"] = "irregular"
        values["blue_whitish_veil"] = INDETERMINATE
        self.assertFalse(score_criteria(values).decidable)
        self.assertEqual(verdict_hinges(values), [])

    def test_a_verdict_far_from_threshold_has_no_hinges(self):
        values = {
            "pigment_network": "atypical", "blue_whitish_veil": "present",
            "vascular_structures": "dotted", "streaks": "irregular",
            "pigmentation": "diffuse irregular", "dots_and_globules": "irregular",
            "regression_structures": "blue areas",
        }
        self.assertEqual(score_criteria(values).score, 10)
        self.assertEqual(verdict_hinges(values), [])

    def test_a_criterion_carrying_the_verdict_from_above_erodes(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["pigment_network"] = "atypical"     # 2
        values["streaks"] = "irregular"            # 1 -> total 3, exactly at threshold
        base = score_criteria(values)
        self.assertTrue(base.meets_threshold)

        hinges = {h["name"]: h for h in verdict_hinges(values)}
        self.assertEqual(set(hinges), {"pigment_network", "streaks"})
        self.assertEqual(hinges["pigment_network"]["value"], "atypical")
        self.assertEqual(hinges["pigment_network"]["points_contributed"], 2)
        self.assertEqual(hinges["pigment_network"]["flipped_verdict_direction"], "erodes")
        self.assertEqual(hinges["streaks"]["flipped_verdict_direction"], "erodes")

    def test_an_absent_major_criterion_below_threshold_elevates(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["streaks"] = "irregular"    # 1 point, base score 1, below threshold
        base = score_criteria(values)
        self.assertFalse(base.meets_threshold)

        hinges = {h["name"]: h for h in verdict_hinges(values)}
        self.assertEqual(
            set(hinges), {"pigment_network", "blue_whitish_veil", "vascular_structures"}
        )
        for name in hinges:
            self.assertEqual(hinges[name]["points_contributed"], 0)
            self.assertEqual(hinges[name]["flipped_verdict_direction"], "elevates")
        self.assertNotIn("pigmentation", hinges)

    def test_hinges_are_returned_in_canonical_criterion_order(self):
        values = {name: "absent" for name in CRITERION_NAMES}
        values["streaks"] = "irregular"
        hinges = verdict_hinges(values)
        names = [h["name"] for h in hinges]
        self.assertEqual(names, [n for n in CRITERION_NAMES if n in names])


if __name__ == "__main__":
    unittest.main()
