"""Per-criterion model margins: the optional uncertainty channel the promoted perception path adds.

Three surfaces are covered:

* ``verdict_hinges(criteria, margins)`` -- ascending-margin ordering when margins are supplied,
  byte-identical canonical order when they are not (every existing caller relies on the latter).
* ``DecisionContract.criterion_margins`` -- round-trips through ``build_contract`` /
  ``contract_to_dict`` and stays optional: absent when ``None`` (so the pinned pre-promotion
  ``cdss_contracts_valid.json`` serialisation is unchanged), present and rounded when supplied.
  ``verification_failures`` stays permissive for ``None`` and rejects a margins map that names an
  unknown criterion.
* ``render._hinge_line`` -- ranks the hinge line by ascending margin (most-uncertain first) when the
  contract carries margins, and falls back to the pre-promotion point-weight ranking when it does not.
"""
from __future__ import annotations

import unittest

from medguard_v2.criteria.aggregate import verdict_hinges
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.evidence.contract import build_contract, contract_to_dict
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog
from medguard_v2.evidence.render import render_markdown

CLEAN = {name: "absent" for name in CRITERION_NAMES}


def _two_major_hinge_criteria():
    """pigment_network(+2) and blue_whitish_veil(+2) -> score 4 >= 3; removing either widens the
    range across the threshold, so BOTH are hinges with equal points_contributed. That equal-point
    tie is exactly what margin ranking is there to break."""
    values = dict(CLEAN)
    values["pigment_network"] = "atypical"
    values["blue_whitish_veil"] = "present"
    return values


class VerdictHingesMarginOrderingTest(unittest.TestCase):
    def test_no_margins_keeps_canonical_order_and_no_margin_key(self):
        hinges = verdict_hinges(_two_major_hinge_criteria())
        names = [h["name"] for h in hinges]
        self.assertEqual(names, ["pigment_network", "blue_whitish_veil"])  # canonical order
        for h in hinges:
            self.assertNotIn("margin", h)  # byte-identical to the pre-margin behaviour

    def test_margins_sort_ascending_most_uncertain_first(self):
        # blue_whitish_veil is the more uncertain reading (smaller margin) -> it must come first,
        # ahead of the higher-confidence pigment_network, inverting the canonical order.
        margins = {"pigment_network": 0.90, "blue_whitish_veil": 0.10}
        hinges = verdict_hinges(_two_major_hinge_criteria(), margins)
        self.assertEqual([h["name"] for h in hinges], ["blue_whitish_veil", "pigment_network"])
        self.assertEqual(hinges[0]["margin"], 0.10)

    def test_a_hinge_criterion_without_a_supplied_margin_sorts_last(self):
        # Only pigment_network has a margin; blue_whitish_veil (no margin) is treated as maximally
        # certain (inf) and sorts after it.
        margins = {"pigment_network": 0.30}
        hinges = verdict_hinges(_two_major_hinge_criteria(), margins)
        self.assertEqual([h["name"] for h in hinges], ["pigment_network", "blue_whitish_veil"])

    def test_equal_margins_fall_back_to_canonical_order(self):
        margins = {"pigment_network": 0.4, "blue_whitish_veil": 0.4}
        hinges = verdict_hinges(_two_major_hinge_criteria(), margins)
        self.assertEqual([h["name"] for h in hinges], ["pigment_network", "blue_whitish_veil"])


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class ContractCriterionMarginsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, criteria, margins=None, diagnosis="MEL"):
        return build_contract(
            case_id="margin-case",
            classifier_diagnosis=diagnosis,
            classifier_confidence=0.8,
            criteria=criteria,
            catalog=self.catalog,
            criterion_margins=margins,
        )

    def test_absent_by_default_and_serialisation_omits_the_key(self):
        contract = self._contract(_two_major_hinge_criteria())
        self.assertIsNone(contract.criterion_margins)
        self.assertNotIn("criterion_margins", contract_to_dict(contract))

    def test_present_margins_round_trip_and_are_rounded(self):
        margins = {name: 0.123456789 for name in CRITERION_NAMES}
        contract = self._contract(_two_major_hinge_criteria(), margins)
        payload = contract_to_dict(contract)
        self.assertIn("criterion_margins", payload)
        self.assertEqual(set(payload["criterion_margins"]), set(CRITERION_NAMES))
        self.assertEqual(payload["criterion_margins"]["pigment_network"], round(0.123456789, 6))

    def test_margins_are_advisory_and_do_not_change_safety(self):
        # A perfectly valid contract stays safe_to_display whether or not margins are attached.
        without = self._contract(_two_major_hinge_criteria())
        with_margins = self._contract(
            _two_major_hinge_criteria(), {name: 0.5 for name in CRITERION_NAMES}
        )
        self.assertTrue(without.safe_to_display)
        self.assertTrue(with_margins.safe_to_display)

    def test_unknown_criterion_in_margins_is_flagged(self):
        contract = self._contract(
            _two_major_hinge_criteria(), {"not_a_real_criterion": 0.2}
        )
        self.assertTrue(
            any("criterion_margins references unknown criteria" in f
                for f in contract.verification_failures())
        )
        self.assertFalse(contract.safe_to_display)


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class HingeLineMarginRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, margins):
        return build_contract(
            case_id="hinge-case",
            classifier_diagnosis="MEL",
            classifier_confidence=0.8,
            criteria=_two_major_hinge_criteria(),
            catalog=self.catalog,
            criterion_margins=margins,
        )

    def test_hinge_line_names_the_most_uncertain_criterion_when_margins_present(self):
        # blue_whitish_veil is the least-certain hinge -> the rendered hinge line must name it,
        # even though pigment_network is the canonical/point-weight choice.
        rendered = render_markdown(self._contract(
            {"pigment_network": 0.90, "blue_whitish_veil": 0.10}
        ))
        self.assertIn("This verdict hinges on blue whitish veil", rendered)
        self.assertNotIn("This verdict hinges on pigment network", rendered)

    def test_margin_ranking_can_invert_the_point_weight_choice(self):
        # Same criteria, opposite margins -> the hinge line follows the margin, not the points.
        low_pn = render_markdown(self._contract(
            {"pigment_network": 0.10, "blue_whitish_veil": 0.90}
        ))
        self.assertIn("This verdict hinges on pigment network", low_pn)

    def test_falls_back_to_point_weight_when_no_margins(self):
        # No margins -> pre-promotion behaviour: canonical/highest-point pick = pigment_network.
        contract = build_contract(
            case_id="hinge-case",
            classifier_diagnosis="MEL",
            classifier_confidence=0.8,
            criteria=_two_major_hinge_criteria(),
            catalog=self.catalog,
        )
        rendered = render_markdown(contract)
        self.assertIn("This verdict hinges on pigment network", rendered)


if __name__ == "__main__":
    unittest.main()
