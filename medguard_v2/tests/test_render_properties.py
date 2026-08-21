"""Property tests for render.py: the R1-R5 surfaces plus the three-tier report structure.

Rather than pin exact strings for every branch (test_evidence_contract.py already pins several
of the load-bearing ones -- conflict wording, uncovered-condition wording), this file asserts the
STRUCTURAL invariants the spec calls out, across a spread of contracts chosen to hit every
branch: agreement, explicit conflict, an undecidable score, guideline-uncovered, a case with a
LOCO hinge, and a case robust to every single criterion. Pinning structure rather than prose means
a future wording tweak does not have to touch this file, but a future branch that silently stops
rendering its required section will.

The tier redesign adds three families of properties:

- **Boring-for-boring**: a routine dual-view case (no conflict, decidable) reaches its
  "### Next step" line within eight non-blank lines and carries no warning furniture; the
  conflict warning, when it fires, sits immediately after Tier 1.
- **Hash discipline**: Tiers 1-2 carry "citation · version · ✓", never a hex hash prefix; the
  hex prefixes (and the remaining recommendations, and the full LOCO surface) live behind
  ``include_appendix=True`` only.
- **Single-view**: the classifier-only contract renders with the ``single-view`` trust chip, the
  one-line checklist-unavailable disclosure in the table slot, the mandatory
  clinical-correlation sentence in its Next step, and the same withheld-on-failure behaviour.
"""
from __future__ import annotations

import dataclasses
import re
import unittest

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE
from medguard_v2.evidence.contract import SimilarCase, SimilarCasesContext, build_contract
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog
from medguard_v2.evidence.render import render_headline, render_markdown
from medguard_v2.evidence.single_view import build_single_view_contract

CLEAN = {name: "absent" for name in CRITERION_NAMES}

#: 16 lowercase hex chars as a standalone token -- the content-hash prefix format. Tier 1-2 must
#: never contain one; Tier 3 must.
HEX16 = re.compile(r"\b[0-9a-f]{16}\b")

SINGLE_VIEW_REASON = (
    "this imaging source carries no dermoscopic-criteria annotations; single-view decision"
)


def _melanoma_criteria():
    values = dict(CLEAN)
    values["pigment_network"] = "atypical"   # 2
    values["blue_whitish_veil"] = "present"  # 2 -> 4, above threshold; pigment_network hinges
    return values


def _all_positive_criteria():
    return {
        "pigment_network": "atypical",
        "blue_whitish_veil": "present",
        "vascular_structures": "dotted",
        "streaks": "irregular",
        "pigmentation": "diffuse irregular",
        "dots_and_globules": "irregular",
        "regression_structures": "blue areas",
    }


def _undecidable_criteria():
    values = dict(CLEAN)
    values["dots_and_globules"] = "irregular"     # 1
    values["streaks"] = "irregular"                # 1 -> 2, below threshold on its own
    values["blue_whitish_veil"] = INDETERMINATE    # worth 2 -> range [2, 4] spans 3
    return values


def _nonblank(rendered: str):
    return [l for l in rendered.splitlines() if l.strip()]


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class RenderPropertiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis, criteria, confidence=0.8, case_id="prop-case"):
        return build_contract(
            case_id=case_id,
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            criteria=criteria,
            catalog=self.catalog,
        )

    def _spread(self):
        """(label, contract) pairs hitting every branch R1-R5 need to handle."""
        agree_covered = self._contract("MEL", _melanoma_criteria())         # decidable, hinge
        conflict = self._contract("NEV", _melanoma_criteria())               # VIEWS DISAGREE
        undecidable = self._contract("NEV", _undecidable_criteria())         # checklist unresolved
        uncovered = self._contract("SK", CLEAN)                              # decidable, no guideline
        no_hinge = self._contract("MEL", _all_positive_criteria())           # decidable, robust
        classifier_conflict = self._contract("MEL", CLEAN)                   # classifier flags, checklist doesn't
        return [
            ("agree_covered_hinge", agree_covered),
            ("conflict", conflict),
            ("undecidable", undecidable),
            ("uncovered_routine", uncovered),
            ("no_hinge_robust", no_hinge),
            ("classifier_only_conflict", classifier_conflict),
        ]

    def _routine(self):
        """The boring subset: decidable, no conflict, no escalation -- Tier 1 is the whole read."""
        routine = [
            (label, c)
            for label, c in self._spread()
            if c.score.decidable and not c.conflict.present and not c.conflict.escalate
        ]
        assert routine, "spread lost its routine cases; the budget tests below would be vacuous"
        return routine

    # -- sanity on the spread itself, so a change to build_contract's own logic that quietly
    # collapses one of these into a different branch fails loudly here rather than only in the
    # assertions below being vacuously satisfied.
    def test_the_spread_actually_covers_every_branch(self):
        by_label = dict(self._spread())
        self.assertTrue(by_label["agree_covered_hinge"].score.decidable)
        self.assertFalse(by_label["agree_covered_hinge"].conflict.present)
        self.assertTrue(by_label["conflict"].conflict.present)
        self.assertFalse(by_label["undecidable"].score.decidable)
        self.assertTrue(by_label["uncovered_routine"].score.decidable)
        self.assertFalse(by_label["uncovered_routine"].guideline.covered)
        self.assertFalse(by_label["uncovered_routine"].conflict.escalate)
        self.assertTrue(by_label["no_hinge_robust"].score.decidable)
        self.assertTrue(by_label["classifier_only_conflict"].conflict.present)

    def test_headline_is_the_first_non_header_line(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                lines = _nonblank(render_markdown(contract))
                self.assertTrue(lines[0].startswith(f"## Case {contract.case_id}"))
                self.assertEqual(lines[1], render_headline(contract))
                self.assertTrue(lines[1].startswith("**Assessment:"))

    def test_hinge_or_robustness_line_present_iff_decidable(self):
        hinge_marker = "This verdict hinges on"
        robust_marker = "No single criterion changes this verdict."
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                has_hinge_surface = (hinge_marker in rendered) or (robust_marker in rendered)
                self.assertEqual(
                    has_hinge_surface,
                    contract.score.decidable,
                    f"{label}: hinge/robustness line presence must match decidable={contract.score.decidable}",
                )

    def test_resolution_protocol_present_iff_undecidable(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                self.assertEqual(
                    "_Resolve " in rendered,
                    not contract.score.decidable,
                    f"{label}: resolution protocol presence must match undecidable",
                )

    def test_next_step_section_always_present(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                self.assertIn("### Next step", render_markdown(contract))

    def test_citation_id_or_explicit_absence_always_present(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                if contract.guideline.covered:
                    self.assertIn(contract.guideline.recommendations[0].citation_id, rendered)
                else:
                    self.assertIn("No recommendation available", rendered)

    def test_withheld_contracts_render_the_withheld_block_only(self):
        contract = self._contract("NEV", CLEAN)
        tampered = dataclasses.replace(contract, score=score_criteria(_melanoma_criteria()))
        self.assertFalse(tampered.safe_to_display)

        rendered = render_markdown(tampered)
        self.assertTrue(rendered.startswith(f"## Case {tampered.case_id} — OUTPUT WITHHELD"))
        self.assertIn("OUTPUT WITHHELD", rendered)
        for marker in ("**Assessment:", "### Next step", "### Dermoscopic criteria", "This verdict hinges", "_Resolve "):
            self.assertNotIn(marker, rendered)

    def test_rendered_line_count_is_at_most_60(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                n_lines = len(render_markdown(contract).splitlines())
                self.assertLessEqual(n_lines, 60, f"{label}: rendered {n_lines} lines")

        # withheld path too -- trivially short, but keep it in the same budget.
        contract = self._contract("NEV", CLEAN)
        tampered = dataclasses.replace(contract, score=score_criteria(_melanoma_criteria()))
        self.assertLessEqual(len(render_markdown(tampered).splitlines()), 60)

    # ------------------------------------------------------------------------------------------
    # tier redesign: boring-for-boring budget and warning placement
    # ------------------------------------------------------------------------------------------

    def test_routine_case_reaches_next_step_within_eight_nonblank_lines(self):
        # TIER 1 budget: a boring case must read as boring -- assessment and action inside the
        # 3-second glance, before any table.
        for label, contract in self._routine():
            with self.subTest(label=label):
                lines = _nonblank(render_markdown(contract))
                self.assertIn("### Next step", lines)
                position = lines.index("### Next step") + 1  # 1-indexed
                self.assertLessEqual(
                    position, 8,
                    f"{label}: '### Next step' is non-blank line {position}, budget is 8",
                )

    def test_routine_case_carries_no_warning_furniture(self):
        for label, contract in self._routine():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                self.assertNotIn("⚠", rendered)
                self.assertNotIn("> ###", rendered)
                self.assertNotIn("VIEWS DISAGREE", rendered)

    def test_conflict_warning_sits_immediately_after_tier_one(self):
        # The warning block still fires on every conflict case, right after Tier 1 (headline /
        # brief / next step) and before the two-view table: nothing but the next-step action
        # paragraph may sit between the tier-1 header lines and the ⚠ block.
        checked = 0
        for label, contract in self._spread():
            if not contract.conflict.present:
                continue
            checked += 1
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                lines = rendered.splitlines()
                i_next = lines.index("### Next step")
                i_warn = next(i for i, l in enumerate(lines) if "⚠" in l)
                i_table = next(i for i, l in enumerate(lines) if l.startswith("| assessment |"))
                self.assertLess(i_next, i_warn, f"{label}: warning must follow Tier 1")
                self.assertLess(i_warn, i_table, f"{label}: warning must precede the audit table")
                between = [l for l in lines[i_next + 1:i_warn] if l.strip()]
                self.assertLessEqual(len(between), 2, f"{label}: {between}")
                for l in between:
                    self.assertFalse(
                        l.startswith(("#", "|", ">")),
                        f"{label}: unexpected section between Tier 1 and the warning: {l!r}",
                    )
        self.assertGreaterEqual(checked, 2, "spread lost its conflict cases; test is vacuous")

    # ------------------------------------------------------------------------------------------
    # tier redesign: hash discipline and Tier-3 deferral
    # ------------------------------------------------------------------------------------------

    def test_default_view_shows_check_mark_never_a_hash_prefix(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                self.assertIsNone(
                    HEX16.search(rendered),
                    f"{label}: default view leaked a 16-hex hash prefix",
                )
                covered_somewhere = contract.guideline.covered or (
                    contract.contingent_guideline is not None
                    and contract.contingent_guideline.covered
                )
                if covered_somewhere:
                    self.assertIn("· ✓", rendered, f"{label}: verified-citation mark missing")

    def test_appendix_carries_the_hash_prefixes(self):
        checked = 0
        for label, contract in self._spread():
            recs = []
            if contract.guideline.covered:
                recs += list(contract.guideline.recommendations)
            if contract.contingent_guideline is not None and contract.contingent_guideline.covered:
                recs += list(contract.contingent_guideline.recommendations)
            if not recs:
                continue
            checked += 1
            with self.subTest(label=label):
                appendix = render_markdown(contract, include_appendix=True)
                for rec in recs:
                    self.assertIn(rec.content_sha256[:16], appendix)
        self.assertGreaterEqual(checked, 2, "spread lost its covered cases; test is vacuous")

    def test_only_the_first_recommendation_is_quoted_in_the_default_view(self):
        saw_multi = False
        for label, contract in self._spread():
            if not contract.guideline.covered:
                continue
            recs = contract.guideline.recommendations
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                self.assertIn(recs[0].text.strip(), rendered)
                for rec in recs[1:]:
                    self.assertNotIn(rec.text.strip(), rendered)
                if len(recs) > 1:
                    saw_multi = True
                    self.assertIn(f"_+{len(recs) - 1} more recommendation(s) on request._", rendered)
                    appendix = render_markdown(contract, include_appendix=True)
                    for rec in recs[1:]:
                        self.assertIn(rec.text.strip(), appendix)
        self.assertTrue(saw_multi, "no covered case binds >1 recommendation; deferral untested")

    def test_appendix_carries_the_full_loco_surface_for_dual_view(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                self.assertNotIn("Per-criterion influence", render_markdown(contract))
                appendix = render_markdown(contract, include_appendix=True)
                self.assertIn("Per-criterion influence", appendix)
                if contract.score.decidable:
                    for name in CRITERION_NAMES:
                        self.assertIn(name.replace("_", " "), appendix)
                else:
                    self.assertIn("undecidable", appendix)


def _toy_similar_cases(query_case_id="999", reference="MEL"):
    n1 = SimilarCase(case_id="101", diagnosis="MEL", similarity=0.93, same_label=reference == "MEL")
    n2 = SimilarCase(case_id="102", diagnosis="NEV", similarity=0.88, same_label=reference == "NEV")
    return SimilarCasesContext(
        query_case_id=query_case_id,
        reference_diagnosis=reference,
        neighbors=(n1, n2),
        nearest_different_label=n2 if reference == "MEL" else n1,
        feature_space="dermlip (512-d frozen image embedding; image-only)",
        bank_name="toy@dermlip",
        bank_hash="ab" * 32,
        k=2,
    )


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class SingleViewRenderPropertiesTest(unittest.TestCase):
    """The classifier-only report: compact, disclosed, and withheld on failure."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis, confidence=0.8, reason=SINGLE_VIEW_REASON, similar_cases=None):
        return build_single_view_contract(
            case_id="sv-case",
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            catalog=self.catalog,
            reason=reason,
            similar_cases=similar_cases,
        )

    def _spread(self):
        return [
            ("covered_mel", self._contract("MEL")),
            ("covered_bcc", self._contract("BCC", confidence=0.55)),
            ("uncovered_nev", self._contract("NEV")),
            ("uncovered_sk", self._contract("SK", confidence=0.4)),
        ]

    def test_headline_first_with_the_single_view_chip(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                lines = _nonblank(render_markdown(contract))
                self.assertTrue(lines[0].startswith(f"## Case {contract.case_id}"))
                self.assertEqual(lines[1], render_headline(contract))
                self.assertTrue(lines[1].startswith("**Assessment:"))
                self.assertTrue(lines[1].endswith("· single-view"))
                self.assertNotIn("views agree", lines[1])

    def test_disclosure_occupies_exactly_one_line_in_the_table_slot(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                carrying = [l for l in rendered.splitlines() if SINGLE_VIEW_REASON in l]
                self.assertEqual(len(carrying), 1, f"{label}: disclosure must be exactly one line")
                self.assertTrue(
                    carrying[0].startswith("| 7-point checklist | _unavailable"),
                    f"{label}: the disclosure must occupy the checklist's table slot: {carrying[0]!r}",
                )

    def test_next_step_carries_the_clinical_correlation_sentence(self):
        sentence = (
            "Single-view decision: no independent checklist cross-check was possible — clinical "
            "correlation carries more weight than on a dual-view case."
        )
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                self.assertIn("### Next step", rendered)
                self.assertIn(sentence, rendered)
                # inside the next-step block: after the header, before the audit table.
                self.assertLess(rendered.index("### Next step"), rendered.index(sentence))
                self.assertLess(rendered.index(sentence), rendered.index("| assessment |"))

    def test_single_view_is_compact_and_carries_no_warning_furniture(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                lines = _nonblank(rendered)
                self.assertLessEqual(lines.index("### Next step") + 1, 8, label)
                self.assertNotIn("⚠", rendered)
                self.assertNotIn("> ###", rendered)
                self.assertLessEqual(len(rendered.splitlines()), 60, label)

    def test_no_dual_view_surfaces_are_faked(self):
        # A single view must never render the checklist's own surfaces: no criteria table, no
        # hinge/robustness line, no resolution protocol, no score row.
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                for marker in (
                    "### Dermoscopic criteria",
                    "This verdict hinges",
                    "No single criterion changes this verdict",
                    "_Resolve ",
                    "**score",
                ):
                    self.assertNotIn(marker, rendered, f"{label}: leaked dual-view surface {marker!r}")

    def test_citation_or_explicit_absence_and_hash_discipline(self):
        for label, contract in self._spread():
            with self.subTest(label=label):
                rendered = render_markdown(contract)
                self.assertIsNone(HEX16.search(rendered), f"{label}: hash prefix in default view")
                if contract.guideline.covered:
                    first = contract.guideline.recommendations[0]
                    self.assertIn(first.citation_id, rendered)
                    self.assertIn("· ✓", rendered)
                    appendix = render_markdown(contract, include_appendix=True)
                    self.assertIn(first.content_sha256[:16], appendix)
                else:
                    self.assertIn("No recommendation available", rendered)

    def test_verification_failure_renders_the_withheld_block_only(self):
        withheld = self._contract("MEL", reason="   ")
        self.assertFalse(withheld.safe_to_display)
        rendered = render_markdown(withheld)
        self.assertTrue(rendered.startswith(f"## Case {withheld.case_id} — OUTPUT WITHHELD"))
        for marker in ("**Assessment:", "### Next step", "| assessment |", "Authoritative guidance"):
            self.assertNotIn(marker, rendered)

    def test_default_view_is_byte_independent_of_attached_similar_cases(self):
        with_ctx = render_markdown(self._contract("MEL", similar_cases=_toy_similar_cases()))
        without = render_markdown(self._contract("MEL"))
        self.assertEqual(with_ctx, without)
        self.assertNotIn("Similar confirmed cases", with_ctx)

    def test_similar_cases_render_only_behind_the_appendix(self):
        contract = self._contract("MEL", similar_cases=_toy_similar_cases())
        appendix = render_markdown(contract, include_appendix=True)
        self.assertIn("### Similar confirmed cases (on request)", appendix)
        self.assertIn("Nearest different-label case", appendix)
        # single-view appendix carries no dual-view-only blocks: no saliency pointer (its
        # faithfulness evidence is Derm7pt-specific) and no LOCO surface (no criteria exist).
        self.assertNotIn("Saliency", appendix)
        self.assertNotIn("Per-criterion influence", appendix)


if __name__ == "__main__":
    unittest.main()
