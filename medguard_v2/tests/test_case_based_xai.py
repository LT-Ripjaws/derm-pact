"""Case-based explanation (faithful-by-construction XAI) wired into the clinician surface.

Five surfaces, mirroring the criterion_margins optional-field discipline:

* ``case_memory.build_similar_cases_context`` -- retrieves over the frozen TRAIN-only
  :class:`FrozenKNNMemory`, is leave-one-out leak-free (a case never retrieves itself), orders by
  descending cosine, computes ``same_label`` against the reference diagnosis, and finds the nearest
  different-label case as the boundary counterfactual.
* ``DecisionContract.similar_cases`` -- optional, round-trips through ``build_contract`` /
  ``contract_to_dict`` (present -> serialised; ``None`` -> the key is omitted so a contract without it
  is byte-identical to what it serialised before the field existed), and stays advisory: a valid
  context does not change ``safe_to_display``, but a *malformed* one (self-leak, out-of-range cosine)
  is host-rejected.
* ``render.render_markdown`` -- the appendix appears ONLY when the caller asks (``include_appendix``)
  AND the context is present; it is NEVER in the default 10-second view (MG52 progressive disclosure).
* the saliency caveat text is present, behind the on-demand block, with its faithfulness caveat.
* ``clinician_agent.explain_reasoning`` -- gains an on-demand step 9 that surfaces the similar-cases
  context and the saliency caveat, leaving steps 1-8 unchanged.
"""
from __future__ import annotations

import unittest

import numpy as np

from medguard_v2.agent.case_memory import build_similar_cases_context
from medguard_v2.agent.clinician_agent import AgentTools
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.evidence.contract import (
    SimilarCase,
    SimilarCasesContext,
    build_contract,
    contract_to_dict,
)
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog
from medguard_v2.evidence.render import render_markdown
from medguard_v2.perception.knn_memory import FrozenKNNMemory

CLEAN = {name: "absent" for name in CRITERION_NAMES}


def _melanoma_criteria():
    values = dict(CLEAN)
    values["pigment_network"] = "atypical"   # 2
    values["blue_whitish_veil"] = "present"  # 2 -> 4, above threshold
    return values


def _toy_bank() -> FrozenKNNMemory:
    """Three well-separated label clusters, two entries each -- same shape as test_knn_memory."""
    feats = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, -0.1]]
    )
    ids = ["10", "20", "30", "40", "50", "60"]
    labels = {"diagnosis": ["NEV", "NEV", "MEL", "MEL", "SK", "SK"]}
    return FrozenKNNMemory(feats, ids, labels, name="toy@dermlip")


class RetrievalBuilderTest(unittest.TestCase):
    """The retrieval primitive: leak-free, ordered, correctly labelled, boundary-aware."""

    def test_neighbors_are_ordered_by_descending_similarity(self):
        ctx = build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis="NEV",
            bank=_toy_bank(),
            query_case_id="999",
            k=4,
        )
        sims = [n.similarity for n in ctx.neighbors]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_same_label_is_measured_against_the_reference_diagnosis(self):
        ctx = build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis="NEV",
            bank=_toy_bank(),
            query_case_id="999",
            k=2,
        )
        # The two nearest are the NEV cluster (10, 20), so both are same_label under reference NEV.
        self.assertTrue(all(n.same_label for n in ctx.neighbors))
        self.assertEqual({n.diagnosis for n in ctx.neighbors}, {"NEV"})
        self.assertEqual(ctx.same_label_neighbor_rate(), 1.0)

    def test_nearest_different_label_is_the_boundary_case(self):
        ctx = build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis="NEV",
            bank=_toy_bank(),
            query_case_id="999",
            k=2,
        )
        self.assertIsNotNone(ctx.nearest_different_label)
        # The nearest non-NEV point to [1, 0.05] is the MEL entry 40 at [0.1, 0.9] (closer than the
        # MEL entry 30 at [0, 1]); it must NOT be one of the same-label neighbours.
        self.assertNotEqual(ctx.nearest_different_label.diagnosis, "NEV")
        self.assertEqual(ctx.nearest_different_label.case_id, "40")
        self.assertFalse(ctx.nearest_different_label.same_label)

    def test_leave_one_out_makes_retrieval_leak_free(self):
        # Query is exactly entry '20'. With exclusion on, '20' must never appear among its own
        # neighbours OR as the boundary case -- the property the whole "train-only" claim rests on.
        bank = _toy_bank()
        ctx = build_similar_cases_context(
            query_embedding=np.array([0.9, 0.1]),
            reference_diagnosis="NEV",
            bank=bank,
            query_case_id="20",
            k=5,
        )
        retrieved_ids = {n.case_id for n in ctx.neighbors}
        self.assertNotIn("20", retrieved_ids)
        if ctx.nearest_different_label is not None:
            self.assertNotEqual(ctx.nearest_different_label.case_id, "20")

    def test_exclude_query_false_can_return_self(self):
        # The flag genuinely controls exclusion: with it off, the exact-match self IS the top hit.
        ctx = build_similar_cases_context(
            query_embedding=np.array([0.9, 0.1]),
            reference_diagnosis="NEV",
            bank=_toy_bank(),
            query_case_id="20",
            k=1,
            exclude_query=False,
        )
        self.assertEqual(ctx.neighbors[0].case_id, "20")

    def test_k_caps_the_neighbour_count(self):
        ctx = build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis="NEV",
            bank=_toy_bank(),
            query_case_id="999",
            k=3,
        )
        self.assertEqual(len(ctx.neighbors), 3)

    def test_context_carries_the_bank_provenance(self):
        bank = _toy_bank()
        ctx = build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis="NEV",
            bank=bank,
            query_case_id="999",
            feature_space="dermlip (512-d frozen image embedding; image-only)",
        )
        self.assertEqual(ctx.bank_hash, bank.bank_hash)
        self.assertEqual(ctx.bank_name, "toy@dermlip")
        self.assertIn("dermlip", ctx.as_dict()["feature_space"])
        self.assertIn("identity", ctx.as_dict()["faithful_by_construction"].lower())


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class ContractSimilarCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _context(self, query_case_id="999", reference="NEV"):
        return build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis=reference,
            bank=_toy_bank(),
            query_case_id=query_case_id,
            k=3,
            feature_space="dermlip (512-d)",
        )

    def _contract(self, similar_cases=None, diagnosis="NEV"):
        return build_contract(
            case_id="derm7pt-999",
            classifier_diagnosis=diagnosis,
            classifier_confidence=0.8,
            criteria=_melanoma_criteria(),
            catalog=self.catalog,
            similar_cases=similar_cases,
        )

    def test_absent_by_default_and_serialisation_omits_the_key(self):
        contract = self._contract()
        self.assertIsNone(contract.similar_cases)
        self.assertNotIn("similar_cases", contract_to_dict(contract))

    def test_serialisation_is_byte_identical_when_absent(self):
        # Attaching the field is a strict superset: a contract without it must serialise to exactly
        # what it did before the field existed (the cdss_contracts_valid.json digest depends on this).
        without_field = self._contract()
        # Reconstruct the same contract via a caller that never mentions similar_cases at all.
        legacy = build_contract(
            case_id="derm7pt-999",
            classifier_diagnosis="NEV",
            classifier_confidence=0.8,
            criteria=_melanoma_criteria(),
            catalog=self.catalog,
        )
        self.assertEqual(contract_to_dict(without_field), contract_to_dict(legacy))

    def test_present_context_round_trips_and_is_rounded(self):
        contract = self._contract(self._context())
        payload = contract_to_dict(contract)
        self.assertIn("similar_cases", payload)
        sc = payload["similar_cases"]
        self.assertEqual(sc["query_case_id"], "999")
        self.assertEqual(sc["reference_diagnosis"], "NEV")
        self.assertEqual(len(sc["neighbors"]), 3)
        # similarities are rounded to 4 dp in the serialised form.
        for nb in sc["neighbors"]:
            self.assertEqual(nb["similarity"], round(nb["similarity"], 4))
        self.assertIsNotNone(sc["nearest_different_label"])
        self.assertIn("faithful_by_construction", sc)

    def test_a_valid_context_does_not_change_safe_to_display(self):
        without = self._contract()
        with_ctx = self._contract(self._context())
        self.assertTrue(without.safe_to_display)
        self.assertTrue(with_ctx.safe_to_display)

    def test_a_self_leak_is_host_rejected(self):
        # A neighbour whose id equals the query id violates the leave-one-out/train-only property.
        # The builder can never produce this; a wiring bug could, and the host must catch it.
        leaky = SimilarCasesContext(
            query_case_id="999",
            reference_diagnosis="NEV",
            neighbors=(
                SimilarCase("999", "NEV", 1.0, same_label=True),  # the query, leaked in
                SimilarCase("20", "NEV", 0.9, same_label=True),
            ),
            nearest_different_label=SimilarCase("40", "MEL", 0.2, same_label=False),
            feature_space="dermlip (512-d)",
            bank_name="toy@dermlip",
            bank_hash="deadbeef",
            k=2,
        )
        contract = self._contract(leaky)
        self.assertFalse(contract.safe_to_display)
        self.assertTrue(
            any("leaks the query case" in f for f in contract.verification_failures())
        )

    def test_an_out_of_range_cosine_is_host_rejected(self):
        bad = SimilarCasesContext(
            query_case_id="999",
            reference_diagnosis="NEV",
            neighbors=(SimilarCase("20", "NEV", 1.7, same_label=True),),  # not a cosine
            nearest_different_label=None,
            feature_space="dermlip (512-d)",
            bank_name="toy@dermlip",
            bank_hash="deadbeef",
            k=1,
        )
        contract = self._contract(bad)
        self.assertFalse(contract.safe_to_display)
        self.assertTrue(
            any("outside the cosine range" in f for f in contract.verification_failures())
        )

    def test_an_empty_neighbour_list_is_host_rejected(self):
        empty = SimilarCasesContext(
            query_case_id="999",
            reference_diagnosis="NEV",
            neighbors=(),
            nearest_different_label=None,
            feature_space="dermlip (512-d)",
            bank_name="toy@dermlip",
            bank_hash="deadbeef",
            k=3,
        )
        contract = self._contract(empty)
        self.assertFalse(contract.safe_to_display)
        self.assertTrue(any("no neighbours" in f for f in contract.verification_failures()))


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class RenderAppendixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _context(self, reference="NEV"):
        return build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis=reference,
            bank=_toy_bank(),
            query_case_id="999",
            k=3,
            feature_space="dermlip (512-d frozen image embedding; image-only)",
        )

    def _contract(self, similar_cases=None):
        return build_contract(
            case_id="derm7pt-999",
            classifier_diagnosis="NEV",
            classifier_confidence=0.8,
            criteria=_melanoma_criteria(),
            catalog=self.catalog,
            similar_cases=similar_cases,
        )

    APPENDIX_HEADER = "### Similar confirmed cases (on request)"

    def test_default_view_never_shows_the_appendix_even_when_present(self):
        # MG52: the case-based explanation is ON-DEMAND. Present on the contract, but the default
        # 10-second render must not surface it.
        contract = self._contract(self._context())
        rendered = render_markdown(contract)
        self.assertNotIn(self.APPENDIX_HEADER, rendered)
        self.assertNotIn("Saliency", rendered)

    def test_default_view_is_byte_identical_whether_or_not_context_is_attached(self):
        # The progressive-disclosure invariant: attaching similar_cases must not perturb the
        # default clinician view by a single byte.
        with_ctx = render_markdown(self._contract(self._context()))
        without = render_markdown(self._contract())
        self.assertEqual(with_ctx, without)

    def test_appendix_appears_only_when_requested_and_present(self):
        contract = self._contract(self._context())
        rendered = render_markdown(contract, include_appendix=True)
        self.assertIn(self.APPENDIX_HEADER, rendered)
        # neighbour rows (confirmed diagnoses) and the boundary counterfactual are both shown.
        self.assertIn("train case", rendered)
        self.assertIn("Nearest different-label case", rendered)
        self.assertIn("by construction", rendered.lower())

    def test_requesting_the_appendix_with_no_context_omits_the_similar_cases_block(self):
        # include_appendix=True but similar_cases is None: the tier redesign made Tier 3 carry
        # MORE than the case-based explanation (saliency pointer, remaining recommendations,
        # record hashes, the LOCO surface), so the appendix is no longer empty without a
        # retrieval context -- but the similar-cases block itself must not appear, and the
        # DEFAULT view must remain byte-identical (the progressive-disclosure invariant the
        # sibling test above pins is about the default view, and still holds).
        contract = self._contract()
        rendered = render_markdown(contract, include_appendix=True)
        self.assertNotIn(self.APPENDIX_HEADER, rendered)
        self.assertNotIn("train case", rendered)
        # Tier-3 material that does not depend on the retrieval context still renders on request.
        self.assertIn("Saliency (caveated appendix)", rendered)
        self.assertIn("Per-criterion influence", rendered)

    def test_saliency_caveat_text_is_present_behind_the_on_demand_block(self):
        rendered = render_markdown(self._contract(self._context()), include_appendix=True)
        self.assertIn("Saliency (caveated appendix)", rendered)
        self.assertIn("not a substitute for clinician localization", rendered)
        self.assertIn("spatially diffuse", rendered)
        # the deletion/insertion faithfulness evidence is cited by file, and the p-value stated.
        self.assertIn("attribution_faithfulness_v1.json", rendered)
        self.assertIn("p<1e-5", rendered)

    def test_default_view_stays_within_the_ten_second_line_budget_with_context_attached(self):
        # The default render must remain the compact 10-second view even for a contract that carries
        # the (unshown) case-based explanation.
        contract = self._contract(self._context())
        self.assertLessEqual(len(render_markdown(contract).splitlines()), 60)


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class ExplainReasoningCaseBasedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _context(self):
        return build_similar_cases_context(
            query_embedding=np.array([1.0, 0.05]),
            reference_diagnosis="NEV",
            bank=_toy_bank(),
            query_case_id="999",
            k=3,
            feature_space="dermlip (512-d)",
        )

    def _contract(self, similar_cases=None):
        return build_contract(
            case_id="derm7pt-999",
            classifier_diagnosis="NEV",
            classifier_confidence=0.8,
            criteria=_melanoma_criteria(),
            catalog=self.catalog,
            similar_cases=similar_cases,
        )

    def test_steps_one_through_eight_are_unchanged(self):
        # Backward-compatibility guard: the case-based step is purely additive.
        steps = AgentTools(self._contract(), self.catalog).explain_reasoning().payload["steps"]
        self.assertEqual(len(steps), 9)
        self.assertEqual(steps[7]["name"], "contract_verification")
        self.assertEqual([s["step"] for s in steps[:8]], list(range(1, 9)))

    def test_step_nine_surfaces_the_context_when_present(self):
        tools = AgentTools(self._contract(self._context()), self.catalog)
        step9 = tools.explain_reasoning().payload["steps"][8]
        self.assertEqual(step9["name"], "case_based_explanation")
        self.assertTrue(step9["on_demand"])
        self.assertTrue(step9["similar_cases_available"])
        self.assertIsNotNone(step9["similar_cases"])
        self.assertEqual(len(step9["similar_cases"]["neighbors"]), 3)

    def test_step_nine_reports_unavailable_but_still_carries_the_caveat_when_absent(self):
        tools = AgentTools(self._contract(), self.catalog)
        step9 = tools.explain_reasoning().payload["steps"][8]
        self.assertFalse(step9["similar_cases_available"])
        self.assertIsNone(step9["similar_cases"])
        # The saliency caveat and faithfulness note ride the reasoning step regardless of whether a
        # bank was wired for this session.
        self.assertIn("not a substitute for clinician localization", step9["saliency_caveat"])
        self.assertIn("attribution_faithfulness_v1.json", step9["saliency_caveat"])
        self.assertIn("identity", step9["faithful_by_construction"].lower())


if __name__ == "__main__":
    unittest.main()
