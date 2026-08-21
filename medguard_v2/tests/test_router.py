"""allowed_retrievals() is the policy `search_guidelines` enforces: what a gated tool call may
serve for a given router state. `route()` already executes retrieval; this is the same decision
expressed as a lookup table instead, so it is worth pinning each of the four states directly
rather than only through the indirection of AgentTools.search_guidelines.
"""
from __future__ import annotations

import unittest

from medguard_v2.evidence.guideline_binding import MANAGEMENT_QUESTION_TYPES
from medguard_v2.evidence.router import QUESTION_TYPES_FOR_STATE, RetrievalState, allowed_retrievals


class AllowedRetrievalsTest(unittest.TestCase):
    def test_no_coverage_allows_nothing(self):
        # The only honest action is absence; nothing is indicated, so nothing is in-plan.
        self.assertEqual(allowed_retrievals(RetrievalState.NO_COVERAGE, "NEV"), set())

    def test_routine_allows_only_the_committed_conditions_management_types(self):
        plan = allowed_retrievals(RetrievalState.ROUTINE, "MEL")
        self.assertEqual(
            plan,
            {(qt_group, qt) for qt_group in ("MEL",) for qt in QUESTION_TYPES_FOR_STATE[RetrievalState.ROUTINE]},
        )
        # No wildcard: ROUTINE knows exactly what it needs.
        self.assertNotIn(("MEL", "*"), plan)
        self.assertNotIn(("BCC", "initial_local_management"), plan)

    def test_routine_mirrors_guideline_bindings_management_question_types_exactly(self):
        # F9: ROUTINE's plan used to hand-list a subset that omitted follow_up_surveillance even
        # though search_guidelines' own tool doc lists it as a valid question type -- a
        # ROUTINE-state agent asking for it by name got gated. Pinned explicitly (not just via
        # the dynamically-derived assertion above) so a future edit that reintroduces drift
        # between the two tuples is caught here.
        self.assertEqual(QUESTION_TYPES_FOR_STATE[RetrievalState.ROUTINE], MANAGEMENT_QUESTION_TYPES)
        self.assertIn("follow_up_surveillance", QUESTION_TYPES_FOR_STATE[RetrievalState.ROUTINE])
        self.assertIn(("MEL", "follow_up_surveillance"), allowed_retrievals(RetrievalState.ROUTINE, "MEL"))

    def test_conflict_allows_both_sides_of_the_differential_with_a_wildcard(self):
        plan = allowed_retrievals(RetrievalState.CONFLICT, "NEV")
        self.assertEqual(plan, {("NEV", "*"), ("MEL", "*")})

    def test_conflict_collapses_to_one_entry_when_the_committed_group_is_already_mel(self):
        plan = allowed_retrievals(RetrievalState.CONFLICT, "MEL")
        self.assertEqual(plan, {("MEL", "*")})

    def test_undecidable_allows_both_sides_of_the_differential_with_a_wildcard(self):
        plan = allowed_retrievals(RetrievalState.UNDECIDABLE, "SK")
        self.assertEqual(plan, {("SK", "*"), ("MEL", "*")})


if __name__ == "__main__":
    unittest.main()
