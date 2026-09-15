"""Contract tests for the Phase 2.4 DermPact perception module."""
from __future__ import annotations

import unittest

from dermpact.perception.core import DermPactPerception, ObservedCase


class DermPactPhase24PerceptionTest(unittest.TestCase):
    def test_label_bearing_facts_are_forwarded_without_rejection(self):
        perception = DermPactPerception(lambda case: case.clinical_facts)
        case = ObservedCase(
            case_id="case-001",
            dermoscopic_image=object(),
            clinical_facts={
                "diagnosis_label": "melanoma",
                "checklist_labels": {"streaks": True},
            },
        )

        self.assertEqual(perception.assess(case), case.clinical_facts)

    def test_clinical_photo_presence_must_match_the_observed_image(self):
        def inference_must_not_run(_: ObservedCase):
            raise AssertionError("inference ran with a contradictory clinical-photo mask")

        perception = DermPactPerception(inference_must_not_run)

        with self.assertRaisesRegex(ValueError, "clinical image presence"):
            case = ObservedCase(
                case_id="case-002",
                dermoscopic_image=object(),
                clinical_image=object(),
                clinical_image_present=False,
            )
            perception.assess(case)


if __name__ == "__main__":
    unittest.main()
