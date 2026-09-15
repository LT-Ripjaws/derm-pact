"""Focused tests for Phase 2.4A controller receipt-reuse contracts."""
from __future__ import annotations

import copy
import runpy
import unittest

from dermpact.eval import dermpact_phase2_4a_training as phase24a


class Phase24AControllerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller = runpy.run_path(
            str(
                phase24a.ROOT
                / "research/scripts/dermpact_phase2_4a_full_run_controller.py"
            )
        )
        cls.validate_reload = staticmethod(
            controller["_canonical_reload_contract_valid"]
        )

    @staticmethod
    def valid_payload():
        return {
            "residual_state_reload_exact": True,
            "output_reload_within_tolerance": True,
            "output_reload_tolerance": 1e-6,
            "output_reload_max_abs": {
                "diagnosis": 0.0,
                "melanoma": 1e-7,
                "criteria": 5e-7,
                "criterion_residual": 9e-7,
                "criterion_attention": 1e-6,
            },
        }

    def test_accepts_complete_finite_reload_contract(self):
        self.assertTrue(self.validate_reload(self.valid_payload()))

    def test_rejects_legacy_missing_and_invalid_reload_contracts(self):
        mutations = (
            ("legacy flag", lambda value: value.update(
                {"residual_reload_exact": True, "residual_state_reload_exact": False}
            )),
            ("missing output", lambda value: value["output_reload_max_abs"].pop(
                "criterion_attention"
            )),
            ("over tolerance", lambda value: value["output_reload_max_abs"].update(
                {"criteria": 1.000001e-6}
            )),
            ("negative", lambda value: value["output_reload_max_abs"].update(
                {"melanoma": -1e-12}
            )),
            ("non-finite", lambda value: value["output_reload_max_abs"].update(
                {"diagnosis": float("nan")}
            )),
            ("boolean", lambda value: value["output_reload_max_abs"].update(
                {"diagnosis": False}
            )),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(self.valid_payload())
                mutate(payload)
                self.assertFalse(self.validate_reload(payload))


if __name__ == "__main__":
    unittest.main()
