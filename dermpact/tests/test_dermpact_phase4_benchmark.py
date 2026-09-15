"""Structural and scope checks on the built Phase 4 applicability benchmark."""
from __future__ import annotations

import json
import unittest
from collections import Counter

from dermpact.eval import dermpact_phase4_benchmark as bench

PRESENT = (bench.OUT_DIR / "benchmark_manifest.json").exists()


@unittest.skipUnless(PRESENT, "Phase 4 benchmark not built")
class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((bench.OUT_DIR / "benchmark_manifest.json").read_text(encoding="utf-8"))
        cls.runs = bench.load_benchmark()

    def test_hard_scope(self):
        hs = self.manifest["hard_scope_checks"]
        self.assertEqual(hs["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(hs["ph2_rows_loaded"], 0)
        self.assertEqual(hs["derm7pt_development_rows"], 616)
        self.assertTrue(all(r["case_id"].startswith("derm7pt-case-") for r in self.runs))

    def test_exact_denominators(self):
        self.assertEqual(len(self.runs), 616 * len(bench.VARIANTS))
        self.assertEqual(len({r["run_id"] for r in self.runs}), len(self.runs))
        per_case = Counter(r["case_id"] for r in self.runs)
        self.assertTrue(all(v == len(bench.VARIANTS) for v in per_case.values()))

    def test_gold_shapes(self):
        for r in self.runs:
            self.assertEqual(set(r["gold"]), set(r["query_rule_ids"]))
            for g in r["gold"].values():
                self.assertIn(g["state"], ("applicable", "not_applicable", "applicability_unknown", "no_catalogue_coverage"))
                if g["state"] == "applicable":
                    self.assertTrue(g["action"])
                if g["state"] == "applicability_unknown":
                    self.assertTrue(g["missing"])
            if r["variant"] == "wrong_entity":
                self.assertTrue(r["expected_refusal"])
                self.assertNotEqual(r["expected_case_id"], r["case_id"])
            else:
                self.assertEqual(r["expected_case_id"], r["case_id"])

    def test_variant_gold_invariants(self):
        for r in self.runs:
            states = {rid: g["state"] for rid, g in r["gold"].items()}
            if r["variant"] in ("stale_2027", "jurisdiction_us"):
                self.assertTrue(all(s == "no_catalogue_coverage" for s in states.values()), r["run_id"])
            if r["variant"] == "scope_query_staging":
                self.assertEqual(states, {"eado-mel-p1-2024:R1": "no_catalogue_coverage"})
            if r["variant"] == "rcm_unavailable":
                self.assertEqual(states["eado-mel-p1-2024:R7"], "not_applicable")
            if r["variant"] == "high_risk_nevus_100":
                self.assertEqual(states["eado-mel-p1-2024:R6"], "applicable")
            if r["variant"] == "high_risk_nevus_99":
                self.assertEqual(states["eado-mel-p1-2024:R6"], "not_applicable")
            if r["variant"] == "no_image":
                self.assertEqual(states["eado-mel-p1-2024:R4"], "not_applicable")
                self.assertEqual(states["eado-mel-p1-2024:R3"], "not_applicable")

    def test_true_label_view_present(self):
        self.assertTrue(all("gold_true_label_view" in r for r in self.runs))


if __name__ == "__main__":
    unittest.main()
