"""The override-annotation store: capture is validated, append-only, and strictly opt-in.

Two properties carry the weight here:

1. **Garbage never enters or leaves the log.** ``record`` rejects unknown criteria and
   out-of-vocabulary values before touching the file; ``load`` refuses a tampered file rather
   than silently serving a subset.
2. **Zero behaviour change when absent.** The ``AgentTools`` hook exists only when a store is
   passed; the default (None) path must be byte-identical, and a REJECTED override must never
   deposit an annotation even when a store is attached.

No network, no paid calls. The store tests need no catalog; the hook tests use the same pinned
catalog fixture as test_clinician_agent.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from medguard_v2.agent.annotation_store import (
    DEFAULT_PATH,
    RECORD_PROTOCOL,
    AnnotationStore,
)
from medguard_v2.agent.clinician_agent import AgentTools
from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog

CLEAN = {name: "absent" for name in CRITERION_NAMES}


class AnnotationStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "annotations.jsonl"
        self.store = AnnotationStore(self.path)

    def test_default_path_is_the_derm7pt_evidence_log(self):
        self.assertEqual(AnnotationStore().path, DEFAULT_PATH)
        self.assertEqual(DEFAULT_PATH.name, "override_annotations.jsonl")

    def test_record_load_round_trip(self):
        first = self.store.record(
            "derm7pt-42", "pigment_network", "atypical", "absent",
            contract_receipt={"split": "valid", "seed": 20260810},
        )
        second = self.store.record(
            "derm7pt-43", "blue_whitish_veil", "present", "absent",
            source="panel_review",
        )
        loaded = self.store.load()
        self.assertEqual(loaded, [first, second])  # chronological, append order
        self.assertEqual(loaded[0]["case_id"], "derm7pt-42")
        self.assertEqual(loaded[0]["corrected_value"], "atypical")
        self.assertEqual(loaded[0]["prior_value"], "absent")
        self.assertEqual(loaded[0]["source"], "clinician_override")
        self.assertEqual(loaded[0]["protocol"], RECORD_PROTOCOL)
        self.assertEqual(
            loaded[0]["contract_provenance"], {"split": "valid", "seed": 20260810}
        )
        self.assertRegex(
            loaded[0]["recorded_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        )
        self.assertEqual(loaded[1]["source"], "panel_review")
        self.assertIsNone(loaded[1]["contract_provenance"])

    def test_load_of_a_missing_file_is_empty(self):
        self.assertEqual(self.store.load(), [])
        self.assertFalse(self.path.exists())  # load never creates the file

    def test_record_rejects_an_unknown_criterion_and_writes_nothing(self):
        with self.assertRaises(ValueError):
            self.store.record("derm7pt-1", "asymmetry", "present", None)
        self.assertFalse(self.path.exists())

    def test_record_rejects_an_out_of_vocabulary_value_and_writes_nothing(self):
        with self.assertRaises(ValueError) as ctx:
            self.store.record("derm7pt-1", "pigment_network", "very atypical", "absent")
        self.assertIn("not a permitted value", str(ctx.exception))
        self.assertFalse(self.path.exists())

    def test_record_rejects_an_out_of_vocabulary_prior_value(self):
        with self.assertRaises(ValueError):
            self.store.record("derm7pt-1", "pigment_network", "atypical", "blurry")
        self.assertFalse(self.path.exists())

    def test_record_rejects_a_blank_case_id_and_a_blank_source(self):
        with self.assertRaises(ValueError):
            self.store.record("  ", "pigment_network", "atypical", "absent")
        with self.assertRaises(ValueError):
            self.store.record("derm7pt-1", "pigment_network", "atypical", "absent", source=" ")
        self.assertFalse(self.path.exists())

    def test_record_accepts_indeterminate_and_a_none_prior(self):
        entry = self.store.record("derm7pt-1", "blue_whitish_veil", INDETERMINATE, None)
        self.assertEqual(entry["corrected_value"], INDETERMINATE)
        self.assertIsNone(entry["prior_value"])
        self.assertEqual(len(self.store.load()), 1)

    def test_last_write_per_case_criterion_wins_in_label_updates(self):
        self.store.record("derm7pt-1", "pigment_network", "atypical", "absent")
        self.store.record("derm7pt-1", "blue_whitish_veil", "present", "absent")
        self.store.record("derm7pt-1", "pigment_network", "typical", "atypical")
        self.assertEqual(
            self.store.as_label_updates(),
            {"derm7pt-1": {"pigment_network": "typical", "blue_whitish_veil": "present"}},
        )

    def test_a_final_indeterminate_suppresses_the_earlier_concrete_value(self):
        # The clinician's LAST word on the pair was "cannot judge": no label may survive, and in
        # particular the earlier concrete value must NOT resurface once indeterminate is filtered.
        self.store.record("derm7pt-1", "pigment_network", "atypical", "absent")
        self.store.record("derm7pt-1", "pigment_network", INDETERMINATE, "atypical")
        self.assertEqual(self.store.as_label_updates(), {})
        self.assertEqual(
            self.store.as_label_updates(include_indeterminate=True),
            {"derm7pt-1": {"pigment_network": INDETERMINATE}},
        )

    def test_append_only_across_store_instances_on_the_same_file(self):
        self.store.record("derm7pt-1", "pigment_network", "atypical", "absent")
        again = AnnotationStore(self.path)
        again.record("derm7pt-2", "streaks", "irregular", "regular")
        self.assertEqual(
            [(r["case_id"], r["criterion"]) for r in again.load()],
            [("derm7pt-1", "pigment_network"), ("derm7pt-2", "streaks")],
        )

    def test_load_refuses_a_tampered_line_naming_its_number(self):
        self.store.record("derm7pt-1", "pigment_network", "atypical", "absent")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        with self.assertRaises(ValueError) as ctx:
            self.store.load()
        self.assertIn(":2:", str(ctx.exception))

    def test_load_refuses_an_out_of_vocabulary_value_smuggled_into_the_file(self):
        record = self.store.record("derm7pt-1", "pigment_network", "atypical", "absent")
        forged = dict(record, corrected_value="very atypical")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(forged) + "\n")
        with self.assertRaises(ValueError):
            self.store.load()

    def test_contract_provenance_is_trimmed_to_scalars(self):
        entry = self.store.record(
            "derm7pt-1", "pigment_network", "atypical", "absent",
            contract_receipt={
                "split": "valid",
                "seed": 20260810,
                "diagnosis_C": 100.0,
                "criteria_C": {"pigment_network": 100.0},   # nested: dropped
                "operating_point": {"SK": 0.5},             # nested: dropped
            },
        )
        self.assertEqual(
            entry["contract_provenance"],
            {"split": "valid", "seed": 20260810, "diagnosis_C": 100.0},
        )

    def test_summary_counts(self):
        self.store.record("derm7pt-1", "pigment_network", "atypical", "absent")
        self.store.record("derm7pt-1", "pigment_network", "typical", "atypical")
        self.store.record("derm7pt-1", "blue_whitish_veil", INDETERMINATE, "absent")
        self.store.record("derm7pt-2", "streaks", "irregular", None, source="panel_review")
        summary = self.store.summary()
        self.assertEqual(summary["n_records"], 4)
        self.assertEqual(summary["n_cases"], 2)
        self.assertEqual(summary["n_case_criterion_pairs"], 3)
        # blue_whitish_veil's final value is indeterminate -> not a label update.
        self.assertEqual(summary["n_label_updates"], 2)
        self.assertEqual(summary["per_criterion"]["pigment_network"], 2)
        self.assertEqual(
            summary["per_source"], {"clinician_override": 3, "panel_review": 1}
        )


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class AgentHookTest(unittest.TestCase):
    """The AgentTools hook: fires on success, never on rejection, absent by default."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "annotations.jsonl"
        self.store = AnnotationStore(self.path)

    def _contract(self, receipt=None):
        return build_contract(
            case_id="derm7pt-7",
            classifier_diagnosis="NEV",
            classifier_confidence=0.8,
            criteria=dict(CLEAN),
            catalog=self.catalog,
            receipt=receipt,
        )

    def test_successful_override_is_recorded_with_case_id_prior_and_provenance(self):
        receipt = {"split": "valid", "seed": 20260810, "criteria_C": {"streaks": 100.0}}
        tools = AgentTools(self._contract(receipt), self.catalog, annotation_store=self.store)
        result = tools.override_criterion("pigment_network", "atypical")
        self.assertTrue(result.ok)

        records = self.store.load()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["case_id"], "derm7pt-7")
        self.assertEqual(records[0]["criterion"], "pigment_network")
        self.assertEqual(records[0]["corrected_value"], "atypical")
        self.assertEqual(records[0]["prior_value"], "absent")   # the contract's value
        self.assertEqual(records[0]["source"], "clinician_override")
        self.assertEqual(
            records[0]["contract_provenance"], {"split": "valid", "seed": 20260810}
        )

    def test_a_rejected_override_records_nothing(self):
        tools = AgentTools(self._contract(), self.catalog, annotation_store=self.store)
        self.assertFalse(tools.override_criterion("pigment_network", "very atypical").ok)
        self.assertFalse(tools.override_criterion("asymmetry", "present").ok)
        self.assertEqual(self.store.load(), [])
        self.assertFalse(self.path.exists())  # nothing was ever written

    def test_a_reoverride_appends_a_second_record_with_the_true_prior(self):
        tools = AgentTools(self._contract(), self.catalog, annotation_store=self.store)
        tools.override_criterion("pigment_network", "atypical")
        tools.override_criterion("pigment_network", "typical")
        records = self.store.load()
        self.assertEqual(
            [(r["corrected_value"], r["prior_value"]) for r in records],
            [("atypical", "absent"), ("typical", "atypical")],
        )
        # Last write wins when replayed as labels.
        self.assertEqual(
            self.store.as_label_updates(), {"derm7pt-7": {"pigment_network": "typical"}}
        )

    def test_default_none_leaves_the_override_path_byte_identical(self):
        with_store = AgentTools(
            self._contract(), self.catalog, annotation_store=self.store
        )
        without_store = AgentTools(self._contract(), self.catalog)
        self.assertIsNone(without_store.annotation_store)

        a = with_store.override_criterion("blue_whitish_veil", "present")
        b = without_store.override_criterion("blue_whitish_veil", "present")
        self.assertEqual(a.payload, b.payload)   # identical tool behaviour
        self.assertEqual(with_store.overrides, without_store.overrides)
        self.assertEqual(with_store.override_sequence, without_store.override_sequence)
        # Only the attached store captured anything; the default path wrote no file.
        self.assertEqual(len(self.store.load()), 1)


if __name__ == "__main__":
    unittest.main()
