import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medguard_v2.data.derm7pt import (
    CRITERION_VOCABULARY,
    DEFAULT_ROOT,
    DERM7PT_CLASS_ORDER,
    DIAGNOSIS_GROUPS,
    DIAGNOSIS_GROUP_ORDER,
    Derm7ptDataError,
    EXPECTED_SPLIT_SIZES,
    SEVEN_CRITERIA,
    SPLITS,
    analysis_cohort,
    cases_without_distinct_views,
    class_counts,
    load_cases,
    split_counts,
)
from medguard_v2.data.manifest import (
    MANIFEST_SCHEMA,
    build_manifest,
    canonical_body_bytes,
    freeze_manifest,
)


CASES = load_cases()


class Derm7ptLoaderTest(unittest.TestCase):
    def test_all_1011_cases_load(self):
        self.assertEqual(len(CASES), 1011)
        self.assertEqual([case.case_index for case in CASES], list(range(1011)))
        self.assertEqual(sorted(case.case_num for case in CASES), list(range(1, 1012)))

    def test_splits_have_the_official_sizes_and_partition_the_set(self):
        self.assertEqual(split_counts(CASES), EXPECTED_SPLIT_SIZES)
        self.assertEqual(sum(EXPECTED_SPLIT_SIZES.values()), 1011)
        by_split = {split: set() for split in SPLITS}
        for case in CASES:
            self.assertIn(case.split, SPLITS)
            by_split[case.split].add(case.case_index)
        self.assertEqual(
            by_split["train"] | by_split["valid"] | by_split["test"], set(range(1011))
        )
        self.assertEqual(by_split["train"] & by_split["valid"], set())
        self.assertEqual(by_split["train"] & by_split["test"], set())
        self.assertEqual(by_split["valid"] & by_split["test"], set())

    def test_every_image_path_exists_on_disk(self):
        images_root = (DEFAULT_ROOT / "images").resolve()
        for case in CASES:
            for path in (case.clinic_path, case.derm_path):
                self.assertTrue(path.is_file(), f"missing image {path}")
                self.assertTrue(path.is_absolute(), path)
                # resolved against the true on-disk casing, so this holds even
                # on a case-sensitive filesystem
                self.assertIn(path.name, {p.name for p in path.parent.iterdir()})
                path.relative_to(images_root)

    def test_split_index_verification_rejects_a_broken_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = root / "meta"
            meta.mkdir(parents=True)
            (root / "images").mkdir()
            (meta / "meta.csv").write_text("case_num\n", encoding="utf-8")
            for split, size in EXPECTED_SPLIT_SIZES.items():
                rows = "\n".join(str(i) for i in range(size))
                (meta / f"{split}_indexes.csv").write_text(
                    "indexes\n" + rows + "\n", encoding="utf-8"
                )
            with self.assertRaises(Derm7ptDataError) as ctx:
                load_cases(root)
            self.assertIn("partition", str(ctx.exception))


class Derm7ptVocabularyTest(unittest.TestCase):
    def test_all_twenty_diagnosis_strings_map_to_a_group(self):
        observed = {case.diagnosis for case in CASES}
        self.assertEqual(len(observed), 20)
        self.assertEqual(observed, set(DIAGNOSIS_GROUPS))
        for diagnosis, group in DIAGNOSIS_GROUPS.items():
            self.assertIn(group, DIAGNOSIS_GROUP_ORDER, diagnosis)
        for case in CASES:
            self.assertEqual(case.diagnosis_group, DIAGNOSIS_GROUPS[case.diagnosis])

    def test_class_counts_are_the_frozen_five_class_grouping(self):
        self.assertEqual(
            class_counts(CASES),
            {"BCC": 42, "NEV": 575, "MEL": 252, "SK": 45, "MISC": 97},
        )
        self.assertEqual(sum(class_counts(CASES).values()), 1011)

    def test_criterion_values_come_from_the_expected_vocabularies(self):
        self.assertEqual(tuple(CRITERION_VOCABULARY), SEVEN_CRITERIA)
        seen = {name: set() for name in SEVEN_CRITERIA}
        for case in CASES:
            self.assertEqual(tuple(case.criteria), SEVEN_CRITERIA)
            for name, value in case.criteria.items():
                self.assertIn(value, CRITERION_VOCABULARY[name], f"{name}={value!r}")
                seen[name].add(value)
        for name in SEVEN_CRITERIA:
            self.assertEqual(seen[name], set(CRITERION_VOCABULARY[name]), name)

    def test_scalar_fields_are_well_formed(self):
        for case in CASES:
            self.assertIn(case.management, {"clinical follow up", "excision",
                                            "no further examination"})
            self.assertIn(case.difficulty, {"low", "medium", "high"})
            self.assertIn(case.elevation, {"flat", "nodular", "palpable"})
            self.assertIn(case.sex, {"female", "male"})
            self.assertTrue(0 <= case.seven_point_score <= 10, case.seven_point_score)
            self.assertTrue(case.location)

    def test_nine_cases_reuse_the_dermoscopic_image_as_the_clinical_one(self):
        shared = cases_without_distinct_views(CASES)
        self.assertEqual(len(shared), 9)
        self.assertEqual(
            sorted(case.case_num for case in shared),
            [565, 800, 824, 827, 833, 834, 839, 844, 851],
        )


class Derm7ptAnalysisCohortTest(unittest.TestCase):
    """The A6 cohort filter must state its own denominator and its own skew."""

    NO_DISTINCT_VIEW_CASE_NUMS = [565, 800, 824, 827, 833, 834, 839, 844, 851]

    def test_class_order_alias_is_the_same_canonical_tuple(self):
        self.assertIs(DIAGNOSIS_GROUP_ORDER, DERM7PT_CLASS_ORDER)
        self.assertEqual(DERM7PT_CLASS_ORDER, ("BCC", "NEV", "MEL", "SK", "MISC"))

    def test_default_keeps_every_case_and_reports_what_it_kept(self):
        kept, report = analysis_cohort(CASES)
        self.assertEqual(len(kept), 1011)
        self.assertEqual([c.case_num for c in kept], [c.case_num for c in CASES])
        self.assertFalse(report["require_distinct_views"])
        self.assertEqual(report["filter"], "none")
        self.assertEqual(report["n_input"], 1011)
        self.assertEqual(report["n_kept"], 1011)
        self.assertEqual(report["n_dropped"], 0)
        self.assertEqual(report["dropped_case_nums"], [])
        self.assertEqual(report["dropped_class_counts"], {})
        self.assertIsNone(report["reason"])
        # "nothing was dropped" must not be mistaken for "nothing to drop"
        self.assertEqual(report["retained_without_distinct_views"], 9)

    def test_requiring_distinct_views_drops_exactly_the_nine_documented_cases(self):
        kept, report = analysis_cohort(CASES, require_distinct_views=True)
        self.assertEqual(report["n_dropped"], 9)
        self.assertEqual(len(kept), 1011 - 9)
        self.assertEqual(report["n_kept"], len(kept))
        self.assertEqual(report["dropped_case_nums"], self.NO_DISTINCT_VIEW_CASE_NUMS)
        self.assertEqual(
            report["dropped_case_nums"],
            sorted(case.case_num for case in cases_without_distinct_views(CASES)),
        )
        self.assertTrue(all(case.has_distinct_views for case in kept))
        self.assertEqual(report["retained_without_distinct_views"], 0)
        self.assertIn("no-op", str(report["reason"]))

    def test_the_dropped_nine_are_three_mel_and_six_misc(self):
        """The exclusion is NOT distribution-neutral -- that is why it is reported."""
        _, report = analysis_cohort(CASES, require_distinct_views=True)
        self.assertEqual(report["dropped_class_counts"], {"MEL": 3, "MISC": 6})
        self.assertEqual(sum(report["dropped_class_counts"].values()), 9)
        for group in ("NEV", "BCC", "SK"):
            self.assertNotIn(group, report["dropped_class_counts"])

    def test_kept_class_counts_are_the_full_table_and_account_for_every_case(self):
        _, full = analysis_cohort(CASES)
        _, filtered = analysis_cohort(CASES, require_distinct_views=True)
        self.assertEqual(list(filtered["kept_class_counts"]), list(DERM7PT_CLASS_ORDER))
        self.assertEqual(sum(filtered["kept_class_counts"].values()), 1002)
        self.assertEqual(full["kept_class_counts"], class_counts(CASES))
        for group in DERM7PT_CLASS_ORDER:
            dropped = filtered["dropped_class_counts"].get(group, 0)
            self.assertEqual(
                filtered["kept_class_counts"][group] + dropped,
                full["kept_class_counts"][group],
                group,
            )

    def test_the_filter_applies_to_whatever_subset_it_is_handed(self):
        """Callers filter a split, not always the whole dataset."""
        test_split = [case for case in CASES if case.split == "test"]
        kept, report = analysis_cohort(test_split, require_distinct_views=True)
        expected = [
            case.case_num for case in cases_without_distinct_views(test_split)
        ]
        self.assertEqual(report["n_input"], len(test_split))
        self.assertEqual(report["dropped_case_nums"], sorted(expected))
        self.assertEqual(len(kept), len(test_split) - len(expected))

    def test_an_empty_cohort_is_reported_rather_than_crashing(self):
        kept, report = analysis_cohort([], require_distinct_views=True)
        self.assertEqual(kept, [])
        self.assertEqual(report["n_input"], 0)
        self.assertEqual(report["n_dropped"], 0)
        self.assertEqual(sum(report["kept_class_counts"].values()), 0)

    def test_the_report_is_json_serialisable_for_an_evidence_artefact(self):
        _, report = analysis_cohort(CASES, require_distinct_views=True)
        round_tripped = json.loads(json.dumps(report))
        self.assertEqual(round_tripped["n_dropped"], 9)
        self.assertEqual(round_tripped["dropped_class_counts"], {"MEL": 3, "MISC": 6})


class Derm7ptManifestTest(unittest.TestCase):
    def test_manifest_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "a" / "manifest.json"
            second = Path(folder) / "b" / "manifest.json"
            freeze_manifest(first)
            freeze_manifest(second)
            first_bytes = first.read_bytes()
            self.assertEqual(first_bytes, second.read_bytes())
            self.assertEqual(
                hashlib.sha256(first_bytes).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            )

    def test_manifest_body_digest_covers_the_body(self):
        manifest = build_manifest(CASES)
        body = {k: v for k, v in manifest.items() if k != "body_sha256"}
        self.assertEqual(
            manifest["body_sha256"],
            hashlib.sha256(canonical_body_bytes(body)).hexdigest(),
        )
        mutated = json.loads(json.dumps(body))
        mutated["cases"][0]["derm_image"]["sha256"] = "0" * 64
        self.assertNotEqual(
            manifest["body_sha256"],
            hashlib.sha256(canonical_body_bytes(mutated)).hexdigest(),
        )

    def test_manifest_records_the_required_fields_and_no_image_bytes(self):
        manifest = build_manifest(CASES)
        self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
        self.assertEqual(manifest["n_records"], 1011)
        self.assertEqual(len(manifest["cases"]), 1011)
        self.assertEqual(manifest["split_counts"], EXPECTED_SPLIT_SIZES)
        self.assertEqual(sum(manifest["class_counts"].values()), 1011)
        for record in manifest["cases"]:
            self.assertEqual(tuple(record["criteria"]), SEVEN_CRITERIA)
            self.assertIn(record["split"], SPLITS)
            self.assertIn(record["diagnosis_group"], DIAGNOSIS_GROUP_ORDER)
            self.assertTrue(record["management"])
            self.assertTrue(record["difficulty"])
            for side in ("clinic_image", "derm_image"):
                digest = record[side]["sha256"]
                self.assertEqual(len(digest), 64, digest)
                self.assertEqual(digest, digest.lower().strip())
                self.assertFalse(Path(record[side]["path"]).is_absolute())
        # metadata + digests only: no machine-specific paths, no image bytes
        blob = json.dumps(manifest)
        self.assertNotIn(str(DEFAULT_ROOT), blob)
        self.assertNotIn(DEFAULT_ROOT.drive or "\\\\", blob)
        self.assertLess(len(blob), 2_000_000)

    def test_image_digests_match_the_files_for_a_sample(self):
        manifest = build_manifest(CASES)
        images_root = (DEFAULT_ROOT / "images").resolve()
        for record in manifest["cases"][::200]:
            for side in ("clinic_image", "derm_image"):
                path = images_root / record[side]["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    record[side]["sha256"],
                    path,
                )


if __name__ == "__main__":
    unittest.main()
