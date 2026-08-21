"""Tests for the G1c few-shot extensions and the G1c ladder.

Three properties carry the gate's whole argument, and each gets a test that fails loudly:

1. **The zero-shot arm must be G1's own request.**  If ``FewShotAssessor`` with no sheets built a
   request that differed from ``CriterionAssessor``'s by so much as a space, the "baseline" row
   would be a fresh paid run that could drift from the published number it is being compared to.
2. **No exemplar may come from valid or test.**  Cheap to break by accident, impossible to detect
   afterwards from the numbers.
3. **The few-shot prompt must be the zero-shot prompt plus images.**  If a definition changed, F2
   would be varying two things at once and the effect could not be attributed.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Optional

from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES, INDETERMINATE
from medguard_v2.eval.gates import g1_criterion_readability as g1
from medguard_v2.eval.gates import g1c_stronger_reader as g1c
from medguard_v2.perception import fewshot
from medguard_v2.perception.criterion_assessor import (
    AssessorCache,
    CriterionAssessor,
    build_user_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures that touch neither the network nor the dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeCase:
    case_index: int
    case_num: int
    split: str
    criteria: Mapping[str, str]
    derm_path: Path
    clinic_path: Path
    diagnosis: str = "melanoma"
    diagnosis_group: str = "MEL"
    difficulty: str = "low"
    seven_point_score: int = 0

    @property
    def has_distinct_views(self) -> bool:
        return self.clinic_path != self.derm_path


def _write_image(path: Path, size=(96, 64), colour=(120, 90, 70)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="PNG")
    return path


class FakeClient:
    """Returns a canned, schema-valid completion.  Records every call for assertions."""

    def __init__(self, values: Optional[Mapping[str, str]] = None):
        self.values = dict(values or {name: "absent" for name in CRITERION_NAMES})
        self.calls: List[Dict[str, Any]] = []
        self.call_log: List[Dict[str, Any]] = []
        self.last_call_metadata: Dict[str, Any] = {}

    def vision(self, prompt, image_path=None, *, system="", model=None, max_tokens=512,
               temperature=0.2, seed=None) -> str:
        paths = list(image_path or [])
        if len(set(paths)) != len(paths):
            raise ValueError("vision request contains duplicate image paths")
        self.calls.append({
            "prompt": prompt, "images": paths, "system": system, "model": model,
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        })
        self.call_log.append({"succeeded": True})
        self.last_call_metadata = {
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "latency_seconds": 0.01, "usage_available": True,
        }
        return json.dumps({
            name: {"value": self.values[name], "confidence": 0.5, "justification": "x"}
            for name in CRITERION_NAMES
        })


# ---------------------------------------------------------------------------
# F2 geometry and cost
# ---------------------------------------------------------------------------
class GridGeometryTest(unittest.TestCase):
    def test_grid_never_degenerates_to_a_ribbon(self):
        for n in range(1, 9):
            cols, rows = fewshot.grid_shape(n)
            self.assertGreaterEqual(cols * rows, n)
            width = cols * fewshot.PANEL_PX
            height = rows * (fewshot.PANEL_PX + fewshot.CAPTION_PX)
            ratio = max(width, height) / min(width, height)
            self.assertLessEqual(
                ratio, fewshot.MAX_SHEET_ASPECT + 1e-9,
                f"{n} panels produced a {cols}x{rows} sheet with edge ratio {ratio:.2f}",
            )

    def test_five_panels_prefer_a_grid_over_a_single_column(self):
        # A single column would be one cell cheaper and is exactly what a naive
        # minimise-cells rule picks; the aspect cap is what stops it.
        self.assertEqual(fewshot.grid_shape(5), (3, 2))

    def test_token_cost_matches_the_qwen_grid_exactly(self):
        for n in range(1, 9):
            cols, rows = fewshot.grid_shape(n)
            width = cols * fewshot.PANEL_PX
            height = rows * (fewshot.PANEL_PX + fewshot.CAPTION_PX)
            expected = (width * height) // (fewshot.SMART_RESIZE_FACTOR ** 2) + 2
            self.assertEqual(fewshot.sheet_token_cost(n), expected)

    def test_full_vocabulary_cost_is_reported_and_bounded(self):
        cost = fewshot.fewshot_image_token_cost()
        self.assertEqual(cost["n_sheets"], len(CRITERION_NAMES))
        # The design exists because 3 exemplars x 28 values x ~490 tokens is unaffordable.
        # Guard the property that motivated it rather than a magic number.
        self.assertLess(cost["total_image_tokens"], 3 * 28 * 490 / 10)


# ---------------------------------------------------------------------------
# F2 exemplar selection
# ---------------------------------------------------------------------------
class ExemplarSelectionTest(unittest.TestCase):
    def _cases(self, split: str) -> List[FakeCase]:
        out = []
        for i in range(6):
            criteria = {name: sorted(CRITERIA_BY_NAME[name].vocabulary)[i % 2]
                        for name in CRITERION_NAMES}
            out.append(FakeCase(i, 100 + i, split, criteria, Path(f"d{i}.png"), Path(f"c{i}.png")))
        return out

    def test_refuses_non_train_cases(self):
        cases = self._cases("valid")
        features = {str(c.case_num): [float(i), 0.0] for i, c in enumerate(cases)}
        with self.assertRaises(ValueError) as ctx:
            fewshot.select_exemplars(cases, features)
        self.assertIn("train", str(ctx.exception))

    def test_picks_the_medoid_and_is_deterministic(self):
        cases = self._cases("train")
        # Value A is carried by cases 0, 2, 4 (even i). Place 2 at the centroid of {0,2,4}.
        features = {str(c.case_num): [float(c.case_index), 0.0] for c in cases}
        first = fewshot.select_exemplars(cases, features)
        second = fewshot.select_exemplars(cases, features)
        self.assertEqual(first, second)
        name = CRITERION_NAMES[0]
        value_a = sorted(CRITERIA_BY_NAME[name].vocabulary)[0]
        self.assertEqual(first[name][value_a], 102)  # case_index 2 -> case_num 102

    def test_gaps_are_reported_not_hidden(self):
        cases = self._cases("train")
        features = {str(c.case_num): [float(c.case_index), 0.0] for c in cases}
        chosen = fewshot.select_exemplars(cases, features)
        gaps = fewshot.exemplar_gaps(chosen)
        # Every criterion with >2 permitted values is only half covered by this fixture.
        self.assertIn("vascular_structures", gaps)
        self.assertTrue(all(gaps[name] for name in gaps))


# ---------------------------------------------------------------------------
# F2 prompt construction
# ---------------------------------------------------------------------------
class FewShotPromptTest(unittest.TestCase):
    def _sheets(self):
        return [
            fewshot.ExemplarSheet(
                criterion=name,
                path=Path(f"{name}.png"),
                sha256="0" * 64,
                values=tuple(sorted(CRITERIA_BY_NAME[name].vocabulary)),
                exemplar_case_nums=tuple(range(len(CRITERIA_BY_NAME[name].vocabulary))),
                columns=2,
                rows=1,
                image_tokens=100,
            )
            for name in CRITERION_NAMES
        ]

    def test_criterion_definitions_are_byte_identical_to_the_zero_shot_prompt(self):
        base = build_user_prompt(has_clinical=False)
        _, _, body = base.partition("\n\n")
        prompt = fewshot.build_fewshot_user_prompt(self._sheets(), has_clinical=False)
        self.assertTrue(
            prompt.endswith(body),
            "the few-shot prompt must be the zero-shot prompt with a new image inventory, "
            "not a rewritten one -- otherwise F2 varies the definitions as well as the exemplars",
        )

    def test_inventory_counts_every_image_that_will_be_sent(self):
        sheets = self._sheets()
        mono = fewshot.build_fewshot_user_prompt(sheets, has_clinical=False)
        duo = fewshot.build_fewshot_user_prompt(sheets, has_clinical=True)
        self.assertIn(f"You are given {1 + len(sheets)} images", mono)
        self.assertIn(f"You are given {2 + len(sheets)} images", duo)
        self.assertIn(f"IMAGE {1 + len(sheets)} -- reference sheet", mono)
        self.assertIn(f"IMAGE {2 + len(sheets)} -- reference sheet", duo)

    def test_unillustrated_values_are_disclosed(self):
        sheets = self._sheets()
        name = "vascular_structures"
        trimmed = [
            s if s.criterion != name else fewshot.ExemplarSheet(
                s.criterion, s.path, s.sha256, s.values[:2], s.exemplar_case_nums[:2],
                s.columns, s.rows, s.image_tokens,
            )
            for s in sheets
        ]
        prompt = fewshot.build_fewshot_user_prompt(trimmed, has_clinical=False)
        self.assertIn("no reference example available", prompt)
        self.assertIn("still permitted answers", prompt)

    def test_rejects_a_drifted_base_prompt(self):
        original = fewshot.build_user_prompt
        try:
            fewshot.build_user_prompt = lambda **_: "Some other opening line without a break"
            with self.assertRaises(RuntimeError):
                fewshot.build_fewshot_user_prompt(self._sheets())
        finally:
            fewshot.build_user_prompt = original


# ---------------------------------------------------------------------------
# The assessor
# ---------------------------------------------------------------------------
class FewShotAssessorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.derm = _write_image(self.root / "derm.png")
        self.clinic = _write_image(self.root / "clinic.png", colour=(20, 40, 60))
        self.case = FakeCase(
            0, 4242, "valid",
            {name: "absent" for name in CRITERION_NAMES},
            self.derm, self.clinic,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _sheet_files(self, n: int = 3):
        sheets = []
        for index, name in enumerate(CRITERION_NAMES[:n]):
            path = _write_image(self.root / f"sheet_{name}.png", colour=(index * 20, 10, 10))
            sheets.append(fewshot.ExemplarSheet(
                criterion=name, path=path, sha256=f"{index:064d}",
                values=tuple(sorted(CRITERIA_BY_NAME[name].vocabulary)),
                exemplar_case_nums=(1, 2), columns=2, rows=1, image_tokens=100,
            ))
        return sheets

    def test_zero_shot_request_is_identical_to_the_parent(self):
        cache = AssessorCache(self.root / "cache")
        common = dict(model="m", temperature=0.0, max_tokens=900, seed=7, cache=cache)
        parent = CriterionAssessor(client=None, **common).build_request(self.case)
        child = fewshot.FewShotAssessor(client=None, sheets=None, **common).build_request(self.case)
        self.assertEqual(parent.cache_key(), child.cache_key())
        self.assertEqual(parent.identity(), child.identity())

    def test_sheets_change_the_cache_key(self):
        cache = AssessorCache(self.root / "cache")
        common = dict(model="m", temperature=0.0, max_tokens=900, seed=7, cache=cache)
        plain = fewshot.FewShotAssessor(client=None, sheets=None, **common).build_request(self.case)
        shot = fewshot.FewShotAssessor(
            client=None, sheets=self._sheet_files(), **common
        ).build_request(self.case)
        self.assertNotEqual(plain.cache_key(), shot.cache_key())
        self.assertEqual(len(shot.image_paths), 1 + 3)
        self.assertEqual(shot.image_roles[0], "dermoscopic")
        self.assertTrue(all(role.startswith("reference:") for role in shot.image_roles[1:]))

    def test_clinical_view_is_skipped_when_the_file_is_a_copy(self):
        shared = FakeCase(1, 99, "valid", self.case.criteria, self.derm, self.derm)
        cache = AssessorCache(self.root / "cache")
        assessor = fewshot.FewShotAssessor(
            client=None, model="m", cache=cache, include_clinical=True,
            sheets=self._sheet_files(),
        )
        request = assessor.build_request(shared)
        self.assertFalse(request.has_clinical)
        self.assertEqual(request.image_roles[0], "dermoscopic")
        self.assertNotIn("clinical", request.image_roles)
        # And the distinct-view case does get both.
        both = assessor.build_request(self.case)
        self.assertTrue(both.has_clinical)
        self.assertEqual(both.image_roles[:2], ("dermoscopic", "clinical"))

    def test_end_to_end_with_a_fake_client_sends_every_sheet(self):
        cache = AssessorCache(self.root / "cache")
        client = FakeClient()
        sheets = self._sheet_files()
        assessor = fewshot.FewShotAssessor(
            client=client, model="m", cache=cache, sheets=sheets,
        )
        assessment = assessor.assess(self.case)
        self.assertTrue(assessment.ok)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]["images"]), 1 + len(sheets))
        self.assertEqual(client.calls[0]["images"][0], str(self.derm))
        # And a replay is free.
        replay = fewshot.FewShotAssessor(
            client=client, model="m", cache=cache, sheets=sheets,
        ).assess(self.case)
        self.assertTrue(replay.stats.cache_hit)
        self.assertEqual(len(client.calls), 1)

    def test_parallel_results_stay_aligned_with_their_cases(self):
        cache = AssessorCache(self.root / "cache")
        cases = []
        for index in range(9):
            derm = _write_image(self.root / f"p{index}.png", colour=(index * 11, 5, 5))
            cases.append(FakeCase(index, 500 + index,
                                  "valid", {n: "absent" for n in CRITERION_NAMES}, derm, derm))
        assessor = fewshot.FewShotAssessor(
            client=None, model="m", cache=cache, sheets=None,
            workers=4, client_factory=FakeClient,
        )
        results = assessor.assess_cases(cases)
        self.assertEqual([r.case_num for r in results], [c.case_num for c in cases])
        self.assertTrue(all(r.ok for r in results))


# ---------------------------------------------------------------------------
# F1 model profiles
# ---------------------------------------------------------------------------
class ModelProfileTest(unittest.TestCase):
    def test_anthropic_profile_drops_seed_and_gpt5_pins_temperature(self):
        self.assertIsNone(fewshot.profile_for("anthropic/claude-opus-5").seed)
        self.assertEqual(fewshot.profile_for("openai/gpt-5.6-terra").temperature, 1.0)

    def test_baseline_profile_reproduces_g1_decoding(self):
        profile = fewshot.profile_for("qwen/qwen3-vl-32b-instruct")
        self.assertEqual((profile.temperature, profile.seed, profile.max_tokens),
                         (0.0, 20260810, 900))

    def test_reasoning_models_get_headroom_beyond_g1s_900(self):
        for model in ("google/gemini-3.1-pro-preview", "x-ai/grok-4.5", "openai/gpt-5.6-terra"):
            self.assertGreater(fewshot.profile_for(model).max_tokens, 900, model)

    def test_unprobed_model_keeps_g1_defaults_rather_than_guessing(self):
        profile = fewshot.profile_for("some/unseen-model")
        self.assertEqual((profile.temperature, profile.seed), (0.0, 20260810))
        self.assertIsNone(profile.price_in_per_mtok)

    def test_cost_is_none_without_a_rate_card(self):
        self.assertIsNone(fewshot.profile_for("some/unseen-model").cost_usd(1000, 1000))
        self.assertAlmostEqual(
            fewshot.profile_for("qwen/qwen3-vl-32b-instruct").cost_usd(1_000_000, 1_000_000),
            0.104 + 0.416,
        )


# ---------------------------------------------------------------------------
# The cohort
# ---------------------------------------------------------------------------
class SubsetSelectionTest(unittest.TestCase):
    def _valid(self, n: int = 203) -> List[FakeCase]:
        groups = ["NEV"] * 100 + ["MEL"] * 61 + ["MISC"] * 25 + ["SK"] * 10 + ["BCC"] * 7
        difficulties = ["low", "medium", "high"]
        out = []
        for index in range(n):
            out.append(FakeCase(
                index, 1000 + index, "valid",
                {name: "absent" for name in CRITERION_NAMES},
                Path(f"d{index}.png"), Path(f"c{index}.png"),
                diagnosis_group=groups[index], difficulty=difficulties[index % 3],
            ))
        return out

    def test_selection_is_deterministic_and_exact_in_size(self):
        cases = self._valid()
        first, record = g1c.select_subset(cases, 72, 20260810)
        second, _ = g1c.select_subset(cases, 72, 20260810)
        self.assertEqual(len(first), 72)
        self.assertEqual([c.case_num for c in first], [c.case_num for c in second])
        self.assertEqual(record["n_selected"], 72)

    def test_a_different_seed_gives_a_different_draw(self):
        cases = self._valid()
        a, _ = g1c.select_subset(cases, 72, 1)
        b, _ = g1c.select_subset(cases, 72, 2)
        self.assertNotEqual([c.case_num for c in a], [c.case_num for c in b])

    def test_strata_are_proportional_within_one_case(self):
        cases = self._valid()
        subset, _ = g1c.select_subset(cases, 72, 20260810)
        for group in {"NEV", "MEL", "MISC", "SK", "BCC"}:
            in_split = sum(1 for c in cases if c.diagnosis_group == group)
            in_subset = sum(1 for c in subset if c.diagnosis_group == group)
            expected = in_split * 72 / len(cases)
            self.assertLessEqual(abs(in_subset - expected), 2.0, group)

    def test_full_split_request_returns_every_case(self):
        cases = self._valid()
        subset, _ = g1c.select_subset(cases, len(cases), 20260810)
        self.assertEqual(len(subset), len(cases))
        self.assertEqual({c.case_num for c in subset}, {c.case_num for c in cases})


class ConfigParsingTest(unittest.TestCase):
    def test_flags_are_parsed_and_validated(self):
        configs = g1c.parse_configs("a:m1,b:m2:fewshot,c:m3:fewshot:clinical")
        self.assertEqual([c.name for c in configs], ["a", "b", "c"])
        self.assertEqual([c.few_shot for c in configs], [False, True, True])
        self.assertEqual([c.clinical for c in configs], [False, False, True])
        with self.assertRaises(SystemExit):
            g1c.parse_configs("a:m1:nonsense")
        with self.assertRaises(SystemExit):
            g1c.parse_configs("justaname")

    def test_default_ladder_covers_the_three_factors(self):
        configs = g1c.parse_configs(g1c.DEFAULT_LADDER)
        self.assertTrue(any(c.model == g1c.BASELINE_MODEL and not c.few_shot for c in configs))
        self.assertTrue(any(c.few_shot for c in configs))
        self.assertGreater(len({c.model for c in configs}), 1)


class IndeterminateAccountingTest(unittest.TestCase):
    """G1's two secondary failures are what G1c must be able to see move."""

    def test_indeterminate_is_a_permitted_value_the_gate_counts(self):
        block = g1._criterion_block(
            "streaks",
            gold=["absent", "irregular", "regular", "absent"],
            pred=["absent", INDETERMINATE, "regular", "absent"],
            train_majority="absent",
            eval_majority="absent",
        )
        self.assertEqual(block["n_indeterminate"], 1)
        self.assertEqual(block["indeterminate_rate"], 0.25)
        self.assertEqual(block["determinate_only"]["accuracy"], 1.0)
        self.assertEqual(block["strict_indeterminate_counts_as_wrong"]["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
