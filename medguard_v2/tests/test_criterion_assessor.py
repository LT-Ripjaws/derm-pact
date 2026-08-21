"""G1 assessor: parsing, the repair retry, non-coercion, abstention, and the cache.

No network. Every test drives a scripted fake client, so the number of remote calls is an
assertable fact rather than a hope. The properties under test are the ones that decide whether
G1's numbers mean anything:

* a schema violation costs exactly one repair attempt, not zero and not an unbounded loop;
* a persistent violation produces ``values=None`` -- never ``absent``, which would fabricate a
  confident 7-point score of 0 out of a parsing bug;
* ``indeterminate`` survives all the way into a score *range*;
* a cache hit is free, and the cache key moves when anything that could change the answer moves.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import (
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    CRITERION_SPECS,
    INDETERMINATE,
)
from medguard_v2.perception.criterion_assessor import (
    VALUE_DEFINITIONS,
    AssessorCache,
    CriterionAssessor,
    build_repair_prompt,
    build_user_prompt,
    parse_response,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClient:
    """Returns scripted completions and records every call. Never touches the network."""

    def __init__(self, responses: Sequence[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.call_log: List[Dict[str, Any]] = []
        self.last_call_metadata: Dict[str, Any] = {}

    def vision(self, prompt: str, image_path=None, *, system: str = "", model=None,
               max_tokens: int = 512, temperature: float = 0.2, seed=None) -> str:
        self.calls.append({
            "prompt": prompt, "image_path": list(image_path or []), "system": system,
            "model": model, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        })
        if not self._responses:
            raise AssertionError("fake client ran out of scripted responses")
        response = self._responses.pop(0)
        metadata = {
            "model": model, "attempt": 1, "succeeded": True, "latency_seconds": 0.125,
            "prompt_tokens": 1500, "completion_tokens": 250, "total_tokens": 1750,
            "usage_available": True, "seed": seed,
        }
        self.call_log.append(metadata)
        self.last_call_metadata = dict(metadata)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def n_calls(self) -> int:
        return len(self.calls)


@dataclass
class FakeCase:
    case_index: int
    case_num: int
    derm_path: Path
    clinic_path: Path
    criteria: Mapping[str, str]
    seven_point_score: int = 0
    split: str = "train"
    diagnosis: str = "melanoma"
    diagnosis_group: str = "MEL"

    @property
    def has_distinct_views(self) -> bool:
        return self.clinic_path != self.derm_path


GOLD = {
    "pigment_network": "atypical",
    "blue_whitish_veil": "present",
    "vascular_structures": "dotted",
    "streaks": "irregular",
    "pigmentation": "diffuse irregular",
    "dots_and_globules": "irregular",
    "regression_structures": "absent",
}


def response_json(values: Mapping[str, str], *, confidence: float = 0.8) -> str:
    return json.dumps({
        name: {"value": values[name], "confidence": confidence,
               "justification": f"visual note for {name}"}
        for name in CRITERION_NAMES
    })


class AssessorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.derm = root / "derm.jpg"
        self.clinic = root / "clinic.jpg"
        self.derm.write_bytes(b"\xff\xd8\xff\xe0derm-image-bytes")
        self.clinic.write_bytes(b"\xff\xd8\xff\xe0clinic-image-bytes")
        self.cache_dir = root / "cache"
        self.case = FakeCase(
            case_index=7, case_num=42, derm_path=self.derm, clinic_path=self.clinic,
            criteria=dict(GOLD), seven_point_score=9,
        )
        self.addCleanup(self._tmp.cleanup)

    def make_assessor(self, client: Any, **kwargs: Any) -> CriterionAssessor:
        options: Dict[str, Any] = {
            "cache": AssessorCache(self.cache_dir),
            "model": "test/model",
        }
        options.update(kwargs)
        return CriterionAssessor(client, **options)


# ---------------------------------------------------------------------------
# Prompt integrity
# ---------------------------------------------------------------------------
class PromptTests(unittest.TestCase):
    def test_every_permitted_value_appears_verbatim_and_is_defined(self) -> None:
        prompt = build_user_prompt()
        for spec in CRITERION_SPECS:
            self.assertIn(f'"{spec.name}"', prompt)
            for value in spec.vocabulary:
                self.assertIn(f'"{value}"', prompt, f"{spec.name}: {value!r} missing from prompt")
                self.assertIn(
                    VALUE_DEFINITIONS[spec.name][value], prompt,
                    f"{spec.name}: {value!r} offered without a definition",
                )

    def test_prompt_never_reveals_weights_or_threshold(self) -> None:
        """The model must not be able to reverse-engineer the host's arithmetic."""
        prompt = (build_user_prompt() + build_repair_prompt(previous="x", error="y")).lower()
        for forbidden in ("2 point", "1 point", "major criterion", "minor criterion",
                          "seven-point score", "7-point score", "threshold", "melanoma"):
            self.assertNotIn(forbidden, prompt, f"prompt leaks {forbidden!r}")

    def test_prompt_offers_indeterminate(self) -> None:
        self.assertIn(INDETERMINATE, build_user_prompt())

    def test_value_definitions_cover_the_schema_exactly(self) -> None:
        self.assertEqual(set(VALUE_DEFINITIONS), {spec.name for spec in CRITERION_SPECS})
        for spec in CRITERION_SPECS:
            self.assertEqual(set(VALUE_DEFINITIONS[spec.name]), set(spec.vocabulary))

    def test_prompt_is_deterministic_across_calls(self) -> None:
        # The prompt is part of the cache key; frozenset iteration order must not leak into it.
        self.assertEqual(build_user_prompt(), build_user_prompt())


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class ParseTests(unittest.TestCase):
    def test_valid_response_parses(self) -> None:
        parsed = parse_response(response_json(GOLD))
        self.assertEqual(parsed.values, dict(GOLD))
        self.assertEqual(set(parsed.confidences), set(CRITERION_NAMES))
        self.assertEqual(parsed.soft_defects, [])

    def test_fenced_and_prefixed_json_is_recovered(self) -> None:
        text = "Here is my assessment:\n```json\n" + response_json(GOLD) + "\n```\nDone."
        self.assertEqual(parse_response(text).values, dict(GOLD))

    def test_surface_variants_are_normalised_and_recorded(self) -> None:
        noisy = dict(GOLD)
        noisy["vascular_structures"] = "Linear-Irregular."
        noisy["pigmentation"] = "  DIFFUSE_IRREGULAR "
        parsed = parse_response(response_json(noisy))
        self.assertEqual(parsed.values["vascular_structures"], "linear irregular")
        self.assertEqual(parsed.values["pigmentation"], "diffuse irregular")
        self.assertIn("vascular_structures", parsed.normalisations)
        self.assertIn("pigmentation", parsed.normalisations)

    def test_normalisation_never_crosses_between_vocabulary_values(self) -> None:
        """The dangerous rewrites: regular -> irregular, typical -> atypical."""
        for name, value, expected in (
            ("pigment_network", "typical", "typical"),
            ("streaks", "regular", "regular"),
            ("dots_and_globules", "Regular", "regular"),
        ):
            values = dict(GOLD)
            values[name] = value
            self.assertEqual(parse_response(response_json(values)).values[name], expected)

    def test_out_of_vocabulary_value_raises(self) -> None:
        values = dict(GOLD)
        values["pigment_network"] = "reticular"
        with self.assertRaises(ValueError) as ctx:
            parse_response(response_json(values))
        self.assertIn("pigment_network", str(ctx.exception))

    def test_missing_and_unknown_criteria_raise(self) -> None:
        payload = {name: {"value": GOLD[name], "confidence": 0.5, "justification": "x"}
                   for name in CRITERION_NAMES}
        partial = {k: v for k, v in payload.items() if k != "streaks"}
        with self.assertRaises(ValueError):
            parse_response(json.dumps(partial))
        extra = dict(payload, asymmetry={"value": "present", "confidence": 1.0})
        with self.assertRaises(ValueError):
            parse_response(json.dumps(extra))

    def test_unparseable_text_raises_valueerror(self) -> None:
        for text in ("", "no json here", "{unbalanced", "[1,2,3]"):
            with self.assertRaises(ValueError):
                parse_response(text)

    def test_confidence_defects_are_recorded_not_fatal(self) -> None:
        payload = {name: {"value": GOLD[name], "confidence": "high", "justification": ""}
                   for name in CRITERION_NAMES}
        parsed = parse_response(json.dumps(payload))
        self.assertEqual(parsed.values, dict(GOLD))
        self.assertIsNone(parsed.confidences["streaks"])
        self.assertTrue(any("confidence" in defect for defect in parsed.soft_defects))
        self.assertTrue(any("empty justification" in defect for defect in parsed.soft_defects))

    def test_indeterminate_is_accepted(self) -> None:
        values = dict(GOLD)
        values["blue_whitish_veil"] = INDETERMINATE
        self.assertEqual(
            parse_response(response_json(values)).values["blue_whitish_veil"], INDETERMINATE
        )


# ---------------------------------------------------------------------------
# The assessor: retry, non-coercion, abstention, cache
# ---------------------------------------------------------------------------
class AssessorTests(AssessorTestBase):
    def test_valid_response_costs_one_call(self) -> None:
        client = FakeClient([response_json(GOLD)])
        assessment = self.make_assessor(client).assess(self.case)
        self.assertEqual(assessment.status, "ok")
        self.assertEqual(assessment.values, dict(GOLD))
        self.assertEqual(client.n_calls, 1)
        self.assertEqual(assessment.stats.remote_calls, 1)
        self.assertEqual(assessment.stats.repair_attempts, 0)
        self.assertFalse(assessment.stats.cache_hit)
        self.assertEqual(assessment.stats.prompt_tokens, 1500)
        self.assertEqual(assessment.stats.completion_tokens, 250)
        self.assertGreater(assessment.stats.latency_seconds, 0.0)
        self.assertEqual(assessment.case_num, 42)

    def test_only_the_dermoscopic_image_is_sent_by_default(self) -> None:
        client = FakeClient([response_json(GOLD)])
        self.make_assessor(client).assess(self.case)
        self.assertEqual(client.calls[0]["image_path"], [str(self.derm)])

    def test_clinical_view_is_added_only_when_it_is_a_distinct_file(self) -> None:
        client = FakeClient([response_json(GOLD)])
        self.make_assessor(client, include_clinical=True).assess(self.case)
        self.assertEqual(client.calls[0]["image_path"], [str(self.derm), str(self.clinic)])

        shared = FakeCase(case_index=8, case_num=43, derm_path=self.derm, clinic_path=self.derm,
                          criteria=dict(GOLD))
        client2 = FakeClient([response_json(GOLD)])
        self.make_assessor(client2, include_clinical=True).assess(shared)
        self.assertEqual(client2.calls[0]["image_path"], [str(self.derm)])

    def test_out_of_vocabulary_triggers_exactly_one_repair_retry(self) -> None:
        bad = dict(GOLD)
        bad["pigment_network"] = "reticular"
        client = FakeClient([response_json(bad), response_json(GOLD)])
        assessment = self.make_assessor(client).assess(self.case)

        self.assertEqual(assessment.status, "ok")
        self.assertEqual(assessment.values, dict(GOLD))
        self.assertEqual(client.n_calls, 2)
        self.assertEqual(assessment.stats.repair_attempts, 1)
        self.assertEqual(assessment.stats.remote_calls, 2)
        # The repair call quotes the rejected response and the validator's own error back.
        repair_prompt = client.calls[1]["prompt"]
        self.assertIn("REJECTED BY THE SCHEMA VALIDATOR", repair_prompt)
        self.assertIn("reticular", repair_prompt)
        self.assertEqual(assessment.attempts[0]["kind"], "initial")
        self.assertEqual(assessment.attempts[1]["kind"], "repair")

    def test_persistent_violation_is_a_parse_failure_and_is_never_coerced(self) -> None:
        bad = dict(GOLD)
        bad["streaks"] = "radial-streaming"
        client = FakeClient([response_json(bad), response_json(bad)])
        assessment = self.make_assessor(client).assess(self.case)

        self.assertEqual(assessment.status, "parse_failure")
        self.assertIsNone(assessment.values)          # the whole point: no fabricated values
        self.assertFalse(assessment.ok)
        self.assertIn("streaks", assessment.failure_reason or "")
        self.assertEqual(client.n_calls, 2)           # 1 initial + 1 repair, then stop
        self.assertEqual(assessment.stats.repair_attempts, 1)
        # And nothing downstream can turn it into a score of 0.
        with self.assertRaises(TypeError):
            score_criteria(assessment.values)         # type: ignore[arg-type]

    def test_transport_failure_is_an_api_failure_and_is_not_cached(self) -> None:
        client = FakeClient([RuntimeError("gateway 502")])
        assessor = self.make_assessor(client)
        assessment = assessor.assess(self.case)
        self.assertEqual(assessment.status, "api_failure")
        self.assertIsNone(assessment.values)
        self.assertIn("502", assessment.failure_reason or "")
        self.assertEqual(assessment.stats.remote_calls, 1)
        self.assertEqual(assessor.cache.count(), 0)

        # A later run must be free to try again rather than replay the outage.
        retry_client = FakeClient([response_json(GOLD)])
        retried = self.make_assessor(retry_client).assess(self.case)
        self.assertEqual(retried.status, "ok")
        self.assertEqual(retry_client.n_calls, 1)

    def test_indeterminate_propagates_into_a_score_range(self) -> None:
        values = dict(GOLD)
        values["blue_whitish_veil"] = INDETERMINATE      # a 2-point criterion
        values["streaks"] = INDETERMINATE                # a 1-point criterion
        client = FakeClient([response_json(values)])
        assessment = self.make_assessor(client).assess(self.case)

        self.assertEqual(assessment.status, "ok")
        self.assertEqual(
            set(assessment.indeterminate_criteria), {"blue_whitish_veil", "streaks"}
        )
        scored = score_criteria(assessment.values)
        self.assertFalse(scored.determinate)
        self.assertEqual(scored.score_max - scored.score_min, 3)   # 2 + 1 unresolved points
        self.assertLess(scored.score_min, scored.score_max)
        with self.assertRaises(ValueError):
            _ = scored.score                                       # no point estimate exists

    def test_indeterminate_can_still_be_decidable(self) -> None:
        """Abstention only blocks a verdict when it could change it."""
        values = {name: "absent" for name in CRITERION_NAMES}
        values["pigment_network"] = "atypical"       # 2 points
        values["streaks"] = "irregular"              # 1 point -> already at the threshold
        values["regression_structures"] = INDETERMINATE
        client = FakeClient([response_json(values)])
        assessment = self.make_assessor(client).assess(self.case)
        scored = score_criteria(assessment.values)
        self.assertTrue(scored.indeterminate)
        self.assertTrue(scored.decidable)
        self.assertTrue(scored.meets_threshold)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class CacheTests(AssessorTestBase):
    def test_cache_hit_returns_without_a_second_call(self) -> None:
        first_client = FakeClient([response_json(GOLD)])
        first = self.make_assessor(first_client).assess(self.case)
        self.assertEqual(first_client.n_calls, 1)
        self.assertFalse(first.stats.cache_hit)

        second_client = FakeClient([])          # any call at all raises inside the fake
        second = self.make_assessor(second_client).assess(self.case)
        self.assertEqual(second_client.n_calls, 0)
        self.assertTrue(second.stats.cache_hit)
        self.assertEqual(second.status, "ok")
        self.assertEqual(second.values, first.values)
        self.assertEqual(second.confidences, first.confidences)

    def test_cache_hit_carries_the_asking_case_identity_not_the_stored_one(self) -> None:
        self.make_assessor(FakeClient([response_json(GOLD)])).assess(self.case)
        twin = FakeCase(case_index=99, case_num=1234, derm_path=self.derm,
                        clinic_path=self.clinic, criteria=dict(GOLD))
        hit = self.make_assessor(FakeClient([])).assess(twin)
        self.assertTrue(hit.stats.cache_hit)
        self.assertEqual(hit.case_num, 1234)
        self.assertEqual(hit.case_index, 99)

    def test_parse_failures_are_cached_so_a_rerun_is_identical(self) -> None:
        bad = dict(GOLD)
        bad["streaks"] = "radial-streaming"
        first = self.make_assessor(FakeClient([response_json(bad)] * 2)).assess(self.case)
        self.assertEqual(first.status, "parse_failure")

        replay_client = FakeClient([])
        replay = self.make_assessor(replay_client).assess(self.case)
        self.assertEqual(replay_client.n_calls, 0)
        self.assertEqual(replay.status, "parse_failure")
        self.assertIsNone(replay.values)
        self.assertTrue(replay.stats.cache_hit)

    def test_refresh_ignores_the_cache_and_rewrites_it(self) -> None:
        self.make_assessor(FakeClient([response_json(GOLD)])).assess(self.case)
        changed = dict(GOLD)
        changed["pigment_network"] = "typical"
        client = FakeClient([response_json(changed)])
        refreshed = self.make_assessor(client, refresh=True).assess(self.case)
        self.assertEqual(client.n_calls, 1)
        self.assertEqual(refreshed.values["pigment_network"], "typical")
        replay = self.make_assessor(FakeClient([])).assess(self.case)
        self.assertEqual(replay.values["pigment_network"], "typical")

    def test_cache_only_mode_refuses_to_pay(self) -> None:
        client = FakeClient([response_json(GOLD)])
        assessor = self.make_assessor(client, allow_remote=False)
        with self.assertRaises(RuntimeError):
            assessor.assess(self.case)
        self.assertEqual(client.n_calls, 0)

    def test_cache_key_moves_with_everything_that_can_change_the_answer(self) -> None:
        base = self.make_assessor(None)
        key = base.build_request(self.case).cache_key()

        variants = {
            "model": self.make_assessor(None, model="other/model"),
            "temperature": self.make_assessor(None, temperature=0.7),
            "max_tokens": self.make_assessor(None, max_tokens=1200),
            "seed": self.make_assessor(None, seed=1),
            "clinical_view": self.make_assessor(None, include_clinical=True),
            "repair_budget": self.make_assessor(None, max_repair_attempts=3),
        }
        for label, assessor in variants.items():
            self.assertNotEqual(
                key, assessor.build_request(self.case).cache_key(),
                f"cache key ignores {label}",
            )

        # ...and with the image bytes, but not with the case number that asked.
        other_image = self.derm.parent / "other.jpg"
        other_image.write_bytes(b"\xff\xd8\xff\xe0different-bytes")
        moved = FakeCase(case_index=1, case_num=2, derm_path=other_image,
                         clinic_path=self.clinic, criteria=dict(GOLD))
        self.assertNotEqual(key, base.build_request(moved).cache_key())
        twin = FakeCase(case_index=500, case_num=600, derm_path=self.derm,
                        clinic_path=self.clinic, criteria=dict(GOLD))
        self.assertEqual(key, base.build_request(twin).cache_key())

    def test_corrupt_cache_entry_raises_rather_than_silently_re_paying(self) -> None:
        assessor = self.make_assessor(FakeClient([response_json(GOLD)]))
        assessment = assessor.assess(self.case)
        path = assessor.cache.entry_path(assessment.cache_key)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["assessment"]["values"]["pigment_network"] = "reticular"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.make_assessor(FakeClient([])).assess(self.case)

    def test_batch_reports_per_case_cache_hits(self) -> None:
        cases = [self.case,
                 FakeCase(case_index=8, case_num=43, derm_path=self.derm,
                          clinic_path=self.clinic, criteria=dict(GOLD))]
        client = FakeClient([response_json(GOLD)])
        results = self.make_assessor(client).assess_cases(cases)
        self.assertEqual(client.n_calls, 1)          # identical images -> one paid call
        self.assertEqual([r.stats.cache_hit for r in results], [False, True])
        self.assertEqual([r.case_num for r in results], [42, 43])


# ---------------------------------------------------------------------------
# Gate scorer wiring (offline)
# ---------------------------------------------------------------------------
class GateScorerTests(AssessorTestBase):
    def _cases(self) -> List[FakeCase]:
        benign = {name: "absent" for name in CRITERION_NAMES}
        out = []
        for index in range(4):
            image = self.derm.parent / f"case{index}.jpg"
            image.write_bytes(b"\xff\xd8\xff\xe0" + bytes([index]) * 32)
            criteria = dict(GOLD) if index % 2 else dict(benign)
            out.append(FakeCase(
                case_index=index, case_num=100 + index, derm_path=image, clinic_path=image,
                criteria=criteria,
                seven_point_score=score_criteria(criteria).score_min,
            ))
        return out

    def test_live_report_shape_and_chance_baseline(self) -> None:
        from medguard_v2.eval.gates import g1_criterion_readability as gate

        cases = self._cases()
        client = FakeClient([response_json(case.criteria) for case in cases])
        assessor = self.make_assessor(client)
        report = gate.run_live(cases, assessor, cases, verbose=False)

        self.assertEqual(report["reliability"]["n_scored"], 4)
        self.assertEqual(report["reliability"]["parse_failures"], 0)
        self.assertEqual(report["macro_across_seven"]["accuracy_strict"], 1.0)
        self.assertEqual(report["seven_point_score_agreement"]["fully_determinate"]["mae"], 0.0)
        # A perfect assessor must still be scored against a stated constant floor.
        baseline = report["seven_point_score_agreement_chance_baseline"]["fully_determinate"]
        self.assertGreater(baseline["mae"], 0.0)
        # A constant predictor scores kappa 0 wherever the gold actually varies. Where the gold
        # column is itself constant the constant is trivially perfect; that must be flagged, not
        # silently counted as the assessor losing to chance.
        for name in CRITERION_NAMES:
            block = report["per_criterion"][name]
            chance = block["chance_baseline"]["train_majority"]
            if block["gold_is_constant"]:
                self.assertEqual(chance["accuracy"], 1.0)
            else:
                self.assertAlmostEqual(chance["cohen_kappa"], 0.0, places=6)
                self.assertLess(chance["accuracy"], 1.0)
        self.assertGreater(report["macro_across_seven"]["lift_over_chance_strict"], 0.0)

    def test_paid_and_cached_usage_are_reported_separately(self) -> None:
        from medguard_v2.eval.gates import g1_criterion_readability as gate

        cases = self._cases()
        assessor = self.make_assessor(FakeClient([response_json(c.criteria) for c in cases]))
        gate.run_live(cases, assessor, cases, verbose=False)

        replay = self.make_assessor(FakeClient([]))
        report = gate.run_live(cases, replay, cases, verbose=False)
        self.assertEqual(report["usage"]["paid_this_run"]["cases"], 0)
        self.assertEqual(report["usage"]["paid_this_run"]["prompt_tokens"], 0)
        self.assertEqual(report["usage"]["replayed_from_cache"]["cases"], 4)
        self.assertGreater(report["usage"]["replayed_from_cache"]["prompt_tokens"], 0)
        self.assertEqual(report["usage"]["cache_hit_rate"], 1.0)

    def test_parse_failures_are_excluded_from_accuracy_and_listed(self) -> None:
        from medguard_v2.eval.gates import g1_criterion_readability as gate

        cases = self._cases()
        bad = dict(GOLD)
        bad["streaks"] = "radial-streaming"
        responses: List[str] = []
        for index, case in enumerate(cases):
            if index == 0:
                responses += [response_json(bad), response_json(bad)]
            else:
                responses.append(response_json(case.criteria))
        assessor = self.make_assessor(FakeClient(responses))
        report = gate.run_live(cases, assessor, cases, verbose=False)

        self.assertEqual(report["reliability"]["parse_failures"], 1)
        self.assertEqual(report["reliability"]["n_scored"], 3)
        self.assertEqual(report["reliability"]["failed_case_nums"], [100])
        self.assertEqual(report["per_criterion"]["streaks"]["n_assessed"], 3)
        failed = [entry for entry in report["per_case"] if entry["case_num"] == 100][0]
        self.assertIsNone(failed["criteria"])
        self.assertEqual(failed["status"], "parse_failure")

    def test_report_survives_a_cohort_where_nothing_was_scored(self) -> None:
        """A total gateway outage must produce a readable report, not a traceback."""
        from medguard_v2.eval.gates import g1_criterion_readability as gate

        cases = self._cases()
        client = FakeClient([RuntimeError("gateway down")] * len(cases))
        report = gate.run_live(cases, self.make_assessor(client), cases, verbose=False)
        self.assertEqual(report["reliability"]["n_scored"], 0)
        self.assertEqual(report["reliability"]["api_failures"], len(cases))
        self.assertEqual(report["per_criterion"], {})
        self.assertIsNone(report["seven_point_score_agreement"])
        self.assertIsNone(report["macro_across_seven"]["accuracy_determinate"])
        self.assertIn("G1 LIVE", gate.summarise(report))

    def test_dry_run_makes_no_calls_and_projects_cost(self) -> None:
        from medguard_v2.eval.gates import g1_criterion_readability as gate

        cases = self._cases()
        client = FakeClient([])
        report = gate.run_dry(cases, self.make_assessor(client), price_in=0.1, price_out=0.4)
        self.assertEqual(client.n_calls, 0)
        self.assertEqual(report["boundaries"]["api_calls"], 0)
        self.assertGreater(report["per_case_tokens"]["prompt_tokens_mean"], 0)
        self.assertEqual(report["cache"]["requests_needing_a_paid_call"], len(cases))
        projection = report["projection"]["valid_split_203_cases"]
        self.assertGreater(projection["input_tokens"], 0)
        self.assertIn("cost_usd_at_supplied_rate", projection)
        self.assertIn("G1 DRY RUN", gate.summarise(report))

    def test_selection_is_deterministic_and_declares_representativeness(self) -> None:
        from medguard_v2.eval.gates import g1_criterion_readability as gate

        cases = self._cases()
        head, head_record = gate.select_cases(cases, 2, "head")
        even, even_record = gate.select_cases(cases, 2, "even")
        spread, spread_record = gate.select_cases(cases, 2, "score-spread")

        self.assertEqual([c.case_num for c in head], [100, 101])
        self.assertEqual([c.case_num for c in even], [100, 103])
        self.assertEqual(gate.select_cases(cases, 2, "even")[0], even)
        self.assertTrue(even_record["representative_of_split"])
        self.assertFalse(head_record["representative_of_split"])
        self.assertFalse(spread_record["representative_of_split"])
        self.assertEqual(len(spread), 2)
        self.assertLessEqual(spread[0].seven_point_score, spread[-1].seven_point_score)


class MajorityBaselineTests(unittest.TestCase):
    def test_majority_is_reproducible_under_ties(self) -> None:
        from medguard_v2.eval.gates.g1_criterion_readability import majority_by_criterion

        @dataclass
        class Tiny:
            criteria: Mapping[str, str]

        a = Tiny({name: sorted(CRITERIA_BY_NAME[name].vocabulary)[0] for name in CRITERION_NAMES})
        b = Tiny({name: sorted(CRITERIA_BY_NAME[name].vocabulary)[-1] for name in CRITERION_NAMES})
        first = majority_by_criterion([a, b])
        second = majority_by_criterion([b, a])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
