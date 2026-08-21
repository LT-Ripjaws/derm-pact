"""Two ablation tables: perception (assembled) and system-level (computed).

Table A reads three already-written evidence artifacts and assembles the four melanoma-
detection arms (classifier alone / criteria path alone / union rule / stacked ensemble) into
one comparable table. It computes nothing new -- every number already exists on disk; this
only extracts and lines them up, plus recomputes Youden's J from sensitivity/specificity by
the formula the caller asked for, so all four arms use one consistent definition instead of
whatever each source script happened to store.

Table B runs `medguard_v2.pipeline.run_cohort.run_cohort("valid")` and four variants of it,
to measure what each pipeline component is actually contributing. Two of the five arms turned
out to be more interesting than "toggle a flag and diff the numbers":

**"No router" is not an ablation of the shipped system -- it already IS the shipped system.**
`medguard_v2/evidence/router.py` implements gated retrieval (four states: NO_COVERAGE /
CONFLICT / UNDECIDABLE / ROUTINE, each with its own retrieval action), but
`medguard_v2/pipeline/run_cohort.py` never imports it. `build_contract()` calls
`catalog.bind(classifier_diagnosis)` unconditionally on every case instead. A repo-wide grep
for `RetrievalState` / `classify_state` / `evidence.router` turns up exactly one file: the
module's own source. So arm 1 ("full system", `run_cohort` as-is) and arm 2 ("no router",
always retrieve) are the same pipeline. This script reports the real numbers for both sides of
that ablation anyway, by running the actual `router.route()` against the same contracts arm 1
built, so the comparison is informative rather than a diff against itself.

**Same shape of finding for "no abstention".** `medguard_v2/perception/criterion_abstention.py`
implements margin-based abstention with a policy already fit on Derm7pt train
(`research/evidence/derm7pt_v1/criterion_abstention_v1.json`) -- but `run_cohort.py`'s criterion
heads are plain `LogisticRegression.predict()`, which can only return a class it was trained on,
and INDETERMINATE is never a training label. Its own module docstring says as much: "the
observed undecidable rate is 0.0000." Confirmed empirically below. `criterion_abstention.py` is
used exactly once in the whole package, by `medguard_v2/eval/agent_eval.py`, not by
`run_cohort.py`. So arm 5 as literally named is arm 1 again; this script applies the real fitted
policy to get the actual counterfactual and measures its effect against Derm7pt's gold
`seven_point_score`, which is available ground truth, not a proxy.

Neither finding required speculation -- both come from reading the two "missing" modules and
grepping their call sites.

Nothing here modifies `run_cohort.py`, `contract.py`, `router.py` or `criterion_abstention.py`.
Every arm below is a new function that calls those existing pieces differently.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover -- environment guard, not a code path under test
    raise SystemExit(
        "numpy/scikit-learn are not on this interpreter. Run this script with the project\n"
        "virtualenv, e.g. from the repo root:\n"
        r'  "venv_win\Scripts\python.exe" scripts\dev\build_ablation_table.py' "\n"
        f"(import failed with: {exc})"
    ) from exc

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import (
    CRITERION_NAMES,
    INDETERMINATE,
    MELANOMA_SUSPICION_THRESHOLD,
)
from medguard_v2.data.derm7pt import Derm7ptCase, load_cases
from medguard_v2.evidence.contract import DecisionContract, build_contract
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.evidence.router import route
from medguard_v2.perception.criterion_abstention import load_policy
from medguard_v2.pipeline.run_cohort import BASE_C, SEED, _load_features, fit_readers, run_cohort

EVIDENCE_DIR = REPO / "research" / "evidence" / "derm7pt_v1"
OUT_PATH = EVIDENCE_DIR / "ablation_table.json"


# =====================================================================================
# shared helpers
# =====================================================================================


def _round4(x: Optional[float]) -> Optional[float]:
    return round(float(x), 4) if x is not None else None


def _youden(sens: Optional[float], spec: Optional[float]) -> Optional[float]:
    if sens is None or spec is None:
        return None
    return _round4(sens + spec - 1)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def _json_safe(obj):
    """Recursively cast numpy scalars to native Python types before json.dumps.

    Cheaper and more correct than ``default=str`` on json.dumps, which would silently turn a
    numpy float into a *string* rather than a number wherever one slipped through.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _decision_summary(contracts: Sequence[DecisionContract]) -> Dict[str, object]:
    """The four rate fields every Table B arm reports, computed the same way each time."""
    n = len(contracts)
    return {
        "n_cases": n,
        "contracts_safe_to_display": sum(1 for c in contracts if c.safe_to_display),
        "undecidable_rate": _round4(sum(1 for c in contracts if not c.score.decidable) / n) if n else None,
        "conflict_rate": _round4(sum(1 for c in contracts if c.conflict.present) / n) if n else None,
        "escalation_rate": _round4(sum(1 for c in contracts if c.conflict.escalate) / n) if n else None,
    }


def _router_summary(contracts: Sequence[DecisionContract], catalog: GuidelineCatalog) -> Dict[str, object]:
    """Run the real, unwired `router.route()` against a set of contracts.

    ``retrieval_calls_as_shipped_ungated`` mirrors exactly what ``build_contract`` actually does
    today: one ``catalog.bind(classifier_diagnosis)`` call per case, plus one more
    ``catalog.bind("MEL")`` call for cases where the checklist flags melanoma against a
    non-MEL committed diagnosis (the contingent-guidance path).
    """
    decisions = [
        route(
            classifier_diagnosis=c.classifier_diagnosis,
            classifier_confidence=c.classifier_confidence,
            score=c.score,
            catalog=catalog,
        )
        for c in contracts
    ]
    states = Counter(d.state.value for d in decisions)
    gated_calls = sum(d.retrieval_calls for d in decisions)
    ungated_calls = sum(
        1 + (1 if (c.conflict.present and c.conflict.criteria_path_flags_melanoma
                   and c.classifier_diagnosis != "MEL") else 0)
        for c in contracts
    )
    return {
        "router_state_distribution": dict(states),
        "retrieval_calls_if_gated_by_router": gated_calls,
        "retrieval_calls_as_shipped_ungated": ungated_calls,
        "extra_calls_from_shipping_without_the_router": ungated_calls - gated_calls,
    }


def _build_contracts(
    cohort: Sequence[Derm7ptCase],
    *,
    diagnosis,
    posterior: np.ndarray,
    predicted: np.ndarray,
    criteria_by_case: Sequence[Dict[str, str]],
    catalog: GuidelineCatalog,
    receipt_extra: Dict[str, object],
) -> List[DecisionContract]:
    return [
        build_contract(
            case_id=f"derm7pt-{case.case_num}",
            classifier_diagnosis=str(predicted[i]),
            classifier_confidence=float(posterior[i].max()),
            criteria=dict(criteria_by_case[i]),
            catalog=catalog,
            receipt={"split": "valid", "fit_split": "train", **receipt_extra},
        )
        for i, case in enumerate(cohort)
    ]


# =====================================================================================
# Table A -- perception ablation, assembled from existing artifacts (nothing recomputed
# except Youden's J, by the formula given: sens + spec - 1)
# =====================================================================================


def build_table_a() -> Dict[str, object]:
    test = json.loads((EVIDENCE_DIR / "confirmatory_test_395.json").read_text(encoding="utf-8"))
    stacked_doc = json.loads((EVIDENCE_DIR / "confirmatory_p1_completion.json").read_text(encoding="utf-8"))
    g6 = json.loads((EVIDENCE_DIR / "g6_multiview_stacking.json").read_text(encoding="utf-8"))

    p1 = test["P1_melanoma_detection"]
    decision = test["decision_level"]
    ref = g6["reference_points"]
    verdict = g6["verdict"]
    stacked_cmp = stacked_doc["P1_registered_comparison"]

    contract_layer_context = {
        "note": (
            "safe_to_display / conflict / escalation / undecidable / citation rates are properties "
            "of the full decision-contract pipeline (classifier diagnosis + complete criteria table "
            "+ guideline catalog) evaluated once on the confirmatory run. They are not computed "
            "per melanoma-detection arm below and do not vary across classifier-alone / "
            "criteria-alone / union / stacked, because none of those four variants is what the "
            "contract is built from -- every contract in the confirmatory run uses the classifier's "
            "committed diagnosis and the full seven-criterion table regardless of which of the four "
            "sensitivity/specificity numbers you are reading. Reported once here, attached to all "
            "four arms below as shared context rather than duplicated with false per-arm precision."
        ),
        "test_split_n395": {
            "contracts_safe_to_display": decision["safe_to_display"],
            "of_n_cases": test["n_cases"],
            "conflict_rate": decision["conflict_rate"],
            "escalation_rate": decision["escalation_rate"],
            "undecidable_rate": decision["undecidable_rate"],
            "citation_rate": decision["citation_rate"],
        },
    }

    def make_arm(
        *,
        sens: float, spec: float, confusion: Dict[str, int], auroc: Optional[float],
        valid_sens: Optional[float], valid_spec: Optional[float], valid_auroc: Optional[float],
        auroc_note: Optional[str] = None, extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        d: Dict[str, object] = {
            "test_n395": {
                "sensitivity": sens,
                "specificity": spec,
                "youden_j": _youden(sens, spec),
                "confusion": confusion,
                "melanoma_auroc": auroc,
            },
            "valid_n203_reference_point": {
                "sensitivity": valid_sens,
                "specificity": valid_spec,
                "youden_j": _youden(valid_sens, valid_spec),
                "melanoma_auroc": valid_auroc,
                "source": "g6_multiview_stacking.json (reference_points / verdict)",
            },
            "contracts_safe_to_display": contract_layer_context["test_split_n395"]["contracts_safe_to_display"],
            "contracts_safe_to_display_of": contract_layer_context["test_split_n395"]["of_n_cases"],
        }
        if auroc_note:
            d["auroc_note"] = auroc_note
        if extra:
            d.update(extra)
        return d

    classifier_alone = p1["classifier_alone"]
    criteria_path = p1["criteria_path"]
    union = p1["union_either_flags"]
    stacked_at_thr = stacked_cmp["stacked_at_registered_threshold"]

    arms = {
        "classifier_alone": make_arm(
            sens=classifier_alone["sensitivity"], spec=classifier_alone["specificity"],
            confusion=classifier_alone["confusion"], auroc=p1["classifier_auroc"],
            valid_sens=ref["classifier_alone"]["sensitivity"], valid_spec=ref["classifier_alone"]["specificity"],
            valid_auroc=None,
            extra={"description": "Diagnosis classifier's own MEL prediction. Ranking score exists (posterior)."},
        ),
        "criteria_path_alone": make_arm(
            sens=criteria_path["sensitivity"], spec=criteria_path["specificity"],
            confusion=criteria_path["confusion"], auroc=None,
            valid_sens=ref["criteria_alone"]["sensitivity"], valid_spec=ref["criteria_alone"]["specificity"],
            valid_auroc=None,
            auroc_note="No AUROC: 'seven-point score >= threshold' is a binary rule, not a ranked score to sweep.",
            extra={"description": "Seven-point checklist score (from trained criterion heads) >= threshold 3."},
        ),
        "union_rule": make_arm(
            sens=union["sensitivity"], spec=union["specificity"],
            confusion=union["confusion"], auroc=None,
            valid_sens=ref["hand_written_union"]["sensitivity"], valid_spec=ref["hand_written_union"]["specificity"],
            valid_auroc=None,
            auroc_note="No AUROC: 'classifier flags OR criteria flags' is a binary OR of two flags, not a ranked score.",
            extra={"description": "Flags melanoma if EITHER the classifier OR the criteria path does."},
        ),
        "stacked_ensemble": make_arm(
            sens=stacked_at_thr["sensitivity"], spec=stacked_at_thr["specificity"],
            confusion=stacked_at_thr["confusion"], auroc=stacked_cmp["stacked_auroc"],
            valid_sens=verdict["valid_sensitivity_at_train_fixed_threshold"],
            valid_spec=verdict["valid_specificity_at_train_fixed_threshold"],
            valid_auroc=g6["melanoma"]["auroc"],
            extra={
                "description": "Logistic-regression meta-learner over multi-view predictions, registered threshold 0.2981.",
                "vs_classifier_alone_delong_test": stacked_cmp["delong"],
                "disclosure": stacked_doc["disclosure"],
            },
        ),
    }

    return {
        "description": (
            "Four melanoma-detection arms assembled from three already-written evidence artifacts. "
            "Primary numbers are the one-shot confirmatory test split (n=395); the matching valid-"
            "split (n=203) numbers from G6 are attached as a reference point, not a second endpoint."
        ),
        "source_artifacts": [
            "research/evidence/derm7pt_v1/confirmatory_test_395.json",
            "research/evidence/derm7pt_v1/confirmatory_p1_completion.json",
            "research/evidence/derm7pt_v1/g6_multiview_stacking.json",
        ],
        "youden_j_formula": "sensitivity + specificity - 1, recomputed here from each source's own rounded sens/spec",
        "youden_j_rounding_note": (
            "Recomputing from already-rounded (4dp) sensitivity/specificity can differ from a "
            "source file's own stored youden_j by up to 0.0001 (rounding order). E.g. "
            "criteria_path_alone test: 0.5743+0.8946-1=0.4689 here vs 0.4688 as stored in "
            "confirmatory_test_395.json. Immaterial, disclosed rather than silently reconciled."
        ),
        "contract_layer_context": contract_layer_context,
        "arms": arms,
    }


# =====================================================================================
# Table B -- system-level ablation, computed fresh on Derm7pt valid (n=203, fit on train)
# =====================================================================================


def arm_full_system(full_contracts: List[DecisionContract], catalog: GuidelineCatalog,
                     run_cohort_report: Dict[str, object]) -> Dict[str, object]:
    summary = _decision_summary(full_contracts)
    summary.update(_router_summary(full_contracts, catalog))
    summary["citation_rate"] = run_cohort_report["guideline_citation"]["citation_rate"]
    summary["cases_with_contingent_melanoma_guidance"] = (
        run_cohort_report["guideline_citation"]["cases_with_contingent_melanoma_guidance"]
    )
    behaviour = run_cohort_report["decision_behaviour"]
    summary["cross_check_against_run_cohort_direct_call"] = {
        "safe_to_display_matches": summary["contracts_safe_to_display"] == behaviour["safe_to_display"],
        "conflict_rate_matches": summary["conflict_rate"] == behaviour["conflict_rate"],
        "undecidable_rate_matches": summary["undecidable_rate"] == behaviour["undecidable_rate"],
    }
    summary["note"] = (
        "This IS run_cohort('valid'), unmodified. Flagged for arms 2 and 5 below: build_contract() "
        "calls catalog.bind(classifier_diagnosis) unconditionally on every case here -- "
        "medguard_v2/evidence/router.py is never imported by run_cohort.py -- and these criterion "
        "heads are argmax LogisticRegression classifiers that structurally cannot emit "
        "INDETERMINATE, since that label never appears in their training data -- "
        "medguard_v2/perception/criterion_abstention.py is never imported by run_cohort.py either "
        "(it is used exactly once in the package, by eval/agent_eval.py). So on the retrieval-"
        "gating and abstention dimensions specifically, this 'full system' arm already IS what "
        "arms 2 and 5 ask to ablate down to. Both arms report the real counterfactual (router-"
        "gated retrieval / policy-applied abstention) instead of a diff against itself."
    )
    return summary


def arm_no_router(full_contracts: List[DecisionContract], catalog: GuidelineCatalog) -> Dict[str, object]:
    result = _decision_summary(full_contracts)
    result.update(_router_summary(full_contracts, catalog))
    result["note"] = (
        "'No router' (always trigger retrieval) is what ships today -- see arm 1's note. This arm "
        "is therefore numerically identical to arm 1 on every field above "
        "retrieval_calls_as_shipped_ungated. 'retrieval_calls_if_gated_by_router' is the real "
        "counterfactual: running medguard_v2/evidence/router.route() against these same 203 "
        "contracts (same classifier_diagnosis/confidence/score, nothing refit). Gating mostly "
        "saves calls on NO_COVERAGE cases (NEV/SK/MISC, ~2/3 of valid), where build_contract "
        "issues a catalog.bind() that is known in advance to return nothing."
    )
    return result


def arm_no_criteria(
    valid: Sequence[Derm7ptCase], train: Sequence[Derm7ptCase],
    features: Dict[str, np.ndarray], catalog: GuidelineCatalog,
    full_summary_citation_rate: float,
) -> Dict[str, object]:
    """Classifier only. Criterion heads are never fit -- not fit-then-ignored, never fit at all."""
    x_train = np.stack([features[str(c.case_num)] for c in train])
    x_valid = np.stack([features[str(c.case_num)] for c in valid])
    diagnosis = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
    diagnosis.fit(x_train, [c.diagnosis_group for c in train])  # the only model fit in this arm
    posterior = diagnosis.predict_proba(x_valid)
    predicted = diagnosis.predict(x_valid)

    # INDETERMINATE is the system's own first-class "not assessed" value (schema.py), so this is
    # how "skip the criterion heads" is represented to build_contract, which requires a complete
    # seven-key table and has no separate "criteria absent" input.
    blank = {name: INDETERMINATE for name in CRITERION_NAMES}
    contracts = _build_contracts(
        valid, diagnosis=diagnosis, posterior=posterior, predicted=predicted,
        criteria_by_case=[blank] * len(valid), catalog=catalog,
        receipt_extra={"ablation": "no_criteria", "criterion_heads_fit": False},
    )

    result = _decision_summary(contracts)
    result.update(_router_summary(contracts, catalog))

    n = len(valid)
    classifier_mel_rate = _round4(sum(1 for c in contracts if c.classifier_diagnosis == "MEL") / n)
    citation_rate = _round4(sum(1 for c in contracts if c.guideline.covered) / n)
    contingent = sum(1 for c in contracts if c.contingent_guideline is not None)

    result["citation_rate"] = citation_rate
    result["citation_rate_unchanged_from_full_system"] = citation_rate == full_summary_citation_rate
    result["cases_with_contingent_melanoma_guidance"] = contingent
    result["classifier_predicts_mel_rate"] = classifier_mel_rate
    result["note"] = (
        "Every criterion is INDETERMINATE, so score_min=0 and score_max=10=MAX_SCORE for all 203 "
        "cases: the checklist can never clear the threshold (score_min>=3) and can never rule it "
        "out (score_max<3), so undecidable_rate above is exactly 1.0 by construction -- this is "
        "the intended, not a bug. citation_rate is unchanged from the full system because "
        "guideline binding follows classifier_diagnosis only, never the criteria table; "
        "contingent guidance drops from the full system's count to 0, because it requires the "
        "checklist to affirmatively flag melanoma (score_min>=3), which is now impossible. "
        ""
        "DESIGN INCONSISTENCY SURFACED BY THIS ABLATION: conflict_rate/escalation_rate above "
        "(from DecisionContract.conflict, contract.py's own _build_conflict) equal "
        "classifier_predicts_mel_rate, because contract.py flags a conflict whenever the "
        "classifier says MEL and score_min<3 -- it does not check score.decidable, so 'checklist "
        "never actually assessed this' and 'checklist actively disagrees' both count as the same "
        "kind of conflict there. router_state_distribution above tells a different story: it is "
        "100% UNDECIDABLE and 0% CONFLICT, because router.classify_state() additionally requires "
        "score.decidable before it will call something a CONFLICT (see router.py's classify_state, "
        "the classifier==MEL branch). Two modules in this codebase both compute 'is this a "
        "conflict' and disagree once criteria go indeterminate; it is invisible on the full system "
        "(undecidable_rate 0.0000 there) and only shows up under this ablation."
    )
    return result


def arm_no_contracts(full_contracts: List[DecisionContract]) -> Dict[str, object]:
    """Skip verification_failures() gating -- force every contract to display regardless."""
    n = len(full_contracts)
    measured = _decision_summary(full_contracts)
    forced_display_count = n  # by definition: the gate is skipped, so nothing is ever withheld

    # ---- what actually changes on this cohort -------------------------------------------------
    delta = forced_display_count - measured["contracts_safe_to_display"]

    # ---- adversarial illustration: the guard is not vacuous in general, just never triggered by
    # honest build_contract() output on this cohort. Same technique as
    # medguard_v2/tests/test_evidence_contract.py::test_a_contract_whose_score_disagrees... :
    # take a real contract and tamper it via dataclasses.replace, which build_contract() itself
    # can never produce, to show what the check catches when something upstream corrupts a
    # contract after the fact.
    base = next(c for c in full_contracts if c.guideline.covered and c.guideline.recommendations)

    mismatched = score_criteria({**{n_: "absent" for n_ in CRITERION_NAMES},
                                  "pigment_network": "atypical", "blue_whitish_veil": "present"})
    if (mismatched.score_min, mismatched.score_max) == (base.score.score_min, base.score.score_max):
        mismatched = score_criteria({n_: "absent" for n_ in CRITERION_NAMES})  # guaranteed different: score 0
    demo_score_mismatch = dataclasses.replace(base, score=mismatched)

    tampered_rec = dataclasses.replace(base.guideline.recommendations[0], content_sha256="0" * 64)
    tampered_guideline = dataclasses.replace(
        base.guideline, recommendations=(tampered_rec,) + base.guideline.recommendations[1:]
    )
    demo_hash_tamper = dataclasses.replace(base, guideline=tampered_guideline)

    demo_empty_conflict_description = dataclasses.replace(
        base, conflict=dataclasses.replace(base.conflict, present=True, description="")
    )

    demonstrations = {
        "score_disagrees_with_its_own_criteria_table": {
            "how_it_was_produced": "dataclasses.replace(contract, score=<score of a different criteria table>)",
            "verification_failures": demo_score_mismatch.verification_failures(),
            "safe_to_display": demo_score_mismatch.safe_to_display,
            "would_display_if_the_check_were_skipped": True,
        },
        "guideline_record_content_hash_tampered_after_load": {
            "how_it_was_produced": "dataclasses.replace(recommendation, content_sha256='0'*64)",
            "verification_failures": demo_hash_tamper.verification_failures(),
            "safe_to_display": demo_hash_tamper.safe_to_display,
            "would_display_if_the_check_were_skipped": True,
        },
        "conflict_flagged_with_no_description": {
            "how_it_was_produced": "dataclasses.replace(conflict, present=True, description='')",
            "verification_failures": demo_empty_conflict_description.verification_failures(),
            "safe_to_display": demo_empty_conflict_description.safe_to_display,
            "would_display_if_the_check_were_skipped": True,
        },
    }

    return {
        **measured,
        "forced_display_count_if_check_skipped": forced_display_count,
        "additional_cases_displayed_by_skipping_the_check": delta,
        "adversarial_demonstrations": demonstrations,
        "note": (
            f"Measured effect on this cohort: {delta} additional cases would display if "
            f"verification_failures() were skipped -- the check never fires through the normal "
            f"build_contract() path, because build_contract's own score always matches the "
            f"criteria table it just computed, guideline hashes are pre-verified once at "
            f"GuidelineCatalog.load() (a tampered record raises GuidelineIntegrityError there, "
            f"before any contract exists), and _build_conflict() never returns present=True with "
            f"an empty description. It is a defense-in-depth invariant on this pipeline, not an "
            f"active filter -- '0 change' is the correct measurement, not a null result. "
            f"adversarial_demonstrations above shows the check is not vacuous in general: each "
            f"demo takes a real contract and corrupts it the same way the project's own test "
            f"suite does (dataclasses.replace, which build_contract() itself cannot produce), and "
            f"the check catches all three. Skipping verification would let each one display."
        ),
    }


def arm_no_abstention(
    valid: Sequence[Derm7ptCase], train: Sequence[Derm7ptCase],
    features: Dict[str, np.ndarray], catalog: GuidelineCatalog,
) -> Dict[str, object]:
    """Apply the REAL fitted margin-abstention policy (train-fit, loaded not refit) and measure
    its effect against Derm7pt's gold seven_point_score -- genuine ground truth, not a proxy."""
    x_train = np.stack([features[str(c.case_num)] for c in train])
    x_valid = np.stack([features[str(c.case_num)] for c in valid])
    diagnosis = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
    diagnosis.fit(x_train, [c.diagnosis_group for c in train])
    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
        head.fit(x_train, [str(c.criteria[name]) for c in train])
        heads[name] = head
    posterior = diagnosis.predict_proba(x_valid)
    predicted = diagnosis.predict(x_valid)

    argmax_pred = {name: list(heads[name].predict(x_valid)) for name in CRITERION_NAMES}

    policy = load_policy()  # fit on train, out-of-fold; loaded as-is, nothing refit here
    abst_values: Dict[str, List[str]] = {}
    for name in CRITERION_NAMES:
        values, _margin = policy.apply(heads[name], x_valid, name)
        abst_values[name] = values

    gold = {name: [str(c.criteria[name]) for c in valid] for name in CRITERION_NAMES}

    # -- per-criterion: is the argmax head actually worse on exactly the cases it abstains on? --
    per_criterion: Dict[str, object] = {}
    for name in CRITERION_NAMES:
        abstained_mask = [v == INDETERMINATE for v in abst_values[name]]
        correct = [argmax_pred[name][i] == gold[name][i] for i in range(len(valid))]
        per_criterion[name] = {
            "margin_threshold": policy.margins[name],
            "n_abstained": sum(abstained_mask),
            "abstention_rate": _round4(sum(abstained_mask) / len(valid)),
            "argmax_accuracy_overall": _safe_mean(correct),
            "argmax_accuracy_on_what_it_retains": _safe_mean(
                c for c, a in zip(correct, abstained_mask) if not a
            ),
            "argmax_would_be_accuracy_on_what_it_abstains_on": _safe_mean(
                c for c, a in zip(correct, abstained_mask) if a
            ),
        }
    macro_abstention_rate = _safe_mean(v["abstention_rate"] for v in per_criterion.values())

    # -- case level: build contracts both ways, score against gold seven_point_score -------------
    argmax_criteria_by_case = [{name: argmax_pred[name][i] for name in CRITERION_NAMES} for i in range(len(valid))]
    abst_criteria_by_case = [{name: abst_values[name][i] for name in CRITERION_NAMES} for i in range(len(valid))]

    argmax_contracts = _build_contracts(
        valid, diagnosis=diagnosis, posterior=posterior, predicted=predicted,
        criteria_by_case=argmax_criteria_by_case, catalog=catalog,
        receipt_extra={"ablation": "no_abstention_argmax_restated"},
    )
    abstaining_contracts = _build_contracts(
        valid, diagnosis=diagnosis, posterior=posterior, predicted=predicted,
        criteria_by_case=abst_criteria_by_case, catalog=catalog,
        receipt_extra={"ablation": "with_real_fitted_abstention_policy"},
    )

    gold_verdict = [c.seven_point_score >= MELANOMA_SUSPICION_THRESHOLD for c in valid]
    argmax_verdict = [c.score.meets_threshold for c in argmax_contracts]  # always decidable
    argmax_case_accuracy = _safe_mean(a == g for a, g in zip(argmax_verdict, gold_verdict))

    decidable_mask = [c.score.decidable for c in abstaining_contracts]
    abst_case_accuracy_on_decidable = _safe_mean(
        abstaining_contracts[i].score.meets_threshold == gold_verdict[i]
        for i in range(len(valid)) if decidable_mask[i]
    )
    newly_undecidable = [i for i in range(len(valid)) if not decidable_mask[i]]
    argmax_error_rate_overall = _safe_mean(a != g for a, g in zip(argmax_verdict, gold_verdict))
    argmax_error_rate_on_cases_abstention_flags = _safe_mean(
        argmax_verdict[i] != gold_verdict[i] for i in newly_undecidable
    )

    argmax_summary = _decision_summary(argmax_contracts)
    argmax_summary.update(_router_summary(argmax_contracts, catalog))
    abst_summary = _decision_summary(abstaining_contracts)
    abst_summary.update(_router_summary(abstaining_contracts, catalog))

    return {
        "as_literally_named_no_abstention": {
            **argmax_summary,
            "note": "Identical to arm 1 -- see arm 1's note. Restated here so this arm is self-contained.",
        },
        "real_counterfactual_with_fitted_policy_applied": {
            **abst_summary,
            "policy_source": "research/evidence/derm7pt_v1/criterion_abstention_v1.json (fit on train, out-of-fold; loaded, not refit)",
        },
        "score_accuracy_impact": {
            "per_criterion": per_criterion,
            "macro_abstention_rate": macro_abstention_rate,
            "case_level_melanoma_verdict_vs_gold_seven_point_score": {
                "argmax_case_accuracy": argmax_case_accuracy,
                "abstention_case_accuracy_on_decidable_subset": abst_case_accuracy_on_decidable,
                "abstention_coverage": _round4((len(valid) - len(newly_undecidable)) / len(valid)),
                "argmax_forced_verdict_error_rate_overall": argmax_error_rate_overall,
                "argmax_forced_verdict_error_rate_on_cases_abstention_would_flag_undecidable": (
                    argmax_error_rate_on_cases_abstention_flags
                ),
                "n_cases_abstention_flags_undecidable": len(newly_undecidable),
            },
        },
        "note": (
            "Per-criterion: argmax accuracy on the subset each head abstains on vs. the subset it "
            "keeps, both computed against Derm7pt's real per-criterion gold labels (not a proxy). "
            "Case level: the melanoma-threshold verdict (score>=3) compared against gold "
            "seven_point_score>=3, which schema.py's own docstring states this checklist "
            "reproduces on 1011/1011 Derm7pt cases, so it is exact ground truth here, not "
            "recomputed by this script. 'argmax_forced_verdict_error_rate_on_cases_abstention_"
            "would_flag_undecidable' vs '..._overall' is the key comparison: if the first is "
            "higher, abstention is concentrating on exactly the cases the forced argmax verdict "
            "gets wrong, i.e. buying real safety rather than just adding caution."
        ),
    }


# =====================================================================================
# orchestration
# =====================================================================================


def build_table_b() -> Dict[str, object]:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    features = _load_features()
    catalog = GuidelineCatalog.load()

    print(f"  cohort: valid n={len(valid)}, fit on train n={len(train)}")

    # Arm 1: run_cohort('valid') exactly as shipped -- both the direct call (for the printed
    # summary and the cross-check) and the equivalent contracts as objects (needed by arms
    # 2 and 4, which inspect fields run_cohort's own dict serialisation drops, e.g.
    # ConflictState.criteria_path_flags_melanoma and the DecisionContract objects themselves).
    run_cohort_report = run_cohort("valid")
    diagnosis, heads = fit_readers(train, features)
    x_valid = np.stack([features[str(c.case_num)] for c in valid])
    posterior = diagnosis.predict_proba(x_valid)
    predicted = diagnosis.predict(x_valid)
    head_pred = {name: list(head.predict(x_valid)) for name, head in heads.items()}
    full_contracts = _build_contracts(
        valid, diagnosis=diagnosis, posterior=posterior, predicted=predicted,
        criteria_by_case=[{name: head_pred[name][i] for name in CRITERION_NAMES} for i in range(len(valid))],
        catalog=catalog, receipt_extra={"ablation": "full_system"},
    )

    print("  arm 1: full system")
    arm1 = arm_full_system(full_contracts, catalog, run_cohort_report)
    print("  arm 2: no router")
    arm2 = arm_no_router(full_contracts, catalog)
    print("  arm 3: no criteria")
    arm3 = arm_no_criteria(valid, train, features, catalog, arm1["citation_rate"])
    print("  arm 4: no contracts")
    arm4 = arm_no_contracts(full_contracts)
    print("  arm 5: no abstention")
    arm5 = arm_no_abstention(valid, train, features, catalog)

    return {
        "description": (
            "Five run_cohort('valid') variants, computed fresh (not read from an existing "
            "artifact). Arm 1 is the real function call; arms 2-5 are new functions in this "
            "script that call build_contract/router/criterion_abstention differently, per the "
            "task's instruction not to modify run_cohort.py/contract.py/router.py/"
            "criterion_abstention.py themselves."
        ),
        "cohort": {"split": "valid", "n_cases": len(valid), "fit_split": "train", "n_fit_cases": len(train)},
        "arms": {
            "1_full_system": arm1,
            "2_no_router": arm2,
            "3_no_criteria": arm3,
            "4_no_contracts": arm4,
            "5_no_abstention": arm5,
        },
        "integration_findings": [
            (
                "medguard_v2/evidence/router.py (gated retrieval: NO_COVERAGE/CONFLICT/UNDECIDABLE/"
                "ROUTINE) is fully implemented but not imported anywhere in the executed pipeline -- "
                "not by run_cohort.py, not by eval/confirmatory.py, not by eval/agent_eval.py, not "
                "by agent/clinician_agent.py. Verified by grep across medguard_v2/**/*.py: the only "
                "hit is the module's own definition. run_cohort's shipped behaviour already matches "
                "what arm 2 ('no router') was specified to be."
            ),
            (
                "medguard_v2/perception/criterion_abstention.py (margin-based INDETERMINATE "
                "producer, policy already fit on train) is used in exactly one place in the whole "
                "package: eval/agent_eval.py. run_cohort.py's criterion heads are plain "
                "LogisticRegression.predict() calls, which cannot emit a class absent from their "
                "training labels, so run_cohort's undecidable_rate is structurally 0.0000 -- "
                "confirmed both by criterion_abstention.py's own docstring and by direct "
                "measurement here. run_cohort's shipped behaviour already matches what arm 5 ('no "
                "abstention') was specified to be."
            ),
            (
                "contract.py's DecisionContract.conflict (_build_conflict) and router.py's "
                "classify_state() both compute 'is this case a conflict' and disagree once "
                "criteria go indeterminate: _build_conflict only checks score_min>=threshold, so "
                "classifier=MEL + an unassessed checklist reads as a conflict; classify_state "
                "additionally requires score.decidable, so the same case reads as UNDECIDABLE, not "
                "CONFLICT. Invisible on the shipped pipeline (undecidable_rate 0.0000 there); "
                "visible and quantified under arm 3 (no_criteria) above, where conflict_rate/"
                "escalation_rate (contract.py's definition) and router_state_distribution "
                "(router.py's definition) tell different stories about the same 203 contracts."
            ),
        ],
    }


def main() -> None:
    print("Table A: assembling from existing artifacts...")
    table_a = build_table_a()

    print("Table B: computing system-level ablation on valid...")
    table_b = build_table_b()

    result = {
        "protocol": "medguard-v2-derm7pt-ablation-table-v1",
        "generated_by": "scripts/dev/build_ablation_table.py",
        "table_a_perception_ablation": table_a,
        "table_b_system_ablation": table_b,
    }
    result = _json_safe(result)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH.relative_to(REPO)}")

    print("\n--- Table A: melanoma sensitivity / specificity / Youden J (test n=395) ---")
    for name, arm in table_a["arms"].items():
        t = arm["test_n395"]
        print(f"  {name:<22} sens={t['sensitivity']}  spec={t['specificity']}  "
              f"J={t['youden_j']}  AUROC={t['melanoma_auroc']}")

    print("\n--- Table B: decision behaviour per arm (valid n=203) ---")
    for name, arm in table_b["arms"].items():
        flat = arm if "contracts_safe_to_display" in arm else arm.get("as_literally_named_no_abstention", {})
        print(f"  {name:<18} safe_to_display={flat.get('contracts_safe_to_display')} "
              f"conflict={flat.get('conflict_rate')} undecidable={flat.get('undecidable_rate')} "
              f"router_states={flat.get('router_state_distribution')}")

    print("\n--- integration findings ---")
    for finding in table_b["integration_findings"]:
        print(f"  - {finding[:160]}...")


if __name__ == "__main__":
    main()
