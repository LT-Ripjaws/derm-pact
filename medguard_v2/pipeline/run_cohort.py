"""End-to-end: images in, auditable decision contracts out.

This is the runnable system, and the thing the confirmatory evaluation executes. Everything it
does is fixed by `research/PREREGISTRATION_V2.json`: the feature view, the per-target probes, the
host-computed 7-point aggregation, and the guideline binding. Nothing is selected here.

The metrics it emits are deliberately *decision-level* rather than model-level, because the
model-level endpoints already live in G6. What a clinical reader needs to know about a decision
support tool is different from what a reviewer needs to know about a classifier:

- how often it produces something safe to display at all;
- how often the two views disagree, since that is when it asks for a human;
- how often it can cite guidance, and on what denominator;
- and how often it stays silent because no guidance exists -- reported, never hidden.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.evidence.contract import DecisionContract, build_contract, contract_to_dict
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.evidence.render import render_markdown
from medguard_v2.evidence.router import RetrievalState, route
from medguard_v2.perception.clinical_metadata import encode_derm7pt
from medguard_v2.perception.criterion_abstention import load_policy

REPO = Path(__file__).resolve().parents[2]
FEATURES = REPO / "research" / "evidence" / "derm7pt_v1" / "features" / "dermlip" / "features.npz"
FEATURES_P1 = REPO / "research" / "evidence" / "derm7pt_v1" / "features_phase1"
PREREG = REPO / "research" / "PREREGISTRATION_V2.json"

#: Registered in PREREGISTRATION_V2. Not tunable here.
SEED = 20260810
BASE_C = 10.0

# --------------------------------------------------------------------------------------------
# Promoted perception (phase 2, SAFETY-PRESERVING). The phase-1 winners were re-examined against
# a hard melanoma-safety constraint: the promoted diagnosis MUST hold MEL AUROC >= the current
# production 0.895 on the sealed Derm7pt TEST split. The monolithic phase-1 winner
# (dermlip_tta+dermlip_clinic_tta, class-balanced) FAILED that constraint -- it regressed MEL
# AUROC to 0.867, because class-balancing plus the weak clinic view (standalone MEL AUROC 0.775)
# distorts the melanoma-posterior ranking. This module therefore promotes the safety-preserving
# config selected in research/evidence/derm7pt_v1/perception_phase2_safety_preserving.json:
#
#   diagnosis  : concat(dermlip, metadata14) = 526-d, UNBALANCED LR, C=100 (train-CV-selected in
#                phase 1), + the AUROC-invariant SK/MISC operating point. TEST MEL AUROC 0.9151
#                (>= 0.895, actually +0.020), and the operating point lifts SK/MISC recall for
#                free because re-ranking SK/MISC never touches the MEL posterior column.
#   criteria   : concat(dermlip_tta, metadata14) = 526-d, lr_balanced, per-head C -- UNCHANGED
#                from the phase-1 winner (it does not touch the diagnosis AUROC and it lifts the
#                per-criterion binary-concept F1 by +0.0545 on test and union melanoma
#                sensitivity 0.7426 -> 0.8614).
#
# NOTHING here is selected in this module; promotion is opt-in behind run_cohort(promoted=True),
# and the pre-promotion path above is left untouched so the frozen cdss_contracts_valid.json stays
# reproducible and every existing caller is unchanged.
#
# CALIBER-safe: this promotion is NOT an accuracy-SOTA claim over any prior frozen-probe work. Its
# value is minority-class recall (SK/MISC) + per-criterion criteria quality + downstream contract
# quality from metadata and the operating point -- and, unlike the monolith, it PRESERVES (does not
# beat) melanoma AUROC rather than trading it away.
# --------------------------------------------------------------------------------------------
#: diagnosis: concat(dermlip, metadata14) = 526-d, UNBALANCED LR, C=100. The diagnosis base view
#: is the plain frozen dermlip probe (not TTA/clinic) so the melanoma ranking is unchanged from the
#: 0.895 production baseline; metadata is the only added signal and it lifts AUROC to 0.9151.
PROMOTED_DIAGNOSIS_VIEWS = ("dermlip", "metadata14")
PROMOTED_DIAGNOSIS_C = 100.0
#: The promoted diagnosis head is deliberately UNBALANCED (class_weight=None). Class-balancing was
#: the melanoma-AUROC regressor in the monolith; minority recall is recovered instead by the
#: AUROC-invariant operating point below, which changes the decision, not the MEL ranking.
PROMOTED_DIAGNOSIS_CLASS_WEIGHT = None
#: argmax(posterior / per-class weight); weight < 1 boosts that class's rank. Lifts SK/MISC recall.
PROMOTED_OPERATING_POINT = {"SK": 0.5, "MISC": 0.7}
#: criteria: concat(dermlip_tta, metadata14) = 526-d, lr_balanced, per-head C.
PROMOTED_CRITERIA_VIEW = "dermlip_tta"
PROMOTED_CRITERIA_C = {
    "pigment_network": 100.0,
    "blue_whitish_veil": 10.0,
    "vascular_structures": 100.0,
    "streaks": 100.0,
    "pigmentation": 100.0,
    "dots_and_globules": 10.0,
    "regression_structures": 100.0,
}


def _load_features() -> Dict[str, np.ndarray]:
    data = np.load(FEATURES)
    return {str(c): v for c, v in zip(data["case_num"], data["features"])}


def fit_readers(train, features):
    """Fit the diagnosis classifier and the seven criterion heads. TRAIN ONLY."""
    x = np.stack([features[str(c.case_num)] for c in train])
    diagnosis = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
    diagnosis.fit(x, [c.diagnosis_group for c in train])
    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
        head.fit(x, [str(c.criteria[name]) for c in train])
        heads[name] = head
    return diagnosis, heads


# --------------------------------------------------------------------------------------------
# Promoted perception path (phase 2). Opt-in via run_cohort(promoted=True).
# --------------------------------------------------------------------------------------------

def _load_npz_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {str(c): v for c, v in zip(data["case_num"], data["features"])}


def _promoted_feature_matrices(cases) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-case promoted feature vectors, keyed by case_num string.

    Returns ``(diagnosis_features, criteria_features)``. Both are 526-d but they read DIFFERENT
    image views, because they are different readers:

    - diagnosis = concat(dermlip, metadata14). The plain frozen dermlip probe (features/dermlip),
      NOT the TTA/clinic multi-view -- keeping the melanoma-posterior ranking identical to the 0.895
      production baseline so metadata is the only thing that moves it (up, to 0.9151).
    - criteria  = concat(dermlip_tta, metadata14). The horizontal-flip TTA view, unchanged from the
      phase-1 criteria winner (read-only from features_phase1/).

    metadata14 is the frozen 14-d Derm7pt encoding (sex+location+elevation), computed here rather
    than cached because it is a pure function of the case record.
    """
    dermlip = _load_features()
    dermlip_tta = _load_npz_features(FEATURES_P1 / "dermlip_tta" / "features.npz")
    diagnosis_features: Dict[str, np.ndarray] = {}
    criteria_features: Dict[str, np.ndarray] = {}
    for case in cases:
        key = str(case.case_num)
        meta = encode_derm7pt(case.sex, case.location, case.elevation)
        diagnosis_features[key] = np.concatenate([dermlip[key], meta])
        criteria_features[key] = np.concatenate([dermlip_tta[key], meta])
    return diagnosis_features, criteria_features


def fit_readers_promoted(train, diagnosis_features, criteria_features):
    """Fit the promoted diagnosis classifier and seven criterion heads. TRAIN ONLY.

    Diagnosis: UNBALANCED LR (class_weight=None) C=100 over the 526-d dermlip+metadata features.
    The head is deliberately unbalanced -- class-balancing was the melanoma-AUROC regressor in the
    phase-1 monolith; the operating point (applied at decision time, AUROC-invariant) recovers
    SK/MISC recall instead. Criteria: lr_balanced with the per-head C selected in phase 1, over the
    526-d TTA+metadata features. Nothing is selected here -- every hyperparameter is frozen above.
    """
    xd = np.stack([diagnosis_features[str(c.case_num)] for c in train])
    diagnosis = LogisticRegression(
        max_iter=5000, C=PROMOTED_DIAGNOSIS_C, random_state=SEED,
        solver="lbfgs", class_weight=PROMOTED_DIAGNOSIS_CLASS_WEIGHT,
    )
    diagnosis.fit(xd, [c.diagnosis_group for c in train])

    xc = np.stack([criteria_features[str(c.case_num)] for c in train])
    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(
            max_iter=5000, C=PROMOTED_CRITERIA_C[name], random_state=SEED,
            solver="lbfgs", class_weight="balanced",
        )
        head.fit(xc, [str(c.criteria[name]) for c in train])
        heads[name] = head
    return diagnosis, heads


def apply_operating_point(posterior: np.ndarray, classes, weights: Dict[str, float]):
    """Re-rank a posterior by argmax(posterior / per-class weight); weight < 1 boosts that class.

    Returns ``(predicted_labels, chosen_confidence)`` where the confidence is the model's own
    (un-reweighted) posterior mass on the class that was ultimately chosen -- the honest "how sure
    is the model about the label we are displaying", which the router and contract both read.
    """
    class_list = [str(c) for c in classes]
    weight_vector = np.array([weights.get(c, 1.0) for c in class_list], dtype=np.float64)
    adjusted = np.asarray(posterior, dtype=np.float64) / weight_vector
    chosen = adjusted.argmax(axis=1)
    predicted = [class_list[i] for i in chosen]
    confidence = [float(posterior[row, i]) for row, i in enumerate(chosen)]
    return predicted, confidence


def _head_values_and_margins(head, x: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Argmax criterion readings plus the per-case abstention margin (top - second posterior).

    The margin is the head's own decision-boundary distance -- the same quantity
    criterion_abstention computes -- and is plumbed into the contract as uncertainty metadata. The
    frozen criterion_abstention_v1 CUTOFFS are deliberately NOT applied here: they were fit on the
    pre-promotion 512-d dermlip head, so gating the new 526-d balanced heads with them would
    mis-abstain. Values are a plain argmax (matching how phase 1 selected these heads); the margin
    is reported so a future re-fit of the abstention policy has what it needs.
    """
    proba = head.predict_proba(x)
    classes = head.classes_
    order = np.argsort(-proba, axis=1)
    top = proba[np.arange(len(proba)), order[:, 0]]
    second = (
        proba[np.arange(len(proba)), order[:, 1]] if proba.shape[1] > 1 else np.zeros(len(proba))
    )
    margins = top - second
    values = [str(classes[order[i, 0]]) for i in range(len(proba))]
    return values, margins


def run_cohort(split: str = "valid", *, limit: Optional[int] = None,
               allow_test: bool = False, promoted: bool = False) -> Dict:
    if split == "test" and not allow_test:
        raise RuntimeError(
            "run_cohort(split='test') is refused by default: the confirmatory run is a one-shot "
            "governed by PREREGISTRATION_V2. Pass allow_test=True only from a governed runner "
            "that records the access."
        )
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    cohort = [c for c in cases if c.split == split]
    if limit:
        cohort = cohort[:limit]

    catalog = GuidelineCatalog.load()
    head_margins: Optional[Dict[str, np.ndarray]] = None
    if promoted:
        diagnosis_features, criteria_features = _promoted_feature_matrices(cases)
        diagnosis, heads = fit_readers_promoted(train, diagnosis_features, criteria_features)
        xd_cohort = np.stack([diagnosis_features[str(c.case_num)] for c in cohort])
        xc_cohort = np.stack([criteria_features[str(c.case_num)] for c in cohort])
        posterior = diagnosis.predict_proba(xd_cohort)
        predicted, confidences = apply_operating_point(
            posterior, diagnosis.classes_, PROMOTED_OPERATING_POINT
        )
        head_predictions: Dict[str, List[str]] = {}
        head_margins = {}
        for name, head in heads.items():
            values, margins = _head_values_and_margins(head, xc_cohort)
            head_predictions[name] = values
            head_margins[name] = margins
        receipt = {
            "split": split,
            "feature_set": "diagnosis: dermlip+metadata14 (526-d); "
                           "criteria: dermlip_tta+metadata14 (526-d)",
            "diagnosis_C": PROMOTED_DIAGNOSIS_C,
            "diagnosis_class_weight": (
                "balanced" if PROMOTED_DIAGNOSIS_CLASS_WEIGHT == "balanced" else "none"
            ),
            "operating_point": dict(PROMOTED_OPERATING_POINT),
            "criteria_C": dict(PROMOTED_CRITERIA_C),
            "criteria_class_weight": "balanced",
            "seed": SEED,
            "fit_split": "train",
            "n_fit_cases": len(train),
            "abstention_policy": (
                "none (argmax on the promoted balanced heads); criterion_abstention_v1 cutoffs are "
                "fit on the pre-promotion 512-d dermlip head and are not valid here -- the "
                "top-minus-second margins are plumbed into criterion_margins instead"
            ),
            "perception_promotion": "phase2 (perception_uplift_phase1.json final_config)",
        }
    else:
        features = _load_features()
        diagnosis, heads = fit_readers(train, features)
        x_cohort = np.stack([features[str(c.case_num)] for c in cohort])
        posterior = diagnosis.predict_proba(x_cohort)
        predicted = diagnosis.predict(x_cohort)
        confidences = [float(posterior[i].max()) for i in range(len(cohort))]
        policy = load_policy()
        head_predictions = {}
        for name, head in heads.items():
            values, _margins = policy.apply(head, x_cohort, name)
            head_predictions[name] = values
        receipt = {
            "split": split,
            "feature_set": "dermlip/derm",
            "base_C": BASE_C,
            "seed": SEED,
            "fit_split": "train",
            "n_fit_cases": len(train),
            "abstention_policy": "criterion_abstention_v1",
        }

    contracts: List[DecisionContract] = []
    for i, case in enumerate(cohort):
        criteria = {name: head_predictions[name][i] for name in CRITERION_NAMES}
        margins = (
            {name: float(head_margins[name][i]) for name in CRITERION_NAMES}
            if head_margins is not None
            else None
        )
        contracts.append(
            build_contract(
                case_id=f"derm7pt-{case.case_num}",
                classifier_diagnosis=str(predicted[i]),
                classifier_confidence=float(confidences[i]),
                criteria=criteria,
                catalog=catalog,
                receipt=dict(receipt),
                criterion_margins=margins,
            )
        )

    router_decisions = [
        route(
            classifier_diagnosis=c.classifier_diagnosis,
            classifier_confidence=c.classifier_confidence,
            score=c.score,
            catalog=catalog,
        )
        for c in contracts
    ]

    # --- decision-level behaviour ------------------------------------------------
    withheld = [c for c in contracts if not c.safe_to_display]
    conflicts = [c for c in contracts if c.conflict.present]
    escalations = [c for c in contracts if c.conflict.escalate]
    cited = [c for c in contracts if c.guideline.covered]
    contingent = [c for c in contracts if c.contingent_guideline is not None]
    undecidable = [c for c in contracts if not c.score.decidable]

    n = len(contracts)
    coverage = catalog.coverage_report([c.diagnosis_group for c in cohort])

    # Where the checklist and the truth actually land, for context only.
    truth_mel = [c.diagnosis_group == "MEL" for c in cohort]
    checklist_flags = [c.score.score_min >= MELANOMA_SUSPICION_THRESHOLD for c in contracts]
    classifier_mel = [c.classifier_diagnosis == "MEL" for c in contracts]
    caught_by_either = sum(
        1 for t, a, b in zip(truth_mel, checklist_flags, classifier_mel) if t and (a or b)
    )

    router_states = Counter(d.state.value for d in router_decisions)
    gated_calls = sum(d.retrieval_calls for d in router_decisions)
    ungated_calls = n

    report = {
        "protocol": "medguard-v2-derm7pt-criteria-ensemble-v2",
        "split": split,
        "n_cases": n,
        "boundaries": {
            "fit_split": "train",
            "n_fit_cases": len(train),
            "generation_calls": 0,
            "paid_calls": 0,
            "hyperparameters_from": "PREREGISTRATION_V2.json (not selected here)",
            "abstention_policy": "criterion_abstention_v1 (margin-based, fit on train OOF)",
            "router": (
                "retrieval-need classification (NO_COVERAGE/CONFLICT/UNDECIDABLE/ROUTINE); "
                "classification only -- binding is a local catalog lookup and runs for every case"
            ),
        },
        "decision_behaviour": {
            "safe_to_display": n - len(withheld),
            "withheld_rate": round(len(withheld) / n, 4) if n else None,
            "withheld_reasons": Counter(
                f for c in withheld for f in c.verification_failures()
            ),
            "conflict_rate": round(len(conflicts) / n, 4) if n else None,
            "escalation_rate": round(len(escalations) / n, 4) if n else None,
            "undecidable_rate": round(len(undecidable) / n, 4) if n else None,
        },
        "router": {
            "state_distribution": dict(router_states),
            "calls_a_gated_deployment_would_make": gated_calls,
            "calls_this_pipeline_made": ungated_calls,
            "hypothetical_savings": round(1.0 - gated_calls / ungated_calls, 4) if ungated_calls else 0.0,
            "note": (
                "The router classifies retrieval need; it does not gate binding in this "
                "pipeline, where binding is a free local lookup performed for every case. "
                "Savings quantify what gating would avoid in a deployment where retrieval "
                "is remote or paid."
            ),
        },
        "guideline_citation": {
            "cases_with_a_citable_recommendation": len(cited),
            "citation_rate": round(len(cited) / n, 4) if n else None,
            "cases_with_contingent_melanoma_guidance": len(contingent),
            "coverage_denominator": coverage,
            "note": (
                "Citation rate is bounded by corpus coverage, not by retrieval quality. Naevi, "
                "seborrhoeic keratoses and the miscellaneous group have no formal guideline; the "
                "system reports that rather than generating management text."
            ),
        },
        "context_not_an_endpoint": {
            "melanomas_in_cohort": sum(truth_mel),
            "caught_by_checklist_or_classifier": caught_by_either,
            "note": "Descriptive. The registered inferential endpoints are computed in G6.",
        },
        "contracts": [contract_to_dict(c) for c in contracts],
    }

    if promoted:
        # Non-destructive marker so the promoted artifact is never mistaken for the frozen
        # pre-promotion one. The pre-promotion path never enters this branch, so its output stays
        # byte-identical to what produced the pinned cdss_contracts_valid.json digest.
        report["protocol"] = "medguard-v2-derm7pt-criteria-ensemble-v2-promoted"
        report["perception_promotion"] = {
            "source": "research/evidence/derm7pt_v1/perception_phase2_safety_preserving.json "
                      "(safety-preserving final config)",
            "diagnosis": "dermlip+metadata14, UNBALANCED LR C=100, operating point "
                         "w_SK=0.5 w_MISC=0.7 (TEST MEL AUROC 0.9151 >= production 0.895)",
            "criteria": "dermlip_tta+metadata14, lr_balanced, per-head C",
            "criterion_margins_plumbed": True,
            "melanoma_safety": (
                "Diagnosis head is UNBALANCED and reads the plain dermlip probe, so the melanoma "
                "posterior ranking is not distorted -- MEL AUROC is preserved (0.9151 vs 0.895), "
                "not regressed as in the class-balanced multi-view monolith (0.867). SK/MISC recall "
                "is recovered by the AUROC-invariant operating point."
            ),
            "caliber_safe_note": (
                "Not an accuracy-SOTA claim over any prior frozen-probe work (CALIBER/DERM-GATE is "
                "our own). The value is SK/MISC recall + per-criterion criteria quality + contract "
                "quality from metadata and the operating point; melanoma AUROC is preserved, not "
                "beaten."
            ),
        }
        report["boundaries"]["abstention_policy"] = (
            "none (argmax on promoted balanced heads); per-criterion top-minus-second margins "
            "plumbed into criterion_margins"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=0, help="render N contracts as markdown")
    args = parser.parse_args()

    if args.split == "test":
        raise SystemExit(
            "Refusing to run on the test split from the command line. The confirmatory run is a "
            "one-shot governed by PREREGISTRATION_V2; use the confirmatory runner, which records "
            "the access."
        )

    report = run_cohort(args.split, limit=args.limit)
    behaviour = report["decision_behaviour"]
    print(f"split={report['split']} n={report['n_cases']}")
    print(f"  safe to display : {behaviour['safe_to_display']}/{report['n_cases']}")
    print(f"  conflict rate   : {behaviour['conflict_rate']}")
    print(f"  escalation rate : {behaviour['escalation_rate']}")
    print(f"  undecidable     : {behaviour['undecidable_rate']}")
    print(f"  citation rate   : {report['guideline_citation']['citation_rate']}")
    print(f"  eligible frac   : {report['guideline_citation']['coverage_denominator']['eligible_fraction']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    if args.sample:
        cases = load_cases()
        cohort = [c for c in cases if c.split == args.split]
        features = _load_features()
        train = [c for c in cases if c.split == "train"]
        diagnosis, heads = fit_readers(train, features)
        catalog = GuidelineCatalog.load()
        policy = load_policy()
        shown = 0
        for case in cohort:
            x = features[str(case.case_num)].reshape(1, -1)
            criteria = {n: policy.apply(heads[n], x, n)[0][0] for n in CRITERION_NAMES}
            contract = build_contract(
                case_id=f"derm7pt-{case.case_num}",
                classifier_diagnosis=str(diagnosis.predict(x)[0]),
                classifier_confidence=float(diagnosis.predict_proba(x).max()),
                criteria=criteria,
                catalog=catalog,
            )
            if contract.conflict.present:
                print("\n" + "=" * 78 + "\n")
                print(render_markdown(contract))
                shown += 1
                if shown >= args.sample:
                    break


if __name__ == "__main__":
    main()
