"""Complete the registered P1 comparison, which the confirmatory runner failed to execute.

`PREREGISTRATION_V2` defines P1 as the AUROC of the **stacked ensemble** against the
diagnosis-classifier-alone arm, with the operating point taken from the registered threshold
0.2981. `medguard_v2/eval/confirmatory.py` computed the classifier's AUROC and threshold statistics
for the criteria path and the hand-written union rule, but never built the meta-feature matrix --
an implementation gap, not a substitution.

This module finishes that analysis on the same fitted readers and the same split, using the
registered construction (`g6.build_meta_features`), the registered meta-learner (`C=100.0`) and
the registered threshold. It selects nothing.

**Disclosure that travels with the result:** the union-rule outcome was already observed when this
was computed. That is weaker than producing both blind, and it is why both are reported together
and neither is presented as the headline in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.confirmatory import _append_log, verify_log
from medguard_v2.eval.gates.g6_multiview_stacking import build_meta_features, discover_feature_sets
from medguard_v2.eval.statistics import binary_auc, delong_auc_test

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "confirmatory_p1_completion.json"
CONFIRMATORY = REPO / "research" / "evidence" / "derm7pt_v1" / "confirmatory_test_395.json"

# All registered in PREREGISTRATION_V2 schema 2. None is chosen here.
REGISTERED_VIEW = "dermlip"
REGISTERED_META_C = 100.0
REGISTERED_THRESHOLD = 0.2981
SEED = 20260810


def _stats(flag, truth) -> Dict:
    tp = sum(1 for f, t in zip(flag, truth) if f and t)
    fn = sum(1 for f, t in zip(flag, truth) if not f and t)
    fp = sum(1 for f, t in zip(flag, truth) if f and not t)
    tn = sum(1 for f, t in zip(flag, truth) if not f and not t)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "youden_j": round(sens + spec - 1, 4) if None not in (sens, spec) else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def run() -> Dict:
    state = verify_log()
    if not state["valid"]:
        raise RuntimeError(f"access log broken: {state}")

    grant = _append_log(
        "confirmatory_p1_completion",
        "Completing the registered P1 stacked-ensemble comparison, which confirmatory.py did not "
        "implement. Same fitted readers, same split, registered hyperparameters and threshold. "
        "Disclosed: the union-rule result was already observed when this ran.",
        {"registered_threshold": REGISTERED_THRESHOLD, "meta_C": REGISTERED_META_C},
    )

    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]

    available = discover_feature_sets()
    feature_sets = {REGISTERED_VIEW: available[REGISTERED_VIEW]}

    # Out-of-fold on train, full-train refit for test -- the registered construction.
    x_train, x_test, feature_names = build_meta_features(feature_sets, train, test)

    y_train = ["MEL" if c.diagnosis_group == "MEL" else "other" for c in train]
    meta = LogisticRegression(max_iter=5000, C=REGISTERED_META_C, random_state=SEED)
    meta.fit(x_train, y_train)
    mel_index = list(meta.classes_).index("MEL")
    stacked_score = meta.predict_proba(x_test)[:, mel_index]

    truth_mel = [c.diagnosis_group == "MEL" for c in test]

    # Classifier-alone comparator, refit identically to the confirmatory run.
    from medguard_v2.pipeline.run_cohort import _load_features, fit_readers

    features = _load_features()
    diagnosis, _ = fit_readers(train, features)
    x_raw_test = np.stack([features[str(c.case_num)] for c in test])
    classifier_score = diagnosis.predict_proba(x_raw_test)[
        :, list(diagnosis.classes_).index("MEL")
    ]

    stacked_auroc = binary_auc(truth_mel, [float(s) for s in stacked_score])
    classifier_auroc = binary_auc(truth_mel, [float(s) for s in classifier_score])
    delong = delong_auc_test(
        truth_mel, [float(s) for s in stacked_score], [float(s) for s in classifier_score]
    )

    confirmatory = json.loads(CONFIRMATORY.read_text(encoding="utf-8"))
    union = confirmatory["P1_melanoma_detection"]["union_either_flags"]
    classifier_alone = confirmatory["P1_melanoma_detection"]["classifier_alone"]

    result = {
        "purpose": "registered P1 completion",
        "protocol_id": "medguard-v2-derm7pt-criteria-ensemble-v1",
        "preregistration_schema": 2,
        "split": "test",
        "n_cases": len(test),
        "melanomas": sum(truth_mel),
        "access_grant": {"sequence": grant["sequence"], "entry_sha256": grant["entry_sha256"]},
        "registered_configuration": {
            "view": REGISTERED_VIEW,
            "meta_C": REGISTERED_META_C,
            "threshold": REGISTERED_THRESHOLD,
            "meta_feature_count": len(feature_names),
            "base_predictions": "out-of-fold on train",
        },
        "P1_registered_comparison": {
            "stacked_auroc": round(stacked_auroc, 4),
            "classifier_alone_auroc": round(classifier_auroc, 4),
            "auroc_difference": round(stacked_auroc - classifier_auroc, 4),
            "delong": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in delong.items()
            }
            if isinstance(delong, dict)
            else delong,
            "stacked_at_registered_threshold": _stats(
                [s >= REGISTERED_THRESHOLD for s in stacked_score], truth_mel
            ),
        },
        "reported_alongside_from_the_confirmatory_run": {
            "classifier_alone": classifier_alone,
            "hand_written_union": union,
        },
        "disclosure": (
            "The confirmatory runner did not implement the registered stacked comparison; this "
            "module completes it on the same fitted readers and split. The union-rule outcome was "
            "already observed when this was computed, so the two are reported together and "
            "neither is presented as a blind result. The comparator's test metrics were also "
            "visible during design (see test_split_exposure in the preregistration)."
        ),
    }
    return result


def main() -> None:
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    print(json.dumps(result["P1_registered_comparison"], indent=2))


if __name__ == "__main__":
    main()
