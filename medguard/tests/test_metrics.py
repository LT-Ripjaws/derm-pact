"""Hand-computed checks for diagnostic, selective, and evidence metrics."""
from medguard.config import MALIGNANT_LABELS
from medguard.eval.metrics import (
    classification_summary, confusion_matrix, row_normalize,
    ece, aurc, risk_coverage_curve, selective_accuracy_at_coverage,
    faithfulness, over_reliance, copy_reference_ratio, completeness,
    acuity_ordinal, quadratic_weighted_kappa, triage_rates,
    actionable_confusion, is_actionable, sensitivity_at_fixed_specificity,
)


def test_multiclass_confusion_matrix_and_summary():
    labels = ("melanoma", "bcc", "nevus")
    truth = ["melanoma", "melanoma", "bcc", "nevus"]
    pred = ["melanoma", "nevus", "bcc", "nevus"]
    matrix = confusion_matrix(truth, pred, labels)
    assert matrix == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    assert row_normalize(matrix)[0] == [0.5, 0.0, 0.5]
    summary = classification_summary(truth, pred, labels)
    assert summary["accuracy"] == 0.75
    assert summary["per_class"]["melanoma"]["recall"] == 0.5


def test_confusion_matrix_rejects_unknown_and_mismatched_labels():
    for truth, pred in [(["melanoma"], ["other"]), (["melanoma"], [])]:
        try:
            confusion_matrix(truth, pred, ("melanoma", "nevus"))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid confusion-matrix input was accepted")


def test_ece_calibration():
    assert ece([0.5, 0.5], [True, False]) == 0.0          # acc 0.5 == conf 0.5
    assert abs(ece([0.9, 0.9], [True, True]) - 0.1) < 1e-6  # acc 1.0 vs conf 0.9


def test_risk_coverage_and_aurc():
    curve = risk_coverage_curve([0.9, 0.8, 0.7], [True, True, False])
    assert curve == [(0.3333, 0.0), (0.6667, 0.0), (1.0, 0.3333)]
    assert abs(aurc([0.9, 0.8, 0.7], [True, True, False]) - 0.1111) < 1e-3
    assert selective_accuracy_at_coverage([0.9, 0.1], [True, False], coverage=0.5) == 1.0


def test_faithfulness_and_over_reliance():
    ans = "melanoma [1] requires biopsy"
    ch = [{"text": "melanoma requires biopsy, it is malignant"}]
    assert faithfulness([ans], [ch]) >= 0.9
    assert over_reliance([True, True, False], [False, True, False]) == round(1 / 3, 4)


def test_copy_ref_and_completeness():
    assert copy_reference_ratio("a b c d e f", [{"text": "a b c d e f g"}]) == 1.0
    assert copy_reference_ratio("x y z p q r", [{"text": "no overlap here at all"}]) == 0.0
    assert abs(completeness("melanoma malignant", [{"text": "melanoma malignant biopsy"}]) - 0.6667) < 1e-3


def test_qwk_and_acuity_ordinal():
    assert acuity_ordinal("melanoma") == 2 and acuity_ordinal("akiec") == 1 and acuity_ordinal("nevus") == 0
    assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) == 1.0        # perfect agreement
    assert quadratic_weighted_kappa([0, 0, 2], [2, 2, 0]) < 0.0        # systematic disagreement


def test_triage_rates_flags_missed_cancer():
    r = triage_rates(pred_labels=["melanoma", "nevus", "melanoma"],
                     true_labels=["melanoma", "melanoma", "nevus"])
    assert r["under_triage"] == 0.5          # 1 of 2 malignant missed
    assert r["over_triage"] == 1.0           # 1 of 1 benign over-called
    assert r["high_risk_detection"] == 0.5


def test_actionable_confusion_matrix_and_rates():
    true = ["melanoma", "nevus", "bcc", "bkl", "akiec"]
    pred = ["melanoma", "melanoma", "nevus", "bkl", "akiec"]
    out = actionable_confusion(pred, true)
    #                 pred_act pred_ben
    #  true_act (TP=2)   2        1  (FN: bcc called nevus)
    #  true_ben (FP=1)   1        1  (FP: nevus called melanoma)
    assert out["confusion_matrix"] == [[2, 1], [1, 1]]
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (2, 1, 1, 1)
    assert out["sensitivity"] == 0.6667 and out["specificity"] == 0.5
    assert out["ppv"] == 0.6667 and out["npv"] == 0.5
    assert out["false_negative_rate"] == 0.3333 and out["accuracy"] == 0.6
    assert out["auroc"] is None                     # no scores supplied


def test_actionable_undefined_rates_are_none_not_zero():
    out = actionable_confusion(["nevus", "melanoma"], ["nevus", "bkl"])
    assert out["sensitivity"] is None and out["false_negative_rate"] is None
    assert out["prevalence"] == 0.0                 # zero positives, but a real denominator
    assert out["specificity"] == 0.5 and out["npv"] == 1.0
    assert out["auroc"] is None                     # single class -> undefined


def test_actionable_auroc_uses_the_actionable_score():
    out = actionable_confusion(
        ["melanoma", "nevus", "bcc", "nevus"], ["melanoma", "nevus", "bcc", "nevus"],
        scores=[0.9, 0.2, 0.8, 0.1])
    assert out["auroc"] == 1.0                       # positives rank above negatives


def test_actionable_positive_set_is_configurable_for_bcn_scc():
    assert not is_actionable("scc")                                  # benign under HAM7 set
    assert is_actionable("scc", MALIGNANT_LABELS | {"scc"})          # actionable for BCN
    default = actionable_confusion(["scc"], ["scc"])
    assert default["tn"] == 1 and default["tp"] == 0                 # scc benign by default
    bcn = actionable_confusion(["scc"], ["scc"], positive_labels=MALIGNANT_LABELS | {"scc"})
    assert bcn["tp"] == 1


def test_sensitivity_at_fixed_specificity_picks_highest_sensitivity_point():
    truth = [True, True, False, False]
    scores = [0.9, 0.6, 0.4, 0.1]
    at_100 = sensitivity_at_fixed_specificity(truth, scores, 1.0)
    assert at_100["sensitivity"] == 1.0 and at_100["specificity"] == 1.0
    assert at_100["threshold"] == 0.6
    at_50 = sensitivity_at_fixed_specificity(truth, scores, 0.5)
    assert at_50["sensitivity"] == 1.0 and at_50["specificity"] == 0.5


def test_sensitivity_at_fixed_specificity_requires_both_classes():
    assert sensitivity_at_fixed_specificity([True, True], [0.5, 0.6], 0.5) is None
