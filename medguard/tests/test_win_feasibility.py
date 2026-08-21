from __future__ import annotations

import json

from medguard.eval.win_feasibility_v1 import build_report


def test_shared_perception_floor_is_the_tie_every_arm_inherits():
    report = build_report()
    floor = report["shared_perception_floor"]
    # 122/160 correct commitments; this is the number no arm can beat on diagnosis.
    assert abs(floor["accuracy"] - 122 / 160) < 1e-9
    assert "cannot be a head-to-head win" in floor["note"]


def test_axis_a_confirms_selective_signal_but_keeps_the_shared_curve_caveat():
    report = build_report()
    axis = report["axes"]["A_selective_prediction_aurc"]
    assert axis["verdict"] == "GO"
    # The confidence must beat a shuffled-confidence floor, or abstention is pointless.
    assert axis["aurc_observed"] < axis["aurc_shuffled_confidence_mean"] - 0.01
    # Abstaining on the low-confidence half must genuinely raise accuracy.
    assert axis["headroom_selacc50_minus_overall"] >= 0.05
    curve = axis["selective_accuracy_by_coverage"]
    assert curve["selective_accuracy_at_20pct_coverage"] >= curve["selective_accuracy_at_100pct_coverage"]
    # The caveat that makes this honest: the curve is shared across arms.
    assert "shared by every arm" in axis["reading"]


def test_axis_b_over_reliance_is_net_protective_on_the_asymmetry_not_the_gap():
    report = build_report()
    axis = report["axes"]["B_over_reliance_headline"]
    assert axis["verdict"] in ("GO", "GO-CONDITIONAL")
    # The robust signal is that conflicting exemplar evidence is more often wrong
    # than right when it disagrees with a correct commitment.
    assert axis["protected_correct_cases"] > axis["sacrificed_recoverable_cases"]
    assert axis["net_protection"] == (
        axis["protected_correct_cases"] - axis["sacrificed_recoverable_cases"])
    # The attack surface must be non-trivial for the immunity to be worth claiming.
    assert axis["attack_surface_fraction"] >= 0.15
    # The honest limit: this bounds the image channel; the text channel needs generation.
    assert "text channel" in axis["reading"]


def test_axis_c_provenance_is_exclusive_and_labelled_as_such():
    report = build_report()
    axis = report["axes"]["C_provenance_exclusive"]
    assert axis["medguard_source_exact_capable"] == 92
    assert axis["baseline_source_exact_capable"] == 0
    # Exclusive axes differentiate but cannot be a fair standalone win.
    assert "cannot be a fair head-to-head" in axis["reading"]


def test_report_declares_dev_label_use_and_no_generation():
    report = build_report()
    boundaries = report["boundaries"]
    assert boundaries["generation_calls"] == 0
    assert boundaries["sealed_test_access"] is False
    # This analysis legitimately uses development labels; it must say so, not hide it.
    assert boundaries["development_labels_used"] is True


def test_overall_recommendation_is_earned_not_assumed():
    report = build_report()
    overall = report["overall"]
    assert overall["recommendation"].startswith("PROCEED")
    # The recommendation must not be read as "the paper is already won".
    assert "does NOT establish the arm-to-arm deltas" in overall["what_this_does_not_prove"]


def test_report_is_byte_reproducible():
    a = json.dumps(build_report(), sort_keys=True)
    b = json.dumps(build_report(), sort_keys=True)
    assert a == b
