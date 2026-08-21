"""Post-hoc comparability re-score: our Derm7pt criteria heads under Kawahara's GROUPED labels.

Our seven criterion heads are fit and evaluated on the fine Derm7pt vocabulary
(``medguard_v2.data.derm7pt.CRITERION_VOCABULARY`` -- e.g. ``vascular_structures`` has 8 classes).
Kawahara et al. 2019 (the dataset paper) and DermFormer 2025 report per-criterion accuracy on
COARSER, GROUPED labels (Table I's ``abbrev`` column -- e.g. ``vascular_structures`` grouped to
ABS/REG/IR, 3 classes). The two numbers are not the same axis, so a bare accuracy comparison is
apples-to-oranges. This script re-scores our EXISTING test-set predictions under Kawahara's exact
grouping so the numbers land on a comparable axis. Nothing is refit, tuned, or selected here: the
heads are the frozen promoted/base configurations, fit on TRAIN exactly as they always are; the
only new step is a coarser label MAP applied at scoring time, and a companion sensitivity check
(one-vs-rest AUROC of the melanoma-indicative grouped label, using the SUMMED member-class
probability as the ranking score -- the axis Wang et al. Table I and Kawahara Table III report).

This is intentionally a CONSERVATIVE comparison for us: Kawahara/DermFormer/Wang train their heads
DIRECTLY on the grouped task, so their decision boundary is optimized for exactly that split. Ours
is optimized for the FINE split and only coarsened post-hoc by grouping the argmax fine prediction
-- a handicap relative to training on the grouped task directly, not an advantage.

Kawahara Table I grouping (papers/raw_texts/141.txt ll. 434-638), verbatim:
  pigment_network        ABS=absent, TYP=typical, ATP=atypical                (unchanged, 3)
  blue_whitish_veil      ABS=absent, PRS=present                              (unchanged, 2)
  streaks                ABS=absent, REG=regular, IR=irregular                (unchanged, 3)
  dots_and_globules      ABS=absent, REG=regular, IR=irregular                (unchanged, 3)
  vascular_structures    ABS={absent}; REG={arborizing,comma,hairpin,within regression,wreath};
                          IR={dotted,linear irregular}                        (8 -> 3)
  pigmentation           ABS={absent}; REG={diffuse regular,localized regular};
                          IR={diffuse irregular,localized irregular}          (5 -> 3)
  regression_structures  ABS={absent}; PRS={blue areas,white areas,combinations}  (4 -> 2)

Feature/fit recipe is REUSED, not reimplemented: ``derm7pt_p2_safety_promote.Features`` and
``derm7pt_p2_safety_promote._fit_heads`` (imported below) are the objects the task named for this
reuse. The sealed fine-grained numbers this script's wiring self-check reproduces were actually
written by ``build_promoted_management_and_score.py``, which independently re-derives the same
recipe via ``run_cohort.fit_readers_promoted``/``fit_readers``; the two code paths were verified
bit-identical here (``np.array_equal`` on the fitted feature matrices) before being relied on. The
promoted criteria arm is ``dermlip_tta+metadata14``, ``lr_balanced``, per-head C from
``perception_phase2_safety_preserving.json -> final_config.criteria.C_per_head``
(== ``medguard_v2.pipeline.run_cohort.PROMOTED_CRITERIA_C``). The base (pre-promotion) arm is the
driver's "current production" criteria arm (``perception_phase2_safety_preserving.json ->
criteria_comparison.checklist_only_classifier_independent.current_criteria``): plain ``dermlip``
(512-d), unbalanced LR C=10.0 -- identical to ``run_cohort.fit_readers``'s heads.

Before any grouped number is emitted, this script refits both arms and checks the FINE-grained
per-criterion accuracy against ``paper_metrics_summary.json ->
perception_derm7pt.per_criterion_summary[_base_arm_reference]`` (tol 0.0006). If that fails, it
refuses to write output -- a wiring drift must be reported, not silently absorbed into a "fix".

Writes:
  * research/evidence/derm7pt_v1/criteria_regrouped_kawahara.json

Run::
    py -3.13 scripts/dev/build_criteria_regrouped_kawahara.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from medguard_v2.criteria.schema import CRITERION_NAMES  # noqa: E402
from medguard_v2.data.derm7pt import CRITERION_VOCABULARY, load_cases  # noqa: E402
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci  # noqa: E402

import derm7pt_p2_safety_promote as driver  # noqa: E402  (reuse Features + _fit_heads; no copy-paste)

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT = EVIDENCE / "criteria_regrouped_kawahara.json"

N_BOOT = 1000
BOOT_SEED = 20260712          # project-wide bootstrap-seed convention (see e.g. test_metrics_complete.json)
TOL = 0.0006                   # same tolerance the driver's own wiring self-check uses

# --------------------------------------------------------------------------------------------
# Kawahara Table I grouping (papers/raw_texts/141.txt ll. 434-638), verbatim. Every fine label in
# CRITERION_VOCABULARY must appear here exactly once -- checked at import time below.
# --------------------------------------------------------------------------------------------
KAWAHARA_GROUP: dict[str, dict[str, str]] = {
    "pigment_network": {
        "absent": "ABS", "typical": "TYP", "atypical": "ATP",
    },
    "blue_whitish_veil": {
        "absent": "ABS", "present": "PRS",
    },
    "streaks": {
        "absent": "ABS", "regular": "REG", "irregular": "IR",
    },
    "dots_and_globules": {
        "absent": "ABS", "regular": "REG", "irregular": "IR",
    },
    "vascular_structures": {
        "absent": "ABS",
        "arborizing": "REG", "comma": "REG", "hairpin": "REG",
        "within regression": "REG", "wreath": "REG",
        "dotted": "IR", "linear irregular": "IR",
    },
    "pigmentation": {
        "absent": "ABS",
        "diffuse regular": "REG", "localized regular": "REG",
        "diffuse irregular": "IR", "localized irregular": "IR",
    },
    "regression_structures": {
        "absent": "ABS",
        "blue areas": "PRS", "white areas": "PRS", "combinations": "PRS",
    },
}

GROUP_ORDER: dict[str, tuple[str, ...]] = {
    "pigment_network": ("ABS", "TYP", "ATP"),
    "blue_whitish_veil": ("ABS", "PRS"),
    "streaks": ("ABS", "REG", "IR"),
    "dots_and_globules": ("ABS", "REG", "IR"),
    "vascular_structures": ("ABS", "REG", "IR"),
    "pigmentation": ("ABS", "REG", "IR"),
    "regression_structures": ("ABS", "PRS"),
}

#: The grouped label that scores in the classic 7-point rule (Argenziano/Kawahara) -- the "positive"
#: class for the one-vs-rest AUROC axis Wang et al. Table I and Kawahara Table III report.
MELANOMA_INDICATIVE_GROUP: dict[str, str] = {
    "pigment_network": "ATP",
    "blue_whitish_veil": "PRS",
    "vascular_structures": "IR",
    "pigmentation": "IR",
    "streaks": "IR",
    "dots_and_globules": "IR",
    "regression_structures": "PRS",
}

CRITERION_ABBREV: dict[str, str] = {
    "pigment_network": "PN",
    "blue_whitish_veil": "BWV",
    "vascular_structures": "VS",
    "streaks": "STR",
    "pigmentation": "PIG",
    "dots_and_globules": "DaG",
    "regression_structures": "RS",
}

#: Criteria whose grouping is a bijection (Kawahara leaves these unchanged: fine == grouped).
UNCHANGED_CRITERIA = frozenset(
    {"pigment_network", "blue_whitish_veil", "streaks", "dots_and_globules"}
)
REGROUPED_CRITERIA = frozenset({"vascular_structures", "pigmentation", "regression_structures"})

# Self-check: the grouping table above must be complete and exactly cover the loader's vocabulary.
for _name in CRITERION_NAMES:
    _fine_vocab = set(CRITERION_VOCABULARY[_name])
    _grouped_vocab = set(KAWAHARA_GROUP[_name])
    if _fine_vocab != _grouped_vocab:
        raise AssertionError(
            f"KAWAHARA_GROUP[{_name!r}] does not match CRITERION_VOCABULARY: "
            f"missing={_fine_vocab - _grouped_vocab} extra={_grouped_vocab - _fine_vocab}"
        )
    _n_groups = len(set(KAWAHARA_GROUP[_name].values()))
    _bijective = _n_groups == len(_fine_vocab)
    if (_name in UNCHANGED_CRITERIA) != _bijective:
        raise AssertionError(f"{_name}: UNCHANGED_CRITERIA membership disagrees with bijectivity")
del _name, _fine_vocab, _grouped_vocab, _n_groups, _bijective

# --------------------------------------------------------------------------------------------
# Sealed fine-grained references this script must reproduce before emitting anything (wiring
# self-check). Source: research/evidence/derm7pt_v1/paper_metrics_summary.json.
# --------------------------------------------------------------------------------------------
REF_PROMOTED_FINE_ACC = {
    "pigment_network": 0.6886, "blue_whitish_veil": 0.8253, "vascular_structures": 0.6835,
    "streaks": 0.7342, "pigmentation": 0.643, "dots_and_globules": 0.6203,
    "regression_structures": 0.6861,
}
REF_BASE_FINE_ACC = {
    "pigment_network": 0.7013, "blue_whitish_veil": 0.8608, "vascular_structures": 0.8051,
    "streaks": 0.7342, "pigmentation": 0.6658, "dots_and_globules": 0.6076,
    "regression_structures": 0.7519,
}
#: DIAG (5-class) accuracy is NOT refit by this script (the criteria heads never touch the
#: diagnosis classifier) -- read verbatim from paper_metrics_summary.json for the 8-task means.
REF_PROMOTED_DIAG_ARGMAX_ACC = 0.762
REF_PROMOTED_DIAG_OP_ACC = 0.7696
REF_BASE_DIAG_ACC = 0.7392

#: KNOWN, FORENSICALLY-DIAGNOSED non-reproducibility (2026-08-15). Investigated live during this
#: script's development, not assumed:
#:   1. Feature arrays: driver.Features().tta_meta_mat(train/test) vs run_cohort.py's own
#:      _promoted_feature_matrices() were compared with np.array_equal -- BIT-IDENTICAL.
#:   2. Fit determinism: refitting twice in-process, forcing single-threaded BLAS
#:      (OMP/OPENBLAS/MKL_NUM_THREADS=1), and swapping the solver entirely (lbfgs -> newton-cg)
#:      all agree with EACH OTHER (271/395 = 0.6861) but not with the sealed 272/395 = 0.6886.
#:   3. Decisive test: importing build_promoted_management_and_score.py (the script that actually
#:      wrote the sealed paper_metrics_summary.json numbers) and calling its own build() fresh, in
#:      this same environment, reproduces 13/14 fine-grained accuracies EXACTLY -- and *also*
#:      gives 0.6861 for pigment_network promoted, not its own previously-sealed 0.6886. The
#:      feature .npz caches are untouched since an earlier commit (git log confirms), ruling out a
#:      stale-cache explanation.
#: Conclusion: this is a pre-existing gap in the sealed artifact's OWN bit-reproducibility (most
#: plausibly lbfgs converging to a very slightly different optimum across independent process
#: launches -- pigment_network's closest test-case decision margin is 0.0018, the tightest of any
#: criterion, so it is the one head where that could flip a prediction), not a defect in this
#: script's reuse of the driver, and not something introduced by this task. Reported upstream via
#: spawn_task; NOT silently patched -- the exact diagnosed value is pinned here so this narrow
#: allowance cannot mask a DIFFERENT, unexplained future drift on this same cell.
KNOWN_NONREPRODUCIBLE_CELLS: dict[tuple[str, str], float] = {
    ("pigment_network", "promoted"): 0.6861,
}


def _round4(x) -> float:
    return round(float(x), 4)


def _accuracy_of_rows(rows) -> float:
    return sum(1 for p, g in rows if p == g) / len(rows)


def _auroc_of_rows(rows) -> float:
    return binary_auc([bool(r[0]) for r in rows], [float(r[1]) for r in rows])


def _bootstrap_ci(rows, cluster_ids, metric) -> dict:
    interval = clustered_bootstrap_ci(rows, cluster_ids, metric, n_boot=N_BOOT, seed=BOOT_SEED)
    return {
        "estimate": _round4(interval.estimate),
        "ci95_low": _round4(interval.low),
        "ci95_high": _round4(interval.high),
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
    }


def _per_criterion_block(name, pred_fine, classes, proba, gold_fine, cluster_ids) -> dict:
    group_map = KAWAHARA_GROUP[name]
    pred_group = [group_map[p] for p in pred_fine]
    gold_group = [group_map[g] for g in gold_fine]

    fine_accuracy = _round4(_accuracy_of_rows(list(zip(pred_fine, gold_fine))))

    acc_boot = _bootstrap_ci(list(zip(pred_group, gold_group)), cluster_ids, _accuracy_of_rows)
    grouped_accuracy = acc_boot["estimate"]

    order = list(GROUP_ORDER[name])
    macro_f1 = _round4(
        f1_score(gold_group, pred_group, labels=order, average="macro", zero_division=0)
    )

    indicative = MELANOMA_INDICATIVE_GROUP[name]
    member_fine = sorted(f for f, g in group_map.items() if g == indicative)
    idxs = [classes.index(f) for f in member_fine if f in classes]
    missing = sorted(f for f in member_fine if f not in classes)
    scores = proba[:, idxs].sum(axis=1).tolist() if idxs else [0.0] * len(pred_fine)
    y_bin = [1 if g == indicative else 0 for g in gold_group]
    auroc_boot = _bootstrap_ci(list(zip(y_bin, scores)), cluster_ids, _auroc_of_rows)

    unchanged = name in UNCHANGED_CRITERIA
    if unchanged:
        if abs(grouped_accuracy - fine_accuracy) > 1e-9:
            raise AssertionError(
                f"{name}: unchanged criterion but grouped accuracy {grouped_accuracy} != "
                f"fine accuracy {fine_accuracy}"
            )
    elif grouped_accuracy < fine_accuracy - 1e-9:
        raise AssertionError(
            f"{name}: grouped accuracy {grouped_accuracy} < fine accuracy {fine_accuracy} "
            "(coarsening a correct fine prediction can never become incorrect -- this should be "
            "mathematically impossible; something is wrong with the grouping map or predictions)"
        )

    return {
        "kawahara_abbreviation": CRITERION_ABBREV[name],
        "n_classes_fine_full_vocabulary": len(CRITERION_VOCABULARY[name]),
        "n_classes_fine_observed_in_test": len(set(gold_fine) | set(pred_fine)),
        "n_classes_grouped": len(order),
        "grouped_labels_order": order,
        "melanoma_indicative_grouped_label": indicative,
        "fine_accuracy": fine_accuracy,
        "grouped_accuracy": grouped_accuracy,
        "grouped_accuracy_ci95": acc_boot,
        "grouped_accuracy_equals_fine_accuracy": unchanged,
        "grouped_accuracy_gte_fine_accuracy": bool(grouped_accuracy >= fine_accuracy - 1e-9),
        "grouped_macro_f1": macro_f1,
        "grouped_macro_f1_note": "macro-F1; DermFormer's exact F-score definition unverified",
        "melanoma_indicative_auroc": auroc_boot["estimate"],
        "melanoma_indicative_auroc_ci95": auroc_boot,
        "melanoma_indicative_auroc_score_definition": (
            "sum of the head's predicted probability over the member fine classes "
            f"{member_fine} (one-vs-rest on the gold grouped label)"
            + (f"; classes never observed in TRAIN, absent from the head's output: {missing}"
               if missing else "")
        ),
    }


# --------------------------------------------------------------------------------------------
# Comparators copied from the papers (task-supplied / independently re-verified below).
# --------------------------------------------------------------------------------------------
def _kawahara_table_iii_comparators() -> dict:
    # papers/raw_texts/141.txt ll. 1027-1112: Table III, experiment "x-combine", the "auroc"
    # sub-row (ll. 1092-1111, 20 values). Column order confirmed from the header (ll. 838-856):
    # BWV[ABS,PRS], DaG[ABS,REG,IR], PIG[ABS,REG,IR], PN[ABS,TYP,ATP], RS[ABS,PRS],
    # STR[ABS,REG,IR], VS[ABS,REG,IR], Avg. -- and cross-checked by the ABS==PRS / ABS==PRS
    # symmetry that a 2-class AUROC must show for BWV and RS, which holds exactly in this row
    # (89.2/89.2 and 82.9/82.9) and in the two other experiment rows read alongside it
    # (x-unbalanced 87.0/87.0, 79.9/79.9; x-balanced -- not needed, same pattern visible).
    all_19 = {
        "BWV_ABS": 89.2, "BWV_PRS": 89.2,
        "DaG_ABS": 74.1, "DaG_REG": 76.5, "DaG_IR": 79.9,
        "PIG_ABS": 79.0, "PIG_REG": 74.9, "PIG_IR": 79.0,
        "PN_ABS": 89.9, "PN_TYP": 84.2, "PN_ATP": 79.9,
        "RS_ABS": 82.9, "RS_PRS": 82.9,
        "STR_ABS": 86.1, "STR_REG": 87.0, "STR_IR": 78.9,
        "VS_ABS": 86.2, "VS_REG": 85.5, "VS_IR": 76.1,
    }
    return {
        "source": (
            "papers/raw_texts/141.txt ll. 1027-1112 (Table III, experiment x-combine, auroc "
            "sub-row ll. 1092-1111); units are percent (0-100), matching the paper"
        ),
        "recoverable": True,
        "melanoma_indicative_label_auroc": {
            "PN_ATP": all_19["PN_ATP"], "BWV_PRS": all_19["BWV_PRS"], "VS_IR": all_19["VS_IR"],
            "PIG_IR": all_19["PIG_IR"], "STR_IR": all_19["STR_IR"], "DaG_IR": all_19["DaG_IR"],
            "RS_PRS": all_19["RS_PRS"],
        },
        "all_19_label_columns_auroc": all_19,
        "paper_reported_avg_across_19_label_columns": 82.2,
        "avg_note": (
            "the paper's own Avg. column for the x-combine auroc row -- averaged over all 19 "
            "label columns (not just the 7 melanoma-indicative ones quoted above)"
        ),
    }


def _comparators() -> dict:
    kawahara_ii = {
        "BWV": 87.1, "DaG": 60.0, "PIG": 66.1, "PN": 70.9, "RS": 77.2, "STR": 74.2, "VS": 79.7,
        "DIAG": 74.2, "avg": 73.7,
    }
    seven_only = _round4(np.mean([kawahara_ii[k] for k in ("BWV", "DaG", "PIG", "PN", "RS", "STR", "VS")]))
    return {
        "kawahara_2019_table_ii_accuracy_x_combine": {
            "source": "papers/raw_texts/141.txt ll. 699-704 (Table II, x-combine row); units percent",
            **kawahara_ii,
            "avg_definition": "paper's own avg.: mean over the 7 criteria + DIAG (8-task mean)",
            "seven_criteria_only_mean": seven_only,
        },
        "kawahara_2019_table_iii_auroc_x_combine": _kawahara_table_iii_comparators(),
        "dermformer_2025_table4_accuracy": {
            "source": "task-supplied (DermFormer Table 4); units fraction [0,1]",
            "DIAG": 0.762, "PN": 0.739, "BWV": 0.924, "VS": 0.828, "PIG": 0.716, "STR": 0.767,
            "DaG": 0.686, "RS": 0.810, "AVG": 0.779,
            "f_score_DIAG": 0.642, "f_score_AVG": 0.684,
            "avg_definition": "paper's own AVG.: mean over DIAG + 7 criteria (8-task mean)",
        },
        "wang_et_al_table_i_auc": {
            "source": "task-supplied (Wang et al. Table I); units percent; melanoma-indicative "
                      "one-vs-rest AUC per criterion (same axis as melanoma_indicative_auroc "
                      "above), MEL = their diagnosis AUC",
            "MEL": 92.0, "PN": 86.7, "STR": 90.1, "PIG": 88.3, "RS": 89.3, "DaG": 86.2,
            "BWV": 93.4, "VS": 83.0, "avg": 88.6,
        },
    }


def build() -> dict:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    if len(test) != 395:
        raise AssertionError(f"expected 395 test cases, got {len(test)}")
    cluster_ids = [str(c.case_num) for c in test]
    gold_fine = {name: [str(c.criteria[name]) for c in test] for name in CRITERION_NAMES}

    F = driver.Features()

    # Promoted arm: dermlip_tta+metadata14, lr_balanced, per-head C -- final_config.criteria.
    heads_promoted = driver._fit_heads(F.tta_meta_mat, train, driver.PROMOTED_CRITERIA_C, "balanced")
    x_promoted_test = F.tta_meta_mat(test)
    # Base arm: the driver's "current production" criteria arm -- plain dermlip, unbalanced C=10.
    heads_base = driver._fit_heads(F.dermlip_mat, train, 10.0, None)
    x_base_test = F.dermlip_mat(test)

    fine_promoted, fine_base = {}, {}
    for name in CRITERION_NAMES:
        hp, hb = heads_promoted[name], heads_base[name]
        fine_promoted[name] = {
            "pred": [str(p) for p in hp.predict(x_promoted_test)],
            "classes": [str(c) for c in hp.classes_],
            "proba": hp.predict_proba(x_promoted_test),
        }
        fine_base[name] = {
            "pred": [str(p) for p in hb.predict(x_base_test)],
            "classes": [str(c) for c in hb.classes_],
            "proba": hb.predict_proba(x_base_test),
        }

    # ---- wiring self-check: fine-grained accuracy must reproduce the sealed summary ----------
    wiring: dict = {}
    failures: list[str] = []
    for name in CRITERION_NAMES:
        acc_p = _round4(_accuracy_of_rows(list(zip(fine_promoted[name]["pred"], gold_fine[name]))))
        acc_b = _round4(_accuracy_of_rows(list(zip(fine_base[name]["pred"], gold_fine[name]))))
        exp_p, exp_b = REF_PROMOTED_FINE_ACC[name], REF_BASE_FINE_ACC[name]
        ok_p, ok_b = abs(acc_p - exp_p) <= TOL, abs(acc_b - exp_b) <= TOL
        wiring[name] = {
            "promoted_fine_accuracy": {"got": acc_p, "expected": exp_p, "reproduced": bool(ok_p)},
            "base_fine_accuracy": {"got": acc_b, "expected": exp_b, "reproduced": bool(ok_b)},
        }
        if not ok_p:
            known = KNOWN_NONREPRODUCIBLE_CELLS.get((name, "promoted"))
            if known is not None and acc_p == known:
                wiring[name]["promoted_fine_accuracy"]["known_nonreproducible_drift"] = True
                wiring[name]["promoted_fine_accuracy"]["diagnosis"] = (
                    "single flipped test case (271/395 vs sealed 272/395); forensically confirmed "
                    "to be a pre-existing gap in the SEALED ARTIFACT's own bit-reproducibility "
                    "(re-running build_promoted_management_and_score.py's own build() fresh "
                    "reproduces this SAME 0.6861, not its own sealed 0.6886), not a defect in this "
                    "script's feature/fit reuse -- see KNOWN_NONREPRODUCIBLE_CELLS docstring."
                )
            else:
                failures.append(f"{name} promoted fine accuracy got={acc_p} expected={exp_p}")
        if not ok_b:
            failures.append(f"{name} base fine accuracy got={acc_b} expected={exp_b}")

    if failures:
        raise SystemExit(
            "WIRING DRIFT: refit did not reproduce paper_metrics_summary.json's "
            f"per_criterion_summary fine-grained accuracies within tol={TOL}: {failures}\n"
            + json.dumps(wiring, indent=2)
        )

    # ---- per-criterion grouped stats, both arms ----------------------------------------------
    per_criterion_promoted, per_criterion_base = {}, {}
    for name in CRITERION_NAMES:
        per_criterion_promoted[name] = _per_criterion_block(
            name, fine_promoted[name]["pred"], fine_promoted[name]["classes"],
            fine_promoted[name]["proba"], gold_fine[name], cluster_ids,
        )
        per_criterion_base[name] = _per_criterion_block(
            name, fine_base[name]["pred"], fine_base[name]["classes"],
            fine_base[name]["proba"], gold_fine[name], cluster_ids,
        )

    seven_p = [per_criterion_promoted[n]["grouped_accuracy"] for n in CRITERION_NAMES]
    seven_b = [per_criterion_base[n]["grouped_accuracy"] for n in CRITERION_NAMES]

    eight_task_mean = {
        "promoted": {
            "seven_criteria_only_mean": _round4(np.mean(seven_p)),
            "eight_task_mean_diag_argmax": _round4(np.mean(seven_p + [REF_PROMOTED_DIAG_ARGMAX_ACC])),
            "eight_task_mean_diag_operating_point": _round4(
                np.mean(seven_p + [REF_PROMOTED_DIAG_OP_ACC])
            ),
            "diag_accuracy_source": (
                "paper_metrics_summary.json -> perception_derm7pt.classifier_5class "
                "(argmax 0.762, operating_point 0.7696); not refit here"
            ),
        },
        "base": {
            "seven_criteria_only_mean": _round4(np.mean(seven_b)),
            "eight_task_mean": _round4(np.mean(seven_b + [REF_BASE_DIAG_ACC])),
            "diag_accuracy_source": (
                "paper_metrics_summary.json -> perception_derm7pt.classifier_5class."
                "base_arm_reference (0.7392); not refit here"
            ),
        },
    }

    protocol = {
        "analysis_type": (
            "post-hoc comparability re-score (coarser label map on existing test predictions; "
            "no fitting or threshold tuning on test)"
        ),
        "kawahara_table_i_grouping_source": "papers/raw_texts/141.txt ll. 434-638 (Table I), verbatim",
        "grouping_fine_to_group": KAWAHARA_GROUP,
        "group_order": {n: list(GROUP_ORDER[n]) for n in CRITERION_NAMES},
        "melanoma_indicative_grouped_label": MELANOMA_INDICATIVE_GROUP,
        "criterion_abbreviation": CRITERION_ABBREV,
        "unchanged_criteria": sorted(UNCHANGED_CRITERIA),
        "regrouped_criteria": sorted(REGROUPED_CRITERIA),
        "feature_recipe": {
            "promoted": (
                "dermlip_tta+metadata14 (526-d), lr_balanced, per-head C "
                "(perception_phase2_safety_preserving.json -> final_config.criteria.C_per_head)"
            ),
            "promoted_C_per_head": dict(driver.PROMOTED_CRITERIA_C),
            "base": (
                "dermlip (512-d), lr_unbalanced C=10.0 -- the driver's 'current production' "
                "criteria arm (derm7pt_p2_safety_promote.py heads_current == "
                "run_cohort.fit_readers heads)"
            ),
            "fit_split": "train",
            "n_fit_cases": len(train),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": {"fit_seed": driver.SEED, "bootstrap_seed": BOOT_SEED},
        "bootstrap": {
            "n_resamples": N_BOOT,
            "confidence": 0.95,
            "method": (
                "case-level nonparametric bootstrap over the 395 test cases "
                "(medguard_v2.eval.statistics.clustered_bootstrap_ci with one case per cluster, "
                "i.e. plain per-case resampling with replacement)"
            ),
        },
        "n_test": len(test),
        "wiring_self_check": {
            "tolerance": TOL,
            "targets_source": (
                "research/evidence/derm7pt_v1/paper_metrics_summary.json -> "
                "perception_derm7pt.per_criterion_summary[_base_arm_reference]"
            ),
            "per_criterion": wiring,
            "all_reproduced_within_tolerance": 0.0006,
            "all_reproduced_or_diagnosed": True,
            "summary": (
                "13/14 fine-grained accuracies reproduce the sealed reference EXACTLY. The 1 "
                "exception (pigment_network, promoted) is a forensically-diagnosed, pre-existing "
                "non-reproducibility in the SEALED ARTIFACT itself (see "
                "KNOWN_NONREPRODUCIBLE_CELLS in this script), not a defect in this script's reuse "
                "of the driver's feature loading and fitting: features were verified bit-identical "
                "(np.array_equal) to run_cohort.py's own feature construction, and re-running "
                "build_promoted_management_and_score.py's own build() fresh in this environment "
                "reproduces the SAME 0.6861 this script gets, not its own previously-sealed "
                "0.6886. The gap is exactly one flipped test case (271/395 vs 272/395)."
            ),
        },
    }

    return {
        "protocol": protocol,
        "per_criterion": {"promoted": per_criterion_promoted, "base": per_criterion_base},
        "eight_task_mean": eight_task_mean,
        "comparators": _comparators(),
    }


def main() -> None:
    started = time.perf_counter()
    report = build()
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    wsc = report["protocol"]["wiring_self_check"]
    print("wiring self-check all_reproduced_or_diagnosed:", wsc["all_reproduced_or_diagnosed"])
    print("wiring self-check summary:", wsc["summary"])
    print("\n=== per-criterion: fine acc -> grouped acc (grouped AUROC)  [promoted | base] ===")
    for name in CRITERION_NAMES:
        p = report["per_criterion"]["promoted"][name]
        b = report["per_criterion"]["base"][name]
        print(
            f"  {CRITERION_ABBREV[name]:4s} "
            f"promoted {p['fine_accuracy']:.4f}->{p['grouped_accuracy']:.4f} "
            f"(auroc {p['melanoma_indicative_auroc']:.4f})  |  "
            f"base {b['fine_accuracy']:.4f}->{b['grouped_accuracy']:.4f} "
            f"(auroc {b['melanoma_indicative_auroc']:.4f})"
        )
    etm = report["eight_task_mean"]
    print(f"\n8-task mean (promoted, DIAG argmax) : {etm['promoted']['eight_task_mean_diag_argmax']}")
    print(f"8-task mean (promoted, DIAG op)     : {etm['promoted']['eight_task_mean_diag_operating_point']}")
    print(f"7-criteria-only mean (promoted)     : {etm['promoted']['seven_criteria_only_mean']}")
    print(f"8-task mean (base)                  : {etm['base']['eight_task_mean']}")
    print(f"7-criteria-only mean (base)          : {etm['base']['seven_criteria_only_mean']}")
    print(f"\ndone in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
