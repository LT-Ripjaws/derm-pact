"""G6 -- learn the ensemble instead of hand-writing it.

Everything combined so far was combined by a rule I wrote: "escalate if either the classifier or
the criteria path flags melanoma".  That rule scored sensitivity 0.7049 at specificity 0.8592, but
it is one point on a surface nobody searched.  G6 replaces it with a stacked model over every view
available, so the combination is fitted rather than asserted.

Three things this must get right, because each has already bitten this project once.

**Stacking leakage.**  A meta-learner trained on base predictions that the base models have already
memorised learns to over-trust them.  Every base prediction feeding the meta-layer is therefore
*out-of-fold* on train.  In-sample base predictions would produce a beautiful, worthless number.

**Selection on the evaluation set.**  All model choice -- base C, meta C, which views to include --
happens inside train via cross-validation.  ``valid`` is scored once at the end for reporting.  The
test split is never opened.

**Feature leakage from expert annotation.**  ``level_of_diagnostic_difficulty`` is an expert
judgement about the case; it is NOT available at inference time and is excluded from the feature
set.  It is used only to stratify the report.  ``sex``, ``location`` and ``elevation`` are genuine
patient metadata and may be used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from medguard_v2.criteria.aggregate import MANAGEMENT_LADDER, score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import classification_summary, quadratic_weighted_kappa
from medguard_v2.eval.statistics import binary_auc

REPO = Path(__file__).resolve().parents[3]
FEATURE_ROOT = REPO / "research" / "evidence" / "derm7pt_v1" / "features"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "g6_multiview_stacking.json"

SEED = 20260810
N_FOLDS = 5
C_GRID = (0.1, 1.0, 10.0, 100.0)
PATIENT_METADATA = ("sex", "location", "elevation")


# --------------------------------------------------------------------------- data


def discover_feature_sets() -> Dict[str, Path]:
    """Whatever has been encoded so far. Runs with one view or with all four."""
    found = {}
    if FEATURE_ROOT.exists():
        for child in sorted(FEATURE_ROOT.iterdir()):
            npz = child / "features.npz"
            if npz.exists():
                found[child.name] = npz
    if not found:
        raise FileNotFoundError(f"no cached feature sets under {FEATURE_ROOT}")
    return found


def load_feature_matrix(npz: Path, case_nums: Sequence[int]) -> np.ndarray:
    data = np.load(npz)
    index = {str(c): i for i, c in enumerate(data["case_num"])}
    features = data["features"]
    missing = [c for c in case_nums if str(c) not in index]
    if missing:
        raise KeyError(f"{npz.parent.name}: missing {len(missing)} cases, e.g. {missing[:3]}")
    return np.stack([features[index[str(c)]] for c in case_nums])


# ----------------------------------------------------------------- base layer


def _select_c(x: np.ndarray, y: Sequence[str]) -> float:
    counts = Counter(y)
    n_splits = min(N_FOLDS, min(counts.values()))
    if n_splits < 2:
        return 10.0
    y_arr = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    best_c, best = 10.0, -1.0
    for c in C_GRID:
        correct = 0
        for tr, va in folds.split(x, y_arr):
            model = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
            model.fit(x[tr], y_arr[tr])
            correct += int((model.predict(x[va]) == y_arr[va]).sum())
        score = correct / len(y_arr)
        if score > best:
            best_c, best = c, score
    return best_c


def oof_and_full(
    x_train: np.ndarray, y_train: Sequence[str], x_eval: np.ndarray, c: float
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Out-of-fold posteriors on train, plus posteriors on eval from a full-train refit.

    The OOF half is what makes stacking honest: the meta-learner only ever sees base predictions
    for cases the base model did not train on.
    """
    y_arr = np.asarray(y_train)
    classes = sorted(set(y_arr))
    counts = Counter(y_arr)
    n_splits = min(N_FOLDS, min(counts.values()))

    oof = np.zeros((len(y_arr), len(classes)), dtype=float)
    if n_splits >= 2:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for tr, va in folds.split(x_train, y_arr):
            model = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
            model.fit(x_train[tr], y_arr[tr])
            proba = model.predict_proba(x_train[va])
            for j, cls in enumerate(model.classes_):
                oof[va, classes.index(cls)] = proba[:, j]
    else:  # degenerate class: fall back to the prior
        oof[:, :] = 1.0 / len(classes)

    full = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
    full.fit(x_train, y_arr)
    proba_eval = full.predict_proba(x_eval)
    eval_matrix = np.zeros((x_eval.shape[0], len(classes)), dtype=float)
    for j, cls in enumerate(full.classes_):
        eval_matrix[:, classes.index(cls)] = proba_eval[:, j]

    return oof, eval_matrix, classes


# ------------------------------------------------------------- meta features


def build_meta_features(
    feature_sets: Dict[str, Path], train, evaluate
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble stacked features: per-view diagnosis and criterion posteriors, plus
    cross-view agreement flags, the derived 7-point score, and patient metadata."""
    train_nums = [c.case_num for c in train]
    eval_nums = [c.case_num for c in evaluate]

    blocks_train: List[np.ndarray] = []
    blocks_eval: List[np.ndarray] = []
    names: List[str] = []

    diag_pred_train: Dict[str, np.ndarray] = {}
    diag_pred_eval: Dict[str, np.ndarray] = {}
    crit_pred_train: Dict[str, Dict[str, np.ndarray]] = {}
    crit_pred_eval: Dict[str, Dict[str, np.ndarray]] = {}

    for view, npz in feature_sets.items():
        x_tr = load_feature_matrix(npz, train_nums)
        x_ev = load_feature_matrix(npz, eval_nums)

        y = [c.diagnosis_group for c in train]
        c_sel = _select_c(x_tr, y)
        oof, ev, classes = oof_and_full(x_tr, y, x_ev, c_sel)
        blocks_train.append(oof)
        blocks_eval.append(ev)
        names += [f"{view}:diag:{cls}" for cls in classes]
        diag_pred_train[view] = np.array([classes[i] for i in oof.argmax(axis=1)])
        diag_pred_eval[view] = np.array([classes[i] for i in ev.argmax(axis=1)])

        crit_pred_train[view] = {}
        crit_pred_eval[view] = {}
        for criterion in CRITERION_NAMES:
            yc = [str(c.criteria[criterion]) for c in train]
            c_sel = _select_c(x_tr, yc)
            oof_c, ev_c, classes_c = oof_and_full(x_tr, yc, x_ev, c_sel)
            blocks_train.append(oof_c)
            blocks_eval.append(ev_c)
            names += [f"{view}:{criterion}:{cls}" for cls in classes_c]
            crit_pred_train[view][criterion] = np.array(
                [classes_c[i] for i in oof_c.argmax(axis=1)]
            )
            crit_pred_eval[view][criterion] = np.array(
                [classes_c[i] for i in ev_c.argmax(axis=1)]
            )

    # derived 7-point score per view -- the architecture's own intermediate quantity
    for view in feature_sets:
        for source, pred, holder in (
            ("train", crit_pred_train[view], blocks_train),
            ("eval", crit_pred_eval[view], blocks_eval),
        ):
            n = len(pred[CRITERION_NAMES[0]])
            scores = np.array(
                [
                    score_criteria({k: pred[k][i] for k in CRITERION_NAMES}).score
                    for i in range(n)
                ],
                dtype=float,
            ).reshape(-1, 1)
            holder.append(scores / 10.0)
        names.append(f"{view}:seven_point_score")

    # cross-view agreement flags: the reliability signal measured earlier
    views = list(feature_sets)
    if len(views) > 1:
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                a, b = views[i], views[j]
                blocks_train.append(
                    (diag_pred_train[a] == diag_pred_train[b]).astype(float).reshape(-1, 1)
                )
                blocks_eval.append(
                    (diag_pred_eval[a] == diag_pred_eval[b]).astype(float).reshape(-1, 1)
                )
                names.append(f"agree:diag:{a}~{b}")

                agree_tr = np.mean(
                    [crit_pred_train[a][k] == crit_pred_train[b][k] for k in CRITERION_NAMES],
                    axis=0,
                ).reshape(-1, 1)
                agree_ev = np.mean(
                    [crit_pred_eval[a][k] == crit_pred_eval[b][k] for k in CRITERION_NAMES],
                    axis=0,
                ).reshape(-1, 1)
                blocks_train.append(agree_tr)
                blocks_eval.append(agree_ev)
                names.append(f"agree:criteria_frac:{a}~{b}")

    # patient metadata -- available at inference. NOT difficulty, which is an expert label.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    meta_tr = encoder.fit_transform([[str(getattr(c, f)) for f in PATIENT_METADATA] for c in train])
    meta_ev = encoder.transform([[str(getattr(c, f)) for f in PATIENT_METADATA] for c in evaluate])
    blocks_train.append(meta_tr)
    blocks_eval.append(meta_ev)
    names += [f"meta:{n}" for n in encoder.get_feature_names_out(list(PATIENT_METADATA))]

    return np.hstack(blocks_train), np.hstack(blocks_eval), names


# ------------------------------------------------------------------ reporting


def _melanoma_stats(flag: Sequence[bool], truth: Sequence[bool]) -> Dict:
    tp = sum(1 for f, t in zip(flag, truth) if f and t)
    fn = sum(1 for f, t in zip(flag, truth) if not f and t)
    fp = sum(1 for f, t in zip(flag, truth) if f and not t)
    tn = sum(1 for f, t in zip(flag, truth) if not f and not t)
    return {
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "ppv": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def select_view_subset(all_views: Dict[str, Path], train) -> Tuple[List[str], List[Dict]]:
    """Choose which views to stack, using TRAIN cross-validation only.

    Adding every available view is not free: each contributes ~37 meta-features, and with 413
    training cases the meta-learner overfits long before the views run out.  Using all four
    scored *worse* on valid than using one -- but "worse on valid" is not a selection criterion
    we are allowed to use, so the subset is chosen here by out-of-fold AUROC on train and valid
    is scored once, afterwards, with whatever this picks.
    """
    import itertools

    names = sorted(all_views)
    y = np.asarray(["MEL" if c.diagnosis_group == "MEL" else "other" for c in train])
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    ranking: List[Dict] = []
    for size in range(1, len(names) + 1):
        for combo in itertools.combinations(names, size):
            subset = {k: all_views[k] for k in combo}
            scores = np.zeros(len(train), dtype=float)
            for tr_idx, va_idx in folds.split(np.zeros(len(train)), y):
                fold_train = [train[i] for i in tr_idx]
                fold_eval = [train[i] for i in va_idx]
                x_tr, x_ev, _ = build_meta_features(subset, fold_train, fold_eval)
                y_tr = ["MEL" if c.diagnosis_group == "MEL" else "other" for c in fold_train]
                model = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
                model.fit(x_tr, y_tr)
                scores[va_idx] = model.predict_proba(x_ev)[
                    :, list(model.classes_).index("MEL")
                ]
            ranking.append(
                {
                    "views": list(combo),
                    "n_views": size,
                    "train_oof_auroc": round(
                        binary_auc([v == "MEL" for v in y], [float(s) for s in scores]), 4
                    ),
                }
            )

    ranking.sort(key=lambda r: (-r["train_oof_auroc"], r["n_views"]))
    return ranking[0]["views"], ranking


def run(select_views: bool = True) -> Dict:
    available = discover_feature_sets()
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]

    view_ranking: List[Dict] = []
    if select_views and len(available) > 1:
        chosen, view_ranking = select_view_subset(available, train)
        feature_sets = {k: available[k] for k in chosen}
    else:
        feature_sets = available

    x_train, x_valid, feature_names = build_meta_features(feature_sets, train, valid)

    truth_mel = [c.diagnosis_group == "MEL" for c in valid]
    report: Dict = {
        "gate_id": "G6_multiview_stacking",
        "question": "Does a fitted ensemble over multiple views beat the hand-written union rule?",
        "boundaries": {
            "feature_sets": sorted(feature_sets),
            "views_available": sorted(available),
            "view_selection": "train out-of-fold AUROC" if view_ranking else "all available",
            "view_selection_ranking": view_ranking[:8],
            "base_predictions": "out-of-fold on train (no stacking leakage)",
            "selection": "all C selection inside train via cross-validation",
            "difficulty_used_as_feature": False,
            "difficulty_note": "expert annotation, unavailable at inference; reporting strata only",
            "test_split_read": False,
            "paid_calls": 0,
        },
        "cohort": {"train": len(train), "valid": len(valid), "melanoma_in_valid": sum(truth_mel)},
        "meta_feature_count": len(feature_names),
    }

    # --- melanoma (primary safety endpoint) ------------------------------------
    y_mel_train = ["MEL" if c.diagnosis_group == "MEL" else "other" for c in train]
    c_meta = _select_c(x_train, y_mel_train)
    meta = LogisticRegression(max_iter=5000, C=c_meta, random_state=SEED)
    meta.fit(x_train, y_mel_train)
    mel_index = list(meta.classes_).index("MEL")
    mel_score = meta.predict_proba(x_valid)[:, mel_index]

    # --- operating point, fixed on TRAIN and applied blind to valid ------------
    #
    # The previous version chose the threshold on valid so that sensitivity landed on 0.7049,
    # then reported the resulting specificity. That comparison is convenient and dishonest: the
    # operating point was fitted to the evaluation set. Here the threshold comes from
    # out-of-fold meta-predictions on train and is applied to valid unchanged, so whatever
    # sensitivity valid produces is a genuine out-of-sample result -- it will NOT land exactly
    # on the target, and that gap is itself informative.
    y_mel_arr = np.asarray(y_mel_train)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(len(y_mel_arr), dtype=float)
    for tr_idx, va_idx in folds.split(x_train, y_mel_arr):
        fold_model = LogisticRegression(max_iter=5000, C=c_meta, random_state=SEED)
        fold_model.fit(x_train[tr_idx], y_mel_arr[tr_idx])
        oof_meta[va_idx] = fold_model.predict_proba(x_train[va_idx])[
            :, list(fold_model.classes_).index("MEL")
        ]

    truth_train = [v == "MEL" for v in y_mel_arr]

    def _threshold_for_sensitivity(scores: np.ndarray, truth: Sequence[bool], target: float) -> float:
        """Lowest score at which sensitivity reaches ``target`` on the given data."""
        order = np.argsort(-scores)
        n_pos = sum(truth)
        need = int(np.ceil(target * n_pos))
        hits = 0
        for i in order:
            if truth[i]:
                hits += 1
                if hits >= need:
                    return float(scores[i])
        return float(scores[order[-1]])

    # Pre-declared target: catch 70% of melanomas. Round, clinically motivated, and fixed
    # before looking at valid.
    TARGET_SENSITIVITY = 0.70
    cut = _threshold_for_sensitivity(oof_meta, truth_train, TARGET_SENSITIVITY)

    # Youden's J on the same out-of-fold train scores: a second pre-specifiable rule that
    # optimises no valid quantity.
    best_j, cut_j = -2.0, 0.5
    for candidate in np.unique(oof_meta):
        pred = oof_meta >= candidate
        tp = int(np.sum(pred & np.asarray(truth_train)))
        fn = int(np.sum(~pred & np.asarray(truth_train)))
        fp = int(np.sum(pred & ~np.asarray(truth_train)))
        tn = int(np.sum(~pred & ~np.asarray(truth_train)))
        if (tp + fn) == 0 or (tn + fp) == 0:
            continue
        j = tp / (tp + fn) + tn / (tn + fp) - 1.0
        if j > best_j:
            best_j, cut_j = j, float(candidate)

    report["melanoma"] = {
        "selected_C": c_meta,
        "auroc": round(binary_auc(truth_mel, [float(s) for s in mel_score]), 4),
        "threshold_policy": (
            "fixed on train out-of-fold meta-predictions, applied unchanged to valid"
        ),
        "at_train_fixed_threshold_target_sens_0_70": {
            "target_sensitivity": TARGET_SENSITIVITY,
            "threshold": round(cut, 4),
            "train_oof": _melanoma_stats([s >= cut for s in oof_meta], truth_train),
            "valid": _melanoma_stats([s >= cut for s in mel_score], truth_mel),
        },
        "at_train_fixed_threshold_youden_j": {
            "threshold": round(cut_j, 4),
            "train_oof_youden_j": round(best_j, 4),
            "train_oof": _melanoma_stats([s >= cut_j for s in oof_meta], truth_train),
            "valid": _melanoma_stats([s >= cut_j for s in mel_score], truth_mel),
        },
        "at_argmax_0_5": _melanoma_stats([s >= 0.5 for s in mel_score], truth_mel),
    }

    # --- management (ordinal) ---------------------------------------------------
    #
    # Stacking the meta-features alone REGRESSED management badly against a plain probe on raw
    # features (QWK 0.3846 vs 0.6550). The meta-layer sees only posteriors, and management
    # evidently needs the representation those posteriors discard. Rather than assert a fix,
    # the representation is selected on train by cross-validated quadratic-weighted kappa --
    # the registered P2 statistic -- across three candidates.
    y_mgmt_train = [c.management for c in train]
    ladder_sel = {name: i for i, name in enumerate(MANAGEMENT_LADDER)}
    raw_train = load_feature_matrix(
        feature_sets[sorted(feature_sets)[0]], [c.case_num for c in train]
    )
    raw_valid = load_feature_matrix(
        feature_sets[sorted(feature_sets)[0]], [c.case_num for c in valid]
    )

    candidates = {
        "raw_features": (raw_train, raw_valid),
        "meta_features": (x_train, x_valid),
        "raw_plus_meta": (np.hstack([raw_train, x_train]), np.hstack([raw_valid, x_valid])),
    }

    # Both the representation AND C are selected by out-of-fold quadratic-weighted kappa on
    # train, because QWK is the registered P2 statistic. Selecting the representation on QWK
    # while selecting C on accuracy -- as an earlier version did -- is incoherent, and it is
    # what produced two different "raw-feature probe" management numbers (0.6550 at C=10 chosen
    # by log-loss, 0.4648 at C=100 chosen by accuracy) from the same representation.
    y_arr = np.asarray(y_mgmt_train)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    mgmt_selection = []
    for name, (mat_train, _) in candidates.items():
        for c_try in C_GRID:
            oof = np.empty(len(y_arr), dtype=object)
            for tr_idx, va_idx in folds.split(mat_train, y_arr):
                model = LogisticRegression(max_iter=5000, C=c_try, random_state=SEED)
                model.fit(mat_train[tr_idx], y_arr[tr_idx])
                oof[va_idx] = model.predict(mat_train[va_idx])
            qwk = quadratic_weighted_kappa(
                [ladder_sel[v] for v in y_arr],
                [ladder_sel[v] for v in oof],
                n_classes=len(MANAGEMENT_LADDER),
            )
            mgmt_selection.append(
                {"representation": name, "selected_C": c_try, "train_oof_qwk": round(qwk, 4)}
            )

    # Ties break toward stronger regularisation, fixed in advance.
    mgmt_selection.sort(key=lambda r: (-r["train_oof_qwk"], r["selected_C"]))
    chosen_repr = mgmt_selection[0]["representation"]
    c_mgmt = mgmt_selection[0]["selected_C"]
    mat_train, mat_valid = candidates[chosen_repr]

    mgmt = LogisticRegression(max_iter=5000, C=c_mgmt, random_state=SEED)
    mgmt.fit(mat_train, y_mgmt_train)
    mgmt_pred = list(mgmt.predict(mat_valid))
    y_mgmt_valid = [c.management for c in valid]
    ladder = {name: i for i, name in enumerate(MANAGEMENT_LADDER)}
    summary = classification_summary(y_mgmt_valid, mgmt_pred, list(MANAGEMENT_LADDER))
    under = sum(
        1
        for t, p, c in zip(y_mgmt_valid, mgmt_pred, valid)
        if c.diagnosis_group == "MEL" and ladder[p] < ladder["excision"]
    )
    report["management"] = {
        "representation": chosen_repr,
        "representation_selected_by": "train out-of-fold quadratic-weighted kappa",
        "representation_ranking": mgmt_selection,
        "selected_C": c_mgmt,
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "quadratic_weighted_kappa": round(
            quadratic_weighted_kappa(
                [ladder[v] for v in y_mgmt_valid],
                [ladder[v] for v in mgmt_pred],
                n_classes=len(MANAGEMENT_LADDER),
            ),
            4,
        ),
        "melanoma_under_managed": under,
        "melanoma_under_management_rate": round(under / sum(truth_mel), 4) if truth_mel else None,
    }

    # --- reference points already measured -------------------------------------
    report["reference_points"] = {
        "classifier_alone": {"sensitivity": 0.5574, "specificity": 0.9437},
        "criteria_alone": {"sensitivity": 0.5574, "specificity": 0.8944},
        "hand_written_union": {"sensitivity": 0.7049, "specificity": 0.8592},
        "management_qwk_dermlip_probe": 0.6550,
        "management_qwk_gold_criteria": 0.6310,
        "note": "All on the same valid split; see G2/G5 and the perception baseline.",
    }

    union = report["reference_points"]["hand_written_union"]
    fixed = report["melanoma"]["at_train_fixed_threshold_target_sens_0_70"]["valid"]

    # Youden's index compares two operating points without assuming either sensitivity or
    # specificity is the thing to hold constant -- necessary here because the train-fixed
    # threshold does not reproduce the union rule's sensitivity exactly.
    def _youden(stats: Dict) -> float:
        if stats["sensitivity"] is None or stats["specificity"] is None:
            return float("nan")
        return stats["sensitivity"] + stats["specificity"] - 1.0

    report["verdict"] = {
        "threshold_is_honest": True,
        "valid_sensitivity_at_train_fixed_threshold": fixed["sensitivity"],
        "valid_specificity_at_train_fixed_threshold": fixed["specificity"],
        "youden_j_stacked": round(_youden(fixed), 4),
        "youden_j_hand_written_union": round(
            union["sensitivity"] + union["specificity"] - 1.0, 4
        ),
        "beats_union_on_youden_j": bool(
            _youden(fixed) > union["sensitivity"] + union["specificity"] - 1.0
        ),
        "caveat": (
            "View subset, meta-C and the threshold are now all chosen on train; valid is scored "
            "once. The union comparator is a fixed rule with no tunable threshold, so the two "
            "are compared by Youden's J rather than by holding one axis constant. Register this "
            "configuration before touching test."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = run()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
