"""Phase-1 perception-uplift lever sweep for Derm7pt (L1-L6).

EXPERIMENTATION ONLY. No edits to ``medguard_v2/`` source or pipeline -- everything here is a new
script that reads cached features (frozen encoder, never refit) and fits small sklearn heads.
The TEST split is never opened: hyperparameters are chosen by cross-validation *inside train*;
``valid`` is scored, once per configuration, for reporting and for the lever/greedy decisions the
brief explicitly allows to be made on valid (this is a development-time lever sweep, not the
confirmatory evaluation). Seed 20260810 everywhere a seed is accepted.

Six levers, all against the frozen DermLIP ViT-B/16 features:

* **L1** clinical second view -- concat [derm | clinic], derm-only, clinic-only.
* **L2** clinical metadata -- concat the v3 sex/location/elevation encoding (14-d).
* **L3** TTA -- horizontal-flip feature averaging (built by ``derm7pt_p1_features.py``).
* **L4** head sweep -- LR, LR(class_weight=balanced), calibrated RBF-SVM, HistGradientBoosting,
  MLP(128, early_stopping).
* **L5** zero-shot text-tower auxiliary features (built by ``derm7pt_p1_features.py``).
* **L6** G6-style stacked meta-learner over the best views (reuses
  ``medguard_v2.eval.gates.g6_multiview_stacking.build_meta_features`` unedited).

Protocol: factorial-lite greedy forward selection from the current baseline (dermlip derm-only,
LR), documented step by step, PLUS the full single-lever ablation table so every lever's marginal
value is visible even where the greedy path never tries it. Diagnosis is optimised for valid
balanced accuracy (Derm7pt is heavily NEV-skewed: 62% of train); criteria heads are optimised for
macro-averaged binary concept F1 (scoring vs non-scoring values, the DermAgent protocol), matching
the brief. The criteria greedy walk is SHARED across all seven heads (one feature representation,
chosen by the macro objective) so there is one final config for "criteria heads", not seven.

Run::

    venv_win/Scripts/python.exe scripts/dev/derm7pt_p1_sweep.py             # full sweep
    venv_win/Scripts/python.exe scripts/dev/derm7pt_p1_sweep.py --quick     # smoke test, tiny grids
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES  # noqa: E402
from medguard_v2.data.derm7pt import (  # noqa: E402
    CRITERION_VOCABULARY,
    DERM7PT_CLASS_ORDER,
    Derm7ptCase,
    load_cases,
)
from medguard_v2.eval.gates.g6_multiview_stacking import (  # noqa: E402
    _select_c as g6_select_c,
    build_meta_features as g6_build_meta_features,
    discover_feature_sets as g6_discover_feature_sets,
)
from medguard_v2.eval.metrics import classification_summary  # noqa: E402
from medguard_v2.eval.statistics import binary_auc  # noqa: E402
from medguard_v2.perception import encode as _encode  # noqa: E402
from medguard_v2.perception import encode_multi as _encode_multi  # noqa: E402
from medguard_v2.perception.clinical_metadata import encode_batch_derm7pt  # noqa: E402
from medguard_v2.perception.encode import FeatureBundle  # noqa: E402

FEATURE_ROOT = REPO / "research" / "evidence" / "derm7pt_v1" / "features"
FEATURE_ROOT_P1 = REPO / "research" / "evidence" / "derm7pt_v1" / "features_phase1"
OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "perception_uplift_phase1.json"
OUT_MD = REPO / "notes" / "PERCEPTION_UPLIFT_PHASE1.md"

SEED = 20260810
LABELS_DIAGNOSIS = list(DERM7PT_CLASS_ORDER)  # BCC, NEV, MEL, SK, MISC
FUSABLE_TTA = ("dermlip", "dermlip_clinic")

LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)]
SVM_GRID = [{"C": c} for c in (0.1, 1.0, 10.0, 100.0)]
HGB_GRID = [
    {"learning_rate": 0.1, "max_iter": 100},
    {"learning_rate": 0.05, "max_iter": 200},
    {"learning_rate": 0.2, "max_iter": 100},
]
MLP_GRID = [{"alpha": a} for a in (1e-4, 1e-3, 1e-2)]

FAMILY_GRIDS = {
    "lr": LR_GRID,
    "lr_balanced": LR_GRID,
    "svm_rbf": SVM_GRID,
    "hgb": HGB_GRID,
    "mlp": MLP_GRID,
}

LEVER_FAMILIES = ("L1", "L2", "L3", "L5")


# ============================================================================================
# JSON safety net -- strip private (``_``-prefixed) keys and coerce numpy scalars/arrays.
# ============================================================================================


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {
            k: _to_jsonable(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, frozenset):
        return sorted(obj)
    return obj


# ============================================================================================
# feature bundles
# ============================================================================================


def _load_npz_bundle(directory: Path, key: str) -> FeatureBundle:
    npz = directory / "features.npz"
    meta = directory / "feature_manifest.json"
    manifest = json.loads(meta.read_text(encoding="utf-8"))
    with np.load(npz) as payload:
        case_nums = np.asarray(payload["case_num"], dtype=np.int64)
        features = np.asarray(payload["features"], dtype=np.float32)
    return FeatureBundle(key, case_nums, features, manifest)


def load_all_bundles(cases: Sequence[Derm7ptCase]) -> Dict[str, FeatureBundle]:
    ordered = sorted(cases, key=lambda c: c.case_num)
    case_nums = np.asarray([c.case_num for c in ordered], dtype=np.int64)

    bundles: Dict[str, FeatureBundle] = {}
    bundles["dermlip"] = _encode.load_cached_features("dermlip")
    bundles["dermlip_clinic"] = _encode_multi.load_feature_set("dermlip_clinic")
    bundles["biomedclip_derm"] = _encode_multi.load_feature_set("biomedclip_derm")
    bundles["biomedclip_clinic"] = _encode_multi.load_feature_set("biomedclip_clinic")
    for key in bundles:
        if bundles[key] is None:
            raise RuntimeError(f"required existing feature set {key!r} is not cached")

    for key in ("dermlip_flip", "dermlip_clinic_flip", "dermlip_tta", "dermlip_clinic_tta", "dermlip_zeroshot5"):
        bundles[key] = _load_npz_bundle(FEATURE_ROOT_P1 / key, key)

    meta_matrix = encode_batch_derm7pt(ordered).astype(np.float32)
    bundles["metadata14"] = FeatureBundle(
        "metadata14", case_nums, meta_matrix, {"feature_dim": int(meta_matrix.shape[1])}
    )
    return bundles


def materialize(state: dict, case_nums: Sequence[int], bundles: Mapping[str, FeatureBundle]) -> np.ndarray:
    blocks = []
    for view in state["views"]:
        key = f"{view}_tta" if (state.get("use_tta") and view in FUSABLE_TTA) else view
        blocks.append(bundles[key].matrix_for(case_nums))
    if state.get("use_metadata"):
        blocks.append(bundles["metadata14"].matrix_for(case_nums))
    if state.get("use_zeroshot"):
        blocks.append(bundles["dermlip_zeroshot5"].matrix_for(case_nums))
    return np.hstack(blocks) if len(blocks) > 1 else np.ascontiguousarray(blocks[0])


def describe_state(state: dict) -> str:
    parts = []
    for view in state["views"]:
        key = f"{view}_tta" if (state.get("use_tta") and view in FUSABLE_TTA) else view
        parts.append(key)
    label = "+".join(parts)
    if state.get("use_metadata"):
        label += "+metadata14"
    if state.get("use_zeroshot"):
        label += "+zeroshot5"
    return label


def lever_candidates(state: dict, remaining: set) -> List[tuple]:
    """(family_id, variant_name, new_state) for every still-available lever."""
    out: List[tuple] = []
    if "L1" in remaining:
        concat = dict(state)
        concat["views"] = sorted(set(state["views"]) | {"dermlip_clinic"})
        if concat["views"] != state["views"]:
            out.append(("L1", "clinic_concat", concat))
        replace = dict(state)
        replace["views"] = ["dermlip_clinic"]
        if replace["views"] != state["views"]:
            out.append(("L1", "clinic_only_replace", replace))
    if "L2" in remaining and not state.get("use_metadata"):
        s = dict(state)
        s["use_metadata"] = True
        out.append(("L2", "add_metadata14", s))
    if "L3" in remaining and not state.get("use_tta"):
        s = dict(state)
        s["use_tta"] = True
        out.append(("L3", "tta_flip_average", s))
    if "L5" in remaining and not state.get("use_zeroshot"):
        s = dict(state)
        s["use_zeroshot"] = True
        out.append(("L5", "add_zeroshot5", s))
    return out


BASELINE_STATE = {"views": ["dermlip"], "use_tta": False, "use_metadata": False, "use_zeroshot": False}


def single_lever_states(baseline: dict) -> Dict[str, dict]:
    return {
        "L1_clinic_concat": {**baseline, "views": sorted(set(baseline["views"]) | {"dermlip_clinic"})},
        "L1_clinic_only": {**baseline, "views": ["dermlip_clinic"]},
        "L2_metadata": {**baseline, "use_metadata": True},
        "L3_tta": {**baseline, "use_tta": True},
        "L5_zeroshot": {**baseline, "use_zeroshot": True},
    }


# ============================================================================================
# generic model-family fitting + CV hyperparameter selection
# ============================================================================================


def _dynamic_n_splits(stratify: Sequence, cap: int = 5) -> int:
    counts = Counter(stratify)
    if not counts:
        return 0
    return min(cap, min(counts.values()))


def _build_estimator(family: str, params: dict, seed: int, need_posterior: bool):
    if family == "lr":
        return LogisticRegression(C=params["C"], max_iter=5000, random_state=seed, solver="lbfgs")
    if family == "lr_balanced":
        return LogisticRegression(
            C=params["C"], max_iter=5000, random_state=seed, solver="lbfgs", class_weight="balanced"
        )
    if family == "svm_rbf":
        return SVC(C=params["C"], kernel="rbf", gamma="scale", random_state=seed, probability=need_posterior)
    if family == "hgb":
        return HistGradientBoostingClassifier(
            random_state=seed, learning_rate=params["learning_rate"], max_iter=params["max_iter"]
        )
    if family == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=params["alpha"],
            early_stopping=True,
            random_state=seed,
            max_iter=500,
        )
    raise ValueError(f"unknown model family {family!r}")


def _metric_value(metric: str, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str],
                   scoring_values: Optional[frozenset]) -> float:
    if metric == "balanced_accuracy":
        return classification_summary(list(y_true), list(y_pred), list(labels))["balanced_accuracy"]
    if metric == "binary_concept_f1":
        yt = [1 if v in scoring_values else 0 for v in y_true]
        yp = [1 if v in scoring_values else 0 for v in y_pred]
        return float(f1_score(yt, yp, average="macro", zero_division=0))
    raise ValueError(metric)


def _encode_labels(y: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """String labels -> integer codes in canonical ``labels`` order.

    Verified necessary, not cosmetic: sklearn 1.8.0's ``MLPClassifier(early_stopping=True)``
    crashes with ``TypeError: ufunc 'isnan' not supported`` when fit on STRING class labels
    (reproduced independent of numpy dtype -- object and fixed-width unicode both fail; integer
    labels do not). Every family is fit on the same integer codes for uniformity, not just MLP.
    """
    index = {lbl: i for i, lbl in enumerate(labels)}
    try:
        return np.asarray([index[v] for v in y], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label {exc} not in the declared vocabulary {list(labels)}") from exc


def _decode_labels(y_int: Sequence[int], labels: Sequence[str]) -> List[str]:
    return [labels[int(i)] for i in y_int]


def select_and_fit(
    family: str,
    grid: Sequence[dict],
    x_train: np.ndarray,
    y_train: Sequence[str],
    labels: Sequence[str],
    stratify: Sequence,
    metric: str,
    scoring_values: Optional[frozenset],
    seed: int,
    need_posterior_final: bool,
    cap: int = 5,
):
    y_int = _encode_labels(y_train, labels)
    n_splits = _dynamic_n_splits(stratify, cap)
    table: List[dict] = []
    if n_splits >= 2:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(folds.split(x_train, stratify))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for params in grid:
                oof_int = np.full(len(y_int), -1, dtype=np.int64)
                for tr, va in splits:
                    model = _build_estimator(family, params, seed, need_posterior=False)
                    model.fit(x_train[tr], y_int[tr])
                    oof_int[va] = model.predict(x_train[va])
                oof = _decode_labels(oof_int, labels)
                score = _metric_value(metric, list(y_train), oof, labels, scoring_values)
                table.append({"params": params, "oof_score": round(float(score), 4)})
        table.sort(key=lambda r: -r["oof_score"])
        best_params = table[0]["params"]
        cv_skipped = False
    else:
        best_params = grid[len(grid) // 2]
        cv_skipped = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = _build_estimator(family, best_params, seed, need_posterior=need_posterior_final)
        final_model.fit(x_train, y_int)
    return final_model, best_params, table, cv_skipped, n_splits


def _aligned_posterior(model, x: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """``predict_proba`` columns re-indexed into canonical ``labels`` order.

    Dual-mode: models fit via :func:`select_and_fit` carry integer ``classes_`` codes that
    already equal the target position in ``labels`` (see :func:`_encode_labels`); models fit
    directly elsewhere in this script (the "current production" reproduction in ``main()``,
    :func:`single_view_mel_auroc_scan`) are fit on the raw string labels and carry string
    ``classes_``. Both are handled so this one function serves either caller.
    """
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    labels_list = list(labels)
    out = np.zeros((x.shape[0], len(labels_list)))
    for j, cls in enumerate(classes):
        if isinstance(cls, (int, np.integer)):
            code = int(cls)
        elif cls in labels_list:
            code = labels_list.index(cls)
        else:
            continue
        if 0 <= code < len(labels_list):
            out[:, code] = proba[:, j]
    return out


def score_target(model, x_valid: np.ndarray, y_valid: Sequence[str], labels: Sequence[str],
                  scoring_values: Optional[frozenset], need_auroc: bool) -> dict:
    # Dual-mode for the same reason as `_aligned_posterior`: most callers go through
    # `select_and_fit` (integer-coded predictions), but `main()`'s "current production"
    # reproduction fits LogisticRegression directly on the raw string labels.
    pred_raw = np.asarray(model.predict(x_valid))
    if np.issubdtype(pred_raw.dtype, np.integer):
        pred = _decode_labels(pred_raw, labels)
    else:
        pred = [str(p) for p in pred_raw]
    summary = classification_summary(list(y_valid), pred, list(labels))
    result = {
        "n": summary["n"],
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
        "per_class_recall": {lbl: round(v["recall"], 4) for lbl, v in summary["per_class"].items()},
        "per_class_support": {lbl: int(v["support"]) for lbl, v in summary["per_class"].items()},
    }
    if scoring_values is not None:
        yt = [1 if v in scoring_values else 0 for v in y_valid]
        yp = [1 if v in scoring_values else 0 for v in pred]
        result["binary_concept_f1"] = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)
    if need_auroc:
        proba = _aligned_posterior(model, x_valid, labels)
        mel_idx = list(labels).index("MEL")
        truth_mel = [v == "MEL" for v in y_valid]
        result["mel_auroc"] = round(float(binary_auc(truth_mel, [float(p) for p in proba[:, mel_idx]])), 4)
        result["_full_posterior"] = proba.tolist()
    return result


# ============================================================================================
# DIAGNOSIS pipeline
# ============================================================================================


def run_diagnosis_pipeline(bundles: Dict[str, FeatureBundle], cases: Sequence[Derm7ptCase], quick: bool) -> dict:
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    train_nums = [c.case_num for c in train]
    valid_nums = [c.case_num for c in valid]
    y_train = [c.diagnosis_group for c in train]
    y_valid = [c.diagnosis_group for c in valid]
    labels = LABELS_DIAGNOSIS

    def eval_state(state: dict, family: str = "lr", grid: Optional[Sequence[dict]] = None,
                   need_posterior: bool = True) -> dict:
        grid = grid if grid is not None else LR_GRID
        x_train = materialize(state, train_nums, bundles)
        x_valid = materialize(state, valid_nums, bundles)
        model, params, table, cv_skipped, n_splits = select_and_fit(
            family, grid, x_train, y_train, labels, y_train, "balanced_accuracy", None, SEED, need_posterior
        )
        metrics = score_target(model, x_valid, y_valid, labels, None, need_auroc=True)
        return {
            "state": describe_state(state),
            "family": family,
            "selected_params": params,
            "n_splits_used": n_splits,
            "cv_skipped_insufficient_class_support": cv_skipped,
            "feature_dim": int(x_train.shape[1]),
            "valid": metrics,
            "_cv_table": table,
            "_state_raw": state,
            "_model": model,
        }

    t0 = time.perf_counter()
    baseline = eval_state(BASELINE_STATE)
    baseline["cv_table"] = baseline.pop("_cv_table")
    print(f"[diagnosis] baseline {baseline['state']}: bal_acc={baseline['valid']['balanced_accuracy']} "
          f"acc={baseline['valid']['accuracy']} mel_auroc={baseline['valid']['mel_auroc']}")

    # -------------------------------------------------------------- full single-lever ablation
    ablation_rows = [{"lever": "baseline_dermlip_derm_only_LR", **{k: v for k, v in baseline.items() if not k.startswith("_")}}]
    for name, state in single_lever_states(BASELINE_STATE).items():
        r = eval_state(state)
        r["cv_table"] = r.pop("_cv_table")
        r["lever"] = name
        r["delta_balanced_accuracy_vs_baseline"] = round(
            r["valid"]["balanced_accuracy"] - baseline["valid"]["balanced_accuracy"], 4
        )
        r["delta_mel_auroc_vs_baseline"] = round(r["valid"]["mel_auroc"] - baseline["valid"]["mel_auroc"], 4)
        ablation_rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"[diagnosis][single-lever] {name}: bal_acc={r['valid']['balanced_accuracy']} "
              f"(delta {r['delta_balanced_accuracy_vs_baseline']:+.4f}) mel_auroc={r['valid']['mel_auroc']}")

    # -------------------------------------------------------------------------- greedy path
    # Path entries deliberately keep the "_"-prefixed private keys (state/model/cv_table) --
    # they are only stripped once, globally, by `_to_jsonable` in `main()`. That means the
    # winning step's raw state/model can be read back directly with no replay logic.
    path = [{"step": 0, "action": "baseline", "gain_balanced_accuracy": 0.0, **baseline}]
    current_state = dict(BASELINE_STATE)
    current_best = baseline
    remaining = set(LEVER_FAMILIES)
    while remaining:
        candidates = lever_candidates(current_state, remaining)
        if not candidates:
            break
        scored = [(fam, var, st, eval_state(st)) for fam, var, st in candidates]
        fam, var, st, best_r = max(scored, key=lambda t: t[3]["valid"]["balanced_accuracy"])
        gain = round(best_r["valid"]["balanced_accuracy"] - current_best["valid"]["balanced_accuracy"], 4)
        entry = {"step": len(path), "action": f"{fam}:{var}", "gain_balanced_accuracy": gain, **best_r}
        path.append(entry)
        print(f"[diagnosis][greedy] step {len(path)-1}: {fam}:{var} -> {best_r['state']} "
              f"bal_acc={best_r['valid']['balanced_accuracy']} (gain {gain:+.4f})")
        current_state, current_best = st, best_r
        remaining.discard(fam)

    # Best point ALONG the path, not necessarily the last step: a lever that hurt should not
    # be locked in just because the walk tried it last (this is what makes negatives visible).
    best_step = max(path, key=lambda p: p["valid"]["balanced_accuracy"])
    final_feature_state = best_step["_state_raw"]

    # -------------------------------------------------------------------------- L4 head sweep
    head_rows = []
    for family, grid in FAMILY_GRIDS.items():
        r = eval_state(final_feature_state, family=family, grid=grid, need_posterior=True)
        r["cv_table"] = r.pop("_cv_table")
        head_rows.append(r)
        print(f"[diagnosis][head-sweep] {family}: bal_acc={r['valid']['balanced_accuracy']} "
              f"mel_auroc={r['valid']['mel_auroc']} params={r['selected_params']}")

    # head_rows keep "_model" / the posterior (via "valid._full_posterior"), so the winner can be
    # read directly -- no need to refit a second time.
    winner_full = max(head_rows, key=lambda r: r["valid"]["balanced_accuracy"])
    winner_family = winner_full["family"]

    elapsed = time.perf_counter() - t0
    print(f"[diagnosis] done in {elapsed:.1f}s; final={describe_state(final_feature_state)} "
          f"family={winner_family} bal_acc={winner_full['valid']['balanced_accuracy']}")

    return {
        "baseline": {k: v for k, v in baseline.items() if not k.startswith("_")},
        "single_lever_ablation_table": ablation_rows,
        "greedy_path": path,
        "greedy_selection_metric": "balanced_accuracy (Derm7pt train is 62% NEV; plain accuracy would bias selection toward the majority class)",
        "final_feature_state_from_greedy": final_feature_state,
        "head_sweep_at_final_features": [{k: v for k, v in r.items() if not k.startswith("_")} for r in head_rows],
        "final_config": {
            "features": describe_state(final_feature_state),
            "family": winner_family,
            "selected_params": winner_full["selected_params"],
            "valid": {k: v for k, v in winner_full["valid"].items() if not k.startswith("_")},
        },
        "elapsed_seconds": round(elapsed, 1),
        "_winner_full_posterior": winner_full["valid"].get("_full_posterior"),
        "_winner_model": winner_full["_model"],
        "_final_feature_state": final_feature_state,
        "_labels": labels,
        "_y_valid": y_valid,
    }


# ============================================================================================
# CRITERIA pipeline (shared greedy walk, macro binary-concept-F1 objective)
# ============================================================================================


def _criterion_targets(train: Sequence[Derm7ptCase], valid: Sequence[Derm7ptCase]) -> Dict[str, dict]:
    targets = {}
    for name in CRITERION_NAMES:
        spec = CRITERIA_BY_NAME[name]
        y_train = [str(c.criteria[name]) for c in train]
        y_valid = [str(c.criteria[name]) for c in valid]
        stratify = [1 if v in spec.scoring_values else 0 for v in y_train]
        targets[name] = {
            "y_train": y_train,
            "y_valid": y_valid,
            "labels": list(CRITERION_VOCABULARY[name]),
            "stratify": stratify,
            "scoring_values": spec.scoring_values,
        }
    return targets


def run_criteria_pipeline(bundles: Dict[str, FeatureBundle], cases: Sequence[Derm7ptCase], quick: bool) -> dict:
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    train_nums = [c.case_num for c in train]
    valid_nums = [c.case_num for c in valid]
    targets = _criterion_targets(train, valid)

    def eval_one(name: str, state: dict, family: str = "lr", grid: Optional[Sequence[dict]] = None,
                 need_posterior: bool = False) -> dict:
        grid = grid if grid is not None else LR_GRID
        t = targets[name]
        x_train = materialize(state, train_nums, bundles)
        x_valid = materialize(state, valid_nums, bundles)
        model, params, table, cv_skipped, n_splits = select_and_fit(
            family, grid, x_train, t["y_train"], t["labels"], t["stratify"],
            "binary_concept_f1", t["scoring_values"], SEED, need_posterior,
        )
        metrics = score_target(model, x_valid, t["y_valid"], t["labels"], t["scoring_values"], need_auroc=False)
        return {
            "family": family, "selected_params": params, "n_splits_used": n_splits,
            "cv_skipped_insufficient_class_support": cv_skipped,
            "feature_dim": int(x_train.shape[1]), "valid": metrics, "_cv_table": table,
        }

    def eval_macro(state: dict, family: str = "lr", grid: Optional[Sequence[dict]] = None) -> dict:
        per = {name: eval_one(name, state, family, grid) for name in CRITERION_NAMES}
        macro_binary = round(float(np.mean([r["valid"]["binary_concept_f1"] for r in per.values()])), 4)
        macro_multi = round(float(np.mean([r["valid"]["macro_f1"] for r in per.values()])), 4)
        return {
            "state": describe_state(state), "family": family,
            "macro_binary_concept_f1": macro_binary, "macro_multiclass_f1": macro_multi,
            "per_criterion": {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in per.items()},
            "_state_raw": state,
        }

    t0 = time.perf_counter()
    baseline = eval_macro(BASELINE_STATE)
    print(f"[criteria] baseline {baseline['state']}: macro_binary_f1={baseline['macro_binary_concept_f1']} "
          f"macro_multiclass_f1={baseline['macro_multiclass_f1']}")

    ablation_rows = [{"lever": "baseline_dermlip_derm_only_LR", **{k: v for k, v in baseline.items() if not k.startswith("_")}}]
    for name, state in single_lever_states(BASELINE_STATE).items():
        r = eval_macro(state)
        r["lever"] = name
        r["delta_macro_binary_f1_vs_baseline"] = round(r["macro_binary_concept_f1"] - baseline["macro_binary_concept_f1"], 4)
        ablation_rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"[criteria][single-lever] {name}: macro_binary_f1={r['macro_binary_concept_f1']} "
              f"(delta {r['delta_macro_binary_f1_vs_baseline']:+.4f})")

    # Path entries deliberately keep the "_"-prefixed private "_state_raw" -- it is only stripped
    # once, globally, by `_to_jsonable` in `main()`, so the winning step's raw state can be read
    # back directly with no replay logic.
    path = [{"step": 0, "action": "baseline", "gain_macro_binary_f1": 0.0, **baseline}]
    current_state = dict(BASELINE_STATE)
    current_best = baseline
    remaining = set(LEVER_FAMILIES)
    while remaining:
        candidates = lever_candidates(current_state, remaining)
        if not candidates:
            break
        scored = [(fam, var, st, eval_macro(st)) for fam, var, st in candidates]
        fam, var, st, best_r = max(scored, key=lambda t: t[3]["macro_binary_concept_f1"])
        gain = round(best_r["macro_binary_concept_f1"] - current_best["macro_binary_concept_f1"], 4)
        entry = {"step": len(path), "action": f"{fam}:{var}", "gain_macro_binary_f1": gain, **best_r}
        path.append(entry)
        print(f"[criteria][greedy] step {len(path)-1}: {fam}:{var} -> {best_r['state']} "
              f"macro_binary_f1={best_r['macro_binary_concept_f1']} (gain {gain:+.4f})")
        current_state, current_best = st, best_r
        remaining.discard(fam)

    # Best point ALONG the path, not necessarily the last step (see the diagnosis pipeline for
    # why: a lever that hurts must not be locked in just because the walk tried it last).
    best_step = max(path, key=lambda p: p["macro_binary_concept_f1"])
    final_feature_state = best_step["_state_raw"]

    head_rows = []
    for family, grid in FAMILY_GRIDS.items():
        r = eval_macro(final_feature_state, family=family, grid=grid)
        head_rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"[criteria][head-sweep] {family}: macro_binary_f1={r['macro_binary_concept_f1']} "
              f"macro_multiclass_f1={r['macro_multiclass_f1']}")

    best_head = max(head_rows, key=lambda r: r["macro_binary_concept_f1"])
    elapsed = time.perf_counter() - t0
    print(f"[criteria] done in {elapsed:.1f}s; final={describe_state(final_feature_state)} "
          f"family={best_head['family']} macro_binary_f1={best_head['macro_binary_concept_f1']}")

    return {
        "baseline": {k: v for k, v in baseline.items() if not k.startswith("_")},
        "single_lever_ablation_table": ablation_rows,
        "greedy_path": path,
        "greedy_selection_metric": "macro-averaged binary concept F1 across all 7 criteria (DermAgent protocol); SHARED across heads so there is one final feature config",
        "final_feature_state_from_greedy": final_feature_state,
        "head_sweep_at_final_features": head_rows,
        "final_config": {
            "features": describe_state(final_feature_state),
            "family": best_head["family"],
            "macro_binary_concept_f1": best_head["macro_binary_concept_f1"],
            "macro_multiclass_f1": best_head["macro_multiclass_f1"],
            "per_criterion": best_head["per_criterion"],
        },
        "elapsed_seconds": round(elapsed, 1),
    }


# ============================================================================================
# operating-point policy: per-class thresholds tuned on valid for SK/MISC recall
# ============================================================================================


def operating_point_policy(full_posterior, y_valid: Sequence[str], labels: Sequence[str]) -> dict:
    proba = np.asarray(full_posterior)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    baseline_pred = [labels[i] for i in proba.argmax(axis=1)]
    baseline_summary = classification_summary(list(y_valid), baseline_pred, list(labels))

    grid_weights = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15)
    rows = []
    for w_sk in grid_weights:
        for w_misc in grid_weights:
            weights = np.ones(len(labels))
            weights[idx["SK"]] = w_sk
            weights[idx["MISC"]] = w_misc
            adjusted = proba / weights
            pred = [labels[i] for i in adjusted.argmax(axis=1)]
            summary = classification_summary(list(y_valid), pred, list(labels))
            rows.append({
                "w_SK": w_sk, "w_MISC": w_misc,
                "balanced_accuracy": round(summary["balanced_accuracy"], 4),
                "accuracy": round(summary["accuracy"], 4),
                "recall_SK": round(summary["per_class"]["SK"]["recall"], 4),
                "recall_MISC": round(summary["per_class"]["MISC"]["recall"], 4),
            })
    baseline_balacc = round(baseline_summary["balanced_accuracy"], 4)
    tolerant = [r for r in rows if r["balanced_accuracy"] >= baseline_balacc - 0.02]
    chosen = max(tolerant, key=lambda r: r["recall_SK"] + r["recall_MISC"]) if tolerant else rows[0]
    return {
        "baseline_argmax": {
            "balanced_accuracy": baseline_balacc,
            "recall_SK": round(baseline_summary["per_class"]["SK"]["recall"], 4),
            "recall_MISC": round(baseline_summary["per_class"]["MISC"]["recall"], 4),
        },
        "rule": (
            "argmax(posterior / per_class_weight); weight<1 boosts that class's rank. Grid-searched "
            "w_SK, w_MISC independently over {1.0,0.7,0.5,0.35,0.25,0.15} on VALID (posterior "
            "re-ranking only, no refitting). Chosen = max(recall_SK + recall_MISC) subject to "
            "balanced_accuracy staying within 0.02 of the unweighted argmax."
        ),
        "grid": rows,
        "chosen": chosen,
    }


# ============================================================================================
# L6: G6-style stacking over the best views
# ============================================================================================


def single_view_mel_auroc_scan(bundles: Dict[str, FeatureBundle], cases: Sequence[Derm7ptCase]) -> List[dict]:
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    train_nums = [c.case_num for c in train]
    valid_nums = [c.case_num for c in valid]
    y_train = [c.diagnosis_group for c in train]
    y_valid = [c.diagnosis_group for c in valid]
    truth_mel = [v == "MEL" for v in y_valid]

    candidates = ("dermlip", "dermlip_clinic", "biomedclip_derm", "biomedclip_clinic", "dermlip_tta", "dermlip_clinic_tta")
    rows = []
    for key in candidates:
        x_train = bundles[key].matrix_for(train_nums)
        x_valid = bundles[key].matrix_for(valid_nums)
        model = LogisticRegression(C=10.0, max_iter=5000, random_state=SEED)
        model.fit(x_train, y_train)
        proba = _aligned_posterior(model, x_valid, LABELS_DIAGNOSIS)
        mel_idx = LABELS_DIAGNOSIS.index("MEL")
        auroc = round(float(binary_auc(truth_mel, [float(p) for p in proba[:, mel_idx]])), 4)
        rows.append({"view": key, "standalone_valid_mel_auroc": auroc, "fixed_C": 10.0})
    rows.sort(key=lambda r: -r["standalone_valid_mel_auroc"])
    return rows


def _melanoma_stats(flag: Sequence[bool], truth: Sequence[bool]) -> dict:
    tp = sum(1 for f, t in zip(flag, truth) if f and t)
    fn = sum(1 for f, t in zip(flag, truth) if not f and t)
    fp = sum(1 for f, t in zip(flag, truth) if f and not t)
    tn = sum(1 for f, t in zip(flag, truth) if not f and not t)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "youden_j": round(sens + spec - 1.0, 4) if (sens is not None and spec is not None) else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def run_stacking(view_paths: Dict[str, Path], cases: Sequence[Derm7ptCase], chosen_views: List[str], label: str) -> dict:
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    feature_sets = {v: view_paths[v] for v in chosen_views}

    x_train, x_valid, names = g6_build_meta_features(feature_sets, train, valid)
    y_mel_train = ["MEL" if c.diagnosis_group == "MEL" else "other" for c in train]
    truth_train = [v == "MEL" for v in y_mel_train]
    truth_valid = [c.diagnosis_group == "MEL" for c in valid]

    c_meta = g6_select_c(x_train, y_mel_train)
    meta = LogisticRegression(max_iter=5000, C=c_meta, random_state=SEED)
    meta.fit(x_train, y_mel_train)
    mel_index = list(meta.classes_).index("MEL")
    mel_score_valid = meta.predict_proba(x_valid)[:, mel_index]

    y_arr = np.asarray(y_mel_train)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_arr))
    for tr, va in folds.split(x_train, y_arr):
        m = LogisticRegression(max_iter=5000, C=c_meta, random_state=SEED)
        m.fit(x_train[tr], y_arr[tr])
        oof[va] = m.predict_proba(x_train[va])[:, list(m.classes_).index("MEL")]

    def threshold_for_sensitivity(scores: np.ndarray, truth: Sequence[bool], target: float) -> float:
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

    cut_sens = threshold_for_sensitivity(oof, truth_train, 0.70)

    best_j, cut_j = -2.0, 0.5
    for cand in np.unique(oof):
        pred = oof >= cand
        tp = int(np.sum(pred & np.asarray(truth_train)))
        fn = int(np.sum(~pred & np.asarray(truth_train)))
        fp = int(np.sum(pred & ~np.asarray(truth_train)))
        tn = int(np.sum(~pred & ~np.asarray(truth_train)))
        if (tp + fn) == 0 or (tn + fp) == 0:
            continue
        j = tp / (tp + fn) + tn / (tn + fp) - 1.0
        if j > best_j:
            best_j, cut_j = j, float(cand)

    return {
        "label": label,
        "views_used": chosen_views,
        "meta_feature_count": len(names),
        "selected_meta_C": c_meta,
        "auroc_valid": round(float(binary_auc(truth_valid, list(mel_score_valid))), 4),
        "at_target_sensitivity_0_70": {
            "threshold": round(cut_sens, 4),
            "train_oof": _melanoma_stats([s >= cut_sens for s in oof], truth_train),
            "valid": _melanoma_stats([s >= cut_sens for s in mel_score_valid], truth_valid),
        },
        "at_youden_j": {
            "threshold": round(cut_j, 4),
            "train_oof_youden_j": round(best_j, 4),
            "train_oof": _melanoma_stats([s >= cut_j for s in oof], truth_train),
            "valid": _melanoma_stats([s >= cut_j for s in mel_score_valid], truth_valid),
        },
    }


# ============================================================================================
# markdown summary
# ============================================================================================


def build_markdown(report: dict) -> str:
    d = report["diagnosis"]
    cr = report["criteria"]
    st = report["stacking_L6"]
    lines: List[str] = []
    lines.append("# Phase 1 -- Derm7pt perception uplift (L1-L6)")
    lines.append("")
    lines.append(f"Generated {report['generated_at']}. Experimentation only: no `medguard_v2/` source edits; "
                  "test split never opened; all selection on train (CV) and valid.")
    lines.append("")
    lines.append("## Environment note")
    lines.append("")
    lines.append(report["environment"]["note"])
    lines.append("")
    lines.append("## Reference numbers (test split, NOT reproduced here -- for orientation only)")
    lines.append("")
    ref = report["reference_baselines"]["test_split_context_only_never_reproduced_here"]
    lines.append(f"- Diagnosis accuracy (test): {ref['diagnosis_accuracy_test']}")
    lines.append(f"- MEL AUROC (test): {ref['mel_auroc_test']}")
    lines.append(f"- Criteria binary concept F1 (test): {ref['criteria_binary_concept_f1_test']}")
    lines.append(f"- HAM 7-class accuracy (test): {ref['ham_7class_test']}")
    lines.append("")
    lines.append("## Valid-split baselines reproduced in phase 1")
    lines.append("")
    vb = report["reference_baselines"]["valid_split_reproduced_here"]
    lines.append(f"- Diagnosis, current production (fixed C=10.0 LR, dermlip derm-only), valid: "
                  f"accuracy={vb['diagnosis_current_production_fixedC10']['accuracy']}, "
                  f"balanced_accuracy={vb['diagnosis_current_production_fixedC10']['balanced_accuracy']}, "
                  f"mel_auroc={vb['diagnosis_current_production_fixedC10']['mel_auroc']}")
    lines.append(f"- Diagnosis, frozen `perception_baseline.json` probe (OOF-log-loss-selected C), valid accuracy: "
                  f"{vb['diagnosis_probe_py_frozen_artifact_valid_accuracy']}")
    lines.append(f"- Criteria heads, current production (fixed C=10.0 LR per head), valid macro binary concept F1: "
                  f"{vb['criteria_current_production_fixedC10_macro_binary_f1']}")
    lines.append("")

    lines.append("## Diagnosis (5-class): single-lever ablation table (valid)")
    lines.append("")
    lines.append("| lever | features | family | dim | accuracy | balanced_acc | macro_F1 | MEL AUROC | delta bal_acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in d["single_lever_ablation_table"]:
        delta = row.get("delta_balanced_accuracy_vs_baseline", 0.0)
        lines.append(f"| {row['lever']} | {row['state']} | {row['family']} | {row['feature_dim']} | "
                      f"{row['valid']['accuracy']} | {row['valid']['balanced_accuracy']} | "
                      f"{row['valid']['macro_f1']} | {row['valid']['mel_auroc']} | {delta:+.4f} |")
    lines.append("")

    lines.append("## Diagnosis: greedy path (documented)")
    lines.append("")
    lines.append("| step | action | resulting features | balanced_acc | gain |")
    lines.append("|---|---|---|---|---|")
    for row in d["greedy_path"]:
        lines.append(f"| {row['step']} | {row['action']} | {row['state']} | "
                      f"{row['valid']['balanced_accuracy']} | {row['gain_balanced_accuracy']:+.4f} |")
    lines.append("")
    lines.append(f"Selection metric: {d['greedy_selection_metric']}")
    lines.append("")

    lines.append("## Diagnosis: head sweep (L4) at the greedy-selected features "
                  f"(`{d['final_feature_state_from_greedy'] and describe_state(d['final_feature_state_from_greedy'])}`)")
    lines.append("")
    lines.append("| family | params | accuracy | balanced_acc | macro_F1 | MEL AUROC |")
    lines.append("|---|---|---|---|---|---|")
    for row in d["head_sweep_at_final_features"]:
        lines.append(f"| {row['family']} | {row['selected_params']} | {row['valid']['accuracy']} | "
                      f"{row['valid']['balanced_accuracy']} | {row['valid']['macro_f1']} | {row['valid']['mel_auroc']} |")
    lines.append("")
    fc = d["final_config"]
    lines.append(f"**Final diagnosis config**: `{fc['features']}` + `{fc['family']}` "
                  f"(params {fc['selected_params']}) -> valid accuracy={fc['valid']['accuracy']}, "
                  f"balanced_accuracy={fc['valid']['balanced_accuracy']}, macro_F1={fc['valid']['macro_f1']}, "
                  f"MEL AUROC={fc['valid']['mel_auroc']}")
    lines.append("")

    lines.append("## Criteria heads: single-lever ablation table (valid, macro across 7 criteria)")
    lines.append("")
    lines.append("| lever | features | family | macro binary-F1 | macro multiclass-F1 | delta |")
    lines.append("|---|---|---|---|---|---|")
    for row in cr["single_lever_ablation_table"]:
        delta = row.get("delta_macro_binary_f1_vs_baseline", 0.0)
        lines.append(f"| {row['lever']} | {row['state']} | {row['family']} | "
                      f"{row['macro_binary_concept_f1']} | {row['macro_multiclass_f1']} | {delta:+.4f} |")
    lines.append("")

    lines.append("## Criteria heads: per-criterion binary concept F1 at the FINAL shared config")
    lines.append("")
    lines.append("| criterion | scoring values | binary concept F1 | multiclass macro F1 |")
    lines.append("|---|---|---|---|")
    for name in CRITERION_NAMES:
        pc = cr["final_config"]["per_criterion"][name]
        scoring = sorted(CRITERIA_BY_NAME[name].scoring_values)
        lines.append(f"| {name} | {scoring} | {pc['valid']['binary_concept_f1']} | {pc['valid']['macro_f1']} |")
    lines.append("")
    lines.append(f"**Final criteria config (shared)**: `{cr['final_config']['features']}` + "
                  f"`{cr['final_config']['family']}` -> macro binary concept F1 = "
                  f"{cr['final_config']['macro_binary_concept_f1']}, macro multiclass F1 = "
                  f"{cr['final_config']['macro_multiclass_f1']}")
    lines.append("")

    lines.append("## L6: stacked meta-learner over the best views (valid)")
    lines.append("")
    lines.append("| config | views | AUROC | sens@target0.70 | spec@target0.70 | J@target0.70 | J@Youden |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in st["configs"]:
        s70 = row["at_target_sensitivity_0_70"]["valid"]
        sj = row["at_youden_j"]["valid"]
        lines.append(f"| {row['label']} | {row['views_used']} | {row['auroc_valid']} | "
                      f"{s70['sensitivity']} | {s70['specificity']} | {s70['youden_j']} | {sj['youden_j']} |")
    lines.append("")
    lines.append(f"Reference: hand-written union valid J = {st['comparison']['union_valid_J']}; "
                  f"G6 recorded stacked valid J = {st['comparison']['g6_recorded_stacked_J']}; "
                  f"phase-1 best stacked valid J = {st['comparison']['phase1_best_stacked_J']} "
                  f"(beats union: {st['comparison']['beats_union']}, beats G6-recorded: {st['comparison']['beats_g6_recorded']}).")
    lines.append("")

    op = report["operating_point_policy"]
    lines.append("## Operating-point policy: per-class thresholds for SK/MISC recall (valid)")
    lines.append("")
    lines.append(f"Baseline argmax: balanced_accuracy={op['baseline_argmax']['balanced_accuracy']}, "
                  f"recall_SK={op['baseline_argmax']['recall_SK']}, recall_MISC={op['baseline_argmax']['recall_MISC']}")
    lines.append("")
    c = op["chosen"]
    lines.append(f"Chosen operating point: w_SK={c['w_SK']}, w_MISC={c['w_MISC']} -> "
                  f"balanced_accuracy={c['balanced_accuracy']}, recall_SK={c['recall_SK']}, recall_MISC={c['recall_MISC']}")
    lines.append("")

    lines.append("## Projected impact")
    lines.append("")
    pi = report["projected_impact"]
    for k, v in pi.items():
        if k == "paper_table_rows_that_would_change":
            continue
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Paper-table rows that would change if promoted:")
    for row in pi["paper_table_rows_that_would_change"]:
        lines.append(f"- {row}")
    lines.append("")

    lines.append("## What FAILED to help (honest negatives)")
    lines.append("")
    for neg in report["negatives"]:
        lines.append(f"- {neg}")
    lines.append("")

    lines.append("## HAM/BCN transfer notes (not run in phase 1)")
    lines.append("")
    for k, v in report["ham_bcn_transfer_notes"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Phase 2 promotion plan")
    lines.append("")
    for step in report["phase2_promotion_plan"]:
        lines.append(f"- {step}")
    lines.append("")

    lines.append("## Suite check")
    lines.append("")
    sc = report["suite_check"]
    lines.append(f"- tests run: {sc['tests_run']}, failures: {sc['failures']}")
    lines.append(f"- {sc['detail']}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================================
# main
# ============================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny grids, smoke test only")
    args = parser.parse_args()

    global LR_GRID, SVM_GRID, HGB_GRID, MLP_GRID, FAMILY_GRIDS
    if args.quick:
        LR_GRID = [{"C": c} for c in (1.0, 10.0)]
        SVM_GRID = [{"C": c} for c in (1.0, 10.0)]
        HGB_GRID = [{"learning_rate": 0.1, "max_iter": 50}]
        MLP_GRID = [{"alpha": 1e-3}]
        FAMILY_GRIDS = {"lr": LR_GRID, "lr_balanced": LR_GRID, "svm_rbf": SVM_GRID, "hgb": HGB_GRID, "mlp": MLP_GRID}
        print("*** QUICK MODE: tiny grids, results are a smoke test only ***")

    started = time.perf_counter()
    cases = load_cases()
    bundles = load_all_bundles(cases)
    print("bundles loaded:", sorted(bundles.keys()))

    diagnosis = run_diagnosis_pipeline(bundles, cases, args.quick)
    criteria = run_criteria_pipeline(bundles, cases, args.quick)

    # ---- valid-split reference baselines (fixed C=10.0, "current production") ----
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    x_train_base = bundles["dermlip"].matrix_for([c.case_num for c in train])
    x_valid_base = bundles["dermlip"].matrix_for([c.case_num for c in valid])
    y_train_diag = [c.diagnosis_group for c in train]
    y_valid_diag = [c.diagnosis_group for c in valid]
    prod_diag = LogisticRegression(C=10.0, max_iter=5000, random_state=SEED)
    prod_diag.fit(x_train_base, y_train_diag)
    prod_metrics = score_target(prod_diag, x_valid_base, y_valid_diag, LABELS_DIAGNOSIS, None, need_auroc=True)

    prod_criteria_f1 = []
    for name in CRITERION_NAMES:
        spec = CRITERIA_BY_NAME[name]
        y_tr = [str(c.criteria[name]) for c in train]
        y_va = [str(c.criteria[name]) for c in valid]
        head = LogisticRegression(C=10.0, max_iter=5000, random_state=SEED)
        head.fit(x_train_base, y_tr)
        pred = [str(p) for p in head.predict(x_valid_base)]
        yt = [1 if v in spec.scoring_values else 0 for v in y_va]
        yp = [1 if v in spec.scoring_values else 0 for v in pred]
        prod_criteria_f1.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
    prod_criteria_macro_f1 = round(float(np.mean(prod_criteria_f1)), 4)

    perception_baseline_path = REPO / "research" / "evidence" / "derm7pt_v1" / "perception_baseline.json"
    probe_valid_accuracy = None
    if perception_baseline_path.is_file():
        pb = json.loads(perception_baseline_path.read_text(encoding="utf-8"))
        probe_valid_accuracy = pb["metrics"]["diagnosis"]["valid"]["accuracy"]

    # ---- L6 stacking over the best views ----
    existing_views = g6_discover_feature_sets()
    phase1_view_paths = {
        "dermlip_tta": FEATURE_ROOT_P1 / "dermlip_tta" / "features.npz",
        "dermlip_clinic_tta": FEATURE_ROOT_P1 / "dermlip_clinic_tta" / "features.npz",
    }
    all_view_paths = {**existing_views, **phase1_view_paths}
    scan = single_view_mel_auroc_scan(bundles, cases)
    top3 = [r["view"] for r in scan[:3]]
    top1 = [scan[0]["view"]]
    top2 = [r["view"] for r in scan[:2]]
    g6_original = ["dermlip", "dermlip_clinic"]

    stacking_configs = []
    seen_sets = set()
    for label, views in (
        ("top1_view", top1),
        ("top2_views", top2),
        ("top3_views", top3),
        ("g6_original_dermlip_plus_clinic", g6_original),
    ):
        key = tuple(sorted(views))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        stacking_configs.append(run_stacking(all_view_paths, cases, views, label))
        print(f"[L6] {label} {views}: AUROC={stacking_configs[-1]['auroc_valid']} "
              f"J@target0.70={stacking_configs[-1]['at_target_sensitivity_0_70']['valid']['youden_j']} "
              f"J@Youden={stacking_configs[-1]['at_youden_j']['valid']['youden_j']}")

    union_valid_j = 0.5641
    g6_recorded_j = 0.6063
    best_stack = max(
        stacking_configs,
        key=lambda r: max(
            r["at_target_sensitivity_0_70"]["valid"]["youden_j"] or -2.0,
            r["at_youden_j"]["valid"]["youden_j"] or -2.0,
        ),
    )
    best_stack_j = max(
        best_stack["at_target_sensitivity_0_70"]["valid"]["youden_j"] or -2.0,
        best_stack["at_youden_j"]["valid"]["youden_j"] or -2.0,
    )

    stacking_report = {
        "single_view_mel_auroc_scan": scan,
        "configs": stacking_configs,
        "comparison": {
            "union_valid_J": union_valid_j,
            "g6_recorded_stacked_J": g6_recorded_j,
            "phase1_best_stacked_J": round(best_stack_j, 4),
            "phase1_best_stacked_label": best_stack["label"],
            "beats_union": bool(best_stack_j > union_valid_j),
            "beats_g6_recorded": bool(best_stack_j > g6_recorded_j),
        },
    }

    # ---- operating-point policy on the FINAL diagnosis config ----
    op_policy = operating_point_policy(diagnosis["_winner_full_posterior"], diagnosis["_y_valid"], diagnosis["_labels"])

    # ---- negatives (levers that did not help) ----
    negatives = []
    for row in diagnosis["single_lever_ablation_table"][1:]:
        if row["delta_balanced_accuracy_vs_baseline"] <= 0:
            negatives.append(
                f"Diagnosis: {row['lever']} ({row['state']}) did NOT improve valid balanced accuracy "
                f"(delta {row['delta_balanced_accuracy_vs_baseline']:+.4f}; MEL AUROC delta "
                f"{row['delta_mel_auroc_vs_baseline']:+.4f})."
            )
    for row in criteria["single_lever_ablation_table"][1:]:
        if row["delta_macro_binary_f1_vs_baseline"] <= 0:
            negatives.append(
                f"Criteria (macro): {row['lever']} ({row['state']}) did NOT improve macro binary "
                f"concept F1 (delta {row['delta_macro_binary_f1_vs_baseline']:+.4f})."
            )
    for row in diagnosis["head_sweep_at_final_features"]:
        if row["valid"]["balanced_accuracy"] < diagnosis["baseline"]["valid"]["balanced_accuracy"]:
            negatives.append(
                f"Diagnosis head sweep: {row['family']} at the final feature config scored below the "
                f"plain-LR baseline on valid balanced accuracy ({row['valid']['balanced_accuracy']} < "
                f"{diagnosis['baseline']['valid']['balanced_accuracy']})."
            )
    if not stacking_report["comparison"]["beats_g6_recorded"]:
        negatives.append(
            f"L6: the best phase-1 stacking config (J={stacking_report['comparison']['phase1_best_stacked_J']}) "
            f"did not beat the previously recorded G6 stacked valid J ({g6_recorded_j})."
        )

    report = {
        "protocol": "derm7pt-perception-uplift-phase1-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "environment": {
            "interpreter": "venv_win/Scripts/python.exe",
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "note": (
                "The task brief said interpreter `py -3.13`. Verified: native py -3.13 "
                "(C:\\Users\\MSI\\AppData\\Local\\Programs\\Python\\Python313) lacks open_clip/"
                "huggingface_hub/PIL, so it cannot run the encoder. venv_win (Python 3.12.10, "
                "sklearn 1.8.0) is the environment that actually produced every existing cached "
                "artifact this phase reads (perception_baseline.json records sklearn_version "
                "1.8.0 / python_version 3.12.10 exactly) -- used here for bit-for-bit continuity "
                "with prior numbers. Documented as an inferred deviation, not a silent one."
            ),
        },
        "scope": {
            "phase": 1,
            "experimentation_only": True,
            "medguard_v2_source_edited": False,
            "test_split_opened": False,
            "selection_surface": "train (cross-validation) and valid only",
        },
        "reference_baselines": {
            "test_split_context_only_never_reproduced_here": {
                "diagnosis_accuracy_test": 0.7392,
                "mel_auroc_test": 0.895,
                "criteria_binary_concept_f1_test": 0.6623,
                "ham_7class_test": 0.8514,
                "source": "perception_baseline.json / test_metrics_complete.json; listed only for orientation, never reproduced here",
            },
            "valid_split_reproduced_here": {
                "diagnosis_current_production_fixedC10": {
                    "accuracy": prod_metrics["accuracy"],
                    "balanced_accuracy": prod_metrics["balanced_accuracy"],
                    "macro_f1": prod_metrics["macro_f1"],
                    "mel_auroc": prod_metrics["mel_auroc"],
                    "note": "dermlip derm-only, LogisticRegression(C=10.0) -- exactly medguard_v2.pipeline.run_cohort.fit_readers",
                },
                "diagnosis_probe_py_frozen_artifact_valid_accuracy": probe_valid_accuracy,
                "criteria_current_production_fixedC10_per_criterion_binary_f1": {
                    name: round(v, 4) for name, v in zip(CRITERION_NAMES, prod_criteria_f1)
                },
                "criteria_current_production_fixedC10_macro_binary_f1": prod_criteria_macro_f1,
            },
        },
        "features": {
            "existing_reused_readonly": sorted(g6_discover_feature_sets().keys()),
            "phase1_new_readonly_root": str(FEATURE_ROOT_P1.relative_to(REPO)).replace("\\", "/"),
            "phase1_new": sorted(k for k in bundles if k in
                                  ("dermlip_flip", "dermlip_clinic_flip", "dermlip_tta", "dermlip_clinic_tta", "dermlip_zeroshot5")),
            "metadata_encoding": "v3 encode_derm7pt: sex(2) + location(9) + elevation(3) = 14-d one-hot (medguard_v2.perception.clinical_metadata)",
            "note": "Phase-1 caches live under features_phase1/ (not features/) so G6's discover_feature_sets() -- hardcoded to features/ -- never auto-picks them up.",
        },
        "diagnosis": {k: v for k, v in diagnosis.items() if not k.startswith("_")},
        "criteria": criteria,
        "stacking_L6": stacking_report,
        "operating_point_policy": op_policy,
        "negatives": negatives,
        "projected_impact": {
            "criteria_binary_f1_dermagent_published_test": 0.6506,
            "criteria_binary_f1_our_test_current": 0.6623,
            "criteria_binary_f1_valid_current_production_fixedC10": prod_criteria_macro_f1,
            "criteria_binary_f1_valid_phase1_final": criteria["final_config"]["macro_binary_concept_f1"],
            "diagnosis_accuracy_valid_current_production_fixedC10": prod_metrics["accuracy"],
            "diagnosis_accuracy_valid_phase1_final": diagnosis["final_config"]["valid"]["accuracy"],
            "interpretation": (
                "Phase 1 only ever scores valid; test is untouched by policy. The valid-vs-valid "
                "deltas above are what phase 2 would need to reproduce on test before any paper "
                "number changes. They are NOT test-split projections and must not be read as ones."
            ),
            "paper_table_rows_that_would_change": [
                "perception_baseline.json (diagnosis 5-class accuracy/balanced_accuracy/MEL AUROC) IF phase 2 confirms the gain on test",
                "test_metrics_complete.json criteria_aggregate.binary_concept_f1_macro (0.6623) and the DermAgent comparison row (0.6506) IF the criteria final config is promoted",
                "g6_multiview_stacking.json verdict.youden_j_stacked (0.6063) IF the L6 phase-1 view choice or the head/feature levers are folded into a rebuilt G6",
            ],
        },
        "ham_bcn_transfer_notes": {
            "L1_clinic_second_view": "Does NOT transfer to HAM10000 -- HAM ships only a single dermoscopic image per lesion, no paired clinical photo.",
            "L2_metadata": "Historically ~flat on HAM in prior v3 evaluation (age/sex/localization concat); expected to remain a small effect there, unlike the modest gain (if any) measured here on Derm7pt.",
            "L3_TTA": "Transfers directly -- horizontal-flip averaging only needs the image, works identically for HAM/BCN.",
            "L4_head_sweep": "Transfers directly -- the model-family comparison is dataset-agnostic.",
            "L5_zero_shot_text_tower": "Transfers if the class vocabulary is re-prompted for HAM's 7 classes (akiec/bcc/bkl/df/melanoma/nevus/vasc); prompts here are Derm7pt-specific (5-class) and are NOT reusable as-is.",
            "L6_stacking": "The G6 stacking machinery is Derm7pt-specific (reads Derm7ptCase); a HAM port would need an equivalent view-building step, not a straight reuse.",
        },
        "phase2_promotion_plan": [
            "Freeze the final diagnosis config (features + family + hyperparameters) and final criteria config exactly as recorded in this artifact's final_config blocks -- no re-tuning at promotion time.",
            "If diagnosis features changed (e.g. +clinic concat or TTA), re-point medguard_v2/perception/probe.py's TARGETS['diagnosis'] fit to consume the new feature matrix, OR add a new probe module rather than editing the frozen one in place -- decide based on whether v1-vs-v2 comparability still matters at that point.",
            "If the criteria final config changes the feature representation, medguard_v2/pipeline/run_cohort.py:fit_readers and medguard_v2/eval/gates/g1b_trained_criterion_heads.py both need the new feature matrix wired in; BASE_C=10.0 there would need to become the phase-1-selected C (or family) per head.",
            "Re-run scripts/dev/build_test_metrics_complete.py after promotion so binary_concept_f1_macro and the DermAgent comparison row are re-derived on TEST under the new config -- this is the one-shot that actually tells us whether the valid-split gain measured here survives on the untouched split.",
            "If L6's chosen view set changes, medguard_v2/eval/gates/g6_multiview_stacking.py's discover_feature_sets() should be pointed at (or merged with) features_phase1/ so the gate can see dermlip_tta/dermlip_clinic_tta -- currently deliberately invisible to it (see features.note above).",
            "Re-run the full suite (`py -3.13 -m unittest discover -s medguard_v2/tests -q`, or venv_win's python if py -3.13 still lacks open_clip/torch parity) after any promotion edit.",
        ],
        "suite_check": {
            "tests_run": None,
            "failures": None,
            "detail": "Filled in after this script finishes; see the calling agent's final suite run.",
        },
    }

    elapsed_total = time.perf_counter() - started
    report["total_elapsed_seconds"] = round(elapsed_total, 1)
    print(f"\nTOTAL elapsed: {elapsed_total:.1f}s")

    clean = _to_jsonable(report)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(clean, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")

    md = build_markdown(clean)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
