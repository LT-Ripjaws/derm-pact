"""Frozen linear probe over the DermLIP features -- the classifier-only baseline.

This is arm ``R0`` (image in, label out, no criteria, no retrieval, no LLM) and
the classifier half of gate ``G5`` (does classifier-vs-criteria disagreement
predict error?).  It deliberately is not a strong model: a multinomial logistic
regression on frozen, L2-normalised features measures how much of the task the
*encoder* already solves, which is the number every later component has to beat.

Two probes are fit, both on the official **train** split only:

* ``diagnosis``   -- 5 classes (BCC / NEV / MEL / SK / MISC)
* ``management`` -- 3 classes (no further examination / clinical follow up / excision)

Discipline
----------
**Nothing is fit on ``valid`` or ``test``.**  That includes the two things it is
tempting to tune there:

* the regularisation strength ``C`` is chosen by 5-fold stratified
  cross-validation *inside train*, by mean out-of-fold log-loss.  Log-loss is a
  proper scoring rule, so the choice does not quietly optimise one of the
  accuracy numbers that get reported;
* the calibration temperature is fit on out-of-fold ``train`` logits, again
  never on ``valid``.  Temperature scaling divides the logits by a positive
  scalar, so it **cannot change any argmax** -- accuracy is identical with and
  without it, and only the confidences move.  Held-out ECE is reported both raw
  and calibrated as a diagnostic, but neither was used to pick anything.

The honest caveat about the out-of-fold temperature: it is fit from models
trained on 4/5 of train, which are slightly less sharp than the final model
trained on all of train, so ``T`` tends to over-smooth a little.  That is the
price of refusing to touch ``valid``, and it is visible in the reported ECEs.

Test-split numbers appear in the artifact because this is a **baseline
characterisation**, not a confirmatory claim; the header marks them as such.

CLI::

    python -m medguard_v2.perception.probe            # writes the frozen artifact
    python -m medguard_v2.perception.probe --dry-run  # report only, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
# NOTE: sklearn's ``log_loss`` is deliberately NOT used here. It re-sorts its ``labels``
# argument lexicographically without reordering the probability columns, which silently
# mis-aligns truth against predictions for any non-lexicographic vocabulary. Out-of-fold
# log-loss is computed by ``_nll`` instead, which indexes both by the same ``labels`` order.
from sklearn.model_selection import StratifiedKFold

from medguard_v2.criteria.aggregate import MANAGEMENT_LADDER
from medguard_v2.data.derm7pt import (
    DIAGNOSIS_GROUP_ORDER,
    Derm7ptCase,
    MANAGEMENT_VOCABULARY,
    SPLITS,
    load_cases,
)
from medguard_v2.eval.metrics import classification_summary, ece, quadratic_weighted_kappa
from medguard_v2.perception.encode import FeatureBundle, encode_cases, feature_path

__all__ = [
    "SEED",
    "C_GRID",
    "CV_FOLDS",
    "TARGETS",
    "ARTIFACT_PATH",
    "ProbeFit",
    "TargetSpec",
    "fit_probe",
    "posterior",
    "build_baseline",
    "write_baseline",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "research" / "evidence" / "derm7pt_v1" / "perception_baseline.json"

SEED = 20260810
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
MAX_ITER = 5000
FIT_SPLIT = "train"
PROTOCOL = "derm7pt-perception-baseline-v1"

#: Posteriors are rounded before they are written, so the file is readable and
#: byte-stable.  Six decimals keeps the sum-to-one error below 5e-6.
POSTERIOR_DECIMALS = 6

if set(MANAGEMENT_LADDER) != set(MANAGEMENT_VOCABULARY):
    raise RuntimeError(
        "MANAGEMENT_LADDER is not a permutation of MANAGEMENT_VOCABULARY: "
        f"{MANAGEMENT_LADDER} vs {MANAGEMENT_VOCABULARY}"
    )


@dataclass(frozen=True)
class TargetSpec:
    name: str
    attribute: str
    labels: tuple[str, ...]
    #: Least-to-most aggressive order, for ordinal metrics.  ``None`` when the
    #: target is nominal (diagnosis has no ordering).
    ordinal: tuple[str, ...] | None = None


TARGETS: Mapping[str, TargetSpec] = {
    "diagnosis": TargetSpec("diagnosis", "diagnosis_group", tuple(DIAGNOSIS_GROUP_ORDER)),
    "management": TargetSpec(
        "management", "management", tuple(MANAGEMENT_VOCABULARY), tuple(MANAGEMENT_LADDER)
    ),
}


@dataclass(frozen=True)
class ProbeFit:
    """One fitted probe plus everything needed to reproduce and audit it."""

    target: str
    labels: tuple[str, ...]
    model: LogisticRegression
    temperature: float
    selected_c: float
    fit_case_nums: tuple[int, ...]
    cv: dict


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def _new_model(c: float) -> LogisticRegression:
    # solver=lbfgs on a multiclass target is multinomial (softmax) in sklearn >=1.5.
    return LogisticRegression(C=float(c), max_iter=MAX_ITER, random_state=SEED, solver="lbfgs")


def _logits(model: LogisticRegression, x: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Decision scores re-indexed into the canonical label order."""
    scores = np.asarray(model.decision_function(x), dtype=np.float64)
    if scores.ndim == 1:  # pragma: no cover - both targets are multiclass
        scores = np.column_stack([-scores, scores])
    order = [list(model.classes_).index(label) for label in labels]
    return scores[:, order]


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(scores: np.ndarray, y_index: np.ndarray, temperature: float) -> float:
    probabilities = _softmax(scores, temperature)
    picked = probabilities[np.arange(len(y_index)), y_index]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def _out_of_fold_logits(
    x: np.ndarray, y: Sequence[str], labels: Sequence[str], c: float
) -> np.ndarray:
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros((len(y), len(labels)), dtype=np.float64)
    y_array = np.asarray(y)
    for train_index, held_index in folds.split(x, y_array):
        model = _new_model(c)
        model.fit(x[train_index], y_array[train_index])
        scores[held_index] = _logits(model, x[held_index], labels)
    return scores


def fit_probe(
    x_train: np.ndarray,
    y_train: Sequence[str],
    labels: Sequence[str],
    *,
    target: str,
    fit_case_nums: Sequence[int],
) -> ProbeFit:
    """Fit one probe on train only: select ``C``, fit ``T``, refit on all of train."""
    labels = tuple(labels)
    y_array = np.asarray(y_train)
    missing = sorted(set(y_array) - set(labels))
    if missing:
        raise ValueError(f"training labels outside the declared vocabulary: {missing}")
    absent = sorted(set(labels) - set(y_array))
    if absent:
        raise ValueError(f"labels with no training example: {absent}")

    grid: list[dict] = []
    oof_by_c: dict[float, np.ndarray] = {}
    y_index = np.asarray([labels.index(value) for value in y_array], dtype=int)
    for c in C_GRID:
        scores = _out_of_fold_logits(x_train, y_array, labels, c)
        oof_by_c[float(c)] = scores
        probabilities = _softmax(scores)
        predicted = [labels[i] for i in probabilities.argmax(axis=1)]
        summary = classification_summary(list(y_array), predicted, list(labels))
        grid.append(
            {
                "C": float(c),
                # Computed directly from the score matrix and ``y_index``, both of which are
                # ordered by ``labels``.  sklearn's ``log_loss`` re-sorts its ``labels``
                # argument lexicographically while leaving the probability columns alone, so
                # passing a canonically ordered vocabulary (BCC, NEV, MEL, SK, MISC) silently
                # mis-aligned truth against probabilities and made this selection meaningless
                # -- it chose the most-regularised C, collapsing diagnosis to the majority
                # class at balanced accuracy 0.2000.
                "oof_log_loss": round(float(_nll(scores, y_index, 1.0)), 6),
                "oof_accuracy": round(float(summary["accuracy"]), 4),
                "oof_balanced_accuracy": round(float(summary["balanced_accuracy"]), 4),
                "oof_macro_f1": round(float(summary["macro_f1"]), 4),
            }
        )

    # Selection rule fixed in advance: lowest out-of-fold log-loss, ties broken
    # toward the smaller C (more regularisation).  Log-loss rather than accuracy
    # so the choice does not pre-optimise a reported headline metric.
    best = min(grid, key=lambda row: (row["oof_log_loss"], row["C"]))
    selected_c = float(best["C"])

    temperature = _fit_temperature(oof_by_c[selected_c], y_index)

    model = _new_model(selected_c)
    model.fit(x_train, y_array)

    cv = {
        "folds": CV_FOLDS,
        "scheme": "StratifiedKFold(shuffle=True, random_state=SEED)",
        "selection_metric": "mean out-of-fold multinomial log-loss (lower is better)",
        "grid": grid,
        "selected_C": selected_c,
        "temperature_fit_on": "out-of-fold train logits at the selected C",
        "oof_nll_before_temperature": round(_nll(oof_by_c[selected_c], y_index, 1.0), 6),
        "oof_nll_after_temperature": round(_nll(oof_by_c[selected_c], y_index, temperature), 6),
    }
    return ProbeFit(
        target=target,
        labels=labels,
        model=model,
        temperature=temperature,
        selected_c=selected_c,
        fit_case_nums=tuple(int(value) for value in fit_case_nums),
        cv=cv,
    )


def _fit_temperature(scores: np.ndarray, y_index: np.ndarray) -> float:
    """Scalar ``T > 0`` minimising NLL of ``softmax(scores / T)``.

    Bounded scalar minimisation over ``log T`` is deterministic and needs no
    starting point; the bounds are wide enough that hitting one would itself be
    a signal that something is wrong (asserted below).
    """
    lo, hi = np.log(0.05), np.log(20.0)
    result = minimize_scalar(
        lambda log_t: _nll(scores, y_index, float(np.exp(log_t))),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-6},
    )
    temperature = float(np.exp(float(result.x)))
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"temperature search returned {temperature!r}")
    if temperature <= 0.06 or temperature >= 19.0:
        raise RuntimeError(
            f"temperature {temperature:.4f} hit the search bounds; the logits are not "
            "on the scale this code assumes"
        )
    return temperature


def posterior(fit: ProbeFit, x: np.ndarray, *, calibrated: bool = True) -> np.ndarray:
    """Class posteriors in ``fit.labels`` order."""
    scores = _logits(fit.model, x, fit.labels)
    return _softmax(scores, fit.temperature if calibrated else 1.0)


# --------------------------------------------------------------------------
# artifact
# --------------------------------------------------------------------------

def _summarise(
    spec: TargetSpec,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidence: Sequence[float],
    raw_confidence: Sequence[float],
) -> dict:
    summary = classification_summary(list(y_true), list(y_pred), list(spec.labels))
    correct = [t == p for t, p in zip(y_true, y_pred)]
    block = {
        "n": int(summary["n"]),
        "accuracy": round(float(summary["accuracy"]), 4),
        "balanced_accuracy": round(float(summary["balanced_accuracy"]), 4),
        "macro_f1": round(float(summary["macro_f1"]), 4),
        "weighted_f1": round(float(summary["weighted_f1"]), 4),
        "cohen_kappa": round(float(summary["cohen_kappa"]), 4),
        "mcc": round(float(summary["mcc"]), 4),
        "labels": list(spec.labels),
        "confusion_matrix": summary["confusion_matrix"],
        "per_class": {
            label: {
                "precision": round(float(values["precision"]), 4),
                "recall": round(float(values["recall"]), 4),
                "f1": round(float(values["f1"]), 4),
                "support": int(values["support"]),
            }
            for label, values in summary["per_class"].items()
        },
        "mean_confidence": round(float(np.mean(confidence)), 4),
        "ece_calibrated": ece(list(confidence), correct),
        "ece_uncalibrated": ece(list(raw_confidence), correct),
    }
    if spec.ordinal is not None:
        ladder = {label: index for index, label in enumerate(spec.ordinal)}
        block["quadratic_weighted_kappa"] = quadratic_weighted_kappa(
            [ladder[value] for value in y_true],
            [ladder[value] for value in y_pred],
            n_classes=len(spec.ordinal),
        )
        block["ordinal_order"] = list(spec.ordinal)
        under = sum(1 for t, p in zip(y_true, y_pred) if ladder[p] < ladder[t])
        over = sum(1 for t, p in zip(y_true, y_pred) if ladder[p] > ladder[t])
        block["under_managed"] = under
        block["over_managed"] = over
        block["under_management_rate"] = round(under / len(y_true), 4)
    return block


def build_baseline(
    cases: Sequence[Derm7ptCase] | None = None,
    bundle: FeatureBundle | None = None,
) -> dict:
    """Fit both probes on train and characterise them on every split."""
    cases = list(cases if cases is not None else load_cases())
    bundle = bundle if bundle is not None else encode_cases(cases)

    ordered = sorted(cases, key=lambda case: case.case_num)
    by_split = {split: [c for c in ordered if c.split == split] for split in SPLITS}
    train_cases = by_split[FIT_SPLIT]
    if not train_cases:
        raise RuntimeError(f"no cases in the {FIT_SPLIT!r} split")

    x_all = bundle.matrix_for([case.case_num for case in ordered])
    x_train = bundle.matrix_for([case.case_num for case in train_cases])

    fits: dict[str, ProbeFit] = {}
    posteriors: dict[str, np.ndarray] = {}
    raw_posteriors: dict[str, np.ndarray] = {}
    for name, spec in TARGETS.items():
        fits[name] = fit_probe(
            x_train,
            [getattr(case, spec.attribute) for case in train_cases],
            spec.labels,
            target=name,
            fit_case_nums=[case.case_num for case in train_cases],
        )
        posteriors[name] = posterior(fits[name], x_all, calibrated=True)
        raw_posteriors[name] = posterior(fits[name], x_all, calibrated=False)

    # Per-case records.  Posteriors are rounded for readability; the argmax label
    # is computed before rounding so a rounding tie can never move a prediction.
    records: list[dict] = []
    predictions: dict[str, list[str]] = {name: [] for name in TARGETS}
    for row, case in enumerate(ordered):
        record: dict = {"case_num": int(case.case_num), "split": case.split}
        for name, spec in TARGETS.items():
            probabilities = posteriors[name][row]
            index = int(probabilities.argmax())
            predictions[name].append(spec.labels[index])
            record[name] = {
                "truth": getattr(case, spec.attribute),
                "prediction": spec.labels[index],
                "confidence": round(float(probabilities[index]), POSTERIOR_DECIMALS),
                "posterior": {
                    label: round(float(value), POSTERIOR_DECIMALS)
                    for label, value in zip(spec.labels, probabilities)
                },
            }
        records.append(record)

    index_by_split = {
        split: [i for i, case in enumerate(ordered) if case.split == split] for split in SPLITS
    }
    metrics: dict[str, dict] = {}
    for name, spec in TARGETS.items():
        per_split: dict[str, dict] = {}
        for split in SPLITS:
            rows = index_by_split[split]
            truth = [getattr(ordered[i], spec.attribute) for i in rows]
            predicted = [predictions[name][i] for i in rows]
            confidence = [float(posteriors[name][i].max()) for i in rows]
            raw_confidence = [float(raw_posteriors[name][i].max()) for i in rows]
            block = _summarise(spec, truth, predicted, confidence, raw_confidence)
            block["in_sample"] = split == FIT_SPLIT
            per_split[split] = block
        metrics[name] = per_split

    header = {
        "protocol": PROTOCOL,
        "artifact": "derm7pt perception baseline (frozen encoder + linear probe)",
        "purpose": (
            "classifier-only arm R0, and the classifier side of gate G5 "
            "(does classifier-vs-criteria disagreement predict error?)"
        ),
        "status": {
            "kind": "baseline_characterisation",
            "confirmatory": False,
            "test_split_reported": True,
            "test_split_used_for_fitting": False,
            "valid_split_used_for_fitting": False,
            "note": (
                "Test-split numbers are reported here because this artifact CHARACTERISES a "
                "baseline; they are not a confirmatory system claim and must not be cited as "
                "one. Nothing -- not the probe weights, not C, not the temperature -- was fit "
                "or selected on valid or test."
            ),
        },
        "dataset": {
            "name": "Derm7pt",
            "license": "CC BY-NC-ND (no image bytes are stored in this artifact)",
            "n_cases": len(ordered),
            "image_view": bundle.manifest.get("image_view"),
            "split_sizes": {split: len(by_split[split]) for split in SPLITS},
            "image_set_sha256": bundle.manifest.get("image_set_sha256"),
            "class_counts": {
                name: {
                    split: {
                        label: sum(
                            1
                            for case in by_split[split]
                            if getattr(case, spec.attribute) == label
                        )
                        for label in spec.labels
                    }
                    for split in SPLITS
                }
                for name, spec in TARGETS.items()
            },
        },
        "encoder": {
            "encoder_key": bundle.encoder_key,
            "model_id": bundle.manifest["provenance"]["model_id"],
            "resolved_model_id": bundle.manifest["provenance"]["resolved_model_id"],
            "loader": bundle.manifest["provenance"]["loader"],
            "selection_reason": bundle.manifest["provenance"].get("selection_reason"),
            "snapshot_revision": bundle.manifest["provenance"]["checkpoint"]["snapshot_revision"],
            "checkpoint_sha256": bundle.manifest["provenance"]["checkpoint"]["combined_sha256"],
            "weight_files": bundle.manifest["provenance"]["checkpoint"]["weight_files"],
            "parameters": bundle.manifest["provenance"]["parameters"],
            "frozen": True,
            "feature_dim": bundle.dim,
            "l2_normalised": True,
            "features_sha256": bundle.manifest.get("features_sha256"),
            "feature_cache": str(feature_path(bundle.encoder_key).relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "open_clip_version": bundle.manifest["provenance"].get("open_clip_version"),
            "torch_version": bundle.manifest["provenance"].get("torch_version"),
        },
        "probe": {
            "estimator": "sklearn.linear_model.LogisticRegression(solver='lbfgs', multinomial)",
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "seed": SEED,
            "max_iter": MAX_ITER,
            "C_grid": list(C_GRID),
            "class_weight": None,
            "fit_split": FIT_SPLIT,
            "n_fit_cases": len(train_cases),
            "fit_case_nums": [int(case.case_num) for case in train_cases],
            "fit_case_nums_sha256": hashlib.sha256(
                ",".join(str(case.case_num) for case in train_cases).encode("ascii")
            ).hexdigest(),
            "calibration": (
                "temperature scaling; T fit on out-of-fold TRAIN logits only. Dividing logits "
                "by T > 0 preserves every argmax, so predictions are identical with and "
                "without it."
            ),
            "posterior_decimals": POSTERIOR_DECIMALS,
            "targets": {
                name: {
                    "labels": list(fits[name].labels),
                    "selected_C": fits[name].selected_c,
                    "temperature": round(fits[name].temperature, 6),
                    "cross_validation": fits[name].cv,
                }
                for name in TARGETS
            },
        },
        "metrics": metrics,
    }
    header["cases"] = records
    return header


def write_baseline(artifact: dict, path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def _report(artifact: dict) -> None:
    encoder = artifact["encoder"]
    print(f"encoder     : {encoder['encoder_key']} ({encoder['model_id']})")
    print(f"reason      : {encoder['selection_reason']}")
    print(f"checkpoint  : {encoder['checkpoint_sha256'][:16]}... rev {encoder['snapshot_revision']}")
    print(f"fit on      : {artifact['probe']['fit_split']} "
          f"n={artifact['probe']['n_fit_cases']} seed={artifact['probe']['seed']}")
    for name in TARGETS:
        target = artifact["probe"]["targets"][name]
        print(f"\n{name}: C={target['selected_C']} T={target['temperature']:.4f}")
        print(f"  {'split':<7}{'n':>5}{'acc':>9}{'bal_acc':>9}{'macroF1':>9}{'ECE':>8}")
        for split in SPLITS:
            block = artifact["metrics"][name][split]
            flag = " (in-sample)" if block["in_sample"] else ""
            print(
                f"  {split:<7}{block['n']:>5}{block['accuracy']:>9.4f}"
                f"{block['balanced_accuracy']:>9.4f}{block['macro_f1']:>9.4f}"
                f"{block['ece_calibrated']:>8.4f}{flag}"
            )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--out", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args()

    artifact = build_baseline()
    _report(artifact)
    if args.dry_run:
        print("\n(dry run: no artifact written)")
        return
    path = write_baseline(artifact, args.out)
    print(f"\nwrote {path}")


if __name__ == "__main__":  # pragma: no cover
    _main()
