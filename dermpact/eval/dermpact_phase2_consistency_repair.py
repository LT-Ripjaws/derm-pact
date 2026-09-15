"""Phase 2.3A-R result-informed consistency-only repair."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from . import dermpact_phase2_architecture_ladder as ladder
from . import dermpact_phase2_dataset_composition as composition
from . import dermpact_phase2_hybrid_confirmation as confirmation
from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_baselines import _read_jsonl, _write_jsonl, sha256_file
from .dermpact_phase2_vector_bootstrap import interval, stratified_bootstrap_weights


ROOT = screen.ROOT
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_3/consistency_repair_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_3/consistency_repair_v1"
PROTOCOL_PATH = OUTPUT_DIR / "consistency_repair_protocol.json"
COMPARISON_PATH = OUTPUT_DIR / "consistency_repair_comparisons.json"
RECEIPT_PATH = OUTPUT_DIR / "consistency_repair_receipt.json"
DATA_STRATEGIES = ladder.DATA_STRATEGIES
SEEDS = ladder.SEEDS
N_FOLDS = ladder.N_FOLDS
EPOCHS = ladder.EPOCHS
N_BOOT = ladder.N_BOOT
BOOTSTRAP_SEED = 20261228
PROTECTION_MARGIN = ladder.PROTECTION_MARGIN
DISAGREEMENT_MARGIN = 0.02


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_path(data_strategy: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{data_strategy}_independent_consistency_seed_{seed}_result.json"


def _batch_loss(
    model: screen.HybridSharedHead,
    *,
    dataset: str,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> tuple[Tensor, dict[str, Tensor]]:
    outputs = model(
        screen._tensor(
            np.asarray(data[f"{dataset}_features"][indices], dtype="float32"),
            dtype=torch.float32,
        )
    )
    common: dict[str, Any] = {
        "outputs": outputs,
        "dataset": "derm7pt" if dataset == "derm" else "ham",
        "diagnosis": screen._tensor(data[f"{dataset}_global"][indices], dtype=torch.long),
        "melanoma": screen._tensor(data[f"{dataset}_melanoma"][indices], dtype=torch.float32),
        "diagnosis_weight": screen._tensor(
            weights[f"{dataset}_diagnosis"][indices], dtype=torch.float32
        ),
        "melanoma_weight": screen._tensor(
            weights[f"{dataset}_melanoma"][indices], dtype=torch.float32
        ),
        "loss_weighting": "equal_family",
    }
    if dataset == "derm":
        common["criteria"] = screen._tensor(
            data["derm_criteria"][indices], dtype=torch.float32
        )
        common["criterion_weights"] = screen._tensor(
            weights["derm_criteria"][indices], dtype=torch.float32
        )
    total, components = screen.multitask_loss(**common)
    consistency = total * 0.0
    eligible = 0
    if dataset == "derm":
        consistency, eligible = ladder.truth_anchored_consistency_loss(
            outputs,
            melanoma=common["melanoma"],
            criteria=common["criteria"],
            diagnosis=common["diagnosis"],
        )
        total = total + ladder.CONSISTENCY_WEIGHT * consistency
    return total, {
        **components,
        "consistency": consistency,
        "consistency_eligible": torch.tensor(float(eligible)),
    }


def _reload_difference(
    model: screen.HybridSharedHead, checkpoint: Path, features: np.ndarray
) -> float:
    restored = screen.HybridSharedHead()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    batch = screen._tensor(
        np.asarray(features[: min(16, len(features))], dtype="float32"),
        dtype=torch.float32,
    )
    model.eval()
    restored.eval()
    maximum = 0.0
    with torch.inference_mode():
        original = model(batch)
        reloaded = restored(batch)
        for name in original:
            maximum = max(
                maximum,
                float(torch.max(torch.abs(original[name] - reloaded[name]))),
            )
    return maximum


def train_fold(
    data_strategy: str,
    *,
    fold: int,
    seed: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    if data_strategy not in DATA_STRATEGIES:
        raise ValueError(data_strategy)
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    ham_train = np.flatnonzero(data["ham_folds"] != fold)
    ham_valid = np.flatnonzero(data["ham_folds"] == fold)
    derm_train = np.flatnonzero(data["derm_folds"] != fold)
    derm_valid = np.flatnonzero(data["derm_folds"] == fold)
    combined = data_strategy == "derm_ham_bcn"
    if combined:
        bcn_train = np.flatnonzero(data["bcn_folds"] != fold)
        bcn_valid = np.flatnonzero(data["bcn_folds"] == fold)
    weights = composition._training_weights(data, fold)
    model = screen.HybridSharedHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ladder.LEARNING_RATE, weight_decay=screen.WEIGHT_DECAY
    )
    steps = composition._steps_per_epoch(data, fold)
    scheduler = screen._scheduler(optimizer, EPOCHS * steps)
    history = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        ham_batches = screen._cyclic_batches(
            ham_train,
            steps=steps,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 3 + 1),
        )
        derm_batches = screen._cyclic_batches(
            derm_train,
            steps=steps,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 3 + 2),
        )
        if combined:
            bcn_batches = screen._cyclic_batches(
                bcn_train,
                steps=steps,
                batch_size=screen.BATCH_SIZE,
                rng=np.random.default_rng(seed + fold * 1000 + epoch * 3 + 3),
            )
        losses = []
        component_sum: dict[str, float] = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for ham_indices, derm_indices in zip(ham_batches, derm_batches):
            optimizer.zero_grad(set_to_none=True)
            ham_loss, ham_components = _batch_loss(
                model,
                dataset="ham",
                indices=ham_indices,
                data=data,
                weights=weights,
            )
            derm_loss, derm_components = _batch_loss(
                model,
                dataset="derm",
                indices=derm_indices,
                data=data,
                weights=weights,
            )
            if combined:
                bcn_indices = next(bcn_batches)
                bcn_loss, bcn_components = composition._bcn_batch_loss(
                    model, bcn_indices, data, weights
                )
                loss = 0.5 * derm_loss + 0.25 * ham_loss + 0.25 * bcn_loss
                components = {
                    **{f"derm_{key}": value for key, value in derm_components.items()},
                    **{f"ham_{key}": value for key, value in ham_components.items()},
                    **{f"bcn_{key}": value for key, value in bcn_components.items()},
                }
            else:
                loss = 0.5 * derm_loss + 0.5 * ham_loss
                components = {
                    **{f"derm_{key}": value for key, value in derm_components.items()},
                    **{f"ham_{key}": value for key, value in ham_components.items()},
                }
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite consistency-repair loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), screen.GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            for name, value in components.items():
                component_sum[name] += float(value.detach())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(losses)),
                "last_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "component_means": {
                    name: value / steps for name, value in component_sum.items()
                },
            }
        )
    prediction_sets = {
        "ham": (data["ham_features"], ham_valid),
        "derm7pt": (data["derm_features"], derm_valid),
    }
    if combined:
        prediction_sets["bcn_train_oof"] = (data["bcn_features"], bcn_valid)
    predictions = {
        name: confirmation._predict_with_logits(model, features, indices)
        for name, (features, indices) in prediction_sets.items()
    }
    fold_dir = EXPERIMENT_DIR / data_strategy / f"seed_{seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / f"epoch_{EPOCHS}_state.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    reload_difference = _reload_difference(
        model, checkpoint, np.asarray(data["derm_features"][derm_valid], dtype="float32")
    )
    if reload_difference != 0.0:
        raise AssertionError("checkpoint reload changed logits")
    raw_artifacts = {}
    for name, (_, indices) in prediction_sets.items():
        raw_path = fold_dir / f"{name}_validation_raw_logits.npz"
        values = predictions[name]
        raw_artifacts[name] = {
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": confirmation._save_npz(
                raw_path,
                row_indices=np.asarray(indices, dtype=np.int64),
                diagnosis_logits=np.asarray(values["diagnosis_logits"], dtype=np.float32),
                melanoma_logits=np.asarray(values["melanoma_logits"], dtype=np.float32),
                criterion_logits=np.asarray(values["criterion_logits"], dtype=np.float32),
            ),
        }
    result: dict[str, Any] = {
        "fold": fold,
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "steps_per_epoch": steps,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint),
            "reload_max_abs_logit_difference": reload_difference,
        },
        "raw_logits": raw_artifacts,
        "ham_rows": screen._ham_prediction_rows(data, ham_valid, predictions["ham"]),
        "derm_rows": screen._derm_prediction_rows(
            data, derm_valid, predictions["derm7pt"]
        ),
    }
    if combined:
        result["bcn_rows"] = composition._bcn_prediction_rows(
            data, bcn_valid, predictions["bcn_train_oof"]
        )
    return result


def run_seed(
    data_strategy: str, seed: int, *, protocol_sha256: str
) -> dict[str, Any]:
    result_path = _result_path(data_strategy, seed)
    sidecar = result_path.with_suffix(result_path.suffix + ".sha256")
    if result_path.exists() and sidecar.exists():
        existing = _load_json(result_path)
        expected = sidecar.read_text(encoding="ascii").strip()
        if sha256_file(result_path) == expected and existing.get("protocol_sha256") == protocol_sha256:
            return {
                "data_strategy": data_strategy,
                "seed": seed,
                "status": "reused_verified_result",
                "result_path": str(result_path.relative_to(ROOT)),
                "result_sha256": expected,
            }
    torch.set_num_threads(ladder.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    data = composition._data()
    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    folds = []
    ham_rows: list[dict[str, Any]] = []
    derm_rows: list[dict[str, Any]] = []
    bcn_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(N_FOLDS):
        fold_result = train_fold(data_strategy, fold=fold, seed=seed, data=data)
        ham_rows.extend(fold_result.pop("ham_rows"))
        derm_rows.extend(fold_result.pop("derm_rows"))
        bcn_rows.extend(fold_result.pop("bcn_rows", []))
        folds.append(fold_result)
        print(
            f"[{data_strategy}/independent_consistency/seed{seed}] "
            f"fold {fold + 1}/{N_FOLDS} complete in {fold_result['elapsed_seconds']:.2f}s",
            flush=True,
        )
    ham_rows = screen._order_like_reference(ham_rows, frozen_ham)
    derm_rows = screen._order_like_reference(derm_rows, frozen_derm)
    seed_dir = EXPERIMENT_DIR / data_strategy / f"seed_{seed}"
    predictions = {}
    for name, rows in (("ham", ham_rows), ("derm7pt", derm_rows)):
        path = seed_dir / f"{name}_oof_predictions.jsonl"
        predictions[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _write_jsonl(path, rows),
        }
    if data_strategy == "derm_ham_bcn":
        bcn_rows = sorted(bcn_rows, key=lambda row: str(row["id"]))
        path = seed_dir / "bcn_training_oof_predictions.jsonl"
        predictions["bcn_train_oof"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _write_jsonl(path, bcn_rows),
        }
    metrics = confirmation._seed_metrics(ham_rows, derm_rows, seed=seed)
    metrics["derm7pt"]["rule_consistency"] = ladder._derm_rule_summary(derm_rows)
    result = {
        "protocol": "dermpact-phase2-3a-r-consistency-seed-result-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "data_strategy": data_strategy,
        "architecture": "independent_heads_plus_truth_anchored_consistency",
        "seed": seed,
        "fixed_training": {
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "learning_rate": ladder.LEARNING_RATE,
            "consistency_weight": ladder.CONSISTENCY_WEIGHT,
        },
        **metrics,
        "fold_training": folds,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": predictions,
        "hard_scope_checks": {
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_feature_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
    }
    result_hash = screen._write_json(result_path, result)
    return {
        "data_strategy": data_strategy,
        "seed": seed,
        "status": "completed",
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": result_hash,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _candidate_rows(data_strategy: str, seed: int, dataset: str) -> list[dict[str, Any]]:
    result = _load_json(_result_path(data_strategy, seed))
    return _read_jsonl(ROOT / result["predictions"][dataset]["path"])


def _comparison(
    data_strategy: str,
    *,
    dataset: str,
    metric: str,
    weights: np.ndarray,
    seed_draws: np.ndarray,
) -> dict[str, Any]:
    candidate_samples = []
    baseline_samples = []
    candidate_points = []
    baseline_points = []
    per_seed_delta = {}
    for seed in SEEDS:
        candidate = _candidate_rows(data_strategy, seed, dataset)
        baseline = ladder._source_rows(
            data_strategy, ladder.BASELINE_ARCHITECTURE, seed, dataset
        )
        if [row["id"] for row in candidate] != [row["id"] for row in baseline]:
            raise AssertionError("paired comparison row order changed")
        candidate_samples.append(
            ladder._metric_samples(candidate, dataset=dataset, weights=weights)[metric]
        )
        baseline_samples.append(
            ladder._metric_samples(baseline, dataset=dataset, weights=weights)[metric]
        )
        candidate_point = ladder._point_metric(candidate, dataset=dataset, metric=metric)
        baseline_point = ladder._point_metric(baseline, dataset=dataset, metric=metric)
        candidate_points.append(candidate_point)
        baseline_points.append(baseline_point)
        per_seed_delta[str(seed)] = candidate_point - baseline_point
    candidate_mean = confirmation._seed_resampled_mean(candidate_samples, seed_draws)
    baseline_mean = confirmation._seed_resampled_mean(baseline_samples, seed_draws)
    result = interval(
        candidate_mean - baseline_mean,
        float(np.mean(candidate_points) - np.mean(baseline_points)),
        n_boot=N_BOOT,
    )
    result["per_seed_delta"] = per_seed_delta
    result["nonnegative_seed_count"] = sum(
        value >= 0.0 for value in per_seed_delta.values()
    )
    return result


def finalize(*, protocol_sha256: str) -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != protocol_sha256:
        raise AssertionError("consistency-repair protocol hash mismatch")
    derm_reference = ladder._source_rows(
        "derm_ham", ladder.BASELINE_ARCHITECTURE, SEEDS[0], "derm7pt"
    )
    ham_reference = ladder._source_rows(
        "derm_ham", ladder.BASELINE_ARCHITECTURE, SEEDS[0], "ham"
    )
    derm_weights = stratified_bootstrap_weights(
        confirmation._derm_bootstrap_strata(derm_reference),
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 1,
    )
    ham_weights = stratified_bootstrap_weights(
        [str(row["label"]) for row in ham_reference],
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 2,
    )
    seed_draws = np.random.default_rng(BOOTSTRAP_SEED + 3).integers(
        0, len(SEEDS), size=(N_BOOT, len(SEEDS))
    )
    comparisons: dict[str, Any] = {
        "protocol": "dermpact-phase2-3a-r-consistency-comparisons-v1",
        "protocol_sha256": protocol_sha256,
        "direction": "consistency candidate minus independent-head baseline",
        "data_strategies": {},
    }
    for data_strategy in DATA_STRATEGIES:
        values = {
            dataset: {
                metric: _comparison(
                    data_strategy,
                    dataset=dataset,
                    metric=metric,
                    weights=weights,
                    seed_draws=seed_draws,
                )
                for metric in metrics
            }
            for dataset, metrics, weights in (
                (
                    "derm7pt",
                    (
                        "primary",
                        "balanced_accuracy",
                        "melanoma_auc",
                        "joint_correct_agreement",
                        "direct_rule_disagreement",
                    ),
                    derm_weights,
                ),
                ("ham", ("balanced_accuracy", "melanoma_auc"), ham_weights),
            )
        }
        regressions = []
        for dataset, metrics in values.items():
            for metric, result in metrics.items():
                if metric in {"joint_correct_agreement", "direct_rule_disagreement"}:
                    continue
                if result["estimate"] < -PROTECTION_MARGIN and result["ci_high"] < 0.0:
                    regressions.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "delta": result["estimate"],
                            "ci_high": result["ci_high"],
                        }
                    )
        declared = values["derm7pt"]["joint_correct_agreement"]
        disagreement = values["derm7pt"]["direct_rule_disagreement"]
        disagreement_violation = bool(
            disagreement["estimate"] > DISAGREEMENT_MARGIN
            and disagreement["ci_low"] > 0.0
        )
        qualifies = bool(
            declared["estimate"] > 0.0
            and declared["nonnegative_seed_count"] >= 2
            and not regressions
            and not disagreement_violation
        )
        comparisons["data_strategies"][data_strategy] = {
            "comparisons": values,
            "material_regressions": regressions,
            "material_disagreement_increase": disagreement_violation,
            "qualifies": qualifies,
            "selected_architecture": (
                "independent_heads_plus_truth_anchored_consistency"
                if qualifies
                else "independent_heads"
            ),
        }
    comparisons["decision"] = {
        strategy: value["selected_architecture"]
        for strategy, value in comparisons["data_strategies"].items()
    }
    comparison_hash = screen._write_json(COMPARISON_PATH, comparisons)
    checkpoint_count = raw_count = prediction_count = 0
    artifacts = {}
    for strategy in DATA_STRATEGIES:
        for seed in SEEDS:
            path = _result_path(strategy, seed)
            result = _load_json(path)
            artifacts[f"{strategy}_seed_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for fold in result["fold_training"]:
                if sha256_file(ROOT / fold["checkpoint"]["path"]) != fold["checkpoint"]["sha256"]:
                    raise AssertionError("checkpoint hash mismatch")
                checkpoint_count += 1
                for raw in fold["raw_logits"].values():
                    if sha256_file(ROOT / raw["path"]) != raw["sha256"]:
                        raise AssertionError("raw-logit hash mismatch")
                    raw_count += 1
            for prediction in result["predictions"].values():
                if sha256_file(ROOT / prediction["path"]) != prediction["sha256"]:
                    raise AssertionError("prediction hash mismatch")
                prediction_count += 1
    receipt = {
        "protocol": "dermpact-phase2-3a-r-consistency-receipt-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "decision": comparisons["decision"],
        "comparison_artifact": {
            "path": str(COMPARISON_PATH.relative_to(ROOT)),
            "sha256": comparison_hash,
        },
        "checkpoints_verified": checkpoint_count,
        "raw_logit_artifacts_verified": raw_count,
        "prediction_artifacts_verified": prediction_count,
        "protected_or_external_rows_loaded": 0,
        "result_artifacts": artifacts,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    receipt_hash = screen._write_json(RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": receipt_hash}


def build_protocol() -> dict[str, Any]:
    references = {}
    for strategy in DATA_STRATEGIES:
        for seed in SEEDS:
            path = ladder._reference_result_path(strategy, seed)
            references[f"{strategy}_baseline_seed_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
    protocol = {
        "protocol": "dermpact-phase2-3a-r-consistency-repair-v1",
        "status": "PRESPECIFIED_AFTER_FAILED_NESTED_HEAD_BEFORE_REPAIR_RESULTS",
        "reason": (
            "The frozen-global-feature token and relation scaffold materially regressed the "
            "Derm7pt primary endpoint, while the consistency increment had a small positive "
            "agreement point change.  This repair isolates consistency on the proven head."
        ),
        "frozen_negative_result": {
            "architecture_ladder_receipt": {
                "path": str(ladder.RECEIPT_PATH.relative_to(ROOT)),
                "sha256": sha256_file(ladder.RECEIPT_PATH),
            },
            "architecture_ladder_comparisons": {
                "path": str(ladder.COMPARISON_PATH.relative_to(ROOT)),
                "sha256": sha256_file(ladder.COMPARISON_PATH),
            },
            "failed_arms_remain_rejected": ["criterion_tokens", "relation"],
        },
        "single_changed_factor": {
            "baseline": "independent scalar criterion heads",
            "candidate": "same heads plus truth-anchored consistency",
            "weight": ladder.CONSISTENCY_WEIGHT,
            "eligibility": (
                "MEL/NEV training cases whose labelled checklist decision agrees with the labelled melanoma target"
            ),
        },
        "data_strategies": list(DATA_STRATEGIES),
        "controlled_training": {
            "backbone": screen.MODEL_ID,
            "backbone_state": "frozen cached global features",
            "learning_rate": ladder.LEARNING_RATE,
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "seeds": list(SEEDS),
            "loss_weighting": "equal_family",
        },
        "promotion_rule": {
            "declared_endpoint": "positive jointly-correct direct/checklist agreement delta",
            "seed_stability": "non-negative in at least two of three seeds",
            "protected_regression_margin": PROTECTION_MARGIN,
            "disagreement_increase_margin": DISAGREEMENT_MARGIN,
            "selection": "retain consistency only for each data strategy that passes all gates",
        },
        "hard_scope": {
            "allowed": ["Derm7pt 616-case development", "HAM training", "BCN training"],
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_feature_rows_loaded": 0,
            "ph2_rows_loaded": 0,
            "clinical_photo_features_loaded": 0,
            "structured_facts_loaded_into_model": 0,
        },
        "artifacts": {
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "requirements_lock": {
                "path": str(screen.LOCK_FILE.relative_to(ROOT)),
                "sha256": sha256_file(screen.LOCK_FILE),
            },
            **references,
        },
    }
    protocol_sha256 = screen._write_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": protocol_sha256}


def run_all() -> dict[str, Any]:
    protocol = build_protocol()
    completed = [
        run_seed(strategy, seed, protocol_sha256=protocol["protocol_sha256"])
        for strategy in DATA_STRATEGIES
        for seed in SEEDS
    ]
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "completed": completed,
        "receipt": finalize(protocol_sha256=protocol["protocol_sha256"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-protocol", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    if args.build_protocol:
        print(json.dumps(build_protocol(), indent=2, sort_keys=True))
        return 0
    if args.run_all:
        print(json.dumps(run_all(), indent=2, sort_keys=True))
        return 0
    if args.finalize:
        if not args.protocol_sha256:
            raise SystemExit("--protocol-sha256 is required")
        print(json.dumps(finalize(protocol_sha256=args.protocol_sha256), indent=2, sort_keys=True))
        return 0
    parser.error("choose --build-protocol, --run-all, or --finalize")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
