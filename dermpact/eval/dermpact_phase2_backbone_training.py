"""Cache, train and evaluate the prespecified Phase 2.3D adaptation arms."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from . import dermpact_phase2_architecture_ladder as ladder
from . import dermpact_phase2_backbone_adaptation as adaptation
from . import dermpact_phase2_consistency_repair as consistency
from . import dermpact_phase2_dataset_composition as composition
from . import dermpact_phase2_hybrid_confirmation as confirmation
from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_baselines import _read_jsonl, _write_jsonl, sha256_file
from .dermpact_phase2_vector_bootstrap import interval, stratified_bootstrap_weights


ROOT = screen.ROOT
OUTPUT_DIR = adaptation.OUTPUT_DIR
EXPERIMENT_DIR = adaptation.EXPERIMENT_DIR
CACHE_DIR = adaptation.CACHE_DIR
COMPARISON_PATH = OUTPUT_DIR / "backbone_adaptation_comparisons.json"
RECEIPT_PATH = OUTPUT_DIR / "backbone_adaptation_receipt.json"
TRAINING_SMOKE_PATH = OUTPUT_DIR / "backbone_training_path_smoke.json"
N_BOOT = 2000
BOOTSTRAP_SEED = 20261229
PROTECTION_MARGIN = 0.02
DISAGREEMENT_MARGIN = 0.02
PREDICTION_BATCH_SIZE = 64
TRAINING_MICROBATCH_SIZE: int | None = None
TRAINING_DEVICE_WARMUP: Callable[
    [torch.nn.Module, Mapping[str, Any], torch.device], None
] | None = None
CANONICAL_ENVIRONMENT = {
    "python": "3.12.13",
    "numpy": "2.4.4",
    "torch": "2.12.1+cpu",
    "device": "cpu",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_sha256() -> str:
    if not adaptation.PROTOCOL_PATH.is_file():
        raise FileNotFoundError("Phase 2.3D protocol has not been frozen")
    digest = sha256_file(adaptation.PROTOCOL_PATH)
    sidecar = adaptation.PROTOCOL_PATH.with_suffix(".json.sha256")
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise AssertionError("Phase 2.3D protocol sidecar mismatch")
    return digest


def _canonical_environment() -> dict[str, Any]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
    }
    return {
        "status": "PASS" if observed == CANONICAL_ENVIRONMENT else "REQUIRES_LOCAL_REEVALUATION",
        "expected": dict(CANONICAL_ENVIRONMENT),
        "observed": observed,
    }


def _cache_paths(dataset: str) -> tuple[Path, Path]:
    return adaptation._cache_paths(dataset)


def cache_prefix_tokens(dataset: str, *, batch_size: int = 16) -> dict[str, Any]:
    """Materialize float32 tokens after frozen visual block ten."""
    from PIL import Image

    if dataset not in {"ham", "derm7pt", "bcn_train"}:
        raise ValueError(dataset)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    rows = adaptation._dataset_rows(dataset)
    identities = [adaptation._row_identity(row, dataset) for row in rows]
    identity_hash = adaptation._canonical_hash(identities)
    final, manifest_path = _cache_paths(dataset)
    if final.is_file() and manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest["record_identity_sha256"] != identity_hash:
            raise AssertionError("cached prefix rows no longer match the source role")
        if manifest["token_file_sha256"] != sha256_file(final):
            raise AssertionError("cached prefix-token hash mismatch")
        tokens = np.load(final, mmap_mode="r")
        if tokens.shape != tuple(manifest["shape"]) or tokens.dtype != np.float32:
            raise AssertionError("cached prefix-token tensor contract changed")
        return {**manifest, "status": "REUSED_VERIFIED"}

    for row in rows:
        path = adaptation._image_path(row, dataset)
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(ladder.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    visual, preprocess, provenance = adaptation._load_hybrid_visual()
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_suffix(".partial.npy")
    if partial.exists():
        partial.unlink()
    values = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype="float32",
        shape=(len(rows), adaptation.TOKEN_COUNT, adaptation.TOKEN_DIM),
    )
    started = time.perf_counter()
    equivalence = 0.0
    try:
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            images = torch.stack([
                preprocess(Image.open(adaptation._image_path(row, dataset)).convert("RGB"))
                for row in rows[start:stop]
            ])
            with torch.inference_mode():
                prefix = adaptation._prefix_tokens(visual, images)
                if start == 0:
                    full = adaptation._full_visual_features(visual, images[:4])
                    reconstructed = adaptation._base_tail_features(visual, prefix[:4])
                    equivalence = float(torch.max(torch.abs(full - reconstructed)))
                    if equivalence > 1e-6:
                        raise AssertionError(
                            f"prefix boundary changed full features by {equivalence}"
                        )
            block = prefix.cpu().numpy().astype("float32", copy=False)
            if block.shape != (stop - start, adaptation.TOKEN_COUNT, adaptation.TOKEN_DIM):
                raise AssertionError(f"invalid prefix-token block: {block.shape}")
            if not np.isfinite(block).all():
                raise FloatingPointError("non-finite prefix-token block")
            values[start:stop] = block
            if stop == len(rows) or stop % max(batch_size * 25, 1) == 0:
                print(f"[{dataset}/prefix] {stop}/{len(rows)}", flush=True)
        values.flush()
        del values
        os.replace(partial, final)
    finally:
        if partial.exists():
            partial.unlink()
    manifest = {
        "protocol": "dermpact-phase2-3d-prefix-token-cache-v1",
        "dataset": dataset,
        "allowed_role": (
            "training only" if dataset != "derm7pt" else "616-case development only"
        ),
        "n_records": len(rows),
        "record_identity_sha256": identity_hash,
        "shape": [len(rows), adaptation.TOKEN_COUNT, adaptation.TOKEN_DIM],
        "dtype": "float32",
        "completed_visual_blocks": adaptation.PREFIX_BLOCKS,
        "full_vs_reconstructed_max_abs_difference": equivalence,
        "token_file": str(final.relative_to(ROOT)),
        "token_file_sha256": sha256_file(final),
        "elapsed_seconds": time.perf_counter() - started,
        "model": provenance,
        "hard_scope_checks": {
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
    }
    digest = screen._write_json(manifest_path, manifest)
    return {**manifest, "manifest_sha256": digest, "status": "COMPLETED"}


def _load_prefix_tokens(dataset: str, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    path, manifest_path = _cache_paths(dataset)
    manifest = _load_json(manifest_path)
    identities = [adaptation._row_identity(row, dataset) for row in rows]
    if manifest["record_identity_sha256"] != adaptation._canonical_hash(identities):
        raise AssertionError(f"{dataset} prefix row identity changed")
    if manifest["token_file_sha256"] != sha256_file(path):
        raise AssertionError(f"{dataset} prefix-token hash changed")
    values = np.load(path, mmap_mode="r")
    if values.shape != (len(rows), adaptation.TOKEN_COUNT, adaptation.TOKEN_DIM):
        raise AssertionError(f"{dataset} prefix shape changed: {values.shape}")
    if values.dtype != np.float32:
        raise AssertionError(f"{dataset} prefix dtype changed")
    return values


def _data() -> dict[str, Any]:
    data = composition._data()
    data["ham_tokens"] = _load_prefix_tokens("ham", data["ham_rows"])
    data["derm_tokens"] = _load_prefix_tokens("derm7pt", data["derm_rows"])
    data["bcn_tokens"] = _load_prefix_tokens("bcn_train", data["bcn_rows"])
    return data


def _batch_loss(
    model: adaptation.AdaptedHybridModel,
    *,
    dataset: str,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> tuple[Tensor, dict[str, Tensor]]:
    device = next(model.parameters()).device
    tokens = torch.from_numpy(
        np.asarray(data[f"{dataset}_tokens"][indices], dtype="float32")
    ).to(device)
    outputs = model(tokens)
    common: dict[str, Any] = {
        "outputs": outputs,
        "dataset": "derm7pt" if dataset == "derm" else "ham",
        "diagnosis": torch.from_numpy(data[f"{dataset}_global"][indices]).long().to(device),
        "melanoma": torch.from_numpy(data[f"{dataset}_melanoma"][indices]).float().to(device),
        "diagnosis_weight": torch.from_numpy(
            weights[f"{dataset}_diagnosis"][indices]
        ).float().to(device),
        "melanoma_weight": torch.from_numpy(
            weights[f"{dataset}_melanoma"][indices]
        ).float().to(device),
        "loss_weighting": "equal_family",
    }
    if dataset == "derm":
        common["criteria"] = torch.from_numpy(
            data["derm_criteria"][indices]
        ).float().to(device)
        common["criterion_weights"] = torch.from_numpy(
            weights["derm_criteria"][indices]
        ).float().to(device)
    total, components = screen.multitask_loss(**common)
    consistency_loss = total * 0.0
    eligible = 0
    if dataset == "derm":
        consistency_loss, eligible = ladder.truth_anchored_consistency_loss(
            outputs,
            melanoma=common["melanoma"],
            criteria=common["criteria"],
            diagnosis=common["diagnosis"],
        )
        total = total + ladder.CONSISTENCY_WEIGHT * consistency_loss
    return total, {
        **components,
        "consistency": consistency_loss,
        "consistency_eligible": torch.tensor(float(eligible)),
    }


def _microbatched_backward(
    model: adaptation.AdaptedHybridModel,
    *,
    dataset: str,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
    coefficient: float,
    microbatch_size: int,
) -> tuple[float, dict[str, Tensor]]:
    """Backpropagate the exact effective-batch objective in smaller graphs."""
    if microbatch_size <= 0 or microbatch_size >= len(indices):
        raise ValueError("microbatch size must be between 1 and batch_size - 1")
    diagnosis_denominator = max(
        float(np.asarray(weights[f"{dataset}_diagnosis"][indices]).sum()), 1e-12
    )
    melanoma_denominator = max(
        float(np.asarray(weights[f"{dataset}_melanoma"][indices]).sum()), 1e-12
    )
    criterion_denominators: np.ndarray | None = None
    consistency_eligible_total = 0
    if dataset == "derm":
        criterion_denominators = np.asarray(
            weights["derm_criteria"][indices], dtype="float64"
        ).sum(axis=0)
        criterion_denominators = np.maximum(criterion_denominators, 1e-12)
        checklist_weights = ladder.CHECKLIST_WEIGHTS.detach().cpu().numpy()
        criteria = np.asarray(data["derm_criteria"][indices], dtype="float32")
        true_rule = criteria @ checklist_weights >= ladder.CHECKLIST_THRESHOLD
        diagnoses = np.asarray(data["derm_global"][indices], dtype="int64")
        melanocytic = np.isin(diagnoses, tuple(ladder.MELANOCYTIC_GLOBAL))
        melanoma = np.asarray(data["derm_melanoma"][indices], dtype="float32")
        consistency_eligible_total = int(
            np.count_nonzero(melanocytic & (true_rule == melanoma))
        )

    diagnosis_numerators: list[Tensor] = []
    melanoma_numerators: list[Tensor] = []
    criterion_numerators_all: list[Tensor] = []
    consistency_sums: list[Tensor] = []
    contributions: list[Tensor] = []
    device = next(model.parameters()).device
    for start in range(0, len(indices), microbatch_size):
        selection = indices[start:start + microbatch_size]
        tokens = torch.from_numpy(
            np.asarray(data[f"{dataset}_tokens"][selection], dtype="float32")
        ).to(device)
        outputs = model(tokens)
        diagnosis = torch.from_numpy(
            np.asarray(data[f"{dataset}_global"][selection], dtype="int64")
        ).long().to(device)
        melanoma = torch.from_numpy(
            np.asarray(data[f"{dataset}_melanoma"][selection], dtype="float32")
        ).to(device)
        diagnosis_weight = torch.from_numpy(
            np.asarray(weights[f"{dataset}_diagnosis"][selection], dtype="float32")
        ).to(device)
        melanoma_weight = torch.from_numpy(
            np.asarray(weights[f"{dataset}_melanoma"][selection], dtype="float32")
        ).to(device)
        diagnosis_numerator = (
            torch.nn.functional.cross_entropy(
                outputs["diagnosis"], diagnosis, reduction="none"
            ) * diagnosis_weight
        ).sum()
        melanoma_numerator = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["melanoma"], melanoma, reduction="none"
            ) * melanoma_weight
        ).sum()
        diagnosis_contribution = diagnosis_numerator / diagnosis_denominator
        melanoma_contribution = melanoma_numerator / melanoma_denominator
        diagnosis_numerators.append(diagnosis_numerator.detach())
        melanoma_numerators.append(melanoma_numerator.detach())

        consistency_contribution = outputs["criteria"].sum() * 0.0
        if dataset == "derm":
            assert criterion_denominators is not None
            criteria = torch.from_numpy(
                np.asarray(data["derm_criteria"][selection], dtype="float32")
            ).to(device)
            criterion_weights = torch.from_numpy(
                np.asarray(weights["derm_criteria"][selection], dtype="float32")
            ).to(device)
            criterion_numerators = torch.stack([
                (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs["criteria"][:, index],
                        criteria[:, index],
                        reduction="none",
                    ) * criterion_weights[:, index]
                ).sum()
                for index in range(len(screen.CRITERIA))
            ])
            denominator_tensor = torch.tensor(
                criterion_denominators, dtype=criterion_numerators.dtype, device=device
            )
            criterion_contributions = criterion_numerators / denominator_tensor
            supervised_contribution = (
                diagnosis_contribution
                + melanoma_contribution
                + criterion_contributions.mean()
            ) / 3.0
            criterion_numerators_all.append(criterion_numerators.detach())

            rule_weights = ladder.CHECKLIST_WEIGHTS.to(device=device, dtype=criteria.dtype)
            selection_criteria = np.asarray(
                data["derm_criteria"][selection], dtype="float32"
            )
            selection_rule = (
                selection_criteria @ ladder.CHECKLIST_WEIGHTS.detach().cpu().numpy()
                >= ladder.CHECKLIST_THRESHOLD
            )
            selection_diagnosis = np.asarray(
                data["derm_global"][selection], dtype="int64"
            )
            selection_melanoma = np.asarray(
                data["derm_melanoma"][selection], dtype="float32"
            )
            eligible_array = np.isin(
                selection_diagnosis, tuple(ladder.MELANOCYTIC_GLOBAL)
            ) & (selection_rule == selection_melanoma)
            eligible_count = int(np.count_nonzero(eligible_array))
            eligible = torch.from_numpy(eligible_array).bool().to(device)
            if eligible_count:
                soft_score = torch.sigmoid(outputs["criteria"]) @ rule_weights
                rule_probability = torch.sigmoid(
                    (soft_score - ladder.CHECKLIST_THRESHOLD)
                    / ladder.CONSISTENCY_TEMPERATURE
                )
                direct_probability = torch.sigmoid(outputs["melanoma"])
                consistency_sum = ladder._bernoulli_js(
                    direct_probability[eligible], rule_probability[eligible]
                ).sum()
                consistency_sums.append(consistency_sum.detach())
                consistency_contribution = consistency_sum / max(
                    consistency_eligible_total, 1
                )
        else:
            supervised_contribution = 0.5 * (
                diagnosis_contribution + melanoma_contribution
            )

        contribution = (
            supervised_contribution
            + ladder.CONSISTENCY_WEIGHT * consistency_contribution
        )
        if os.environ.get("DERMPACT_MICROBATCH_DEBUG") == "1":
            print(
                f"[microbatch-debug] dataset={dataset} rows={start}:"
                f"{start + len(selection)} backward-start",
                flush=True,
            )
        (coefficient * contribution).backward()
        if os.environ.get("DERMPACT_MICROBATCH_DEBUG") == "1":
            print(
                f"[microbatch-debug] dataset={dataset} rows={start}:"
                f"{start + len(selection)} backward-pass",
                flush=True,
            )
        contributions.append(contribution.detach())

    diagnosis_numerator_total = float(
        torch.stack(diagnosis_numerators).sum().cpu()
    )
    melanoma_numerator_total = float(
        torch.stack(melanoma_numerators).sum().cpu()
    )
    criterion_numerator_totals = (
        torch.stack(criterion_numerators_all).sum(dim=0).cpu().numpy()
        if criterion_numerators_all
        else np.zeros(len(screen.CRITERIA), dtype="float64")
    )
    consistency_sum_total = (
        float(torch.stack(consistency_sums).sum().cpu())
        if consistency_sums
        else 0.0
    )
    total_value = float(torch.stack(contributions).sum().cpu())
    diagnosis_value = diagnosis_numerator_total / diagnosis_denominator
    melanoma_value = melanoma_numerator_total / melanoma_denominator
    criteria_value = (
        float(np.mean(criterion_numerator_totals / criterion_denominators))
        if criterion_denominators is not None
        else 0.0
    )
    consistency_value = (
        consistency_sum_total / consistency_eligible_total
        if consistency_eligible_total
        else 0.0
    )
    return total_value, {
        "diagnosis": torch.tensor(diagnosis_value),
        "melanoma": torch.tensor(melanoma_value),
        "criteria_mean": torch.tensor(criteria_value),
        "consistency": torch.tensor(consistency_value),
        "consistency_eligible": torch.tensor(float(consistency_eligible_total)),
    }


def _predict(
    model: adaptation.AdaptedHybridModel,
    tokens: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    diagnosis: list[np.ndarray] = []
    melanoma: list[np.ndarray] = []
    criteria: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), PREDICTION_BATCH_SIZE):
            selection = indices[start:start + PREDICTION_BATCH_SIZE]
            output = model(torch.from_numpy(
                np.asarray(tokens[selection], dtype="float32")
            ))
            diagnosis.append(output["diagnosis"].numpy())
            melanoma.append(output["melanoma"].numpy())
            criteria.append(output["criteria"].numpy())
    diagnosis_logits = np.concatenate(diagnosis)
    melanoma_logits = np.concatenate(melanoma)
    criterion_logits = np.concatenate(criteria)
    return {
        "diagnosis_logits": diagnosis_logits,
        "melanoma_logits": melanoma_logits,
        "criterion_logits": criterion_logits,
        "universal": torch.softmax(torch.from_numpy(diagnosis_logits), dim=1).numpy(),
        "melanoma": torch.sigmoid(torch.from_numpy(melanoma_logits)).numpy(),
        "criteria": torch.sigmoid(torch.from_numpy(criterion_logits)).numpy(),
    }


def _save_checkpoint(
    model: adaptation.AdaptedHybridModel, path: Path, *, arm: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "model_state": model.state_dict(),
        "adaptation_parameter_names": model.visual_tail.adaptation_parameter_names(),
    }
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _cpu_checkpoint_tree(value: Any) -> Any:
    """Detach tensors to CPU so DirectML resume files are portable."""
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_checkpoint_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_checkpoint_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_checkpoint_tree(item) for item in value)
    return value


def _epoch_resume_path(strategy: str, arm: str, seed: int, fold: int) -> Path:
    return (
        EXPERIMENT_DIR / strategy / arm / f"seed_{seed}" / f"fold_{fold}"
        / "epoch_resume_state.pt"
    )


def _write_epoch_resume(
    *,
    strategy: str,
    arm: str,
    seed: int,
    fold: int,
    completed_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    history: Sequence[Mapping[str, Any]],
) -> str:
    """Atomically persist an execution-only, hash-verified epoch checkpoint."""
    path = _epoch_resume_path(strategy, arm, seed, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dermpact-phase2-3d-epoch-resume-v1",
        "protocol_sha256": _protocol_sha256(),
        "strategy": strategy,
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "completed_epochs": completed_epochs,
        "model_state": _cpu_checkpoint_tree(model.state_dict()),
        "optimizer_state": _cpu_checkpoint_tree(optimizer.state_dict()),
        "scheduler_state": scheduler.state_dict(),
        "history": list(history),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _load_epoch_resume(
    *,
    strategy: str,
    arm: str,
    seed: int,
    fold: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    training_device: torch.device,
) -> tuple[int, list[dict[str, Any]]] | None:
    """Restore the last completed epoch only after identity and hash checks."""
    path = _epoch_resume_path(strategy, arm, seed, fold)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        expected_digest = sidecar.read_text(encoding="ascii").strip()
        if sha256_file(path) != expected_digest:
            raise AssertionError("epoch-resume hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity = (
            payload.get("format"),
            payload.get("protocol_sha256"),
            payload.get("strategy"),
            payload.get("arm"),
            payload.get("seed"),
            payload.get("fold"),
        )
        expected_identity = (
            "dermpact-phase2-3d-epoch-resume-v1",
            _protocol_sha256(),
            strategy,
            arm,
            seed,
            fold,
        )
        if identity != expected_identity:
            raise AssertionError(f"epoch-resume identity mismatch: {identity!r}")
        completed_epochs = int(payload["completed_epochs"])
        history = list(payload["history"])
        if not 1 <= completed_epochs <= adaptation.EPOCHS:
            raise AssertionError("epoch-resume count is outside the protocol")
        if len(history) != completed_epochs:
            raise AssertionError("epoch-resume history length mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(training_device)
        scheduler.load_state_dict(payload["scheduler_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        return completed_epochs, history
    except (
        AssertionError,
        EOFError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"[{strategy}/{arm}/seed{seed}] fold {fold + 1} epoch resume "
            f"rejected: {error}; restarting the fold",
            flush=True,
        )
        return None


def _discard_epoch_resume(strategy: str, arm: str, seed: int, fold: int) -> None:
    """Remove transient state only after the durable fold receipt verifies."""
    path = _epoch_resume_path(strategy, arm, seed, fold)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    path.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)


def _reload_difference(
    visual: torch.nn.Module,
    model: adaptation.AdaptedHybridModel,
    checkpoint: Path,
    tokens: np.ndarray,
) -> float:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = adaptation.AdaptedHybridModel(visual, str(payload["arm"]))
    restored.load_state_dict(payload["model_state"], strict=True)
    model.visual_tail.set_adaptation_scale(1.0)
    restored.visual_tail.set_adaptation_scale(1.0)
    sample = torch.from_numpy(
        np.asarray(tokens[: min(8, len(tokens))], dtype="float32").copy()
    )
    model.eval()
    restored.eval()
    maximum = 0.0
    with torch.inference_mode():
        left = model(sample)
        right = restored(sample)
        for key in left:
            maximum = max(maximum, float(torch.max(torch.abs(left[key] - right[key]))))
    return maximum


def _result_path(strategy: str, arm: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{strategy}_{arm}_seed_{seed}_result.json"


def _fold_result_path(strategy: str, arm: str, seed: int, fold: int) -> Path:
    return (
        EXPERIMENT_DIR / strategy / arm / f"seed_{seed}" / f"fold_{fold}"
        / "fold_result.json"
    )


def _verified_completed_fold(
    strategy: str,
    arm: str,
    seed: int,
    fold: int,
    *,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    """Return a completed fold only after its receipt and artifacts verify."""
    path = _fold_result_path(strategy, arm, seed, fold)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        expected = sidecar.read_text(encoding="ascii").strip()
        if sha256_file(path) != expected:
            raise AssertionError("fold-result hash mismatch")
        result = _load_json(path)
        identity = (
            result.get("strategy"),
            result.get("arm"),
            result.get("seed"),
            result.get("fold"),
            result.get("protocol_sha256"),
        )
        expected_identity = (strategy, arm, seed, fold, protocol_sha256)
        if identity != expected_identity:
            raise AssertionError(f"fold identity mismatch: {identity!r}")
        checkpoint = result["checkpoint"]
        if sha256_file(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
            raise AssertionError("fold checkpoint hash mismatch")
        for variant in result["alpha_results"].values():
            for raw in variant["raw_logits"].values():
                if sha256_file(ROOT / raw["path"]) != raw["sha256"]:
                    raise AssertionError("fold raw-logit hash mismatch")
        return result
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        print(
            f"[{strategy}/{arm}/seed{seed}] fold {fold + 1} cannot be resumed: "
            f"{error}; retraining it",
            flush=True,
        )
        return None


def train_fold(
    visual: torch.nn.Module,
    *,
    strategy: str,
    arm: str,
    fold: int,
    seed: int,
    data: Mapping[str, Any],
    training_device: torch.device,
) -> dict[str, Any]:
    if strategy not in adaptation.DATA_STRATEGIES or arm not in adaptation.ARMS:
        raise ValueError((strategy, arm))
    initialization_seed = seed + fold
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    model = adaptation.AdaptedHybridModel(visual, arm).to(training_device)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(), weight_decay=screen.WEIGHT_DECAY
    )
    steps = composition._steps_per_epoch(data, fold)
    scheduler = screen._scheduler(optimizer, adaptation.EPOCHS * steps)
    weights = composition._training_weights(data, fold)
    train_indices = {
        "ham": np.flatnonzero(data["ham_folds"] != fold),
        "derm": np.flatnonzero(data["derm_folds"] != fold),
        "bcn": np.flatnonzero(data["bcn_folds"] != fold),
    }
    valid_indices = {
        "ham": np.flatnonzero(data["ham_folds"] == fold),
        "derm7pt": np.flatnonzero(data["derm_folds"] == fold),
        "bcn_train_oof": np.flatnonzero(data["bcn_folds"] == fold),
    }
    mixtures = (
        (("ham", 0.5), ("derm", 0.5))
        if strategy == "derm_ham"
        else (("ham", 0.25), ("derm", 0.5), ("bcn", 0.25))
    )
    history: list[dict[str, Any]] = []
    resumed_from_epoch = 0
    resume = _load_epoch_resume(
        strategy=strategy,
        arm=arm,
        seed=seed,
        fold=fold,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_device=training_device,
    )
    if resume is not None:
        resumed_from_epoch, history = resume
        print(
            f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
            f"resumed after epoch {resumed_from_epoch}/{adaptation.EPOCHS}",
            flush=True,
        )
    started = time.perf_counter()
    print(
        f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
        f"training start: {adaptation.EPOCHS} epochs, {steps} steps/epoch, "
        f"device={training_device}, effective_batch={screen.BATCH_SIZE}, "
        f"microbatch={TRAINING_MICROBATCH_SIZE or screen.BATCH_SIZE}",
        flush=True,
    )
    for epoch in range(resumed_from_epoch, adaptation.EPOCHS):
        batches = {
            dataset: screen._cyclic_batches(
                train_indices[dataset],
                steps=steps,
                batch_size=screen.BATCH_SIZE,
                rng=np.random.default_rng(
                    seed + fold * 1000 + epoch * 3 + {"ham": 1, "derm": 2, "bcn": 3}[dataset]
                ),
            )
            for dataset, _ in mixtures
        }
        losses = []
        components: dict[str, float] = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for dataset, coefficient in mixtures:
                batch_indices = next(batches[dataset])
                if TRAINING_MICROBATCH_SIZE is None:
                    loss, values = _batch_loss(
                        model,
                        dataset=dataset,
                        indices=batch_indices,
                        data=data,
                        weights=weights,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite adaptation loss")
                    (coefficient * loss).backward()
                    loss_value = float(loss.detach())
                else:
                    loss_value, values = _microbatched_backward(
                        model,
                        dataset=dataset,
                        indices=batch_indices,
                        data=data,
                        weights=weights,
                        coefficient=coefficient,
                        microbatch_size=TRAINING_MICROBATCH_SIZE,
                    )
                    if not np.isfinite(loss_value):
                        raise FloatingPointError("non-finite microbatched adaptation loss")
                step_loss += coefficient * loss_value
                for name, value in values.items():
                    components[f"{dataset}_{name}"] += float(value.detach())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), screen.GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite adaptation gradient")
            optimizer.step()
            scheduler.step()
            losses.append(step_loss)
        epoch_result = {
            "epoch": epoch + 1,
            "mean_training_loss": float(np.mean(losses)),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "component_means": {
                key: value / steps for key, value in components.items()
            },
        }
        history.append(epoch_result)
        resume_digest = _write_epoch_resume(
            strategy=strategy,
            arm=arm,
            seed=seed,
            fold=fold,
            completed_epochs=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
        )
        mean_epoch_seconds = float(np.mean([
            value["elapsed_seconds"] for value in history
        ]))
        remaining_seconds = mean_epoch_seconds * (adaptation.EPOCHS - epoch - 1)
        print(
            f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
            f"epoch {epoch + 1}/{adaptation.EPOCHS} complete: "
            f"loss={epoch_result['mean_training_loss']:.6f}, "
            f"epoch={epoch_result['elapsed_seconds']:.1f}s, "
            f"training_eta={remaining_seconds / 60.0:.1f}m, "
            f"resume={resume_digest[:12]}",
            flush=True,
        )

    print(
        f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
        "training complete; exporting checkpoint and generating CPU predictions",
        flush=True,
    )
    model.to("cpu")
    if training_device.type == "cuda":
        torch.cuda.empty_cache()
    model.visual_tail.capture_adapted_state()
    fold_dir = EXPERIMENT_DIR / strategy / arm / f"seed_{seed}" / f"fold_{fold}"
    checkpoint_path = fold_dir / f"epoch_{adaptation.EPOCHS}_state.pt"
    checkpoint = _save_checkpoint(model, checkpoint_path, arm=arm)
    reload_difference = _reload_difference(
        visual, model, checkpoint_path,
        np.asarray(data["derm_tokens"][valid_indices["derm7pt"]], dtype="float32"),
    )
    if reload_difference != 0.0:
        raise AssertionError(f"checkpoint reload changed logits by {reload_difference}")
    checkpoint["reload_max_abs_logit_difference"] = reload_difference

    prediction_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "ham": (data["ham_tokens"], valid_indices["ham"]),
        "derm7pt": (data["derm_tokens"], valid_indices["derm7pt"]),
    }
    if strategy == "derm_ham_bcn":
        prediction_sets["bcn_train_oof"] = (
            data["bcn_tokens"], valid_indices["bcn_train_oof"]
        )
    alpha_results: dict[str, Any] = {}
    for alpha in adaptation.INTERPOLATION_ALPHAS:
        alpha_key = str(alpha).replace(".", "p")
        print(
            f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
            f"prediction alpha={alpha}",
            flush=True,
        )
        model.visual_tail.set_adaptation_scale(alpha)
        predicted = {
            name: _predict(model, tokens, indices)
            for name, (tokens, indices) in prediction_sets.items()
        }
        raw = {}
        for name, (_, indices) in prediction_sets.items():
            path = fold_dir / f"alpha_{alpha_key}_{name}_raw_logits.npz"
            values = predicted[name]
            raw[name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": confirmation._save_npz(
                    path,
                    row_indices=np.asarray(indices, dtype=np.int64),
                    diagnosis_logits=np.asarray(values["diagnosis_logits"], dtype=np.float32),
                    melanoma_logits=np.asarray(values["melanoma_logits"], dtype=np.float32),
                    criterion_logits=np.asarray(values["criterion_logits"], dtype=np.float32),
                ),
            }
        alpha_results[alpha_key] = {
            "alpha": alpha,
            "raw_logits": raw,
            "ham_rows": screen._ham_prediction_rows(
                data, valid_indices["ham"], predicted["ham"]
            ),
            "derm_rows": screen._derm_prediction_rows(
                data, valid_indices["derm7pt"], predicted["derm7pt"]
            ),
        }
        if strategy == "derm_ham_bcn":
            alpha_results[alpha_key]["bcn_rows"] = composition._bcn_prediction_rows(
                data,
                valid_indices["bcn_train_oof"],
                predicted["bcn_train_oof"],
            )
    model.visual_tail.set_adaptation_scale(1.0)
    return {
        "protocol_sha256": _protocol_sha256(),
        "strategy": strategy,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "initialization_seed": initialization_seed,
        "steps_per_epoch": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "epoch_resume": {
            "format": "dermpact-phase2-3d-epoch-resume-v1",
            "resumed_from_epoch": resumed_from_epoch,
            "checkpointed_after_each_epoch": True,
        },
        "effective_batch_size": screen.BATCH_SIZE,
        "training_microbatch_size": TRAINING_MICROBATCH_SIZE or screen.BATCH_SIZE,
        "checkpoint": checkpoint,
        "alpha_results": alpha_results,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "visual_adaptation_parameter_names": model.visual_tail.adaptation_parameter_names(),
        "training_device": str(training_device),
        "prediction_device": "cpu",
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def run_training_path_smoke() -> dict[str, Any]:
    """Exercise both real-data mixture paths without calculating an endpoint."""
    torch.set_num_threads(ladder.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    data = _data()
    visual, _, provenance = adaptation._load_hybrid_visual()
    results = {}
    smoke_dir = EXPERIMENT_DIR / "training_path_smoke"
    for arm in adaptation.ARMS:
        for strategy in adaptation.DATA_STRATEGIES:
            torch.manual_seed(20260827)
            model = adaptation.AdaptedHybridModel(visual, arm)
            optimizer = torch.optim.AdamW(
                model.optimizer_groups(), weight_decay=screen.WEIGHT_DECAY
            )
            weights = composition._training_weights(data, 0)
            mixtures = (
                (("ham", 0.5), ("derm", 0.5))
                if strategy == "derm_ham"
                else (("ham", 0.25), ("derm", 0.5), ("bcn", 0.25))
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            for dataset, coefficient in mixtures:
                eligible = np.flatnonzero(data[f"{dataset}_folds"] != 0)[:8]
                loss, _ = _batch_loss(
                    model,
                    dataset=dataset,
                    indices=eligible,
                    data=data,
                    weights=weights,
                )
                (coefficient * loss).backward()
                losses[dataset] = float(loss.detach())
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), screen.GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite training-path smoke gradient")
            optimizer.step()
            model.eval()
            model.visual_tail.capture_adapted_state()
            prediction = _predict(
                model,
                data["derm_tokens"],
                np.flatnonzero(data["derm_folds"] == 0)[:8],
            )
            if not all(np.isfinite(value).all() for value in prediction.values()):
                raise FloatingPointError("non-finite training-path smoke prediction")
            checkpoint_path = smoke_dir / f"{strategy}_{arm}.pt"
            checkpoint = _save_checkpoint(model, checkpoint_path, arm=arm)
            reload_difference = _reload_difference(
                visual,
                model,
                checkpoint_path,
                np.asarray(data["derm_tokens"][:8], dtype="float32"),
            )
            if reload_difference != 0.0:
                raise AssertionError("training-path smoke reload changed logits")
            results[f"{strategy}_{arm}"] = {
                "losses": losses,
                "gradient_norm": float(gradient),
                "checkpoint": checkpoint,
                "reload_max_abs_logit_difference": reload_difference,
                "prediction_shapes": {
                    name: list(value.shape) for name, value in prediction.items()
                },
                "status": "PASS",
            }
    payload = {
        "protocol": "dermpact-phase2-3d-training-path-smoke-v1",
        "status": "PASS",
        "scope": "one optimizer step per arm and mixture; no clinical endpoint calculated",
        "model": provenance,
        "results": results,
        "protected_or_external_rows_loaded": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    digest = screen._write_json(TRAINING_SMOKE_PATH, payload)
    return {**payload, "receipt_sha256": digest}


def train_seed(
    strategy: str, arm: str, seed: int, *, device: str = "auto",
) -> dict[str, Any]:
    protocol_sha = _protocol_sha256()
    result_path = _result_path(strategy, arm, seed)
    sidecar = result_path.with_suffix(".json.sha256")
    if result_path.is_file() and sidecar.is_file():
        existing = _load_json(result_path)
        digest = sidecar.read_text(encoding="ascii").strip()
        if sha256_file(result_path) == digest and existing.get("protocol_sha256") == protocol_sha:
            return {"status": "REUSED_VERIFIED", "path": str(result_path.relative_to(ROOT)), "sha256": digest}

    torch.set_num_threads(ladder.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    training_device = _resolve_device(device)
    data = _data()
    visual, _, provenance = adaptation._load_hybrid_visual()
    if TRAINING_DEVICE_WARMUP is not None:
        TRAINING_DEVICE_WARMUP(visual, data, training_device)
    fold_results = []
    combined_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(alpha).replace(".", "p"): {
            "ham": [], "derm7pt": [], "bcn_train_oof": []
        }
        for alpha in adaptation.INTERPOLATION_ALPHAS
    }
    started = time.perf_counter()
    for fold in range(adaptation.N_FOLDS):
        result = _verified_completed_fold(
            strategy,
            arm,
            seed,
            fold,
            protocol_sha256=protocol_sha,
        )
        if result is None:
            result = train_fold(
                visual,
                strategy=strategy,
                arm=arm,
                fold=fold,
                seed=seed,
                data=data,
                training_device=training_device,
            )
            fold_path = _fold_result_path(strategy, arm, seed, fold)
            fold_digest = screen._write_json(fold_path, result)
            print(
                f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
                f"saved with receipt {fold_digest}",
                flush=True,
            )
            committed = _verified_completed_fold(
                strategy,
                arm,
                seed,
                fold,
                protocol_sha256=protocol_sha,
            )
            if committed is None:
                raise AssertionError("new fold receipt failed post-write verification")
            result = committed
        else:
            print(
                f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
                "reused from verified fold receipt",
                flush=True,
            )
        _discard_epoch_resume(strategy, arm, seed, fold)
        for alpha_key, alpha_values in result["alpha_results"].items():
            combined_rows[alpha_key]["ham"].extend(alpha_values.pop("ham_rows"))
            combined_rows[alpha_key]["derm7pt"].extend(alpha_values.pop("derm_rows"))
            combined_rows[alpha_key]["bcn_train_oof"].extend(
                alpha_values.pop("bcn_rows", [])
            )
        fold_results.append(result)
        print(
            f"[{strategy}/{arm}/seed{seed}] fold {fold + 1}/{adaptation.N_FOLDS} "
            f"complete in {result['elapsed_seconds']:.1f}s",
            flush=True,
        )

    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    variants = {}
    seed_dir = EXPERIMENT_DIR / strategy / arm / f"seed_{seed}"
    for alpha_key, rows in combined_rows.items():
        ham_rows = screen._order_like_reference(rows["ham"], frozen_ham)
        derm_rows = screen._order_like_reference(rows["derm7pt"], frozen_derm)
        predictions = {}
        for dataset, values in (("ham", ham_rows), ("derm7pt", derm_rows)):
            path = seed_dir / f"alpha_{alpha_key}_{dataset}_oof_predictions.jsonl"
            predictions[dataset] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _write_jsonl(path, values),
            }
        metrics = confirmation._seed_metrics(ham_rows, derm_rows, seed=seed)
        metrics["derm7pt"]["rule_consistency"] = ladder._derm_rule_summary(derm_rows)
        if strategy == "derm_ham_bcn":
            bcn_rows = sorted(rows["bcn_train_oof"], key=lambda item: str(item["id"]))
            path = seed_dir / f"alpha_{alpha_key}_bcn_training_oof_predictions.jsonl"
            predictions["bcn_train_oof"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _write_jsonl(path, bcn_rows),
            }
            metrics["bcn_train_oof"] = composition._bcn_metrics(bcn_rows, seed=seed)
        variants[alpha_key] = {
            "alpha": float(alpha_key.replace("p", ".")),
            "predictions": predictions,
            **metrics,
        }
    payload = {
        "protocol": "dermpact-phase2-3d-backbone-adaptation-seed-result-v1",
        "protocol_sha256": protocol_sha,
        "status": "PASS",
        "data_strategy": strategy,
        "arm": arm,
        "seed": seed,
        "model": provenance,
        "training_device": str(training_device),
        "prediction_device": "cpu",
        "fixed_training": {
            "epochs": adaptation.EPOCHS,
            "folds": adaptation.N_FOLDS,
            "head_learning_rate": adaptation.HEAD_LR,
            "visual_learning_rate": (
                adaptation.LAST_BLOCK_LR if arm == "last_block" else adaptation.LORA_LR
            ),
            "loss": "equal-family plus truth-anchored consistency",
        },
        "fold_training": fold_results,
        "variants": variants,
        "elapsed_seconds": time.perf_counter() - started,
        "canonical_evaluation": _canonical_environment(),
        "hard_scope_checks": {
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
    }
    digest = screen._write_json(result_path, payload)
    return {
        "status": "COMPLETED",
        "path": str(result_path.relative_to(ROOT)),
        "sha256": digest,
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def _evaluate_checkpoint_fold(
    visual: torch.nn.Module,
    *,
    strategy: str,
    arm: str,
    seed: int,
    fold_result: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Regenerate one fold's logits from its exported state on CPU."""
    fold = int(fold_result["fold"])
    checkpoint = ROOT / str(fold_result["checkpoint"]["path"])
    if sha256_file(checkpoint) != fold_result["checkpoint"]["sha256"]:
        raise AssertionError("adaptation checkpoint hash changed before reevaluation")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["arm"] != arm:
        raise AssertionError("adaptation checkpoint arm changed")
    model = adaptation.AdaptedHybridModel(visual, arm)
    model.load_state_dict(saved["model_state"], strict=True)
    model.visual_tail.capture_adapted_state()
    model.eval()
    valid = {
        "ham": np.flatnonzero(data["ham_folds"] == fold),
        "derm7pt": np.flatnonzero(data["derm_folds"] == fold),
    }
    prediction_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "ham": (data["ham_tokens"], valid["ham"]),
        "derm7pt": (data["derm_tokens"], valid["derm7pt"]),
    }
    if strategy == "derm_ham_bcn":
        valid["bcn_train_oof"] = np.flatnonzero(data["bcn_folds"] == fold)
        prediction_sets["bcn_train_oof"] = (
            data["bcn_tokens"], valid["bcn_train_oof"]
        )
    fold_dir = EXPERIMENT_DIR / strategy / arm / f"seed_{seed}" / f"fold_{fold}"
    alpha_results = {}
    for alpha in adaptation.INTERPOLATION_ALPHAS:
        alpha_key = str(alpha).replace(".", "p")
        model.visual_tail.set_adaptation_scale(alpha)
        predicted = {
            name: _predict(model, tokens, indices)
            for name, (tokens, indices) in prediction_sets.items()
        }
        raw = {}
        for name, (_, indices) in prediction_sets.items():
            path = fold_dir / f"alpha_{alpha_key}_{name}_raw_logits.npz"
            values = predicted[name]
            raw[name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": confirmation._save_npz(
                    path,
                    row_indices=np.asarray(indices, dtype=np.int64),
                    diagnosis_logits=np.asarray(values["diagnosis_logits"], dtype=np.float32),
                    melanoma_logits=np.asarray(values["melanoma_logits"], dtype=np.float32),
                    criterion_logits=np.asarray(values["criterion_logits"], dtype=np.float32),
                ),
            }
        alpha_results[alpha_key] = {
            "alpha": alpha,
            "raw_logits": raw,
            "ham_rows": screen._ham_prediction_rows(data, valid["ham"], predicted["ham"]),
            "derm_rows": screen._derm_prediction_rows(
                data, valid["derm7pt"], predicted["derm7pt"]
            ),
        }
        if strategy == "derm_ham_bcn":
            alpha_results[alpha_key]["bcn_rows"] = composition._bcn_prediction_rows(
                data, valid["bcn_train_oof"], predicted["bcn_train_oof"]
            )
    model.visual_tail.set_adaptation_scale(1.0)
    return alpha_results


def _compile_seed_variants(
    *,
    strategy: str,
    arm: str,
    seed: int,
    combined_rows: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    variants = {}
    seed_dir = EXPERIMENT_DIR / strategy / arm / f"seed_{seed}"
    for alpha_key, rows in combined_rows.items():
        ham_rows = screen._order_like_reference(rows["ham"], frozen_ham)
        derm_rows = screen._order_like_reference(rows["derm7pt"], frozen_derm)
        predictions = {}
        for dataset, values in (("ham", ham_rows), ("derm7pt", derm_rows)):
            path = seed_dir / f"alpha_{alpha_key}_{dataset}_oof_predictions.jsonl"
            predictions[dataset] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _write_jsonl(path, values),
            }
        metrics = confirmation._seed_metrics(ham_rows, derm_rows, seed=seed)
        metrics["derm7pt"]["rule_consistency"] = ladder._derm_rule_summary(derm_rows)
        if strategy == "derm_ham_bcn":
            bcn_rows = sorted(rows["bcn_train_oof"], key=lambda item: str(item["id"]))
            path = seed_dir / f"alpha_{alpha_key}_bcn_training_oof_predictions.jsonl"
            predictions["bcn_train_oof"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _write_jsonl(path, bcn_rows),
            }
            metrics["bcn_train_oof"] = composition._bcn_metrics(bcn_rows, seed=seed)
        variants[alpha_key] = {
            "alpha": float(alpha_key.replace("p", ".")),
            "predictions": predictions,
            **metrics,
        }
    return variants


def reevaluate_seed(strategy: str, arm: str, seed: int) -> dict[str, Any]:
    """Replace all model-selection logits in the canonical local CPU env."""
    environment = _canonical_environment()
    if environment["status"] != "PASS":
        raise RuntimeError(
            "--reevaluate must run in the frozen venv_p2 CPU environment; "
            f"observed {environment['observed']}"
        )
    protocol_sha = _protocol_sha256()
    result_path = _result_path(strategy, arm, seed)
    result = _load_json(result_path)
    if result.get("protocol_sha256") != protocol_sha:
        raise AssertionError("training result belongs to another protocol")
    torch.set_num_threads(ladder.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    data = _data()
    visual, _, provenance = adaptation._load_hybrid_visual()
    combined_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(alpha).replace(".", "p"): {
            "ham": [], "derm7pt": [], "bcn_train_oof": []
        }
        for alpha in adaptation.INTERPOLATION_ALPHAS
    }
    for fold_result in result["fold_training"]:
        alpha_results = _evaluate_checkpoint_fold(
            visual,
            strategy=strategy,
            arm=arm,
            seed=seed,
            fold_result=fold_result,
            data=data,
        )
        fold_result["alpha_results"] = {}
        fold_result["prediction_device"] = "cpu canonical reevaluation"
        for alpha_key, alpha_values in alpha_results.items():
            combined_rows[alpha_key]["ham"].extend(alpha_values.pop("ham_rows"))
            combined_rows[alpha_key]["derm7pt"].extend(alpha_values.pop("derm_rows"))
            combined_rows[alpha_key]["bcn_train_oof"].extend(
                alpha_values.pop("bcn_rows", [])
            )
            fold_result["alpha_results"][alpha_key] = alpha_values
    result["variants"] = _compile_seed_variants(
        strategy=strategy,
        arm=arm,
        seed=seed,
        combined_rows=combined_rows,
    )
    result["model"] = provenance
    result["canonical_evaluation"] = {
        **environment,
        "raw_logits_regenerated": True,
        "all_prediction_devices": ["cpu"],
    }
    digest = screen._write_json(result_path, result)
    return {
        "status": "CANONICALLY_REEVALUATED",
        "path": str(result_path.relative_to(ROOT)),
        "sha256": digest,
        "canonical_environment": environment,
    }


def _variant_rows(
    strategy: str, arm: str, seed: int, alpha_key: str, dataset: str,
) -> list[dict[str, Any]]:
    result = _load_json(_result_path(strategy, arm, seed))
    return _read_jsonl(ROOT / result["variants"][alpha_key]["predictions"][dataset]["path"])


def _comparison(
    strategy: str,
    arm: str,
    alpha_key: str,
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
    for seed in adaptation.SEEDS:
        candidate = _variant_rows(strategy, arm, seed, alpha_key, dataset)
        baseline = consistency._candidate_rows(strategy, seed, dataset)
        if [row["id"] for row in candidate] != [row["id"] for row in baseline]:
            raise AssertionError("paired adaptation row order changed")
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
    result["positive_seed_count"] = sum(value > 0.0 for value in per_seed_delta.values())
    return result


def finalize() -> dict[str, Any]:
    protocol_sha = _protocol_sha256()
    derm_reference = consistency._candidate_rows(
        "derm_ham", adaptation.SEEDS[0], "derm7pt"
    )
    ham_reference = consistency._candidate_rows(
        "derm_ham", adaptation.SEEDS[0], "ham"
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
        0, len(adaptation.SEEDS), size=(N_BOOT, len(adaptation.SEEDS))
    )
    payload: dict[str, Any] = {
        "protocol": "dermpact-phase2-3d-backbone-adaptation-comparisons-v1",
        "protocol_sha256": protocol_sha,
        "direction": "adaptation candidate minus selected frozen Phase 2.3A reference",
        "strategies": {},
    }
    for strategy in adaptation.DATA_STRATEGIES:
        candidates = {}
        for arm in adaptation.ARMS:
            for alpha in adaptation.INTERPOLATION_ALPHAS:
                alpha_key = str(alpha).replace(".", "p")
                key = f"{arm}@{alpha}"
                values = {
                    dataset: {
                        metric: _comparison(
                            strategy,
                            arm,
                            alpha_key,
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
                            regressions.append({
                                "dataset": dataset,
                                "metric": metric,
                                "delta": result["estimate"],
                                "ci_high": result["ci_high"],
                            })
                disagreement = values["derm7pt"]["direct_rule_disagreement"]
                disagreement_violation = bool(
                    disagreement["estimate"] > DISAGREEMENT_MARGIN
                    and disagreement["ci_low"] > 0.0
                )
                primary = values["derm7pt"]["primary"]
                qualifies = bool(
                    alpha > 0.0
                    and primary["estimate"] > 0.0
                    and primary["positive_seed_count"] >= 2
                    and not regressions
                    and not disagreement_violation
                )
                candidates[key] = {
                    "arm": arm,
                    "alpha": alpha,
                    "comparisons": values,
                    "material_regressions": regressions,
                    "material_disagreement_increase": disagreement_violation,
                    "qualifies": qualifies,
                }
        eligible = [
            (key, value) for key, value in candidates.items() if value["qualifies"]
        ]
        selected = (
            max(
                eligible,
                key=lambda item: item[1]["comparisons"]["derm7pt"]["primary"]["estimate"],
            )[0]
            if eligible
            else "frozen_phase2_3a"
        )
        payload["strategies"][strategy] = {
            "candidates": candidates,
            "selected": selected,
        }
    comparison_hash = screen._write_json(COMPARISON_PATH, payload)

    artifacts = {}
    checkpoint_count = raw_count = prediction_count = 0
    for strategy in adaptation.DATA_STRATEGIES:
        for arm in adaptation.ARMS:
            for seed in adaptation.SEEDS:
                path = _result_path(strategy, arm, seed)
                result = _load_json(path)
                if result["protocol_sha256"] != protocol_sha:
                    raise AssertionError("adaptation result used another protocol")
                if result.get("canonical_evaluation", {}).get("status") != "PASS":
                    raise AssertionError(
                        "adaptation result lacks canonical venv_p2 CPU reevaluation"
                    )
                artifacts[f"{strategy}_{arm}_{seed}"] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for fold in result["fold_training"]:
                    checkpoint = fold["checkpoint"]
                    if sha256_file(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
                        raise AssertionError("adaptation checkpoint hash mismatch")
                    checkpoint_count += 1
                    for variant in fold["alpha_results"].values():
                        for raw in variant["raw_logits"].values():
                            if sha256_file(ROOT / raw["path"]) != raw["sha256"]:
                                raise AssertionError("adaptation raw-logit hash mismatch")
                            raw_count += 1
                for variant in result["variants"].values():
                    for prediction in variant["predictions"].values():
                        if sha256_file(ROOT / prediction["path"]) != prediction["sha256"]:
                            raise AssertionError("adaptation prediction hash mismatch")
                        prediction_count += 1
    receipt = {
        "protocol": "dermpact-phase2-3d-backbone-adaptation-receipt-v1",
        "protocol_sha256": protocol_sha,
        "status": "PASS",
        "decision": {
            strategy: values["selected"] for strategy, values in payload["strategies"].items()
        },
        "comparison": {
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
            "device": "cpu canonical evaluation",
        },
    }
    digest = screen._write_json(RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": digest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", choices=("ham", "derm7pt", "bcn_train"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--strategy", choices=adaptation.DATA_STRATEGIES)
    parser.add_argument("--arm", choices=adaptation.ARMS)
    parser.add_argument("--seed", type=int, choices=adaptation.SEEDS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--training-smoke", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.cache:
        payload = cache_prefix_tokens(args.cache, batch_size=args.batch_size)
    elif args.train:
        if args.strategy is None or args.arm is None or args.seed is None:
            parser.error("--train requires --strategy, --arm and --seed")
        payload = train_seed(args.strategy, args.arm, args.seed, device=args.device)
    elif args.finalize:
        payload = finalize()
    elif args.training_smoke:
        payload = run_training_path_smoke()
    elif args.reevaluate:
        if args.strategy is None or args.arm is None or args.seed is None:
            parser.error("--reevaluate requires --strategy, --arm and --seed")
        payload = reevaluate_seed(args.strategy, args.arm, args.seed)
    else:
        parser.error("choose --cache, --train or --finalize")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
