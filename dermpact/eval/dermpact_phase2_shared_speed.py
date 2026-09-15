"""DermPact Phase 2.2A one-epoch training-speed gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dermpact_phase2_baselines import (
    CANDIDATE_DISPLAY,
    CRITERIA,
    FEATURE_DIR,
    OUTPUT_DIR as PHASE2_1_OUTPUT_DIR,
    ROOT,
    _criterion_labels,
    _derm7pt_records,
    _ham_records,
    sha256_file,
)


MODULE = "dermpact.eval.dermpact_phase2_shared_speed"
CANDIDATES = ("dermlip", "panderm")
FOLD = 0
SEED = 20260827
BATCH_SIZE = 64
ADAPTER_DIM = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TORCH_THREADS = 12
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_2"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_2/speed_gate"
FOLD_MANIFEST = (
    PHASE2_1_OUTPUT_DIR / "dermpact_phase2_1_development_folds_v1.json"
)
RESOURCE_SNAPSHOT = (
    ROOT / "Planning discussions/phase2_audits/resource_snapshot_phase2_2_epoch.json"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    os.replace(temporary, path)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


class SharedPerceptionHead(nn.Module):
    """A standard shared adapter plus dataset-native and checklist heads."""

    def __init__(
        self,
        feature_dim: int,
        ham_classes: int,
        derm_classes: int,
        adapter_dim: int = ADAPTER_DIM,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.ham_diagnosis = nn.Linear(adapter_dim, ham_classes)
        self.derm_diagnosis = nn.Linear(adapter_dim, derm_classes)
        self.melanoma = nn.Linear(adapter_dim, 1)
        self.criteria = nn.ModuleList(
            nn.Linear(adapter_dim, 1) for _ in CRITERIA
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        shared = self.adapter(features)
        return {
            "ham_diagnosis": self.ham_diagnosis(shared),
            "derm_diagnosis": self.derm_diagnosis(shared),
            "melanoma": self.melanoma(shared).squeeze(-1),
            "criteria": torch.cat(
                [head(shared) for head in self.criteria], dim=1
            ),
        }


def masked_multitask_loss(
    outputs: Mapping[str, Tensor],
    *,
    dataset: str,
    diagnosis: Tensor,
    melanoma: Tensor,
    criteria: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute dataset-specific diagnosis and explicitly masked criteria loss."""

    if dataset not in {"ham", "derm7pt"}:
        raise ValueError(f"unsupported dataset: {dataset}")
    diagnosis_key = "ham_diagnosis" if dataset == "ham" else "derm_diagnosis"
    diagnosis_loss = F.cross_entropy(outputs[diagnosis_key], diagnosis)
    melanoma_loss = F.binary_cross_entropy_with_logits(
        outputs["melanoma"], melanoma.float()
    )
    if dataset == "ham":
        if criteria is not None:
            raise ValueError("HAM must not receive fabricated criterion labels")
        criterion_loss = outputs["criteria"].sum() * 0.0
        total = 0.5 * (diagnosis_loss + melanoma_loss)
    else:
        if criteria is None or criteria.shape != outputs["criteria"].shape:
            raise ValueError("Derm7pt requires one target for each criterion output")
        criterion_loss = F.binary_cross_entropy_with_logits(
            outputs["criteria"], criteria.float()
        )
        total = (diagnosis_loss + melanoma_loss + criterion_loss) / 3.0
    return total, {
        "diagnosis": diagnosis_loss,
        "melanoma": melanoma_loss,
        "criteria": criterion_loss,
    }


def _class_index(labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    classes = sorted(set(labels))
    index = {label: position for position, label in enumerate(classes)}
    return classes, np.asarray([index[label] for label in labels], dtype="int64")


def _locked_training_rows() -> dict[str, Any]:
    folds = _load_json(FOLD_MANIFEST)
    ham_fold = {str(row["cluster_id"]): int(row["fold"]) for row in folds["ham"]["folds"]}
    derm_fold = {str(row["group_id"]): int(row["fold"]) for row in folds["derm7pt"]["folds"]}
    ham = _ham_records()
    derm = _derm7pt_records()
    if set(ham_fold) != {str(row["cluster_id"]) for row in ham}:
        raise AssertionError("HAM fold manifest does not match the usable training lesions")
    if set(derm_fold) != {str(row["group_id"]) for row in derm}:
        raise AssertionError("Derm7pt fold manifest does not match development cases")
    ham_indices = np.asarray(
        [index for index, row in enumerate(ham) if ham_fold[str(row["cluster_id"])] != FOLD],
        dtype="int64",
    )
    derm_indices = np.asarray(
        [index for index, row in enumerate(derm) if derm_fold[str(row["group_id"])] != FOLD],
        dtype="int64",
    )
    return {
        "ham_records": ham,
        "derm_records": derm,
        "ham_indices": ham_indices,
        "derm_indices": derm_indices,
        "held_out_ham_lesions": sum(fold == FOLD for fold in ham_fold.values()),
        "held_out_derm_cases": sum(fold == FOLD for fold in derm_fold.values()),
    }


def _cyclic_batches(
    indices: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield full shuffled batches, reshuffling whenever the pool is exhausted."""

    if not len(indices) or steps < 1 or batch_size < 1:
        raise ValueError("indices, steps, and batch_size must be non-empty/positive")
    pool = np.asarray(indices, dtype="int64")
    order = rng.permutation(pool)
    cursor = 0
    for _ in range(steps):
        pieces = []
        needed = batch_size
        while needed:
            available = len(order) - cursor
            take = min(needed, available)
            pieces.append(order[cursor:cursor + take])
            cursor += take
            needed -= take
            if cursor == len(order):
                order = rng.permutation(pool)
                cursor = 0
        yield np.concatenate(pieces)


def _labels(rows: Mapping[str, Any]) -> dict[str, Any]:
    ham_labels = [str(row["label"]) for row in rows["ham_records"]]
    derm_labels = [
        str(row["diagnosis"]["harmonized5"]) for row in rows["derm_records"]
    ]
    ham_classes, ham_diagnosis = _class_index(ham_labels)
    derm_classes, derm_diagnosis = _class_index(derm_labels)
    derm_criteria = np.stack(
        [_criterion_labels(rows["derm_records"], criterion) for criterion in CRITERIA],
        axis=1,
    ).astype("float32")
    return {
        "ham_classes": ham_classes,
        "derm_classes": derm_classes,
        "ham_diagnosis": ham_diagnosis,
        "derm_diagnosis": derm_diagnosis,
        "ham_melanoma": np.asarray(
            [label == "melanoma" for label in ham_labels], dtype="float32"
        ),
        "derm_melanoma": np.asarray(
            [label == "MEL" for label in derm_labels], dtype="float32"
        ),
        "derm_criteria": derm_criteria,
    }


def _tensor(array: np.ndarray) -> Tensor:
    return torch.from_numpy(np.asarray(array))


def _feature_matrix(candidate: str, dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    feature_path = FEATURE_DIR / candidate / dataset / "features.npy"
    manifest_path = FEATURE_DIR / candidate / dataset / "feature_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest["feature_file_sha256"] != sha256_file(feature_path):
        raise AssertionError(f"{candidate}/{dataset} feature hash mismatch")
    features = np.load(feature_path, mmap_mode="r")
    expected = 7008 if dataset == "ham" else 616
    if features.shape != (expected, int(manifest["feature_dim"])):
        raise AssertionError(f"unexpected {candidate}/{dataset} feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise AssertionError(f"non-finite {candidate}/{dataset} feature")
    return features, manifest


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _batch_loss(
    model: SharedPerceptionHead,
    *,
    dataset: str,
    features: np.ndarray,
    indices: np.ndarray,
    labels: Mapping[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    outputs = model(_tensor(np.asarray(features[indices], dtype="float32")))
    if dataset == "ham":
        return masked_multitask_loss(
            outputs,
            dataset=dataset,
            diagnosis=_tensor(labels["ham_diagnosis"][indices]),
            melanoma=_tensor(labels["ham_melanoma"][indices]),
        )
    return masked_multitask_loss(
        outputs,
        dataset=dataset,
        diagnosis=_tensor(labels["derm_diagnosis"][indices]),
        melanoma=_tensor(labels["derm_melanoma"][indices]),
        criteria=_tensor(labels["derm_criteria"][indices]),
    )


def run_candidate(candidate: str) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # PyTorch permits setting interop threads only before parallel work starts.
        pass
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rows = _locked_training_rows()
    labels = _labels(rows)
    ham_features, ham_manifest = _feature_matrix(candidate, "ham")
    derm_features, derm_manifest = _feature_matrix(candidate, "derm7pt")
    if ham_features.shape[1] != derm_features.shape[1]:
        raise AssertionError("candidate feature dimensions differ across datasets")
    feature_dim = int(ham_features.shape[1])
    model = SharedPerceptionHead(
        feature_dim,
        len(labels["ham_classes"]),
        len(labels["derm_classes"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    steps = max(
        math.ceil(len(rows["ham_indices"]) / BATCH_SIZE),
        math.ceil(len(rows["derm_indices"]) / BATCH_SIZE),
    )
    ham_batches = list(
        _cyclic_batches(
            rows["ham_indices"], steps=steps, batch_size=BATCH_SIZE,
            rng=np.random.default_rng(SEED + 1),
        )
    )
    derm_batches = list(
        _cyclic_batches(
            rows["derm_indices"], steps=steps, batch_size=BATCH_SIZE,
            rng=np.random.default_rng(SEED + 2),
        )
    )

    model.eval()
    with torch.no_grad():
        ham_initial, _ = _batch_loss(
            model, dataset="ham", features=ham_features,
            indices=ham_batches[0], labels=labels,
        )
        derm_initial, _ = _batch_loss(
            model, dataset="derm7pt", features=derm_features,
            indices=derm_batches[0], labels=labels,
        )
        initial_loss = float(0.5 * (ham_initial + derm_initial))

    start_rss = _rss_bytes()
    peak_rss = start_rss
    step_seconds: list[float] = []
    losses: list[float] = []
    component_sums = {
        "ham_diagnosis": 0.0,
        "ham_melanoma": 0.0,
        "ham_criteria_masked": 0.0,
        "derm_diagnosis": 0.0,
        "derm_melanoma": 0.0,
        "derm_criteria": 0.0,
    }
    model.train()
    epoch_started = time.perf_counter()
    for ham_indices, derm_indices in zip(ham_batches, derm_batches):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        ham_loss, ham_components = _batch_loss(
            model, dataset="ham", features=ham_features,
            indices=ham_indices, labels=labels,
        )
        derm_loss, derm_components = _batch_loss(
            model, dataset="derm7pt", features=derm_features,
            indices=derm_indices, labels=labels,
        )
        loss = 0.5 * (ham_loss + derm_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss for {candidate}")
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise FloatingPointError(f"non-finite gradients for {candidate}")
        optimizer.step()
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach()))
        component_sums["ham_diagnosis"] += float(ham_components["diagnosis"].detach())
        component_sums["ham_melanoma"] += float(ham_components["melanoma"].detach())
        component_sums["ham_criteria_masked"] += float(ham_components["criteria"].detach())
        component_sums["derm_diagnosis"] += float(derm_components["diagnosis"].detach())
        component_sums["derm_melanoma"] += float(derm_components["melanoma"].detach())
        component_sums["derm_criteria"] += float(derm_components["criteria"].detach())
        current_rss = _rss_bytes()
        if current_rss is not None:
            peak_rss = max(peak_rss or 0, current_rss)
    elapsed = time.perf_counter() - epoch_started

    candidate_dir = EXPERIMENT_DIR / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_dir / "one_epoch_smoke_state.pt"
    temporary_checkpoint = checkpoint.with_suffix(".partial.pt")
    torch.save(model.state_dict(), temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)

    reloaded = SharedPerceptionHead(
        feature_dim,
        len(labels["ham_classes"]),
        len(labels["derm_classes"]),
    )
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        reference = model(_tensor(np.asarray(derm_features[derm_batches[0]], dtype="float32")))
        restored = reloaded(_tensor(np.asarray(derm_features[derm_batches[0]], dtype="float32")))
    reload_max_difference = max(
        float(torch.max(torch.abs(reference[key] - restored[key]))) for key in reference
    )
    if reload_max_difference != 0.0:
        raise AssertionError("checkpoint reload did not reproduce logits exactly")

    examples_seen = steps * BATCH_SIZE * 2
    result = {
        "protocol": "dermpact-phase2-2a-one-epoch-speed-result-v1",
        "status": "PASS",
        "evidence_level": "engineering speed gate only; not model-selection evidence",
        "candidate": candidate,
        "candidate_display": CANDIDATE_DISPLAY[candidate],
        "stage_timed": "frozen cached features plus trainable shared adapter and heads",
        "device": "cpu",
        "fold": FOLD,
        "seed": SEED,
        "feature_dim": feature_dim,
        "adapter_dim": ADAPTER_DIM,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_contract": {
            "per_dataset_batch_size": BATCH_SIZE,
            "paired_steps": steps,
            "examples_seen": examples_seen,
            "dataset_balancing": "one full HAM batch plus one full Derm7pt batch per step",
            "ham_unique_training_images": int(len(rows["ham_indices"])),
            "derm7pt_unique_training_cases": int(len(rows["derm_indices"])),
            "held_out_fold_ham_lesions": rows["held_out_ham_lesions"],
            "held_out_fold_derm7pt_cases": rows["held_out_derm_cases"],
        },
        "label_contract": {
            "ham_native_classes": labels["ham_classes"],
            "derm7pt_native_classes": labels["derm_classes"],
            "shared_binary_output": "melanoma",
            "derm7pt_only_criterion_outputs": list(CRITERIA),
            "ham_criterion_targets_fabricated": False,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dataset_loss_weight": 0.5,
        },
        "timing": {
            "epoch_seconds": elapsed,
            "examples_per_second": examples_seen / elapsed,
            "mean_step_seconds": float(np.mean(step_seconds)),
            "median_step_seconds": float(np.median(step_seconds)),
            "p95_step_seconds": float(np.quantile(step_seconds, 0.95)),
        },
        "loss_smoke": {
            "initial_fixed_pair": initial_loss,
            "epoch_mean": float(np.mean(losses)),
            "first_step": losses[0],
            "last_step": losses[-1],
            "component_means": {
                key: value / steps for key, value in component_sums.items()
            },
            "all_losses_finite": True,
            "all_gradients_finite": True,
        },
        "memory": {
            "rss_at_epoch_start_bytes": start_rss,
            "observed_peak_rss_bytes": peak_rss,
            "observed_rss_increase_bytes": (
                None if start_rss is None or peak_rss is None else peak_rss - start_rss
            ),
        },
        "checkpoint_reload": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": checkpoint_hash,
            "maximum_logit_difference": reload_max_difference,
            "usage": "smoke/reload validation only; forbidden for model selection",
        },
        "feature_manifests": {
            "ham": {
                "path": str((FEATURE_DIR / candidate / "ham/feature_manifest.json").relative_to(ROOT)),
                "sha256": sha256_file(FEATURE_DIR / candidate / "ham/feature_manifest.json"),
                "feature_sha256": ham_manifest["feature_file_sha256"],
            },
            "derm7pt": {
                "path": str((FEATURE_DIR / candidate / "derm7pt/feature_manifest.json").relative_to(ROOT)),
                "sha256": sha256_file(FEATURE_DIR / candidate / "derm7pt/feature_manifest.json"),
                "feature_sha256": derm_manifest["feature_file_sha256"],
            },
        },
        "hard_scope_checks": {
            "ham_calibration_records_loaded": 0,
            "final_test_records_loaded": 0,
            "derm7pt_official_test_records_loaded": 0,
            "bcn20000_records_loaded": 0,
            "ph2_records_loaded": 0,
        },
    }
    result_path = OUTPUT_DIR / f"dermpact_phase2_2a_{candidate}_speed_v1.json"
    result["result_sha256"] = _write_json(result_path, result)
    return result


def build_protocol() -> dict[str, Any]:
    protocol = {
        "protocol": "dermpact-phase2-2a-one-epoch-speed-gate-v1",
        "status": "PRESPECIFIED_ENGINEERING_GATE",
        "purpose": (
            "measure one representative fold-training epoch for the initial frozen-backbone "
            "Phase 2.2 stage before committing to the full training schedule"
        ),
        "candidates": list(CANDIDATES),
        "candidate_selection_authority": False,
        "architecture": {
            "shared_adapter": f"LayerNorm(512)-Linear(512,{ADAPTER_DIM})-GELU-Dropout(0.1)",
            "diagnosis_outputs": "shared representation with HAM-7 and Derm7pt-5 projections",
            "melanoma_output": "one shared binary projection",
            "criterion_outputs": "seven independent binary projections, Derm7pt-masked",
        },
        "training": {
            "fold": FOLD,
            "seed": SEED,
            "batch_size_per_dataset": BATCH_SIZE,
            "dataset_balancing": "paired HAM and Derm7pt batches",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "torch_threads": TORCH_THREADS,
        },
        "allowed_data": [
            "HAM Phase 2.1 training-role features outside fold 0",
            "Derm7pt locked development features outside fold 0",
        ],
        "forbidden_data": [
            "HAM calibration",
            "all final-test partitions",
            "Derm7pt official test",
            "BCN20000",
            "PH2",
        ],
        "interpretation": {
            "can_estimate": "frozen-feature shared-head training time",
            "cannot_estimate": [
                "image decoding/augmentation time",
                "backbone forward/backward time",
                "last-block or full-backbone fine-tuning time",
                "candidate accuracy after training",
            ],
            "checkpoint_use": "reload smoke test only",
        },
        "fold_manifest": {
            "path": str(FOLD_MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(FOLD_MANIFEST),
        },
        "resource_snapshot": (
            {
                "path": str(RESOURCE_SNAPSHOT.relative_to(ROOT)),
                "sha256": sha256_file(RESOURCE_SNAPSHOT),
            }
            if RESOURCE_SNAPSHOT.is_file() else None
        ),
    }
    path = OUTPUT_DIR / "dermpact_phase2_2a_speed_protocol_v1.json"
    protocol["protocol_sha256"] = _write_json(path, protocol)
    return protocol


def run_all() -> dict[str, Any]:
    protocol = build_protocol()
    results = [run_candidate(candidate) for candidate in CANDIDATES]
    projections = {}
    for result in results:
        epoch = float(result["timing"]["epoch_seconds"])
        projections[result["candidate"]] = {
            "20_epochs_one_fold_seconds": epoch * 20,
            "20_epochs_five_folds_one_seed_seconds": epoch * 20 * 5,
            "20_epochs_five_folds_three_seeds_seconds": epoch * 20 * 5 * 3,
        }
    full_two_candidate_seconds = sum(
        item["20_epochs_five_folds_three_seeds_seconds"]
        for item in projections.values()
    )
    receipt = {
        "protocol": "dermpact-phase2-2a-one-epoch-speed-receipt-v1",
        "phase": "2.2A",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "CPU",
            "torch_threads": torch.get_num_threads(),
        },
        "protocol_artifact": {
            "path": "research/manifests/dermpact_phase2_2/dermpact_phase2_2a_speed_protocol_v1.json",
            "sha256": protocol["protocol_sha256"],
        },
        "candidate_results": {
            result["candidate"]: {
                "epoch_seconds": result["timing"]["epoch_seconds"],
                "examples_per_second": result["timing"]["examples_per_second"],
                "loss_initial": result["loss_smoke"]["initial_fixed_pair"],
                "loss_last_step": result["loss_smoke"]["last_step"],
                "result_path": (
                    "research/manifests/dermpact_phase2_2/"
                    f"dermpact_phase2_2a_{result['candidate']}_speed_v1.json"
                ),
                "result_sha256": result["result_sha256"],
            }
            for result in results
        },
        "runtime_projections": {
            "assumption": "linear scaling from this one-epoch frozen-feature smoke run",
            "per_candidate": projections,
            "both_candidates_20_epochs_five_folds_three_seeds_seconds": full_two_candidate_seconds,
            "warning": (
                "these projections exclude image loading and backbone backpropagation and "
                "must not be used for selective last-block fine-tuning estimates"
            ),
        },
        "decision": (
            "The initial Phase 2.2 frozen-feature training stage is feasible locally. "
            "This speed gate does not choose DermLIP or the hybrid."
        ),
        "next_gate": (
            "fix the Phase 2.2 training/selection protocol, then run controlled CV; "
            "benchmark last-block adaptation separately on real images before enabling it"
        ),
        "hard_scope_checks": {
            "forbidden_records_loaded": 0,
            "candidate_selected_from_speed": False,
        },
    }
    receipt_path = OUTPUT_DIR / "dermpact_phase2_2a_speed_receipt_v1.json"
    receipt["receipt_sha256"] = _write_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol_only:
        payload = build_protocol()
    elif args.candidate:
        payload = run_candidate(args.candidate)
    else:
        payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
