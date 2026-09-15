"""Isolated Phase 2.4A Tier-2 attention-factor experiment core."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from . import dermpact_phase2_4_localized_training as _base
from . import dermpact_phase2_4a_t2_epoch20 as _c10
from .dermpact_phase2_4a_multidepth_model import Phase24MultiDepthCriterionModel
from .dermpact_phase2_4a_sharp_model import Phase24SharpCriterionModel


ROOT = _base.ROOT
SEED = 20260827
FOLDS = tuple(range(5))
ARMS = ("sharp", "multidepth")
NAMESPACE = "localized_criterion_attention_factor_v1"
MANIFEST_DIR = ROOT / "research/manifests/dermpact_phase2_4" / NAMESPACE
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_4" / NAMESPACE
PROTOCOL_SCHEMA = "dermpact-phase2-4a-attention-factor-protocol-v1"
RESUME_SCHEMA = "dermpact-phase2-4a-attention-factor-resume-v1"
TRAINING_SCHEMA = "dermpact-phase2-4a-attention-factor-fold-training-v1"
CANONICAL_SCHEMA = "dermpact-phase2-4a-attention-factor-canonical-fold-v1"
_VISUAL_TEMPLATE: nn.Module | None = None


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AssertionError(f"required artifact missing: {_relative(path)}")
    return {"path": _relative(path), "sha256": _sha256(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(_canonical_bytes(payload))
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _atomic_torch(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(_cpu_tree(payload), temporary)
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise AssertionError(f"artifact/sidecar missing: {_relative(path)}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    observed = _sha256(path)
    if expected != observed:
        raise AssertionError(f"artifact hash mismatch: {_relative(path)}")
    return observed


def _validate_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown attention-factor arm: {arm}")


def _c10_fold_artifacts(fold: int) -> dict[str, Any]:
    receipt_path = _c10.training_receipt_path("c10", fold)
    receipt = _load_json(receipt_path)
    if (
        receipt.get("status"), receipt.get("candidate_id"), receipt.get("seed"),
        receipt.get("fold"), receipt.get("completed_epochs"),
    ) != ("PASS", "c10", SEED, fold, 20):
        raise AssertionError(f"confirmed c10 fold identity changed: {fold}")
    _verify_sidecar(receipt_path)
    residual = ROOT / receipt["residual_checkpoint"]["path"]
    if _sha256(residual) != receipt["residual_checkpoint"]["sha256"]:
        raise AssertionError(f"confirmed c10 residual changed: {fold}")
    c10_protocol, _, _ = _c10._candidate_protocol("c10")
    retained = c10_protocol["artifacts"]["retained_folds"][fold]["checkpoint"]
    retained_path = ROOT / retained["path"]
    if _sha256(retained_path) != retained["sha256"]:
        raise AssertionError(f"retained Phase 2.3 checkpoint changed: {fold}")
    if receipt["source_checkpoint_sha256"] != retained["sha256"]:
        raise AssertionError(f"c10/retained fold binding changed: {fold}")
    return {
        "fold": fold,
        "retained_core": dict(retained),
        "c10_training_receipt": _artifact(receipt_path),
        "c10_residual": dict(receipt["residual_checkpoint"]),
    }


def generate_protocol_spec() -> dict[str, Any]:
    """Generate the closed scientific design without writing or freezing it."""
    c10_config = _c10.candidate_config("c10")["config"]
    source_paths = {
        "attention_factor_core": Path(__file__),
        "sharp_model": ROOT / "dermpact/eval/dermpact_phase2_4a_sharp_model.py",
        "multidepth_model": ROOT / "dermpact/eval/dermpact_phase2_4a_multidepth_model.py",
        "localized_training": ROOT / "dermpact/eval/dermpact_phase2_4_localized_training.py",
        "directml_launcher": ROOT / "research/scripts/dermpact_phase2_4a_attention_factor_directml_train.py",
        "controller_template": ROOT / "research/scripts/dermpact_phase2_4a_attention_factor_controller.py",
        "full_run_controller": ROOT / "research/scripts/dermpact_phase2_4a_attention_factor_full_run_controller.py",
    }
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "SPECIFIED_NOT_FROZEN",
        "identity": {"seed": SEED, "folds": list(FOLDS), "arms": list(ARMS)},
        "model": {
            "comparator": "confirmed c10@20 from the same seed and fold",
            "retained_core": "matching retained Phase 2.3 core, fully frozen",
            "c10": "matching confirmed c10 residual, fully frozen",
            "c10_attention_dim": int(c10_config["attention_dim"]),
            "arms": {
                "sharp": {
                    "class": "Phase24SharpCriterionModel",
                    "new_trainable": "sharp_residual only",
                    "learned_temperature": True,
                    "initial_temperature": 10.0,
                    "maximum_temperature": 100.0,
                },
                "multidepth": {
                    "class": "Phase24MultiDepthCriterionModel",
                    "new_trainable": "multidepth_residual only",
                    "learned_temperature": False,
                    "token_depth": "frozen block-10 prefix tokens",
                },
            },
        },
        "training": {
            "dataset": "Derm7pt development training rows only",
            "epochs": 20,
            "effective_batch_size": 64,
            "directml_microbatch_size": 8,
            "precision": "fp32",
            "gradient_clip_norm": 1.0,
            "class_balancing": "current per-fold class-balanced BCE semantics",
            "consistency_weight": float(c10_config["consistency_weight"]),
            "optimizer": {
                "projection_group": {
                    "learning_rate": float(c10_config["projection_learning_rate"]),
                    "weight_decay": float(c10_config["projection_weight_decay"]),
                },
                "scale_group": {
                    "learning_rate": float(c10_config["scale_learning_rate"]),
                    "weight_decay": 0.0,
                },
                "sharp_temperature_group": {
                    "learning_rate": float(c10_config["projection_learning_rate"]),
                    "weight_decay": 0.0,
                },
            },
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_ratio": 0.1,
                "minimum_lr_ratio": 0.01,
                "horizon_epochs": 20,
            },
            "early_stopping": False,
            "best_epoch_selection": False,
        },
        "canonical_evaluation": {
            "device": "cpu",
            "raw_oof_required": True,
            "protected_logits": "exact tensor equality with c10",
            "initial_equivalence": "candidate criteria exactly equal c10 before training",
            "attention_evidence": ["entropy", "top_k_mass"],
            "top_k": 10,
        },
        "directml_probe": {
            "required_per_arm": True,
            "initial_cpu_device_max_abs_tolerance": 1e-5,
            "post_step_reload_max_abs_tolerance": 1e-6,
            "full_training_requires_pass": True,
        },
        "comparison": {
            "direction": "candidate minus matching c10@20",
            "paired_by": "identical canonical OOF row indices",
            "selection_forbidden": "never choose a best seed; this screen uses only seed 20260827",
            "safe_arm_rule": "all five canonical folds PASS with diagnosis/melanoma exact; direct_rule_disagreement estimate <= +0.02; joint_correct_agreement estimate >= -0.02",
            "promotion_rule": "criterion_mean_auc candidate-minus-c10 95% CI lower bound > 0",
            "ranking_order": "SAFE promoted arms; decisive sharp-minus-multidepth CI first, otherwise descending c10 criterion estimate, ci_low, arm",
            "arm_to_arm_required": True,
        },
        "hard_scope_checks": {
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
        "artifacts": {
            "fold_inputs": [_c10_fold_artifacts(fold) for fold in FOLDS],
            "sources": {name: _artifact(path) for name, path in source_paths.items()},
        },
    }


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(protocol)).hexdigest()


def validate_protocol_spec(protocol: Mapping[str, Any]) -> str:
    """Fail closed if scientific semantics or any bound input has drifted."""
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise AssertionError("attention-factor protocol schema changed")
    if protocol.get("status") not in {"SPECIFIED_NOT_FROZEN", "FROZEN"}:
        raise AssertionError("attention-factor protocol status is invalid")
    identity = protocol.get("identity", {})
    if identity != {"seed": SEED, "folds": list(FOLDS), "arms": list(ARMS)}:
        raise AssertionError("attention-factor closed identity changed")
    training = protocol.get("training", {})
    if (
        training.get("epochs"), training.get("effective_batch_size"),
        training.get("directml_microbatch_size"), training.get("precision"),
        training.get("early_stopping"), training.get("best_epoch_selection"),
        training.get("gradient_clip_norm"),
    ) != (20, 64, 8, "fp32", False, False, 1.0):
        raise AssertionError("attention-factor training semantics changed")
    c10_config = _c10.candidate_config("c10")["config"]
    expected_optimizer = {
        "projection_group": {
            "learning_rate": float(c10_config["projection_learning_rate"]),
            "weight_decay": float(c10_config["projection_weight_decay"]),
        },
        "scale_group": {
            "learning_rate": float(c10_config["scale_learning_rate"]),
            "weight_decay": 0.0,
        },
        "sharp_temperature_group": {
            "learning_rate": float(c10_config["projection_learning_rate"]),
            "weight_decay": 0.0,
        },
    }
    if training.get("optimizer") != expected_optimizer:
        raise AssertionError("attention-factor optimizer semantics changed")
    if training.get("scheduler") != {
        "name": "warmup_cosine", "warmup_ratio": 0.1,
        "minimum_lr_ratio": 0.01, "horizon_epochs": 20,
    }:
        raise AssertionError("attention-factor scheduler semantics changed")
    if float(training.get("consistency_weight", float("nan"))) != float(c10_config["consistency_weight"]):
        raise AssertionError("attention-factor consistency semantics changed")
    arms = protocol.get("model", {}).get("arms", {})
    if (
        set(arms) != set(ARMS)
        or protocol.get("model", {}).get("c10_attention_dim") != 64
        or arms["sharp"].get("class") != "Phase24SharpCriterionModel"
        or arms["sharp"].get("initial_temperature") != 10.0
        or arms["sharp"].get("maximum_temperature") != 100.0
        or arms["multidepth"].get("class") != "Phase24MultiDepthCriterionModel"
        or arms["multidepth"].get("token_depth") != "frozen block-10 prefix tokens"
    ):
        raise AssertionError("attention-factor topology semantics changed")
    if protocol.get("directml_probe") != {
        "required_per_arm": True,
        "initial_cpu_device_max_abs_tolerance": 1e-5,
        "post_step_reload_max_abs_tolerance": 1e-6,
        "full_training_requires_pass": True,
    }:
        raise AssertionError("attention-factor DirectML probe semantics changed")
    if protocol.get("comparison") != {
        "direction": "candidate minus matching c10@20",
        "paired_by": "identical canonical OOF row indices",
        "selection_forbidden": "never choose a best seed; this screen uses only seed 20260827",
        "safe_arm_rule": "all five canonical folds PASS with diagnosis/melanoma exact; direct_rule_disagreement estimate <= +0.02; joint_correct_agreement estimate >= -0.02",
        "promotion_rule": "criterion_mean_auc candidate-minus-c10 95% CI lower bound > 0",
        "ranking_order": "SAFE promoted arms; decisive sharp-minus-multidepth CI first, otherwise descending c10 criterion estimate, ci_low, arm",
        "arm_to_arm_required": True,
    }:
        raise AssertionError("attention-factor comparison/ranking semantics changed")
    expected_scope = {key: 0 for key in (
        "derm7pt_official_test_rows_loaded", "ham_calibration_or_final_rows_loaded",
        "bcn_validation_or_final_rows_loaded", "ph2_rows_loaded",
    )}
    if protocol.get("hard_scope_checks") != expected_scope:
        raise AssertionError("attention-factor scope checks changed")
    rows = protocol.get("artifacts", {}).get("fold_inputs", [])
    if [row.get("fold") for row in rows] != list(FOLDS):
        raise AssertionError("attention-factor fold inputs changed")
    for fold, row in enumerate(rows):
        if row != _c10_fold_artifacts(fold):
            raise AssertionError(f"attention-factor fold binding changed: {fold}")
        for name in ("retained_core", "c10_training_receipt", "c10_residual"):
            artifact = row[name]
            if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
                raise AssertionError(f"protocol-bound artifact changed: {artifact['path']}")
    sources = protocol.get("artifacts", {}).get("sources", {})
    if set(sources) != {
        "attention_factor_core", "sharp_model", "multidepth_model",
        "localized_training", "directml_launcher", "controller_template",
        "full_run_controller",
    }:
        raise AssertionError("attention-factor protocol source set changed")
    for artifact in sources.values():
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"protocol-bound source changed: {artifact['path']}")
    return protocol_sha256(protocol)


class AttentionFactorAdapter(nn.Module):
    """Expose one consistent training/evidence surface for both closed arms."""

    def __init__(self, arm: str, confirmed: nn.Module) -> None:
        super().__init__()
        _validate_arm(arm)
        self.arm = arm
        self.variant = (
            Phase24SharpCriterionModel(confirmed)
            if arm == "sharp"
            else Phase24MultiDepthCriterionModel(confirmed)
        )

    @property
    def base_model(self):
        return self.variant.confirmed_model.base_model

    @property
    def confirmed_model(self):
        return self.variant.confirmed_model

    @property
    def new_residual(self):
        return (
            self.variant.sharp_residual
            if self.arm == "sharp"
            else self.variant.multidepth_residual
        )

    def forward(self, prefix_tokens: Tensor) -> Mapping[str, Tensor]:
        output = dict(self.variant(prefix_tokens))
        if self.arm == "sharp":
            residual = output["sharp_criterion_residual"]
            attention = output["sharp_criterion_attention"]
            c10_logits = output["c10_criteria"]
        else:
            residual = output["multidepth_criterion_residual"]
            attention = output["multidepth_criterion_attention"]
            c10_logits = output["criteria"] - residual
        output.update({
            "criterion_residual": residual,
            "criterion_attention": attention,
            "base_criteria": c10_logits,
            "c10_criteria": c10_logits,
        })
        return output

    def train(self, mode: bool = True):
        super().train(mode)
        self.variant.train(mode)
        self.confirmed_model.eval()
        return self


def _fold_input(protocol: Mapping[str, Any], fold: int) -> Mapping[str, Any]:
    if fold not in FOLDS:
        raise ValueError(fold)
    return protocol["artifacts"]["fold_inputs"][fold]


def _new_model(protocol: Mapping[str, Any], arm: str, fold: int) -> AttentionFactorAdapter:
    _validate_arm(arm)
    row = _fold_input(protocol, fold)
    retained = torch.load(ROOT / row["retained_core"]["path"], map_location="cpu", weights_only=True)
    if retained.get("arm") != _base.ARM or "model_state" not in retained:
        raise AssertionError("retained Phase 2.3 checkpoint payload changed")
    prototypes = torch.from_numpy(np.load(_base.PROTOTYPES_PATH, allow_pickle=False).copy())
    visual = _load_visual_copy()
    confirmed = _base._new_model(
        visual,
        model_state=retained["model_state"],
        prototypes=prototypes,
        attention_dim=64,
        initialization_seed=SEED + fold,
    )
    c10_state = torch.load(ROOT / row["c10_residual"]["path"], map_location="cpu", weights_only=True)
    confirmed.criterion_residual.load_state_dict(c10_state, strict=True)
    confirmed.eval()
    model = AttentionFactorAdapter(arm, confirmed)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    prefix = "variant.sharp_residual." if arm == "sharp" else "variant.multidepth_residual."
    if not trainable or any(not name.startswith(prefix) for name in trainable):
        raise AssertionError(f"unexpected trainable parameters for {arm}: {trainable}")
    return model


def _load_visual_copy() -> nn.Module:
    """Load official OpenCLIP once, then clone it within the worker process."""
    global _VISUAL_TEMPLATE
    if _VISUAL_TEMPLATE is None:
        visual, _, _ = _base.adaptation._load_hybrid_visual()
        visual.eval()
        _VISUAL_TEMPLATE = visual
    return copy.deepcopy(_VISUAL_TEMPLATE)


def _optimizer(model: AttentionFactorAdapter, protocol: Mapping[str, Any]) -> torch.optim.AdamW:
    config = protocol["training"]["optimizer"]
    residual = model.new_residual
    if model.arm == "sharp":
        projection = [
            *residual.prototype_projection.parameters(),
            *residual.patch_projection.parameters(),
        ]
        groups = [
            {"params": projection, **_group(config["projection_group"])},
            {"params": [residual.residual_scale], **_group(config["scale_group"])},
            {"params": [residual.log_temperature], **_group(config["sharp_temperature_group"])},
        ]
    else:
        inner = residual.intermediate_residual
        projection = [
            *inner.prototype_projection.parameters(), *inner.patch_key.parameters(),
            *inner.patch_value.parameters(),
        ]
        groups = [
            {"params": projection, **_group(config["projection_group"])},
            {"params": [inner.residual_scale], **_group(config["scale_group"])},
        ]
    return torch.optim.AdamW(groups)


def _group(config: Mapping[str, Any]) -> dict[str, float]:
    return {"lr": float(config["learning_rate"]), "weight_decay": float(config["weight_decay"])}


def assert_exact_c10_initialization(
    model: AttentionFactorAdapter, tokens: Tensor, *, criteria_tolerance: float = 1e-6,
) -> dict[str, float | bool]:
    model.eval()
    with torch.no_grad():
        expected = model.confirmed_model(tokens)
        actual = model(tokens)
    for name in ("diagnosis", "melanoma"):
        if not torch.equal(actual[name], expected[name]):
            raise AssertionError(f"{model.arm} is not exactly c10 at initialization: {name}")
    residual = actual["criterion_residual"]
    residual_exact_zero = bool(torch.count_nonzero(residual).item() == 0)
    if not residual_exact_zero:
        raise AssertionError(f"{model.arm} new residual is not exactly zero at initialization")
    criteria_max_abs = float(torch.max(torch.abs(actual["criteria"] - expected["criteria"])).item())
    if not math.isfinite(criteria_max_abs) or criteria_max_abs > criteria_tolerance:
        raise AssertionError(
            f"{model.arm} initialization criteria drift {criteria_max_abs} exceeds {criteria_tolerance}"
        )
    return {
        "diagnosis_exact": True,
        "melanoma_exact": True,
        "new_residual_exact_zero": True,
        "criteria_max_abs": criteria_max_abs,
        "criteria_tolerance": criteria_tolerance,
    }


def attention_evidence(attention: Tensor, *, top_k: int = 10) -> dict[str, np.ndarray]:
    """Return per-case/criterion/anchor normalized entropy and top-k mass."""
    values = attention.detach().float().cpu()
    if values.ndim != 4 or values.shape[-1] < 2:
        raise ValueError("attention must have shape [rows, criteria, anchors, patches]")
    if top_k <= 0 or top_k > values.shape[-1]:
        raise ValueError("top_k must be within the patch dimension")
    probabilities = values.clamp_min(torch.finfo(values.dtype).tiny)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1) / math.log(values.shape[-1])
    top_mass = torch.topk(values, k=top_k, dim=-1).values.sum(dim=-1)
    return {
        "attention_normalized_entropy": entropy.numpy().astype("float32", copy=False),
        "attention_top_k_mass": top_mass.numpy().astype("float32", copy=False),
    }


def _probe(protocol: Mapping[str, Any], arm: str, device: Any, *, fold: int = 0) -> dict[str, Any]:
    """Exercise the actual topology: forward/backward/step/save/reload/parity."""
    digest = validate_protocol_spec(protocol)
    model = _new_model(protocol, arm, fold)
    data = _base._derm_data()
    indices = np.flatnonzero(data["derm_folds"] != fold)[:8]
    tokens = torch.from_numpy(np.asarray(data["derm_tokens"][indices], dtype="float32").copy())
    initialization = assert_exact_c10_initialization(model, tokens)
    with torch.no_grad():
        cpu_initial = {key: value.detach().cpu() for key, value in model(tokens).items() if isinstance(value, Tensor)}
    model.to(device).eval()
    with torch.no_grad():
        device_initial = model(tokens.to(device))
    parity_names = ("diagnosis", "melanoma", "criteria", "criterion_residual", "criterion_attention")
    parity_max_abs = {
        name: float(torch.max(torch.abs(cpu_initial[name] - device_initial[name].detach().cpu())).item())
        for name in parity_names
    }
    parity_tolerance = float(protocol["directml_probe"]["initial_cpu_device_max_abs_tolerance"])
    if any(value > parity_tolerance for value in parity_max_abs.values()):
        raise AssertionError(f"{arm} CPU/device probe parity exceeded {parity_tolerance}: {parity_max_abs}")
    model.train()
    optimizer = _optimizer(model, protocol)
    output = model(tokens.to(device))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["criteria"],
        torch.from_numpy(np.asarray(data["derm_criteria"][indices], dtype="float32")).to(device),
    )
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.new_residual.parameters(), 1.0)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
        raise FloatingPointError(f"non-finite {arm} DirectML probe")
    optimizer.step()
    state = _cpu_tree(model.new_residual.state_dict())
    model.to("cpu").eval()
    with torch.inference_mode():
        post_step = model(tokens)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    reloaded = _new_model(protocol, arm, fold)
    reloaded.new_residual.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
    if not _base._state_dict_exact(state, reloaded.new_residual.state_dict()):
        raise AssertionError(f"{arm} probe save/reload is not exact")
    reloaded.eval()
    with torch.inference_mode():
        post_reload = reloaded(tokens)
    reload_names = ("diagnosis", "melanoma", "criteria", "criterion_residual", "criterion_attention")
    reload_max_abs = {
        name: float(torch.max(torch.abs(post_step[name] - post_reload[name])).item())
        for name in reload_names
    }
    reload_tolerance = float(protocol["directml_probe"]["post_step_reload_max_abs_tolerance"])
    if any(value > reload_tolerance for value in reload_max_abs.values()):
        raise AssertionError(f"{arm} post-step reload exceeded {reload_tolerance}: {reload_max_abs}")
    return {
        "status": "PASS",
        "arm": arm,
        "fold": fold,
        "protocol_sha256": digest,
        "device": str(device),
        "covered": ["actual_model", "forward", "backward", "optimizer", "save_reload", "cpu_device_parity"],
        "initial_c10_equivalence_pass": True,
        "initialization_equivalence": initialization,
        "initial_cpu_device_max_abs": parity_max_abs,
        "initial_cpu_device_tolerance": parity_tolerance,
        "post_step_reload_max_abs": reload_max_abs,
        "post_step_reload_tolerance": reload_tolerance,
        "finite_loss": float(loss.detach().cpu()),
        "finite_gradient_norm": float(gradient.detach().cpu()),
        "initial_output_names": sorted(cpu_initial),
    }


def probe_sharp_directml(protocol: Mapping[str, Any], device: Any, *, fold: int = 0) -> dict[str, Any]:
    return _probe(protocol, "sharp", device, fold=fold)


def probe_multidepth_directml(protocol: Mapping[str, Any], device: Any, *, fold: int = 0) -> dict[str, Any]:
    return _probe(protocol, "multidepth", device, fold=fold)


def _fold_dir(arm: str, fold: int) -> Path:
    return EXPERIMENT_DIR / arm / f"seed_{SEED}" / f"fold_{fold}"


def _resume_path(arm: str, fold: int) -> Path:
    return _fold_dir(arm, fold) / "epoch_resume_state.pt"


def _residual_path(arm: str, fold: int) -> Path:
    return _fold_dir(arm, fold) / "final_new_residual_state.pt"


def training_receipt_path(arm: str, fold: int) -> Path:
    return MANIFEST_DIR / arm / f"seed_{SEED}_fold_{fold}_training.json"


def canonical_receipt_path(arm: str, fold: int) -> Path:
    return MANIFEST_DIR / arm / f"seed_{SEED}_fold_{fold}_canonical.json"


def validate_resume_identity(
    payload: Mapping[str, Any], *, protocol_sha: str, arm: str, fold: int,
    retained_sha: str, c10_sha: str,
) -> None:
    observed = tuple(payload.get(key) for key in (
        "format", "protocol_sha256", "arm", "seed", "fold",
        "retained_core_sha256", "c10_residual_sha256",
    ))
    expected = (RESUME_SCHEMA, protocol_sha, arm, SEED, fold, retained_sha, c10_sha)
    if observed != expected:
        raise AssertionError(f"attention-factor resume identity mismatch: {observed!r}")


def _write_resume(
    arm: str, fold: int, *, protocol_sha: str, retained_sha: str, c10_sha: str,
    completed_epochs: int, model: AttentionFactorAdapter,
    optimizer: torch.optim.Optimizer, scheduler: Any,
    history: Sequence[Mapping[str, Any]],
) -> str:
    return _atomic_torch(_resume_path(arm, fold), {
        "format": RESUME_SCHEMA,
        "protocol_sha256": protocol_sha,
        "arm": arm,
        "seed": SEED,
        "fold": fold,
        "retained_core_sha256": retained_sha,
        "c10_residual_sha256": c10_sha,
        "completed_epochs": completed_epochs,
        "new_residual_state": model.new_residual.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "history": list(history),
    })


def _load_resume(
    arm: str, fold: int, *, protocol_sha: str, retained_sha: str, c10_sha: str,
    model: AttentionFactorAdapter, optimizer: torch.optim.Optimizer,
    scheduler: Any, device: Any, epochs: int,
) -> tuple[int, list[dict[str, Any]]] | None:
    path = _resume_path(arm, fold)
    if not path.exists():
        return None
    _verify_sidecar(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_identity(
        payload, protocol_sha=protocol_sha, arm=arm, fold=fold,
        retained_sha=retained_sha, c10_sha=c10_sha,
    )
    completed, history = int(payload["completed_epochs"]), list(payload["history"])
    if not 1 <= completed <= epochs or len(history) != completed:
        raise AssertionError("attention-factor resume completion/history mismatch")
    model.new_residual.load_state_dict(payload["new_residual_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return completed, history


def verified_training_receipt(
    protocol: Mapping[str, Any], arm: str, fold: int,
) -> dict[str, Any] | None:
    """Return a completed fold only when all identity and artifact checks pass."""
    _validate_arm(arm)
    protocol_sha = validate_protocol_spec(protocol)
    path = training_receipt_path(arm, fold)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    _verify_sidecar(path)
    receipt = _load_json(path)
    row = _fold_input(protocol, fold)
    expected = (
        TRAINING_SCHEMA, "PASS", protocol_sha, arm, SEED, fold, 20,
        row["retained_core"]["sha256"], row["c10_residual"]["sha256"], 64, 8,
    )
    observed = (
        receipt.get("schema_version"), receipt.get("status"),
        receipt.get("protocol_sha256"), receipt.get("arm"), receipt.get("seed"),
        receipt.get("fold"), receipt.get("completed_epochs"),
        receipt.get("retained_core_sha256"), receipt.get("c10_residual_sha256"),
        receipt.get("effective_batch_size"), receipt.get("microbatch_size"),
    )
    if observed != expected or receipt.get("hard_scope_checks") != protocol["hard_scope_checks"]:
        raise AssertionError(f"completed training receipt identity changed: {arm}/{fold}")
    if len(receipt.get("history", [])) != 20:
        raise AssertionError(f"completed training history changed: {arm}/{fold}")
    artifact = receipt.get("new_residual_checkpoint", {})
    expected_path = _residual_path(arm, fold)
    if artifact.get("path") != _relative(expected_path):
        raise AssertionError(f"completed training checkpoint path changed: {arm}/{fold}")
    if _verify_sidecar(expected_path) != artifact.get("sha256"):
        raise AssertionError(f"completed training checkpoint hash changed: {arm}/{fold}")
    return receipt


def verified_canonical_receipt(
    protocol: Mapping[str, Any], arm: str, fold: int,
) -> dict[str, Any] | None:
    """Return canonical evidence only after receipt and raw OOF verification."""
    _validate_arm(arm)
    protocol_sha = validate_protocol_spec(protocol)
    path = canonical_receipt_path(arm, fold)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    _verify_sidecar(path)
    receipt = _load_json(path)
    observed = (
        receipt.get("schema_version"), receipt.get("status"),
        receipt.get("protocol_sha256"), receipt.get("arm"), receipt.get("seed"),
        receipt.get("fold"), receipt.get("protected_logits_equal_c10"),
        receipt.get("new_residual_state_reload_exact"),
    )
    expected = (CANONICAL_SCHEMA, "PASS", protocol_sha, arm, SEED, fold, True, True)
    if observed != expected or receipt.get("hard_scope_checks") != protocol["hard_scope_checks"]:
        raise AssertionError(f"completed canonical receipt identity changed: {arm}/{fold}")
    raw_artifact = receipt.get("raw_oof", {})
    raw = ROOT / raw_artifact.get("path", "__missing__")
    if _verify_sidecar(raw) != raw_artifact.get("sha256"):
        raise AssertionError(f"completed canonical raw OOF changed: {arm}/{fold}")
    with np.load(raw, allow_pickle=False) as arrays:
        required = {
            "row_indices", "diagnosis_logits", "melanoma_logits",
            "c10_criterion_logits", "final_criterion_logits", "new_residual_logits",
            "attention_weights", "attention_normalized_entropy", "attention_top_k_mass",
        }
        if not required.issubset(arrays.files) or len(arrays["row_indices"]) != raw_artifact.get("rows"):
            raise AssertionError(f"completed canonical raw OOF schema changed: {arm}/{fold}")
    return receipt


def train_arm_fold(protocol: Mapping[str, Any], arm: str, fold: int, device: Any) -> dict[str, Any]:
    """Train only the new attention residual for one locked seed/fold."""
    _validate_arm(arm)
    protocol_sha = validate_protocol_spec(protocol)
    reused = verified_training_receipt(protocol, arm, fold)
    if reused is not None:
        return {"execution_status": "REUSED_VERIFIED", **reused}
    row = _fold_input(protocol, fold)
    random.seed(SEED + fold)
    np.random.seed(SEED + fold)
    data = _base._derm_data()
    train_indices = np.flatnonzero(data["derm_folds"] != fold)
    effective_batch = 64
    microbatch = 8
    epochs = 20
    steps = math.ceil(len(train_indices) / effective_batch)
    model = _new_model(protocol, arm, fold)
    probe_tokens = torch.from_numpy(np.asarray(data["derm_tokens"][train_indices[:8]], dtype="float32").copy())
    assert_exact_c10_initialization(model, probe_tokens)
    model.to(device)
    optimizer = _optimizer(model, protocol)
    scheduler = _base.screen._scheduler(optimizer, epochs * steps)
    weights = _base._derm_weights(data, fold)
    resumed = _load_resume(
        arm, fold, protocol_sha=protocol_sha,
        retained_sha=row["retained_core"]["sha256"], c10_sha=row["c10_residual"]["sha256"],
        model=model, optimizer=optimizer, scheduler=scheduler, device=device, epochs=epochs,
    )
    start_epoch, history = resumed if resumed is not None else (0, [])
    started = time.perf_counter()
    original_consistency = _base.legacy_training.ladder.CONSISTENCY_WEIGHT
    _base.legacy_training.ladder.CONSISTENCY_WEIGHT = float(protocol["training"]["consistency_weight"])
    try:
        for epoch in range(start_epoch, epochs):
            batches = _base.screen._cyclic_batches(
                train_indices, steps=steps, batch_size=effective_batch,
                rng=np.random.default_rng(SEED + fold * 1000 + epoch * 3 + 2),
            )
            losses: list[float] = []
            components: dict[str, float] = defaultdict(float)
            model.train()
            epoch_started = time.perf_counter()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss, values = _base.legacy_training._microbatched_backward(
                    model, dataset="derm", indices=next(batches), data=data,
                    weights=weights, coefficient=1.0, microbatch_size=microbatch,
                )
                if not np.isfinite(loss):
                    raise FloatingPointError(f"non-finite {arm} loss")
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.new_residual.parameters(),
                    float(protocol.get("training", {}).get("gradient_clip_norm", 1.0)),
                )
                if not bool(torch.isfinite(gradient).detach().cpu()):
                    raise FloatingPointError(f"non-finite {arm} gradient")
                if any(parameter.grad is not None for parameter in model.confirmed_model.parameters()):
                    raise AssertionError("gradient reached frozen c10")
                optimizer.step()
                scheduler.step()
                losses.append(float(loss))
                for name, value in values.items():
                    components[name] += float(value.detach().cpu())
            history.append({
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(losses)),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "component_means": {name: value / steps for name, value in components.items()},
                "elapsed_seconds": time.perf_counter() - epoch_started,
            })
            _write_resume(
                arm, fold, protocol_sha=protocol_sha,
                retained_sha=row["retained_core"]["sha256"], c10_sha=row["c10_residual"]["sha256"],
                completed_epochs=epoch + 1, model=model, optimizer=optimizer,
                scheduler=scheduler, history=history,
            )
            print(f"[attention-factor/{arm}/fold{fold}] epoch {epoch + 1}/{epochs}", flush=True)
    finally:
        _base.legacy_training.ladder.CONSISTENCY_WEIGHT = original_consistency
    model.to("cpu")
    residual_sha = _atomic_torch(_residual_path(arm, fold), model.new_residual.state_dict())
    receipt = {
        "schema_version": TRAINING_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "protocol_sha256": protocol_sha,
        "arm": arm, "seed": SEED, "fold": fold,
        "retained_core_sha256": row["retained_core"]["sha256"],
        "c10_residual_sha256": row["c10_residual"]["sha256"],
        "completed_epochs": epochs,
        "effective_batch_size": effective_batch,
        "microbatch_size": microbatch,
        "steps_per_epoch": steps,
        "training_rows": len(train_indices),
        "new_residual_checkpoint": {"path": _relative(_residual_path(arm, fold)), "sha256": residual_sha},
        "history": history,
        "training_device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "hard_scope_checks": dict(protocol["hard_scope_checks"]),
    }
    digest = _atomic_json(training_receipt_path(arm, fold), receipt)
    return {**receipt, "receipt_sha256": digest}


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temp_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temp_sidecar, sidecar)
    return digest


def reevaluate_arm_fold(protocol: Mapping[str, Any], arm: str, fold: int) -> dict[str, Any]:
    """Regenerate canonical CPU raw OOF and attention evidence."""
    _validate_arm(arm)
    protocol_sha = validate_protocol_spec(protocol)
    reused = verified_canonical_receipt(protocol, arm, fold)
    if reused is not None:
        return {"execution_status": "REUSED_VERIFIED", **reused}
    receipt_path = training_receipt_path(arm, fold)
    _verify_sidecar(receipt_path)
    training = _load_json(receipt_path)
    if (training.get("status"), training.get("protocol_sha256"), training.get("arm"), training.get("fold")) != ("PASS", protocol_sha, arm, fold):
        raise AssertionError("attention-factor training receipt identity changed")
    checkpoint = ROOT / training["new_residual_checkpoint"]["path"]
    if _sha256(checkpoint) != training["new_residual_checkpoint"]["sha256"]:
        raise AssertionError("new residual checkpoint changed")
    model = _new_model(protocol, arm, fold)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.new_residual.load_state_dict(state, strict=True)
    model.eval()
    data = _base._derm_data()
    valid = np.flatnonzero(data["derm_folds"] == fold)
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(valid), 64):
            selection = valid[start:start + 64]
            tokens = torch.from_numpy(np.asarray(data["derm_tokens"][selection], dtype="float32").copy())
            c10 = model.confirmed_model(tokens)
            output = model(tokens)
            for name in ("diagnosis", "melanoma"):
                if not torch.equal(output[name], c10[name]):
                    raise AssertionError(f"protected {name} logits changed")
            evidence = attention_evidence(output["criterion_attention"], top_k=int(protocol["canonical_evaluation"]["top_k"]))
            named = {
                "diagnosis_logits": output["diagnosis"],
                "melanoma_logits": output["melanoma"],
                "c10_criterion_logits": c10["criteria"],
                "final_criterion_logits": output["criteria"],
                "new_residual_logits": output["criterion_residual"],
                "attention_weights": output["criterion_attention"],
            }
            for name, value in named.items():
                arrays[name].append(value.cpu().numpy().astype("float32", copy=False))
            for name, value in evidence.items():
                arrays[name].append(value)
    combined = {name: np.concatenate(values) for name, values in arrays.items()}
    raw_path = _fold_dir(arm, fold) / "canonical_derm7pt_oof_raw.npz"
    raw_sha = _save_npz(raw_path, row_indices=valid.astype("int64"), **combined)
    reloaded = _new_model(protocol, arm, fold)
    reloaded.new_residual.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if not _base._state_dict_exact(state, reloaded.new_residual.state_dict()):
        raise AssertionError("new residual state reload is not exact")
    receipt = {
        "schema_version": CANONICAL_SCHEMA,
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "arm": arm, "seed": SEED, "fold": fold,
        "raw_oof": {"path": _relative(raw_path), "sha256": raw_sha, "rows": len(valid)},
        "protected_logits_equal_c10": True,
        "new_residual_state_reload_exact": True,
        "attention_evidence": {
            "normalized_entropy_shape": list(combined["attention_normalized_entropy"].shape),
            "top_k_mass_shape": list(combined["attention_top_k_mass"].shape),
            "top_k": int(protocol["canonical_evaluation"]["top_k"]),
        },
        "environment": {
            "device": "cpu", "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
        },
        "hard_scope_checks": dict(protocol["hard_scope_checks"]),
    }
    digest = _atomic_json(canonical_receipt_path(arm, fold), receipt)
    return {**receipt, "receipt_sha256": digest}


def load_canonical_raw(protocol: Mapping[str, Any], arm: str) -> dict[int, Mapping[str, np.ndarray]]:
    protocol_sha = validate_protocol_spec(protocol)
    records: dict[int, Mapping[str, np.ndarray]] = {}
    for fold in FOLDS:
        path = canonical_receipt_path(arm, fold)
        _verify_sidecar(path)
        receipt = _load_json(path)
        if (receipt.get("status"), receipt.get("protocol_sha256"), receipt.get("arm"), receipt.get("fold")) != ("PASS", protocol_sha, arm, fold):
            raise AssertionError(f"canonical fold identity changed: {arm}/{fold}")
        raw = ROOT / receipt["raw_oof"]["path"]
        if _verify_sidecar(raw) != receipt["raw_oof"]["sha256"]:
            raise AssertionError(f"canonical raw hash changed: {arm}/{fold}")
        records[fold] = np.load(raw, allow_pickle=False)
    return records


def c10_reference_raw() -> dict[int, Mapping[str, np.ndarray]]:
    records: dict[int, Mapping[str, np.ndarray]] = {}
    for fold in FOLDS:
        receipt_path = _c10.canonical_receipt_path("c10", fold)
        _verify_sidecar(receipt_path)
        receipt = _load_json(receipt_path)
        raw = ROOT / receipt["raw_oof"]["path"]
        if _verify_sidecar(raw) != receipt["raw_oof"]["sha256"]:
            raise AssertionError(f"confirmed c10 canonical raw changed: {fold}")
        records[fold] = np.load(raw, allow_pickle=False)
    return records


def paired_candidate_vs_c10(
    protocol: Mapping[str, Any], arm: str,
    candidate_raw: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    reference_raw: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Paired comparison hook using the existing prespecified bootstrap engine."""
    _validate_arm(arm)
    candidate = load_canonical_raw(protocol, arm) if candidate_raw is None else candidate_raw
    reference = c10_reference_raw() if reference_raw is None else reference_raw
    c10_protocol, _, _ = _c10._candidate_protocol("c10")
    paired, candidate_rows, reference_rows = _c10._paired_candidate_vs_reference(candidate, reference, c10_protocol)
    return {
        "arm": arm,
        "direction": "candidate_minus_c10",
        "paired": paired,
        "candidate_rows": len(candidate_rows),
        "c10_rows": len(reference_rows),
        "row_identity_exact": [row["id"] for row in candidate_rows] == [row["id"] for row in reference_rows],
    }


def paired_arm_vs_arm(
    protocol: Mapping[str, Any], left_arm: str, right_arm: str,
    left_raw: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    right_raw: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Direct paired comparison between the two attention-factor arms."""
    _validate_arm(left_arm)
    _validate_arm(right_arm)
    if left_arm == right_arm:
        raise ValueError("arm-to-arm comparison requires distinct arms")
    left = load_canonical_raw(protocol, left_arm) if left_raw is None else left_raw
    right = load_canonical_raw(protocol, right_arm) if right_raw is None else right_raw
    c10_protocol, _, _ = _c10._candidate_protocol("c10")
    paired, left_rows, right_rows = _c10._paired_candidate_vs_reference(left, right, c10_protocol)
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "direction": f"{left_arm}_minus_{right_arm}",
        "paired": paired,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "row_identity_exact": [row["id"] for row in left_rows] == [row["id"] for row in right_rows],
    }


def rank_safe_arms(
    comparisons: Mapping[str, Mapping[str, Any]],
    arm_to_arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen safe-arm promotion and deterministic ranking rule."""
    if set(comparisons) != set(ARMS):
        raise AssertionError("ranking requires exactly the two closed arms")
    promoted: list[str] = []
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        comparison = comparisons[arm]
        interval = comparison.get("paired", {}).get("criterion_mean_auc", {})
        estimate = float(interval.get("estimate", float("nan")))
        ci_low = float(interval.get("ci_low", float("nan")))
        ci_high = float(interval.get("ci_high", float("nan")))
        safe = comparison.get("safety_decision") == "SAFE"
        promotion = safe and all(math.isfinite(value) for value in (estimate, ci_low, ci_high)) and ci_low > 0.0
        if promotion:
            promoted.append(arm)
        rows.append({
            "arm": arm, "safety_decision": comparison.get("safety_decision"),
            "criterion_mean_auc": {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high},
            "promoted_over_c10": promotion,
        })
    rows.sort(key=lambda row: (-row["criterion_mean_auc"]["estimate"], -row["criterion_mean_auc"]["ci_low"], row["arm"]))
    winner = "c10@20"
    reason = "NO_SAFE_ARM_BEAT_C10_WITH_POSITIVE_95CI"
    if len(promoted) == 1:
        winner, reason = promoted[0], "ONLY_SAFE_PROMOTED_ARM"
    elif len(promoted) == 2:
        if (
            arm_to_arm.get("left_arm"), arm_to_arm.get("right_arm"),
            arm_to_arm.get("row_identity_exact"),
        ) != ("sharp", "multidepth", True):
            raise AssertionError("ranking arm-to-arm identity changed")
        direct = arm_to_arm.get("paired", {}).get("criterion_mean_auc", {})
        low = float(direct.get("ci_low", float("nan")))
        high = float(direct.get("ci_high", float("nan")))
        if not math.isfinite(low) or not math.isfinite(high):
            raise AssertionError("ranking arm-to-arm interval is nonfinite")
        if low > 0.0:
            winner, reason = "sharp", "DECISIVE_ARM_TO_ARM_95CI"
        elif high < 0.0:
            winner, reason = "multidepth", "DECISIVE_ARM_TO_ARM_95CI"
        else:
            winner, reason = next(row["arm"] for row in rows if row["arm"] in promoted), "INCONCLUSIVE_ARM_TO_ARM_C10_ESTIMATE_TIEBREAK"
    return {
        "status": "PASS",
        "winner": winner,
        "decision_reason": reason,
        "ranked_arms": rows,
        "promoted_arms": promoted,
        "arm_to_arm": dict(arm_to_arm),
    }


def controller_matrix() -> list[dict[str, Any]]:
    """Fixed interruption-safe stage order; this function launches nothing."""
    jobs: list[dict[str, Any]] = []
    for arm in ARMS:
        jobs.append({"stage": "probe", "arm": arm, "fold": 0, "seed": SEED, "environment": "venv_dml"})
        jobs.extend({"stage": "train", "arm": arm, "fold": fold, "seed": SEED, "environment": "venv_dml"} for fold in FOLDS)
        jobs.extend({"stage": "reevaluate", "arm": arm, "fold": fold, "seed": SEED, "environment": "venv_p2"} for fold in FOLDS)
        jobs.append({"stage": "compare", "arm": arm, "fold": None, "seed": SEED, "environment": "venv_p2"})
    return jobs


def _read_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    validate_protocol_spec(protocol)
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-protocol-spec", action="store_true")
    mode.add_argument("--validate-protocol", action="store_true")
    mode.add_argument("--train-fold", action="store_true")
    mode.add_argument("--reevaluate-fold", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.print_protocol_spec:
        payload: Any = generate_protocol_spec()
    else:
        if args.protocol is None:
            parser.error("execution/validation requires --protocol PATH")
        protocol = _read_protocol(args.protocol)
        if args.validate_protocol:
            payload = {"status": "PASS", "protocol_sha256": validate_protocol_spec(protocol)}
        else:
            if args.arm is None:
                parser.error("execution requires --arm")
            if args.train_fold or args.reevaluate_fold:
                if args.fold is None:
                    parser.error("fold stage requires --fold")
                payload = (
                    train_arm_fold(protocol, args.arm, args.fold, args.device)
                    if args.train_fold
                    else reevaluate_arm_fold(protocol, args.arm, args.fold)
                )
            else:
                payload = paired_candidate_vs_c10(protocol, args.arm)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS", "FOLDS", "SEED", "AttentionFactorAdapter",
    "generate_protocol_spec", "validate_protocol_spec", "protocol_sha256",
    "assert_exact_c10_initialization", "attention_evidence",
    "probe_sharp_directml", "probe_multidepth_directml",
    "validate_resume_identity", "train_arm_fold", "reevaluate_arm_fold",
    "paired_candidate_vs_c10", "controller_matrix",
]
