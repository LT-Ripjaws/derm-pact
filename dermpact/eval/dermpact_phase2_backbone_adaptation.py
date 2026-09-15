"""Phase 2.3D conservative adaptation of the selected DermLIP--PanDerm tower."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from . import dermpact_phase2_architecture_ladder as ladder
from . import dermpact_phase2_baselines as baselines
from . import dermpact_phase2_bcn_training_features as bcn_features
from . import dermpact_phase2_consistency_repair as consistency
from . import dermpact_phase2_dataset_composition as composition
from . import dermpact_phase2_hybrid_screen as screen
from .derm_encoders import CANDIDATES, _checkpoint_fingerprint, _snapshot
from .dermpact_phase2_baselines import sha256_file


ROOT = screen.ROOT
MODEL_KEY = "panderm"
MODEL_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
PREFIX_BLOCKS = 10
TOTAL_BLOCKS = 12
TOKEN_COUNT = 197
TOKEN_DIM = 768
FEATURE_DIM = 512
ARMS = ("last_block", "lora_last_two")
LORA_RANK = 4
LORA_ALPHA = 8.0
HEAD_LR = 3e-4
LAST_BLOCK_LR = 1e-5
LORA_LR = 1e-4
EPOCHS = 12
N_FOLDS = 5
SEEDS = ladder.SEEDS
DATA_STRATEGIES = ladder.DATA_STRATEGIES
MICRO_BATCH_CANDIDATES = (8, 16, 32, 64)
SMOKE_STEPS = 3
MAX_LOCAL_PROJECTED_HOURS = 18.0
INTERPOLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_3/backbone_adaptation_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_3/backbone_adaptation_v1"
CACHE_DIR = EXPERIMENT_DIR / "prefix_tokens_block10"
SMOKE_PATH = OUTPUT_DIR / "backbone_adaptation_smoke.json"
PROTOCOL_PATH = OUTPUT_DIR / "backbone_adaptation_protocol.json"
RESOURCE_SNAPSHOT = (
    ROOT / "Planning discussions/phase2_audits/resource_snapshot_phase2_3d.json"
)


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _load_hybrid_visual():
    """Load the pinned hybrid visual tower in a fresh process."""
    official_src = ROOT / "research/third_party/Derm1M_official/src"
    if not official_src.is_dir():
        raise FileNotFoundError("official Derm1M source checkout is missing")
    if "open_clip" in sys.modules:
        raise RuntimeError(
            "hybrid visual loader requires a fresh process before open_clip import"
        )
    import packaging
    import pkg_resources

    pkg_resources.packaging = packaging
    sys.path.insert(0, str(official_src))
    import open_clip

    candidate = CANDIDATES[MODEL_KEY]
    if candidate.model_id != MODEL_ID:
        raise AssertionError("selected hybrid model id changed")
    snapshot = _snapshot(candidate.model_id)
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{candidate.model_id}"
    )
    visual = model.visual
    if len(visual.blocks) != TOTAL_BLOCKS:
        raise AssertionError(f"expected {TOTAL_BLOCKS} visual blocks")
    if visual.head.in_features != TOKEN_DIM or visual.head.out_features != FEATURE_DIM:
        raise AssertionError("hybrid visual projection shape changed")
    visual.eval()
    provenance = {
        "model_id": MODEL_ID,
        "implementation": str(Path(open_clip.__file__).resolve()),
        "checkpoint": _checkpoint_fingerprint(snapshot),
        "visual_type": f"{type(visual).__module__}.{type(visual).__name__}",
        "visual_parameters": sum(parameter.numel() for parameter in visual.parameters()),
        "total_blocks": len(visual.blocks),
        "prefix_blocks": PREFIX_BLOCKS,
    }
    del model
    return visual, preprocess, provenance


def _prefix_tokens(visual: nn.Module, images: Tensor) -> Tensor:
    """Run the immutable image stem and first ten transformer blocks."""
    x = visual.patch_embed(images)
    cls_tokens = visual.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    if visual.pos_embed is not None:
        position = visual.pos_embed.expand(x.shape[0], -1, -1).type_as(x).to(x.device)
        x = x + position.clone().detach()
    x = visual.pos_drop(x)
    shared_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None
    for block in visual.blocks[:PREFIX_BLOCKS]:
        x = block(x, rel_pos_bias=shared_bias)
    return x


def _tail_features(
    blocks: Sequence[nn.Module], norm: nn.Module, projection: nn.Module,
    tokens: Tensor, *, shared_bias: Tensor | None = None,
) -> Tensor:
    x = tokens
    for block in blocks:
        x = block(x, rel_pos_bias=shared_bias)
    x = norm(x)
    x = norm(x[:, 1:, :].mean(dim=1))
    x = projection(x)
    return F.normalize(x, dim=-1)


class LoRAWeight(nn.Module):
    """Low-rank parametrization of a frozen linear weight."""

    def __init__(self, out_features: int, in_features: int, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.scale = float(alpha) / float(rank)
        self.adaptation_scale = 1.0

    def forward(self, weight: Tensor) -> Tensor:
        residual = self.lora_b @ self.lora_a
        return weight + residual * self.scale * self.adaptation_scale


def _replace_attention_linear(block: nn.Module, name: str) -> None:
    layer = getattr(block.attn, name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"expected attention {name} to be Linear")
    for parameter in layer.parameters():
        parameter.requires_grad = False
    parametrize.register_parametrization(
        layer,
        "weight",
        LoRAWeight(
            layer.out_features,
            layer.in_features,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
        ),
    )


class AdaptedVisualTail(nn.Module):
    """The last two visual blocks under one conservative adaptation policy."""

    def __init__(self, visual: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown adaptation arm: {arm}")
        self.arm = arm
        self.blocks = copy.deepcopy(visual.blocks[PREFIX_BLOCKS:])
        self.norm = copy.deepcopy(visual.norm)
        self.projection = copy.deepcopy(visual.head)
        self.shared_rel_pos_bias = (
            copy.deepcopy(visual.rel_pos_bias)
            if visual.rel_pos_bias is not None
            else None
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        if arm == "last_block":
            for parameter in self.blocks[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.norm.parameters():
                parameter.requires_grad = True
            for parameter in self.projection.parameters():
                parameter.requires_grad = True
        else:
            for block in self.blocks:
                _replace_attention_linear(block, "qkv")
                _replace_attention_linear(block, "proj")
        self._anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.endswith(("lora_a", "lora_b"))
        }

    def forward(self, tokens: Tensor) -> Tensor:
        shared_bias = (
            self.shared_rel_pos_bias()
            if self.shared_rel_pos_bias is not None
            else None
        )
        return _tail_features(
            self.blocks, self.norm, self.projection, tokens,
            shared_bias=shared_bias,
        )

    def set_adaptation_scale(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("adaptation scale must be in [0, 1]")
        if self.arm == "lora_last_two":
            for module in self.modules():
                if isinstance(module, LoRAWeight):
                    module.adaptation_scale = float(alpha)
            return
        named = dict(self.named_parameters())
        if set(named).issuperset(self._anchor) is False:
            raise AssertionError("visual anchor no longer matches the tail")
        if not hasattr(self, "_adapted"):
            self._adapted = {
                name: named[name].detach().clone() for name in self._anchor
            }
        with torch.no_grad():
            for name, anchor in self._anchor.items():
                named[name].copy_(anchor.lerp(self._adapted[name], float(alpha)))

    def capture_adapted_state(self) -> None:
        if self.arm == "last_block":
            named = dict(self.named_parameters())
            self._adapted = {
                name: named[name].detach().clone() for name in self._anchor
            }

    def adaptation_parameter_names(self) -> list[str]:
        return [name for name, value in self.named_parameters() if value.requires_grad]


class AdaptedHybridModel(nn.Module):
    """Adapted visual tail followed by the selected multitask head."""

    def __init__(self, visual: nn.Module, arm: str) -> None:
        super().__init__()
        self.task_head = screen.HybridSharedHead(feature_dim=FEATURE_DIM)
        self.visual_tail = AdaptedVisualTail(visual, arm)

    def forward(self, prefix_tokens: Tensor) -> dict[str, Tensor]:
        return self.task_head(self.visual_tail(prefix_tokens))

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual_tail.eval()
        self.task_head.train(mode)
        return self

    def optimizer_groups(self) -> list[dict[str, Any]]:
        visual_lr = LAST_BLOCK_LR if self.visual_tail.arm == "last_block" else LORA_LR
        return [
            {"params": list(self.task_head.parameters()), "lr": HEAD_LR},
            {
                "params": [
                    value for value in self.visual_tail.parameters()
                    if value.requires_grad
                ],
                "lr": visual_lr,
            },
        ]


def _dataset_rows(dataset: str) -> list[dict[str, Any]]:
    if dataset == "ham":
        return baselines._ham_records()
    if dataset == "derm7pt":
        return baselines._derm7pt_records()
    if dataset == "bcn_train":
        rows, _ = bcn_features._source_and_training_rows()
        return rows
    raise ValueError(dataset)


def _image_path(row: Mapping[str, Any], dataset: str) -> Path:
    if dataset in {"ham", "derm7pt"}:
        return baselines._image_path(row, dataset)
    return Path(str(row["image_path"]))


def _row_identity(row: Mapping[str, Any], dataset: str) -> dict[str, str]:
    if dataset == "ham":
        key = str(row["case_id"])
        digest = str(row["image_sha256"])
    elif dataset == "derm7pt":
        key = str(row["group_id"])
        digest = str(row["modalities"]["dermoscopic"]["sha256"])
    else:
        key = str(row["case_id"])
        digest = str(row["image_sha256"])
    return {"id": key, "image_sha256": digest}


def _cache_paths(dataset: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{dataset}_tokens.npy", CACHE_DIR / f"{dataset}_manifest.json"


def _full_visual_features(visual: nn.Module, images: Tensor) -> Tensor:
    return F.normalize(visual(images), dim=-1)


def _base_tail_features(visual: nn.Module, tokens: Tensor) -> Tensor:
    shared_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None
    return _tail_features(
        visual.blocks[PREFIX_BLOCKS:], visual.norm, visual.head, tokens,
        shared_bias=shared_bias,
    )


def _tensor_mb(value: Tensor) -> float:
    return value.numel() * value.element_size() / (1024.0 ** 2)


def run_smoke(*, batch_sizes: Sequence[int] = MICRO_BATCH_CANDIDATES) -> dict[str, Any]:
    """Validate equivalence and measure cached-tail training on real HAM images."""
    from PIL import Image

    torch.set_num_threads(ladder.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    rows = _dataset_rows("ham")[: max(batch_sizes)]
    visual, preprocess, provenance = _load_hybrid_visual()
    images = torch.stack([
        preprocess(Image.open(_image_path(row, "ham")).convert("RGB"))
        for row in rows
    ])
    with torch.inference_mode():
        full = _full_visual_features(visual, images[:4])
        prefix = _prefix_tokens(visual, images[:4])
        reconstructed = _base_tail_features(visual, prefix)
    equivalence = float(torch.max(torch.abs(full - reconstructed)))
    if equivalence > 1e-6:
        raise AssertionError(f"cached boundary changed hybrid features by {equivalence}")

    maximum = max(batch_sizes)
    with torch.inference_mode():
        prefix_all = _prefix_tokens(visual, images[:maximum]).detach()
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_result: dict[str, Any] = {
            "trainable_parameters": None,
            "adaptation_parameter_names": None,
            "batch_sizes": {},
        }
        for batch_size in batch_sizes:
            torch.manual_seed(20260827)
            model = AdaptedHybridModel(visual, arm)
            optimizer = torch.optim.AdamW(
                model.optimizer_groups(), weight_decay=screen.WEIGHT_DECAY
            )
            tokens = prefix_all[:batch_size].clone()
            diagnosis = torch.arange(batch_size, dtype=torch.long) % len(screen.GLOBAL_CLASSES)
            melanoma = (torch.arange(batch_size) % 2).float()
            criteria = torch.stack(
                [((torch.arange(batch_size) + offset) % 2).float()
                 for offset in range(len(baselines.CRITERIA))], dim=1,
            )
            durations = []
            finite = True
            for _ in range(SMOKE_STEPS):
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                outputs = model(tokens)
                loss, _ = screen.multitask_loss(
                    outputs,
                    dataset="derm7pt",
                    diagnosis=diagnosis,
                    melanoma=melanoma,
                    diagnosis_weight=torch.ones(batch_size),
                    melanoma_weight=torch.ones(batch_size),
                    loss_weighting="equal_family",
                    criteria=criteria,
                    criterion_weights=torch.ones_like(criteria),
                )
                consistency_loss, _ = ladder.truth_anchored_consistency_loss(
                    outputs,
                    melanoma=melanoma,
                    criteria=criteria,
                    diagnosis=diagnosis,
                )
                loss = loss + ladder.CONSISTENCY_WEIGHT * consistency_loss
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), screen.GRADIENT_CLIP_NORM
                )
                optimizer.step()
                durations.append(time.perf_counter() - started)
                finite = finite and bool(torch.isfinite(loss) and torch.isfinite(norm))
            trainable = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            )
            arm_result["trainable_parameters"] = trainable
            arm_result["adaptation_parameter_names"] = (
                model.visual_tail.adaptation_parameter_names()
            )
            arm_result["batch_sizes"][str(batch_size)] = {
                "status": "PASS" if finite else "FAIL",
                "median_step_seconds": float(np.median(durations[1:])),
                "all_step_seconds": durations,
                "prefix_tensor_mb": _tensor_mb(tokens),
                "final_loss": float(loss.detach()),
                "gradient_norm": float(norm),
            }
        arms[arm] = arm_result
    steps_per_fold = 88 * EPOCHS
    slowest_selected = max(
        arms[arm]["batch_sizes"][str(maximum)]["median_step_seconds"]
        for arm in ARMS
    )
    projected_hours = slowest_selected * steps_per_fold * (2 + 3) * 15 / 3600.0
    execution = "local" if projected_hours <= MAX_LOCAL_PROJECTED_HOURS else "colab_gpu"
    payload = {
        "protocol": "dermpact-phase2-3d-backbone-smoke-v1",
        "status": "PASS",
        "scope": "runtime and mechanical validation only; no endpoint calculated",
        "resource_snapshot": {
            "path": str(RESOURCE_SNAPSHOT.relative_to(ROOT)),
            "sha256": sha256_file(RESOURCE_SNAPSHOT),
        },
        "model": provenance,
        "boundary": {
            "prefix_blocks": PREFIX_BLOCKS,
            "tail_blocks": TOTAL_BLOCKS - PREFIX_BLOCKS,
            "full_vs_reconstructed_max_abs_difference": equivalence,
            "cache_dtype": "float32",
        },
        "arms": arms,
        "runtime_gate": {
            "projected_full_cpu_hours_conservative": projected_hours,
            "local_limit_hours": MAX_LOCAL_PROJECTED_HOURS,
            "execution_decision": execution,
            "note": "projection excludes one-time prefix-cache extraction",
        },
        "protected_or_external_rows_loaded": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    digest = screen._write_json(SMOKE_PATH, payload)
    return {**payload, "receipt_sha256": digest}


def build_protocol() -> dict[str, Any]:
    if not SMOKE_PATH.is_file():
        raise FileNotFoundError("run the mechanical/runtime smoke before freezing protocol")
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS":
        raise AssertionError("backbone adaptation smoke did not pass")
    references = {}
    for strategy in DATA_STRATEGIES:
        for seed in SEEDS:
            path = consistency._result_path(strategy, seed)
            references[f"{strategy}_seed_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
    implementation_path = Path(__file__).resolve()
    training_path = ROOT / "dermpact/eval/dermpact_phase2_backbone_training.py"
    notebook_path = ROOT / "research/notebooks/DermPact_Phase2_3D_Backbone_Adaptation.ipynb"
    training_smoke_path = OUTPUT_DIR / "backbone_training_path_smoke.json"
    if not training_smoke_path.is_file():
        raise FileNotFoundError("run the real-data training-path smoke before freezing protocol")
    training_smoke = json.loads(training_smoke_path.read_text(encoding="utf-8"))
    if training_smoke.get("status") != "PASS":
        raise AssertionError("real-data training-path smoke did not pass")
    cache_artifacts = {}
    for dataset in ("ham", "derm7pt", "bcn_train"):
        token_path, manifest_path = _cache_paths(dataset)
        if not token_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"derive the verified {dataset} prefix cache before freezing the final protocol"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["token_file_sha256"] != sha256_file(token_path):
            raise AssertionError(f"{dataset} prefix-cache hash mismatch")
        cache_artifacts[dataset] = {
            "token_path": str(token_path.relative_to(ROOT)),
            "token_sha256": manifest["token_file_sha256"],
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
        }
    protocol = {
        "protocol": "dermpact-phase2-3d-backbone-adaptation-v1",
        "status": "PRESPECIFIED_AFTER_RUNTIME_SMOKE_BEFORE_ADAPTATION_RESULTS",
        "question": (
            "Does conservative adaptation of the selected hybrid vision tail improve "
            "the Derm7pt eight-output endpoint without a material protected-task regression?"
        ),
        "fixed_factors": {
            "model_id": MODEL_ID,
            "input": "one dermoscopic image",
            "task_head": "shared adapter, diagnosis/melanoma heads and seven independent criterion heads",
            "loss": "equal-family masked multitask loss plus truth-anchored consistency weight 0.10",
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "seeds": list(SEEDS),
            "head_learning_rate": HEAD_LR,
            "preprocessing": (
                "pinned deterministic hybrid evaluation transform; no random augmentation, "
                "so adaptation depth is the isolated changed factor"
            ),
            "dataset_strategies": list(DATA_STRATEGIES),
        },
        "adaptation_arms": {
            "frozen_reference": "completed Phase 2.3A consistency-selected model",
            "last_block": {
                "trainable": "visual block 12, final visual norm and 768-to-512 projection",
                "learning_rate": LAST_BLOCK_LR,
            },
            "lora_last_two": {
                "trainable": "rank-4 LoRA on attention qkv and output projection in visual blocks 11--12",
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "learning_rate": LORA_LR,
            },
        },
        "frozen_prefix_cache": {
            "completed_blocks": PREFIX_BLOCKS,
            "dtype": "float32",
            "equivalence_gate": "maximum absolute feature difference <= 1e-6",
            "reason": "blocks 1--10 are frozen and deterministic",
        },
        "interpolation": {
            "alphas": list(INTERPOLATION_ALPHAS),
            "meaning": (
                "scale only the learned visual delta while retaining the jointly trained task head; "
                "this is a backbone-delta interpolation, not a claim of exact WiSE-FT replication"
            ),
            "alpha_zero_role": "matched-head frozen-vision control, not the historical frozen reference",
        },
        "selection": {
            "primary": "Derm7pt mean AUROC over melanoma and seven criteria",
            "protected": [
                "Derm7pt five-class balanced accuracy",
                "Derm7pt melanoma AUROC",
                "HAM seven-class balanced accuracy",
                "HAM melanoma AUROC",
                "direct-rule disagreement",
            ],
            "promotion": (
                "positive primary delta versus the frozen Phase 2.3A reference in at least two of "
                "three seeds, no protected deficit larger than 0.02 with its paired 95% interval "
                "below zero, and no direct-rule-disagreement increase larger than 0.02 with its "
                "paired interval above zero"
            ),
            "multiple_alpha_interpretation": (
                "development model selection only; final claims use the later single frozen test read"
            ),
        },
        "compute_gate": smoke["runtime_gate"],
        "canonical_evaluation": {
            "environment": "venv_p2 CPU",
            "python": "3.12.13",
            "numpy": "2.4.4",
            "torch": "2.12.1+cpu",
            "rule": (
                "GPU-trained checkpoints are not eligible for comparison until every OOF logit "
                "has been regenerated by the explicit local --reevaluate command"
            ),
        },
        "hard_scope": {
            "allowed": ["HAM training", "Derm7pt 616-case development", "BCN training"],
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
        "artifacts": {
            "smoke": {
                "path": str(SMOKE_PATH.relative_to(ROOT)),
                "sha256": sha256_file(SMOKE_PATH),
            },
            "training_path_smoke": {
                "path": str(training_smoke_path.relative_to(ROOT)),
                "sha256": sha256_file(training_smoke_path),
            },
            "resource_snapshot": {
                "path": str(RESOURCE_SNAPSHOT.relative_to(ROOT)),
                "sha256": sha256_file(RESOURCE_SNAPSHOT),
            },
            "requirements_lock": {
                "path": str(screen.LOCK_FILE.relative_to(ROOT)),
                "sha256": sha256_file(screen.LOCK_FILE),
            },
            "implementation": {
                "core_path": str(implementation_path.relative_to(ROOT)),
                "core_sha256": sha256_file(implementation_path),
                "training_path": str(training_path.relative_to(ROOT)),
                "training_sha256": sha256_file(training_path),
                "notebook_path": str(notebook_path.relative_to(ROOT)),
                "notebook_sha256": sha256_file(notebook_path),
            },
            "prefix_caches": cache_artifacts,
            "frozen_references": references,
        },
    }
    digest = screen._write_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": digest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        payload = run_smoke()
    elif args.protocol_only:
        payload = build_protocol()
    else:
        parser.error("choose --smoke or --protocol-only")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
