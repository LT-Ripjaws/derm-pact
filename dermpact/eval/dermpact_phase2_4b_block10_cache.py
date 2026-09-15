"""Auditable Phase 2.4B clinical block-10 prefix-cache extraction."""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
from enum import Enum
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import dermpact_phase2_4b_inputs as phase24b_inputs


ROOT = phase24b_inputs.ROOT
DEFAULT_DERM_TOKENS = (
    ROOT
    / "research/experiments/dermpact_phase2_3/backbone_adaptation_v1"
    / "prefix_tokens_block10/derm7pt_tokens.npy"
)
DEFAULT_DERM_MANIFEST = DEFAULT_DERM_TOKENS.with_name("derm7pt_manifest.json")
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "research/experiments/dermpact_phase2_4/clinical_block10_prefix_cache_v1"
)
FORMAT_VERSION = "dermpact-phase2-4b-clinical-block10-cache-v1"
PRODUCTION_TOKEN_COUNT = 197
PRODUCTION_HIDDEN_SIZE = 768


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def _atomic_write_bytes(path: Path, value: bytes, *, with_sidecar: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = phase24b_inputs.sha256_file(path)
    if with_sidecar:
        _atomic_write_bytes(
            _sidecar(path), (digest + "\n").encode("ascii"), with_sidecar=False
        )
    return digest


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _atomic_write_bytes(path, body)


def _atomic_write_npy(path: Path, value: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = phase24b_inputs.sha256_file(path)
    _atomic_write_bytes(
        _sidecar(path), (digest + "\n").encode("ascii"), with_sidecar=False
    )
    return digest


def _verify_sha_sidecar(path: Path) -> str:
    path = Path(path)
    digest = phase24b_inputs.sha256_file(path)
    sidecar = _sidecar(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing SHA-256 sidecar: {sidecar}")
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise AssertionError(f"SHA-256 sidecar mismatch: {path}")
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class CachePaths:
    """All mutable Phase 2.4B cache paths under one output directory."""

    output_dir: Path
    clinical_tokens: Path
    clinical_available: Path
    row_identity: Path
    manifest: Path
    partial_tokens: Path
    progress: Path
    receipts_dir: Path

    @classmethod
    def under(cls, output_dir: Path) -> "CachePaths":
        output_dir = Path(output_dir)
        return cls(
            output_dir=output_dir,
            clinical_tokens=output_dir / "clinical_block10_prefix_tokens.npy",
            clinical_available=output_dir / "clinical_available.npy",
            row_identity=output_dir / "row_identity.json",
            manifest=output_dir / "cache_manifest.json",
            partial_tokens=output_dir / "clinical_block10_prefix_tokens.partial.npy",
            progress=output_dir / "extraction_progress.json",
            receipts_dir=output_dir / "chunk_receipts",
        )


@dataclass(frozen=True)
class PrefixEncoder:
    """Pinned encoder operations plus identities needed for cache evidence."""

    config: phase24b_inputs.Block10CacheConfig
    provenance: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    preprocess_identity: Mapping[str, Any]
    encode_prefix: Callable[[Sequence[Path]], np.ndarray]
    direct_features: Callable[[Sequence[Path]], np.ndarray]
    reconstruct_features: Callable[[np.ndarray], np.ndarray]


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _stable_preprocess_value(value: object) -> Any:
    """Serialize transform configuration without process-specific identity."""
    if isinstance(value, Enum):
        return {
            "enum_type": _qualified_type(value),
            "name": value.name,
            "value": _stable_preprocess_value(value.value),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"path": value.as_posix()}
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [str(key), _stable_preprocess_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if isinstance(value, tuple):
        return {"tuple": [_stable_preprocess_value(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_stable_preprocess_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        members = [_stable_preprocess_value(item) for item in value]
        members.sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        return {"set": members}
    if isinstance(value, functools.partial):
        return {
            "partial": _stable_preprocess_value(value.func),
            "args": _stable_preprocess_value(value.args),
            "keywords": _stable_preprocess_value(value.keywords or {}),
        }
    if inspect.isroutine(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
        if not module or not qualname:
            raise TypeError(f"callable lacks stable module/qualname: {_qualified_type(value)}")
        return {"callable": {"module": module, "qualname": qualname}}
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise TypeError(
            f"unsupported preprocessing configuration value: {_qualified_type(value)}"
        ) from exc
    public = {
        str(name): _stable_preprocess_value(item)
        for name, item in sorted(attributes.items())
        if not str(name).startswith("_")
    }
    return {"type": _qualified_type(value), "parameters": public}


def _preprocess_identity(preprocess: object) -> dict[str, Any]:
    """Deterministic structural identity for the retained transform pipeline."""
    identity = {
        "schema": "structural-preprocess-identity-v1",
        "pipeline": _stable_preprocess_value(preprocess),
    }
    identity["sha256"] = phase24b_inputs.canonical_hash(identity)
    return identity


def load_pinned_hybrid_prefix_encoder() -> PrefixEncoder:
    """Load the selected DermLIP--PanDerm tower and retained preprocessing."""
    import torch
    from PIL import Image

    from . import dermpact_phase2_backbone_adaptation as adaptation

    visual, preprocess, provenance = adaptation._load_hybrid_visual()
    if adaptation.PREFIX_BLOCKS != phase24b_inputs.SHARED_PREFIX_BLOCKS:
        raise AssertionError("shared prefix boundary changed")
    preprocess_identity = _preprocess_identity(preprocess)
    checkpoint = provenance.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping):
        raise ValueError("hybrid loader omitted checkpoint provenance")
    revision = str(checkpoint.get("snapshot_revision") or "")
    checkpoint_sha = str(checkpoint.get("combined_sha256") or "")
    config = phase24b_inputs.Block10CacheConfig(
        model_id=str(provenance.get("model_id") or ""),
        model_revision=revision,
        checkpoint_sha256=checkpoint_sha,
        preprocessing_sha256=str(preprocess_identity["sha256"]),
        hidden_size=adaptation.TOKEN_DIM,
        tokens_per_image=adaptation.TOKEN_COUNT,
    )

    def _images(paths: Sequence[Path]):
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        return torch.stack(tensors)

    def encode_prefix(paths: Sequence[Path]) -> np.ndarray:
        with torch.inference_mode():
            values = adaptation._prefix_tokens(visual, _images(paths))
        return values.cpu().numpy().astype("float32", copy=False)

    def direct_features(paths: Sequence[Path]) -> np.ndarray:
        with torch.inference_mode():
            values = adaptation._full_visual_features(visual, _images(paths))
        return values.cpu().numpy().astype("float32", copy=False)

    def reconstruct_features(tokens: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(tokens, dtype="float32"))
        with torch.inference_mode():
            values = adaptation._base_tail_features(visual, tensor)
        return values.cpu().numpy().astype("float32", copy=False)

    implementation = Path(str(provenance["implementation"])).resolve()
    source_files = {
        "phase2_4b_cache": Path(__file__).resolve(),
        "phase2_4b_inputs": Path(phase24b_inputs.__file__).resolve(),
        "prefix_implementation": Path(adaptation.__file__).resolve(),
        "pinned_open_clip_entrypoint": implementation,
    }
    source_hashes = {
        name: phase24b_inputs.sha256_file(path) for name, path in source_files.items()
    }
    return PrefixEncoder(
        config=config,
        provenance=dict(provenance),
        source_hashes=source_hashes,
        preprocess_identity=preprocess_identity,
        encode_prefix=encode_prefix,
        direct_features=direct_features,
        reconstruct_features=reconstruct_features,
    )


def _derm_row_identity(
    inputs: phase24b_inputs.Phase24BInputSet,
) -> list[dict[str, str]]:
    # This exact schema is the one signed by the Phase 2.3D cache manifest.
    return [
        {"id": case.group_id, "image_sha256": case.dermoscopic_sha256}
        for case in inputs.cases
    ]


def bind_existing_dermoscopic_cache(
    inputs: phase24b_inputs.Phase24BInputSet,
    config: phase24b_inputs.Block10CacheConfig,
    *,
    token_path: Path = DEFAULT_DERM_TOKENS,
    manifest_path: Path = DEFAULT_DERM_MANIFEST,
) -> dict[str, Any]:
    """Verify and bind the existing dermoscopic cache without rewriting it."""
    token_path = Path(token_path)
    manifest_path = Path(manifest_path)
    manifest_sha = _verify_sha_sidecar(manifest_path)
    manifest = _read_json(manifest_path)
    expected_identity = phase24b_inputs.canonical_hash(_derm_row_identity(inputs))
    if manifest.get("record_identity_sha256") != expected_identity:
        raise AssertionError("existing dermoscopic cache row identity changed")
    expected_shape = [len(inputs.cases), config.tokens_per_image, config.hidden_size]
    if manifest.get("shape") != expected_shape or manifest.get("dtype") != "float32":
        raise AssertionError("existing dermoscopic cache tensor contract changed")
    if manifest.get("completed_visual_blocks") != config.shared_prefix_blocks:
        raise AssertionError("existing dermoscopic cache boundary changed")
    model = manifest.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("existing dermoscopic cache model provenance is invalid")
    checkpoint = model.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping):
        raise ValueError("existing dermoscopic cache checkpoint provenance is invalid")
    if model.get("model_id") != config.model_id:
        raise AssertionError("existing dermoscopic cache model changed")
    if checkpoint.get("snapshot_revision") != config.model_revision:
        raise AssertionError("existing dermoscopic cache model revision changed")
    if checkpoint.get("combined_sha256") != config.checkpoint_sha256:
        raise AssertionError("existing dermoscopic cache checkpoint changed")
    token_sha = phase24b_inputs.sha256_file(token_path)
    if manifest.get("token_file_sha256") != token_sha:
        raise AssertionError("existing dermoscopic cache token hash changed")
    values = np.load(token_path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != expected_shape or values.dtype != np.float32:
        raise AssertionError("existing dermoscopic cache array changed")
    return {
        "reuse_policy": "hash-bound existing Phase 2.3D artifact; never recompute",
        "token_path": str(token_path.resolve()),
        "token_sha256": token_sha,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "record_identity_sha256": expected_identity,
        "shape": expected_shape,
        "dtype": "float32",
    }


def verify_frozen_specification(
    path: Path,
    inputs: phase24b_inputs.Phase24BInputSet,
    config: phase24b_inputs.Block10CacheConfig,
) -> tuple[dict[str, Any], str]:
    """Require the exact Phase 2.4B specification and its SHA sidecar."""
    digest = _verify_sha_sidecar(path)
    observed = _read_json(path)
    expected = phase24b_inputs.build_block10_cache_manifest_spec(inputs, config)
    if observed != expected:
        raise AssertionError("frozen cache specification does not match current inputs/model")
    return observed, digest


def _row_identity(inputs: phase24b_inputs.Phase24BInputSet) -> list[dict[str, Any]]:
    return [
        {
            "row": index,
            "case_index": case.case_index,
            "case_num": case.case_num,
            "group_id": case.group_id,
            "fold": case.fold,
            "source_split": case.source_split,
            "clinical_available": case.clinical_available,
            "clinical_sha256": case.clinical_sha256,
        }
        for index, case in enumerate(inputs.cases)
    ]


def _extraction_contract(
    inputs: phase24b_inputs.Phase24BInputSet,
    encoder: PrefixEncoder,
    derm_binding: Mapping[str, Any],
    frozen_spec_sha256: str,
    parity_tolerance: float,
) -> dict[str, Any]:
    rows = _row_identity(inputs)
    clinical_paths = [
        {
            "group_id": case.group_id,
            "relative_path": case.clinical_relative_path,
            "sha256": case.clinical_sha256,
            "available": case.clinical_available,
        }
        for case in inputs.cases
    ]
    folds = [{"group_id": case.group_id, "fold": case.fold} for case in inputs.cases]
    return {
        "format": FORMAT_VERSION,
        "frozen_specification_sha256": frozen_spec_sha256,
        "shape": [
            len(inputs.cases),
            encoder.config.tokens_per_image,
            encoder.config.hidden_size,
        ],
        "dtype": "float32",
        "canonical_parity_tolerance": parity_tolerance,
        "encoder_config": asdict(encoder.config),
        "source_hashes": dict(encoder.source_hashes),
        "source_artifact_hashes": {
            "inventory": inputs.inventory_sha256,
            "fold_manifest": inputs.fold_manifest_sha256,
            "clinical_photo_audit": inputs.clinical_photo_audit_sha256,
            "structured_fact_support": inputs.structured_fact_support_sha256,
        },
        "row_identity_sha256": phase24b_inputs.canonical_hash(rows),
        "clinical_path_identity_sha256": phase24b_inputs.canonical_hash(clinical_paths),
        "fold_identity_sha256": phase24b_inputs.canonical_hash(folds),
        "availability_sha256": phase24b_inputs.canonical_hash(
            [int(case.clinical_available) for case in inputs.cases]
        ),
        "dermoscopic_binding": dict(derm_binding),
    }


def _receipt_path(paths: CachePaths, start: int, stop: int) -> Path:
    return paths.receipts_dir / f"chunk_{start:06d}_{stop:06d}.json"


def _load_and_verify_progress(paths: CachePaths, identity: str) -> dict[str, Any]:
    progress = _read_json(paths.progress)
    _verify_sha_sidecar(paths.progress)
    if progress.get("extraction_identity_sha256") != identity:
        raise AssertionError("partial cache belongs to a different extraction contract")
    return progress


def _verify_receipts(
    paths: CachePaths,
    values: np.ndarray,
    available_positions: Sequence[int],
    progress: Mapping[str, Any],
    identity: str,
) -> int:
    expected_offset = 0
    receipt_names = progress.get("receipts", [])
    if not isinstance(receipt_names, list):
        raise ValueError("invalid receipt list")
    for name in receipt_names:
        receipt_path = paths.receipts_dir / str(name)
        _verify_sha_sidecar(receipt_path)
        receipt = _read_json(receipt_path)
        if receipt.get("extraction_identity_sha256") != identity:
            raise AssertionError("chunk receipt extraction identity changed")
        start = int(receipt["available_offset_start"])
        stop = int(receipt["available_offset_stop"])
        if start != expected_offset or stop <= start:
            raise AssertionError("chunk receipts are not a contiguous prefix")
        positions = list(map(int, receipt.get("row_positions", [])))
        if positions != list(available_positions[start:stop]):
            raise AssertionError("chunk receipt row positions changed")
        if receipt.get("tensor_sha256") != _array_sha256(values[positions]):
            raise AssertionError("partial cache chunk hash changed")
        expected_offset = stop
    if int(progress.get("next_available_offset", -1)) != expected_offset:
        raise AssertionError("progress pointer does not match verified receipts")
    return expected_offset


def _validate_complete_artifacts(
    paths: CachePaths,
    expected_identity: str,
    *,
    inputs: phase24b_inputs.Phase24BInputSet,
    encoder: PrefixEncoder,
    derm_binding: Mapping[str, Any],
    parity_tolerance: float,
    require_production_counts: bool,
) -> dict[str, Any]:
    """Semantically validate a terminal cache before permitting reuse."""
    manifest = _read_json(paths.manifest)
    _verify_sha_sidecar(paths.manifest)
    if manifest.get("status") != "PASS":
        raise AssertionError("completed cache manifest is not PASS")
    if manifest.get("extraction_identity_sha256") != expected_identity:
        raise AssertionError("completed cache belongs to another extraction contract")
    for name, path in {
        "clinical_tokens": paths.clinical_tokens,
        "clinical_available": paths.clinical_available,
        "row_identity": paths.row_identity,
    }.items():
        digest = _verify_sha_sidecar(path)
        if manifest.get("artifacts", {}).get(name, {}).get("sha256") != digest:
            raise AssertionError(f"completed {name} hash changed")

    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("completed cache lacks an extraction contract")
    expected_shape = (
        len(inputs.cases),
        encoder.config.tokens_per_image,
        encoder.config.hidden_size,
    )
    if contract.get("shape") != list(expected_shape) or contract.get("dtype") != "float32":
        raise AssertionError("completed cache contract shape or dtype changed")
    if contract.get("canonical_parity_tolerance") != parity_tolerance:
        raise AssertionError("completed cache parity tolerance changed")

    values = np.load(paths.clinical_tokens, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape or values.dtype != np.float32:
        raise AssertionError("completed clinical token tensor contract changed")
    if not np.isfinite(values).all():
        raise FloatingPointError("completed clinical token tensor is non-finite")

    availability = np.load(paths.clinical_available, allow_pickle=False)
    expected_availability = np.asarray(
        [case.clinical_available for case in inputs.cases], dtype=bool
    )
    if availability.shape != (len(inputs.cases),) or availability.dtype != np.bool_:
        raise AssertionError("completed availability mask contract changed")
    if not np.array_equal(availability, expected_availability):
        raise AssertionError("completed availability mask content changed")
    n_available = int(availability.sum())
    n_unavailable = int((~availability).sum())
    if np.any(np.asarray(values[~availability]) != 0.0):
        raise AssertionError("completed unavailable clinical rows are not exact zeros")

    row_payload = _read_json(paths.row_identity)
    rows = row_payload.get("rows")
    expected_rows = _row_identity(inputs)
    if not isinstance(rows, list) or len(rows) != len(inputs.cases):
        raise AssertionError("completed row-identity count changed")
    if rows != expected_rows:
        raise AssertionError("completed row-identity order or content changed")
    row_hash = phase24b_inputs.canonical_hash(rows)
    if row_hash != contract.get("row_identity_sha256"):
        raise AssertionError("completed row-identity contract hash changed")
    if phase24b_inputs.canonical_hash(availability.astype(int).tolist()) != contract.get(
        "availability_sha256"
    ):
        raise AssertionError("completed availability contract hash changed")

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("completed cache lacks count evidence")
    if (
        counts.get("aligned_rows") != len(inputs.cases)
        or counts.get("unique_clinical_photos_encoded") != n_available
        or counts.get("unavailable_zero_rows") != n_unavailable
    ):
        raise AssertionError("completed cache counts changed")
    if require_production_counts and (
        len(inputs.cases) != phase24b_inputs.N_CASES
        or n_available != phase24b_inputs.N_AVAILABLE_CLINICAL
        or n_unavailable != phase24b_inputs.N_UNAVAILABLE_CLINICAL
        or expected_shape
        != (
            phase24b_inputs.N_CASES,
            PRODUCTION_TOKEN_COUNT,
            PRODUCTION_HIDDEN_SIZE,
        )
    ):
        raise AssertionError("completed production cache is not the required 616/609/7 contract")

    parity = manifest.get("canonical_parity")
    if not isinstance(parity, Mapping) or parity.get("status") != "PASS":
        raise AssertionError("completed cache canonical parity is not PASS")
    if parity.get("tolerance") != parity_tolerance:
        raise AssertionError("completed cache canonical parity tolerance changed")
    maximum_difference = parity.get("maximum_abs_difference")
    if not isinstance(maximum_difference, (int, float)) or not np.isfinite(
        maximum_difference
    ):
        raise AssertionError("completed cache canonical parity result is invalid")
    if float(maximum_difference) > parity_tolerance:
        raise AssertionError("completed cache canonical parity exceeds tolerance")
    samples = parity.get("sample_rows")
    sample_groups = parity.get("sample_group_ids")
    if not isinstance(samples, list) or not samples:
        raise AssertionError("completed cache canonical parity samples are absent")
    if any(not isinstance(row, int) or row < 0 or row >= len(inputs.cases) for row in samples):
        raise AssertionError("completed cache canonical parity sample row is invalid")
    if sample_groups != [inputs.cases[row].group_id for row in samples]:
        raise AssertionError("completed cache canonical parity sample identity changed")
    available_positions = np.flatnonzero(availability).astype(int).tolist()
    expected_sample_offsets = sorted(
        {0, len(available_positions) // 2, len(available_positions) - 1}
    )
    expected_samples = [available_positions[offset] for offset in expected_sample_offsets]
    if samples != expected_samples:
        raise AssertionError("completed cache canonical parity sample selection changed")
    recorded_per_sample = parity.get("per_sample_max_abs_difference")
    if (
        not isinstance(recorded_per_sample, list)
        or len(recorded_per_sample) != len(samples)
        or any(
            not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in recorded_per_sample
        )
    ):
        raise AssertionError("completed cache canonical parity detail is invalid")
    if abs(max(map(float, recorded_per_sample)) - float(maximum_difference)) > 1e-12:
        raise AssertionError("completed cache canonical parity summary changed")
    sample_paths = [Path(inputs.cases[row].clinical_path) for row in samples]  # type: ignore[arg-type]
    direct = np.asarray(encoder.direct_features(sample_paths), dtype="float32")
    reconstructed = np.asarray(
        encoder.reconstruct_features(np.asarray(values[samples], dtype="float32")),
        dtype="float32",
    )
    if direct.shape != reconstructed.shape or direct.ndim != 2:
        raise AssertionError("completed cache parity revalidation shapes differ")
    if not np.isfinite(direct).all() or not np.isfinite(reconstructed).all():
        raise FloatingPointError("completed cache parity revalidation is non-finite")
    revalidated_per_sample = np.max(np.abs(direct - reconstructed), axis=1)
    revalidated_maximum = float(revalidated_per_sample.max(initial=0.0))
    if revalidated_maximum > parity_tolerance:
        raise AssertionError("completed cache fails canonical parity revalidation")
    if not np.allclose(
        revalidated_per_sample,
        np.asarray(recorded_per_sample, dtype="float32"),
        atol=1e-7,
        rtol=0.0,
    ):
        raise AssertionError("completed cache recorded parity no longer reproduces")
    if counts.get("canonical_direct_parity_samples") != len(samples):
        raise AssertionError("completed cache canonical parity sample count changed")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("completed cache lacks artifact bindings")
    bound_derm = artifacts.get("dermoscopic_tokens")
    if not isinstance(bound_derm, Mapping) or dict(bound_derm) != dict(derm_binding):
        raise AssertionError("completed cache dermoscopic binding changed or is absent")
    if contract.get("dermoscopic_binding") != dict(derm_binding):
        raise AssertionError("completed cache contract dermoscopic binding changed")

    hard_scope = manifest.get("hard_scope")
    if not isinstance(hard_scope, Mapping):
        raise ValueError("completed cache lacks hard-scope evidence")
    zero_counters = (
        "derm7pt_official_test_rows_loaded",
        "ham_calibration_or_final_rows_loaded",
        "bcn_validation_or_final_rows_loaded",
        "ph2_rows_loaded",
        "training_steps",
        "predictions_generated",
        "metrics_computed",
    )
    if any(hard_scope.get(name) != 0 for name in zero_counters):
        raise AssertionError("completed cache hard-scope counter is nonzero or absent")
    if hard_scope.get("derm7pt_development_rows_resolved") != len(inputs.cases):
        raise AssertionError("completed cache development scope count changed")
    if hard_scope.get("unique_clinical_images_encoded") != n_available:
        raise AssertionError("completed cache encoded-image count changed")
    return {**manifest, "status": "REUSED_VERIFIED"}


def extract_clinical_prefix_cache(
    inputs: phase24b_inputs.Phase24BInputSet,
    encoder: PrefixEncoder,
    *,
    output_dir: Path,
    derm_binding: Mapping[str, Any],
    frozen_spec_sha256: str,
    batch_size: int = 16,
    parity_tolerance: float = 1e-6,
    require_production_counts: bool = True,
    on_chunk_complete: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Extract or resume the aligned clinical prefix cache."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not _is_sha256(frozen_spec_sha256):
        raise ValueError("frozen_spec_sha256 must be a lowercase SHA-256")
    if encoder.config.dtype != "float32":
        raise ValueError("Phase 2.4B cache dtype must be float32")
    if encoder.config.shared_prefix_blocks != phase24b_inputs.SHARED_PREFIX_BLOCKS:
        raise ValueError("Phase 2.4B requires the immutable block-10 boundary")
    available_positions = [
        index for index, case in enumerate(inputs.cases) if case.clinical_available
    ]
    if require_production_counts and (
        len(inputs.cases) != phase24b_inputs.N_CASES
        or len(available_positions) != phase24b_inputs.N_AVAILABLE_CLINICAL
        or encoder.config.tokens_per_image != PRODUCTION_TOKEN_COUNT
        or encoder.config.hidden_size != PRODUCTION_HIDDEN_SIZE
    ):
        raise AssertionError(
            "production extraction requires exactly 616/609 rows and 197x768 tokens"
        )
    if any(inputs.cases[index].clinical_path is None for index in available_positions):
        raise AssertionError("available clinical row lacks a path")

    # Verify every allowed image by content before loading the model callback.
    for position in available_positions:
        case = inputs.cases[position]
        assert case.clinical_path is not None and case.clinical_sha256 is not None
        if not case.clinical_path.is_file():
            raise FileNotFoundError(case.clinical_path)
        if phase24b_inputs.sha256_file(case.clinical_path) != case.clinical_sha256:
            raise AssertionError(f"clinical image hash changed: {case.group_id}")

    contract = _extraction_contract(
        inputs, encoder, derm_binding, frozen_spec_sha256, parity_tolerance
    )
    extraction_identity = phase24b_inputs.canonical_hash(contract)
    paths = CachePaths.under(output_dir)
    if paths.manifest.is_file():
        return _validate_complete_artifacts(
            paths,
            extraction_identity,
            inputs=inputs,
            encoder=encoder,
            derm_binding=derm_binding,
            parity_tolerance=parity_tolerance,
            require_production_counts=require_production_counts,
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.receipts_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(contract["shape"])

    values: np.ndarray
    if paths.partial_tokens.is_file():
        values = np.lib.format.open_memmap(paths.partial_tokens, mode="r+")
        if values.shape != shape or values.dtype != np.float32:
            raise AssertionError("partial cache tensor contract changed")
        progress = _load_and_verify_progress(paths, extraction_identity)
        next_offset = _verify_receipts(
            paths, values, available_positions, progress, extraction_identity
        )
    elif paths.clinical_tokens.is_file() and paths.progress.is_file():
        values = np.load(paths.clinical_tokens, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or values.dtype != np.float32:
            raise AssertionError("recovered cache tensor contract changed")
        progress = _load_and_verify_progress(paths, extraction_identity)
        next_offset = _verify_receipts(
            paths, values, available_positions, progress, extraction_identity
        )
        if next_offset != len(available_positions):
            raise AssertionError("final cache exists before all chunks completed")
    else:
        if paths.progress.exists() or paths.clinical_tokens.exists():
            raise AssertionError("orphaned cache/progress artifact; refusing ambiguous resume")
        values = np.lib.format.open_memmap(
            paths.partial_tokens, mode="w+", dtype="float32", shape=shape
        )
        values[:] = 0.0
        values.flush()
        progress = {
            "format": FORMAT_VERSION,
            "status": "IN_PROGRESS",
            "extraction_identity_sha256": extraction_identity,
            "next_available_offset": 0,
            "receipts": [],
        }
        _atomic_write_json(paths.progress, progress)
        next_offset = 0

    try:
        for start in range(next_offset, len(available_positions), batch_size):
            stop = min(len(available_positions), start + batch_size)
            positions = available_positions[start:stop]
            image_paths = [inputs.cases[position].clinical_path for position in positions]
            if any(path is None for path in image_paths):
                raise AssertionError("unavailable path entered encoder batch")
            block = np.asarray(
                encoder.encode_prefix([Path(path) for path in image_paths]),
                dtype="float32",
            )
            expected = (stop - start, encoder.config.tokens_per_image, encoder.config.hidden_size)
            if block.shape != expected:
                raise AssertionError(f"invalid block-10 encoder shape: {block.shape}")
            if not np.isfinite(block).all():
                raise FloatingPointError("encoder returned non-finite prefix tokens")
            values[positions] = block
            if hasattr(values, "flush"):
                values.flush()  # type: ignore[union-attr]
            receipt = {
                "format": FORMAT_VERSION,
                "extraction_identity_sha256": extraction_identity,
                "available_offset_start": start,
                "available_offset_stop": stop,
                "row_positions": positions,
                "group_ids_sha256": phase24b_inputs.canonical_hash(
                    [inputs.cases[position].group_id for position in positions]
                ),
                "tensor_sha256": _array_sha256(values[positions]),
                "official_or_final_rows_loaded": 0,
            }
            receipt_path = _receipt_path(paths, start, stop)
            _atomic_write_json(receipt_path, receipt)
            progress = {
                **progress,
                "next_available_offset": stop,
                "receipts": [
                    *list(progress["receipts"]),
                    receipt_path.name,
                ],
            }
            _atomic_write_json(paths.progress, progress)
            if on_chunk_complete is not None:
                on_chunk_complete(dict(receipt))
    finally:
        if hasattr(values, "flush"):
            values.flush()  # type: ignore[union-attr]

    unavailable = np.ones(len(inputs.cases), dtype=bool)
    unavailable[np.asarray(available_positions, dtype=int)] = False
    if np.any(np.asarray(values[unavailable]) != 0.0):
        raise AssertionError("unavailable clinical rows are not exact zeros")
    if not np.isfinite(np.asarray(values[available_positions])).all():
        raise FloatingPointError("completed cache contains non-finite tokens")

    sample_offsets = sorted({0, len(available_positions) // 2, len(available_positions) - 1})
    sample_positions = [available_positions[offset] for offset in sample_offsets]
    sample_paths = [Path(inputs.cases[position].clinical_path) for position in sample_positions]  # type: ignore[arg-type]
    direct = np.asarray(encoder.direct_features(sample_paths), dtype="float32")
    reconstructed = np.asarray(
        encoder.reconstruct_features(np.asarray(values[sample_positions], dtype="float32")),
        dtype="float32",
    )
    if direct.shape != reconstructed.shape or direct.ndim != 2:
        raise AssertionError("canonical parity feature shapes differ")
    if not np.isfinite(direct).all() or not np.isfinite(reconstructed).all():
        raise FloatingPointError("canonical parity produced non-finite features")
    per_sample = np.max(np.abs(direct - reconstructed), axis=1)
    maximum_difference = float(per_sample.max(initial=0.0))
    if maximum_difference > parity_tolerance:
        raise AssertionError(
            f"block-10 cache parity failed: {maximum_difference} > {parity_tolerance}"
        )

    row_identity = _row_identity(inputs)
    availability = np.asarray(
        [case.clinical_available for case in inputs.cases], dtype=bool
    )
    if paths.partial_tokens.is_file():
        del values
        os.replace(paths.partial_tokens, paths.clinical_tokens)
        token_sha = phase24b_inputs.sha256_file(paths.clinical_tokens)
        _atomic_write_bytes(
            _sidecar(paths.clinical_tokens),
            (token_sha + "\n").encode("ascii"),
            with_sidecar=False,
        )
    else:
        token_sha = phase24b_inputs.sha256_file(paths.clinical_tokens)
        if not _sidecar(paths.clinical_tokens).is_file():
            _atomic_write_bytes(
                _sidecar(paths.clinical_tokens),
                (token_sha + "\n").encode("ascii"),
                with_sidecar=False,
            )
    availability_sha = _atomic_write_npy(paths.clinical_available, availability)
    row_sha = _atomic_write_json(paths.row_identity, {"rows": row_identity})
    receipt_bindings = [
        {
            "file": name,
            "sha256": _verify_sha_sidecar(paths.receipts_dir / name),
        }
        for name in progress["receipts"]
    ]
    final_manifest = {
        "format": FORMAT_VERSION,
        "status": "PASS",
        "extraction_identity_sha256": extraction_identity,
        "contract": contract,
        "encoder_provenance": dict(encoder.provenance),
        "preprocess_identity": dict(encoder.preprocess_identity),
        "artifacts": {
            "clinical_tokens": {
                "file": paths.clinical_tokens.name,
                "sha256": token_sha,
                "shape": list(shape),
                "dtype": "float32",
            },
            "clinical_available": {
                "file": paths.clinical_available.name,
                "sha256": availability_sha,
            },
            "row_identity": {"file": paths.row_identity.name, "sha256": row_sha},
            "dermoscopic_tokens": dict(derm_binding),
            "chunk_receipts": receipt_bindings,
        },
        "counts": {
            "aligned_rows": len(inputs.cases),
            "unique_clinical_photos_encoded": len(available_positions),
            "unavailable_zero_rows": len(inputs.cases) - len(available_positions),
            "canonical_direct_parity_samples": len(sample_positions),
        },
        "zero_row_policy": "unavailable rows are exact float32 zeros; no substitution",
        "canonical_parity": {
            "sample_rows": sample_positions,
            "sample_group_ids": [inputs.cases[position].group_id for position in sample_positions],
            "per_sample_max_abs_difference": per_sample.astype(float).tolist(),
            "maximum_abs_difference": maximum_difference,
            "tolerance": parity_tolerance,
            "status": "PASS",
        },
        "hard_scope": {
            "derm7pt_development_rows_resolved": len(inputs.cases),
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
            "unique_clinical_images_encoded": len(available_positions),
            "training_steps": 0,
            "predictions_generated": 0,
            "metrics_computed": 0,
        },
    }
    _atomic_write_json(paths.manifest, final_manifest)
    _atomic_write_json(
        paths.progress,
        {
            **progress,
            "status": "COMPLETE",
            "terminal_manifest_sha256": phase24b_inputs.sha256_file(paths.manifest),
        },
    )
    return {**final_manifest, "manifest_sha256": phase24b_inputs.sha256_file(paths.manifest)}


def _describe() -> dict[str, Any]:
    return {
        "command": "Phase 2.4B clinical block-10 cache extractor",
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "requires_frozen_specification_with_sha_sidecar": True,
        "expected_shape": [
            phase24b_inputs.N_CASES,
            PRODUCTION_TOKEN_COUNT,
            PRODUCTION_HIDDEN_SIZE,
        ],
        "distinct_clinical_photos": phase24b_inputs.N_AVAILABLE_CLINICAL,
        "unavailable_zero_rows": phase24b_inputs.N_UNAVAILABLE_CLINICAL,
        "writes_performed": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true", help="print the no-write plan")
    mode.add_argument("--extract", action="store_true", help="perform authorized extraction")
    parser.add_argument("--frozen-spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--derm-tokens", type=Path, default=DEFAULT_DERM_TOKENS)
    parser.add_argument("--derm-manifest", type=Path, default=DEFAULT_DERM_MANIFEST)
    args = parser.parse_args(argv)
    if args.describe:
        if args.frozen_spec is not None:
            parser.error("--describe does not consume --frozen-spec")
        result = _describe()
    else:
        if args.frozen_spec is None:
            parser.error("--extract requires --frozen-spec")
        inputs = phase24b_inputs.resolve_development_inputs()
        encoder = load_pinned_hybrid_prefix_encoder()
        _, spec_sha = verify_frozen_specification(args.frozen_spec, inputs, encoder.config)
        binding = bind_existing_dermoscopic_cache(
            inputs,
            encoder.config,
            token_path=args.derm_tokens,
            manifest_path=args.derm_manifest,
        )
        result = extract_clinical_prefix_cache(
            inputs,
            encoder,
            output_dir=args.output_dir,
            derm_binding=binding,
            frozen_spec_sha256=spec_sha,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CachePaths",
    "PrefixEncoder",
    "bind_existing_dermoscopic_cache",
    "extract_clinical_prefix_cache",
    "load_pinned_hybrid_prefix_encoder",
    "verify_frozen_specification",
]
