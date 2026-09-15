"""Phase 2.5 high-resolution window prefix-token cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from . import dermpact_phase2_4b_inputs as phase24b_inputs
from . import dermpact_phase2_backbone_adaptation as adaptation


ROOT = phase24b_inputs.ROOT
FORMAT_VERSION = "dermpact-phase2-5-window-prefix-cache-v2"
N_CASES = 616
N_WINDOWS = 16
GRID_SIZE = 4
TARGET_SIZE = 896
WINDOW_SIZE = 224
TOKEN_COUNT = 197
TOKEN_DIM = 768
PATCH_SIZE = 16
PATCHES_PER_SIDE = WINDOW_SIZE // PATCH_SIZE
N_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_OUTPUT_DIR = (
    ROOT / "research/experiments/dermpact_phase2_5/window_prefix_cache_v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def source_bindings(paths: Sequence[Path] | None = None) -> list[dict[str, Any]]:
    """Capture the exact source bytes used by a cache process."""
    selected = list(paths) if paths is not None else [
        Path(__file__).resolve(),
        Path(phase24b_inputs.__file__).resolve(),
        Path(adaptation.__file__).resolve(),
    ]
    bindings: list[dict[str, Any]] = []
    for raw_path in selected:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            display = path.relative_to(ROOT).as_posix()
        except ValueError:
            display = str(path)
        bindings.append({"path": display, "absolute_path": str(path), "sha256": _sha256(path)})
    return bindings


def assert_source_bindings(bindings: Sequence[Mapping[str, Any]]) -> None:
    """Fail if any source captured at process start changed in place."""
    for binding in bindings:
        path = Path(str(binding["absolute_path"]))
        expected = str(binding["sha256"])
        if not path.is_file() or _sha256(path) != expected:
            raise AssertionError(f"source drift detected during cache build: {path}")


def _binding_digest(bindings: Sequence[Mapping[str, Any]]) -> str:
    body = json.dumps(list(bindings), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _atomic_npy(path: Path, array: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    digest = _sha256(path)
    sc_tmp = Path(str(_sidecar(path)) + ".tmp")
    sc_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sc_tmp, _sidecar(path))
    return digest


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with tmp.open("r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    digest = _sha256(path)
    sc_tmp = Path(str(_sidecar(path)) + ".tmp")
    sc_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sc_tmp, _sidecar(path))
    return digest


def aspect_preserve_pad(
    image: Image.Image,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize preserving aspect ratio, symmetric-pad to TARGET_SIZE×TARGET_SIZE."""
    w, h = image.size
    scale = TARGET_SIZE / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left

    padded = np.full(
        (TARGET_SIZE, TARGET_SIZE, 3), IMAGENET_MEAN, dtype=np.float32
    )
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = arr
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def tile_windows(
    padded_image: np.ndarray,
) -> np.ndarray:
    """Tile a TARGET_SIZE×TARGET_SIZE image into GRID_SIZE×GRID_SIZE windows."""
    windows = np.empty(
        (N_WINDOWS, WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.float32
    )
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y0 = r * WINDOW_SIZE
            x0 = c * WINDOW_SIZE
            windows[r * GRID_SIZE + c] = padded_image[
                y0:y0 + WINDOW_SIZE, x0:x0 + WINDOW_SIZE
            ]
    return windows


def compute_patch_valid_mask(
    padding: tuple[int, int, int, int],
) -> np.ndarray:
    """Compute per-patch valid mask for each window."""
    pad_top, pad_bottom, pad_left, pad_right = padding
    valid_top = pad_top
    valid_bottom = TARGET_SIZE - pad_bottom
    valid_left = pad_left
    valid_right = TARGET_SIZE - pad_right

    mask = np.zeros((N_WINDOWS, N_PATCHES), dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            win_y0 = r * WINDOW_SIZE
            win_x0 = c * WINDOW_SIZE
            for pr in range(PATCHES_PER_SIDE):
                for pc in range(PATCHES_PER_SIDE):
                    center_y = win_y0 + pr * PATCH_SIZE + PATCH_SIZE // 2
                    center_x = win_x0 + pc * PATCH_SIZE + PATCH_SIZE // 2
                    patch_idx = pr * PATCHES_PER_SIDE + pc
                    win_idx = r * GRID_SIZE + c
                    if (valid_top <= center_y < valid_bottom and
                            valid_left <= center_x < valid_right):
                        mask[win_idx, patch_idx] = True
    return mask


def window_positions() -> np.ndarray:
    """Return [N_WINDOWS, 2] int array of (row, col) grid positions."""
    pos = np.empty((N_WINDOWS, 2), dtype=np.int32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            pos[r * GRID_SIZE + c] = [r, c]
    return pos


def normalize_for_panDerm(windows: np.ndarray) -> torch.Tensor:
    """Apply ImageNet normalization and convert to [N, 3, H, W] tensor."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    normalized = (windows - mean) / std
    return torch.from_numpy(normalized.transpose(0, 3, 1, 2).copy())


def build_window_cache(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_idx: int = 0,
    count: int | None = None,
) -> dict[str, Any]:
    """Build the 16-window block-10 prefix token cache."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_sources = source_bindings()
    source_bundle_sha = _binding_digest(frozen_sources)
    cache_protocol = {
        "schema_version": "dermpact-phase2-5-window-cache-protocol-v2",
        "source_bindings": frozen_sources,
        "source_bundle_sha256": source_bundle_sha,
        "contract": {
            "n_cases": N_CASES,
            "n_windows": N_WINDOWS,
            "target_size": TARGET_SIZE,
            "window_size": WINDOW_SIZE,
            "dtype": "float16",
            "official_test_rows_loaded": 0,
        },
    }
    protocol_path = output_dir / "window_cache_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != cache_protocol:
            raise AssertionError("existing window-cache protocol does not match current source/config")
    else:
        _atomic_json(protocol_path, cache_protocol)

    print("Resolving authorized development inputs ...", flush=True)
    inputs = phase24b_inputs.resolve_development_inputs()
    cases = inputs.cases
    if len(cases) != N_CASES:
        raise AssertionError(f"expected {N_CASES} cases, got {len(cases)}")

    end_idx = min(start_idx + (count or N_CASES), N_CASES)
    print(f"Processing cases {start_idx} to {end_idx - 1} ...", flush=True)

    print("Loading PanDerm visual encoder ...", flush=True)
    visual, preprocess, provenance = adaptation._load_hybrid_visual()
    visual.eval()

    group_ids = [case.group_id for case in cases]
    group_id_hash = hashlib.sha256(
        json.dumps(group_ids, sort_keys=False).encode("utf-8")
    ).hexdigest()

    tokens_path = output_dir / "window_tokens.npy"
    masks_path = output_dir / "patch_valid_mask.npy"

    # Create or open memory-mapped arrays
    if start_idx == 0 and not tokens_path.exists():
        all_tokens = np.zeros(
            (N_CASES, N_WINDOWS, TOKEN_COUNT, TOKEN_DIM), dtype=np.float16
        )
        np.save(tokens_path, all_tokens)
        del all_tokens
        all_masks = np.zeros((N_CASES, N_WINDOWS, N_PATCHES), dtype=bool)
        np.save(masks_path, all_masks)
        del all_masks

    all_tokens = np.load(tokens_path, mmap_mode="r+")
    all_masks = np.load(masks_path, mmap_mode="r+")
    image_hashes = []

    started = time.perf_counter()
    for idx in range(start_idx, end_idx):
        case = cases[idx]
        image = Image.open(case.dermoscopic_path).convert("RGB")
        img_hash = _sha256(case.dermoscopic_path)
        image_hashes.append(img_hash)

        padded, padding = aspect_preserve_pad(image)
        windows = tile_windows(padded)
        patch_mask = compute_patch_valid_mask(padding)
        all_masks[idx] = patch_mask

        window_tensor = normalize_for_panDerm(windows)

        with torch.inference_mode():
            for w in range(N_WINDOWS):
                tokens = adaptation._prefix_tokens(
                    visual, window_tensor[w:w + 1]
                )
                all_tokens[idx, w] = tokens[0].numpy().astype(np.float16)

        if (idx - start_idx + 1) % 50 == 0 or idx == start_idx:
            elapsed = time.perf_counter() - started
            done = idx - start_idx + 1
            rate = done / elapsed
            remaining = (end_idx - idx - 1) / rate
            print(
                f"  [{idx + 1}/{N_CASES}] {elapsed:.0f}s elapsed, "
                f"~{remaining:.0f}s remaining",
                flush=True,
            )

    all_tokens.flush()
    all_masks.flush()
    del visual

    # Only write manifest if all cases are done
    if start_idx > 0 or end_idx < N_CASES:
        print(f"Chunk {start_idx}-{end_idx - 1} saved. Run remaining chunks.", flush=True)
        return {"status": "PARTIAL", "start": start_idx, "end": end_idx}

    pos = window_positions()
    tokens_path = output_dir / "window_tokens.npy"
    masks_path = output_dir / "patch_valid_mask.npy"
    positions_path = output_dir / "window_positions.npy"

    print("Saving cache ...", flush=True)
    tokens_sha = _atomic_npy(tokens_path, all_tokens)
    masks_sha = _atomic_npy(masks_path, all_masks)
    positions_sha = _atomic_npy(positions_path, pos)

    valid_window_counts = all_masks.any(axis=2).sum(axis=1)
    valid_patch_counts = all_masks.sum(axis=(1, 2))

    assert_source_bindings(frozen_sources)

    manifest = {
        "schema_version": FORMAT_VERSION,
        "status": "PASS",
        "contract": {
            "n_cases": N_CASES,
            "n_windows": N_WINDOWS,
            "grid_size": GRID_SIZE,
            "target_size": TARGET_SIZE,
            "window_size": WINDOW_SIZE,
            "token_shape": [N_CASES, N_WINDOWS, TOKEN_COUNT, TOKEN_DIM],
            "mask_shape": [N_CASES, N_WINDOWS, N_PATCHES],
            "dtype": "float16",
            "official_test_rows_loaded": 0,
        },
        "image_processing": {
            "resize": "aspect_preserving_longer_side_896",
            "padding": "symmetric_imagenet_mean_fill",
            "normalization": "imagenet",
            "tiling": "4x4_non_overlapping_224",
        },
        "statistics": {
            "valid_windows_per_case": {
                "min": int(valid_window_counts.min()),
                "max": int(valid_window_counts.max()),
                "mean": float(valid_window_counts.mean()),
            },
            "valid_patches_per_case": {
                "min": int(valid_patch_counts.min()),
                "max": int(valid_patch_counts.max()),
                "mean": float(valid_patch_counts.mean()),
            },
        },
        "provenance": {
            "group_id_order_sha256": group_id_hash,
            "n_image_hashes": len(image_hashes),
            "image_hash_digest": hashlib.sha256(
                "".join(image_hashes).encode("ascii")
            ).hexdigest(),
            "visual_model": provenance,
        },
        "artifacts": {
            "tokens": {"path": tokens_path.name, "sha256": tokens_sha},
            "mask": {"path": masks_path.name, "sha256": masks_sha},
            "positions": {"path": positions_path.name, "sha256": positions_sha},
        },
        "source": {
            "protocol_path": protocol_path.name,
            "protocol_sha256": _sha256(protocol_path),
            "bundle_sha256": source_bundle_sha,
            "bindings": frozen_sources,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    manifest_path = output_dir / "window_cache_manifest.json"
    manifest_sha = _atomic_json(manifest_path, manifest)
    print(f"Done. Manifest SHA: {manifest_sha[:16]}...", flush=True)
    return {**manifest, "manifest_sha256": manifest_sha}


def validate_window_cache(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    manifest_path = Path(output_dir) / "window_cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = _sha256(manifest_path)
    manifest_sidecar = _sidecar(manifest_path)
    if not manifest_sidecar.is_file() or manifest_sidecar.read_text(encoding="ascii").strip() != manifest_sha:
        raise AssertionError("window cache manifest sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FORMAT_VERSION:
        raise AssertionError("window cache schema version mismatch")
    if manifest.get("status") != "PASS":
        raise AssertionError("window cache status is not PASS")
    for name in ("tokens", "mask", "positions"):
        artifact = manifest["artifacts"][name]
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = Path(output_dir) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != artifact["sha256"]:
            raise AssertionError(f"window cache {name} SHA-256 mismatch")
    return {"status": "PASS", "artifacts": manifest["artifacts"], "manifest": manifest}


def load_window_cache(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    validated = validate_window_cache(output_dir=output_dir)
    def artifact_path(name: str) -> Path:
        path = Path(validated["artifacts"][name]["path"])
        return path if path.is_absolute() else Path(output_dir) / path

    tokens = np.load(artifact_path("tokens"), mmap_mode="r")
    mask = np.load(artifact_path("mask"), allow_pickle=False)
    positions = np.load(artifact_path("positions"), allow_pickle=False)
    return {
        "window_tokens": tokens,
        "patch_valid_mask": mask,
        "window_positions": positions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--validate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args(argv)
    if args.build:
        result = build_window_cache(
            output_dir=args.output_dir,
            start_idx=args.start,
            count=args.count,
        )
    else:
        result = validate_window_cache(output_dir=args.output_dir)
    print(json.dumps(
        {k: v for k, v in result.items() if k != "manifest_sha256"},
        indent=2, sort_keys=True, default=str,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_window_cache",
    "load_window_cache",
    "validate_window_cache",
    "aspect_preserve_pad",
    "tile_windows",
    "compute_patch_valid_mask",
    "window_positions",
    "source_bindings",
    "assert_source_bindings",
]
