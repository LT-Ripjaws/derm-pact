"""Reproducible frozen-feature extraction for the HAM10000 encoder tournament."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..data.manifest import CaseRecord, load_manifest


ROOT = Path(__file__).resolve().parents[2]
DEV_MANIFEST = ROOT / "research/manifests/ham10000_train_calibration_v2.json"
FEATURE_ROOT = ROOT / "research/experiments/derm_perception/v2_tournament/features"


@dataclass(frozen=True)
class Candidate:
    key: str
    model_id: str
    loader: str
    batch_size: int


CANDIDATES = {
    "dermlip": Candidate("dermlip", "redlessone/DermLIP_ViT-B-16", "open_clip", 32),
    "panderm": Candidate(
        "panderm", "redlessone/DermLIP_PanDerm-base-w-PubMed-256", "panderm", 32
    ),
    "biomedclip": Candidate(
        "biomedclip", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "open_clip", 32,
    ),
    "monet": Candidate("monet", "suinleelab/monet", "transformers", 8),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return Path(snapshot_download(model_id, local_files_only=True))


def _checkpoint_fingerprint(snapshot: Path) -> dict[str, object]:
    weight_files = sorted(
        path for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.lower() in {".bin", ".safetensors", ".pt", ".pth"}
    )
    if not weight_files:
        raise FileNotFoundError(f"no weight file in cached snapshot {snapshot}")
    files = [
        {"name": str(path.relative_to(snapshot)), "bytes": path.stat().st_size,
         "sha256": _sha256(path)}
        for path in weight_files
    ]
    joined = "\n".join(f"{item['name']}:{item['sha256']}" for item in files)
    return {
        "snapshot_revision": snapshot.name,
        "weight_files": files,
        "combined_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
    }


def _load_open_clip(candidate: Candidate):
    import open_clip
    import torch
    from PIL import Image

    snapshot = _snapshot(candidate.model_id)
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{candidate.model_id}"
    )
    model.eval()

    def encode(paths: Sequence[str]):
        batch = torch.stack([
            preprocess(Image.open(path).convert("RGB")) for path in paths
        ])
        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype("float32")

    return encode, {
        "implementation": str(Path(open_clip.__file__).resolve()),
        "checkpoint": _checkpoint_fingerprint(snapshot),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _load_panderm(candidate: Candidate):
    official_src = ROOT / "research/third_party/Derm1M_official/src"
    if not official_src.is_dir():
        raise FileNotFoundError("official Derm1M sparse checkout is missing")
    if "open_clip" in sys.modules:
        raise RuntimeError("PanDerm must run in a fresh process before stock open_clip is imported")
    sys.path.insert(0, str(official_src))
    # The official 2025 loader imports packaging through deprecated pkg_resources.
    import packaging
    import pkg_resources

    pkg_resources.packaging = packaging
    import open_clip
    import torch
    from PIL import Image

    snapshot = _snapshot(candidate.model_id)
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{candidate.model_id}"
    )
    model.eval()

    def encode(paths: Sequence[str]):
        batch = torch.stack([
            preprocess(Image.open(path).convert("RGB")) for path in paths
        ])
        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype("float32")

    safe = str((ROOT / "research/third_party/Derm1M_official").as_posix())
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={safe}", "-C",
             str(ROOT / "research/third_party/Derm1M_official"), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return encode, {
        "implementation": str(Path(open_clip.__file__).resolve()),
        "official_derm1m_commit": commit,
        "compatibility_shim": "pkg_resources.packaging alias only",
        "checkpoint": _checkpoint_fingerprint(snapshot),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _load_monet(candidate: Candidate):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    snapshot = _snapshot(candidate.model_id)
    processor = AutoProcessor.from_pretrained(candidate.model_id, local_files_only=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        candidate.model_id, local_files_only=True
    )
    model.eval()

    def encode(paths: Sequence[str]):
        images = [Image.open(path).convert("RGB") for path in paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            vision = model.vision_model(pixel_values=inputs["pixel_values"])
            features = model.visual_projection(vision.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype("float32")

    return encode, {
        "implementation": "transformers.AutoModelForZeroShotImageClassification",
        "feature_path": "vision_model.pooler_output -> visual_projection",
        "checkpoint": _checkpoint_fingerprint(snapshot),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


LOADERS: dict[str, Callable[[Candidate], tuple[Callable, dict[str, object]]]] = {
    "open_clip": _load_open_clip,
    "panderm": _load_panderm,
    "transformers": _load_monet,
}


def load_candidate_encoder(key: str):
    """Load one registered encoder and return its batch encoder plus provenance."""
    if key not in CANDIDATES:
        raise ValueError(f"unknown candidate {key!r}; choose from {sorted(CANDIDATES)}")
    candidate = CANDIDATES[key]
    return LOADERS[candidate.loader](candidate)


def _development_records(manifest: Path) -> list[CaseRecord]:
    records = load_manifest(manifest)
    observed = {record.split for record in records}
    if observed != {"train", "calibration"}:
        raise ValueError(f"tournament requires a test-redacted manifest, got {observed}")
    if any(record.split == "test" for record in records):
        raise AssertionError("final-test record entered feature extraction")
    if any(not record.image_sha256 for record in records):
        raise ValueError("every staged image must be content-hashed")
    return records


def extract_candidate(
    key: str, *, manifest: Path = DEV_MANIFEST, batch_size: int | None = None,
) -> dict[str, object]:
    import numpy as np

    if key not in CANDIDATES:
        raise ValueError(f"unknown candidate {key!r}; choose from {sorted(CANDIDATES)}")
    candidate = CANDIDATES[key]
    records = _development_records(manifest)
    manifest_sha = manifest.with_suffix(manifest.suffix + ".sha256").read_text(
        encoding="ascii"
    ).strip()
    output = FEATURE_ROOT / key
    output.mkdir(parents=True, exist_ok=True)
    final = output / "features.npy"
    meta_path = output / "feature_manifest.json"
    if final.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("development_manifest_sha256") != manifest_sha:
            raise ValueError("cached features belong to a different development manifest")
        if metadata.get("model_key") != key or metadata.get("n_records") != len(records):
            raise ValueError("cached feature metadata mismatch")
        return metadata

    encode, provenance = load_candidate_encoder(key)
    effective_batch = int(batch_size or candidate.batch_size)
    if effective_batch <= 0:
        raise ValueError("batch_size must be positive")
    partial = output / "features.partial.npy"
    progress_path = output / "progress.json"
    next_index = 0
    array = None
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("development_manifest_sha256") != manifest_sha:
            raise ValueError("partial features belong to a different manifest")
        next_index = int(progress["next_index"])
        array = np.lib.format.open_memmap(partial, mode="r+")
        if array.shape[0] != len(records):
            raise ValueError("partial feature row count mismatch")

    started = time.perf_counter()
    for start in range(next_index, len(records), effective_batch):
        stop = min(len(records), start + effective_batch)
        batch = encode([record.image_path for record in records[start:stop]])
        if batch.ndim != 2 or batch.shape[0] != stop - start:
            raise ValueError(f"encoder returned invalid shape {batch.shape}")
        if not np.isfinite(batch).all():
            raise ValueError("encoder returned non-finite features")
        if array is None:
            array = np.lib.format.open_memmap(
                partial, mode="w+", dtype="float32", shape=(len(records), batch.shape[1])
            )
        if batch.shape[1] != array.shape[1]:
            raise ValueError("feature dimension changed mid-run")
        array[start:stop] = batch
        array.flush()
        progress = {
            "model_key": key,
            "development_manifest_sha256": manifest_sha,
            "next_index": stop,
            "n_records": len(records),
            "feature_dim": int(array.shape[1]),
        }
        temp_progress = progress_path.with_suffix(".tmp")
        temp_progress.write_text(json.dumps(progress, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_progress, progress_path)
        if stop == len(records) or stop % max(1, effective_batch * 10) == 0:
            elapsed = time.perf_counter() - started
            print(f"[{key}] {stop}/{len(records)} images; {elapsed:.1f}s", flush=True)

    if array is None:
        raise ValueError("no features extracted")
    del array
    os.replace(partial, final)
    metadata: dict[str, object] = {
        "protocol": "ham10000-derm-encoder-features-v2",
        "model_key": key,
        "model_id": candidate.model_id,
        "loader": candidate.loader,
        "batch_size": effective_batch,
        "development_manifest": str(manifest),
        "development_manifest_sha256": manifest_sha,
        "included_splits": ["train", "calibration"],
        "test_records_loaded": 0,
        "n_records": len(records),
        "feature_dim": int(np.load(final, mmap_mode="r").shape[1]),
        "feature_file_sha256": _sha256(final),
        "elapsed_seconds_current_process": time.perf_counter() - started,
        "provenance": provenance,
    }
    temp_meta = meta_path.with_suffix(".tmp")
    temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_meta, meta_path)
    if progress_path.exists():
        progress_path.unlink()
    return metadata
