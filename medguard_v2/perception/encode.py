"""Frozen image-encoder features for Derm7pt.

This is the perception half of the classifier-only baseline (arm ``R0``) and the
input to gate ``G5`` (does classifier-vs-criteria disagreement predict error?).
Nothing here is trained: a published checkpoint is loaded, put in ``eval`` mode,
and run once over every case's **dermoscopic** image.  The linear probe that
turns these vectors into predictions lives in :mod:`medguard_v2.perception.probe`
and is the only thing that is ever fit.

Two properties this module is built around:

* **Re-runs must be free.**  1,011 forward passes of a ViT-B/16 on CPU costs a
  few minutes.  Features are cached to
  ``research/evidence/derm7pt_v1/features/<encoder>/features.npz`` keyed by
  ``case_num``, next to a manifest that records what produced them.  A cache is
  reused only when the encoder identity, the checkpoint fingerprint, the image
  view and the *content digest of the images themselves* all match; any drift
  invalidates it loudly rather than silently returning stale vectors.
* **Provenance must survive into the artifact.**  The resolved model id, the
  snapshot revision, and a SHA-256 fingerprint of the weight files are recorded
  so a reader can tell exactly which checkpoint produced a number.

Encoder choice
--------------
``dermlip`` (``redlessone/DermLIP_ViT-B-16``) is preferred because v1 used it,
which keeps v1-vs-v2 comparable.  ``biomedclip`` is the documented fallback if
the DermLIP checkpoint is not present locally and cannot be fetched.  Whichever
is used is recorded in the manifest and in the frozen artifact, so the fallback
can never be mistaken for the primary.

Licensing: Derm7pt is CC BY-NC-ND.  No image bytes are copied anywhere by this
module.  Only feature vectors, file-content digests and metadata are written.

CLI::

    python -m medguard_v2.perception.encode            # encode (cached) and report
    python -m medguard_v2.perception.encode --encoder biomedclip
    python -m medguard_v2.perception.encode --force    # ignore an existing cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from medguard_v2.data.derm7pt import Derm7ptCase, load_cases

__all__ = [
    "ENCODERS",
    "PREFERRED_ORDER",
    "EncoderSpec",
    "FeatureBundle",
    "PerceptionEncodeError",
    "FEATURE_ROOT",
    "encoder_snapshot",
    "encoder_available",
    "resolve_encoder",
    "load_encoder",
    "feature_path",
    "manifest_path",
    "image_set_digest",
    "load_cached_features",
    "encode_cases",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = REPO_ROOT / "research" / "evidence" / "derm7pt_v1" / "features"

#: Which image of each case is encoded.  Derm7pt ships a clinical and a
#: dermoscopic photograph; the dermoscopic one is what the seven-point checklist
#: is read from, so it is the only one this baseline sees.
IMAGE_VIEW = "derm"

CACHE_PROTOCOL = "derm7pt-frozen-features-v1"


class PerceptionEncodeError(RuntimeError):
    """Raised when an encoder cannot be resolved or a cache is inconsistent."""


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model_id: str
    loader: str
    batch_size: int
    note: str


ENCODERS: Mapping[str, EncoderSpec] = {
    "dermlip": EncoderSpec(
        key="dermlip",
        model_id="redlessone/DermLIP_ViT-B-16",
        loader="open_clip",
        batch_size=32,
        note="v1's encoder; preferred so v1 and v2 numbers are comparable",
    ),
    "biomedclip": EncoderSpec(
        key="biomedclip",
        model_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        loader="open_clip",
        batch_size=32,
        note="documented fallback: general biomedical CLIP, not dermatology-specific",
    ),
}

#: Tried in this order by :func:`resolve_encoder`.
PREFERRED_ORDER = ("dermlip", "biomedclip")


# --------------------------------------------------------------------------
# checkpoint resolution and fingerprinting
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_snapshot(key: str, *, allow_download: bool = False) -> Path:
    """Resolve the local snapshot directory for one encoder.

    ``allow_download=False`` (the default) never touches the network: it either
    finds the checkpoint in the local Hugging Face cache or raises.
    """
    if key not in ENCODERS:
        raise PerceptionEncodeError(f"unknown encoder {key!r}; choose from {sorted(ENCODERS)}")
    from huggingface_hub import snapshot_download

    if not allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return Path(
            snapshot_download(ENCODERS[key].model_id, local_files_only=not allow_download)
        )
    except Exception as exc:  # noqa: BLE001 - hub raises a family of errors
        raise PerceptionEncodeError(
            f"checkpoint for {key!r} ({ENCODERS[key].model_id}) is not available "
            f"{'online or offline' if allow_download else 'in the local cache'}: {exc}"
        ) from exc


def encoder_available(key: str, *, allow_download: bool = False) -> bool:
    try:
        encoder_snapshot(key, allow_download=allow_download)
    except PerceptionEncodeError:
        return False
    return True


def checkpoint_fingerprint(snapshot: Path) -> dict[str, object]:
    """SHA-256 over every weight file in the snapshot, plus a combined digest."""
    weight_files = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.lower() in {".bin", ".safetensors", ".pt", ".pth"}
    )
    if not weight_files:
        raise PerceptionEncodeError(f"no weight file in cached snapshot {snapshot}")
    files = [
        {
            "name": str(path.relative_to(snapshot)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in weight_files
    ]
    joined = "\n".join(f"{item['name']}:{item['sha256']}" for item in files)
    return {
        "snapshot_revision": snapshot.name,
        "weight_files": files,
        "combined_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
    }


def resolve_encoder(
    preferred: Sequence[str] = PREFERRED_ORDER, *, allow_download: bool = False
) -> tuple[str, str]:
    """Return ``(encoder_key, reason)`` for the first available encoder.

    The reason string is written into the artifact so a fallback is always
    visible to a reader rather than being inferable only from the model id.
    """
    tried: list[str] = []
    for key in preferred:
        if key not in ENCODERS:
            raise PerceptionEncodeError(f"unknown encoder {key!r}")
        if encoder_available(key, allow_download=allow_download):
            if key == preferred[0]:
                reason = (
                    f"{key} resolved from the local checkpoint cache; it is the "
                    "first choice because v1 used it"
                )
            else:
                reason = (
                    f"FALLBACK: {', '.join(tried)} could not be resolved "
                    f"({'download attempted' if allow_download else 'offline, local cache only'}); "
                    f"used {key} instead"
                )
            return key, reason
        tried.append(key)
    raise PerceptionEncodeError(
        "no encoder could be resolved; tried " + ", ".join(tried)
    )


# --------------------------------------------------------------------------
# encoder loading
# --------------------------------------------------------------------------

def load_encoder(key: str, *, allow_download: bool = False):
    """Load one frozen encoder.

    Returns ``(encode_fn, provenance)`` where ``encode_fn`` maps a sequence of
    image paths to an ``(n, d)`` float32 array of L2-normalised features.
    """
    spec = ENCODERS[key]
    if spec.loader != "open_clip":
        raise PerceptionEncodeError(f"unsupported loader {spec.loader!r} for {key!r}")

    import open_clip
    import torch
    from PIL import Image

    snapshot = encoder_snapshot(key, allow_download=allow_download)
    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{spec.model_id}")
    model.eval()

    def encode(paths: Sequence[Path]) -> np.ndarray:
        batch = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths])
        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype("float32")

    provenance = {
        "encoder_key": key,
        "model_id": spec.model_id,
        "resolved_model_id": f"hf-hub:{spec.model_id}",
        "loader": spec.loader,
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "device": "cpu",
        "trainable": False,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "preprocess": str(preprocess),
        "checkpoint": checkpoint_fingerprint(snapshot),
    }
    return encode, provenance


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def feature_path(key: str) -> Path:
    return FEATURE_ROOT / key / "features.npz"


def manifest_path(key: str) -> Path:
    return FEATURE_ROOT / key / "feature_manifest.json"


def _image_path(case: Derm7ptCase) -> Path:
    if IMAGE_VIEW != "derm":  # pragma: no cover - guards a future edit
        raise PerceptionEncodeError(f"unsupported image view {IMAGE_VIEW!r}")
    return case.derm_path


def image_set_digest(cases: Sequence[Derm7ptCase]) -> str:
    """One digest over the *content* of every encoded image.

    Cheap (~0.5 s for the full set) and it is the only thing that can catch an
    image being replaced under a cache.  Only the digest is stored -- never any
    image bytes, which CC BY-NC-ND forbids redistributing.
    """
    lines: list[str] = []
    for case in sorted(cases, key=lambda c: c.case_num):
        path = _image_path(case)
        lines.append(f"{case.case_num}:{path.name}:{_sha256_file(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _features_digest(case_nums: np.ndarray, features: np.ndarray) -> str:
    order = np.argsort(case_nums, kind="stable")
    digest = hashlib.sha256()
    digest.update(np.asarray(case_nums, dtype=np.int64)[order].tobytes())
    digest.update(np.ascontiguousarray(features[order], dtype=np.float32).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureBundle:
    """Cached features for a set of cases, keyed by ``case_num``."""

    encoder_key: str
    case_nums: np.ndarray  # (n,) int64, ascending
    features: np.ndarray  # (n, d) float32, L2-normalised
    manifest: dict

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def index_of(self, case_num: int) -> int:
        position = int(np.searchsorted(self.case_nums, case_num))
        if position >= self.case_nums.size or int(self.case_nums[position]) != case_num:
            raise PerceptionEncodeError(f"no cached feature for case_num {case_num}")
        return position

    def vector_for(self, case_num: int) -> np.ndarray:
        return self.features[self.index_of(case_num)]

    def matrix_for(self, case_nums: Sequence[int]) -> np.ndarray:
        rows = [self.index_of(int(value)) for value in case_nums]
        return np.ascontiguousarray(self.features[rows])


def load_cached_features(key: str) -> FeatureBundle | None:
    """Read the cache for ``key``, or ``None`` if it is absent."""
    path = feature_path(key)
    meta = manifest_path(key)
    if not path.is_file() or not meta.is_file():
        return None
    manifest = json.loads(meta.read_text(encoding="utf-8"))
    with np.load(path) as payload:
        case_nums = np.asarray(payload["case_num"], dtype=np.int64)
        features = np.asarray(payload["features"], dtype=np.float32)
    if case_nums.size != features.shape[0]:
        raise PerceptionEncodeError(f"corrupt feature cache {path}: row count mismatch")
    if not np.all(np.diff(case_nums) > 0):
        raise PerceptionEncodeError(f"corrupt feature cache {path}: case_num not sorted/unique")
    observed = _features_digest(case_nums, features)
    if manifest.get("features_sha256") != observed:
        raise PerceptionEncodeError(
            f"feature cache {path} does not match its manifest digest "
            f"({observed} != {manifest.get('features_sha256')})"
        )
    return FeatureBundle(key, case_nums, features, manifest)


def _cache_is_current(bundle: FeatureBundle, cases: Sequence[Derm7ptCase], digest: str) -> bool:
    manifest = bundle.manifest
    expected = {int(case.case_num) for case in cases}
    return (
        manifest.get("protocol") == CACHE_PROTOCOL
        and manifest.get("image_view") == IMAGE_VIEW
        and manifest.get("image_set_sha256") == digest
        and manifest.get("n_cases") == len(expected)
        and expected.issubset({int(value) for value in bundle.case_nums})
    )


def encode_cases(
    cases: Sequence[Derm7ptCase] | None = None,
    *,
    encoder: str | None = None,
    allow_download: bool = False,
    batch_size: int | None = None,
    force: bool = False,
    verbose: bool = False,
) -> FeatureBundle:
    """Return features for ``cases``, encoding only if the cache cannot serve them.

    ``encoder=None`` resolves the preferred encoder (DermLIP, then the fallback).
    ``force=True`` re-encodes even when a valid cache exists.
    """
    cases = list(cases if cases is not None else load_cases())
    if not cases:
        raise PerceptionEncodeError("no cases to encode")
    case_nums = sorted(int(case.case_num) for case in cases)
    if len(set(case_nums)) != len(case_nums):
        raise PerceptionEncodeError("duplicate case_num in the requested cases")

    if encoder is None:
        # A usable cache is itself proof the encoder is available, so prefer a
        # cached encoder over paying the resolution cost (and over a fallback).
        for key in PREFERRED_ORDER:
            existing = load_cached_features(key)
            if existing is not None and not force:
                digest = image_set_digest(cases)
                if _cache_is_current(existing, cases, digest):
                    return existing
        encoder_key, reason = resolve_encoder(allow_download=allow_download)
    else:
        if encoder not in ENCODERS:
            raise PerceptionEncodeError(f"unknown encoder {encoder!r}")
        encoder_key = encoder
        reason = f"{encoder} requested explicitly"

    digest = image_set_digest(cases)
    if not force:
        existing = load_cached_features(encoder_key)
        if existing is not None and _cache_is_current(existing, cases, digest):
            return existing

    spec = ENCODERS[encoder_key]
    encode, provenance = load_encoder(encoder_key, allow_download=allow_download)
    provenance["selection_reason"] = reason

    ordered = sorted(cases, key=lambda case: case.case_num)
    effective_batch = int(batch_size or spec.batch_size)
    if effective_batch <= 0:
        raise PerceptionEncodeError("batch_size must be positive")

    matrix: np.ndarray | None = None
    started = time.perf_counter()
    for start in range(0, len(ordered), effective_batch):
        chunk = ordered[start : start + effective_batch]
        block = encode([_image_path(case) for case in chunk])
        if block.ndim != 2 or block.shape[0] != len(chunk):
            raise PerceptionEncodeError(f"encoder returned invalid shape {block.shape}")
        if not np.isfinite(block).all():
            failing = [chunk[i].case_num for i in np.where(~np.isfinite(block).all(axis=1))[0]]
            raise PerceptionEncodeError(f"non-finite features for case_num {failing}")
        norms = np.linalg.norm(block, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise PerceptionEncodeError(
                f"features are not L2-normalised (min {norms.min():.4f}, max {norms.max():.4f})"
            )
        if matrix is None:
            matrix = np.zeros((len(ordered), block.shape[1]), dtype=np.float32)
        elif block.shape[1] != matrix.shape[1]:
            raise PerceptionEncodeError("feature dimension changed mid-run")
        matrix[start : start + len(chunk)] = block
        if verbose:
            done = start + len(chunk)
            elapsed = time.perf_counter() - started
            print(f"[{encoder_key}] {done}/{len(ordered)} images; {elapsed:.1f}s", flush=True)

    assert matrix is not None  # guaranteed: `ordered` is non-empty
    keys = np.asarray([case.case_num for case in ordered], dtype=np.int64)
    manifest = {
        "protocol": CACHE_PROTOCOL,
        "encoder_key": encoder_key,
        "image_view": IMAGE_VIEW,
        "n_cases": len(ordered),
        "feature_dim": int(matrix.shape[1]),
        "l2_normalised": True,
        "batch_size": effective_batch,
        "image_set_sha256": digest,
        "features_sha256": _features_digest(keys, matrix),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "provenance": provenance,
    }

    target = feature_path(encoder_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, case_num=keys, features=matrix)
    os.replace(temp, target)
    meta = manifest_path(encoder_key)
    temp_meta = meta.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_meta, meta)
    return FeatureBundle(encoder_key, keys, matrix, manifest)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="ignore any existing cache")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit fetching a checkpoint that is not in the local cache",
    )
    args = parser.parse_args()

    cases = load_cases()
    bundle = encode_cases(
        cases,
        encoder=args.encoder,
        allow_download=args.allow_download,
        batch_size=args.batch_size,
        force=args.force,
        verbose=True,
    )
    provenance = bundle.manifest.get("provenance", {})
    print(f"encoder      : {bundle.encoder_key} ({provenance.get('model_id')})")
    print(f"reason       : {provenance.get('selection_reason')}")
    print(f"snapshot     : {provenance.get('checkpoint', {}).get('snapshot_revision')}")
    print(f"fingerprint  : {provenance.get('checkpoint', {}).get('combined_sha256')}")
    print(f"features     : {bundle.features.shape} -> {feature_path(bundle.encoder_key)}")


if __name__ == "__main__":  # pragma: no cover
    _main()
