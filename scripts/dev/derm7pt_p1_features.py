"""Phase-1 perception-uplift EXTRA feature caches for Derm7pt (L3 TTA, L5 zero-shot text).

Everything ``medguard_v2/perception/encode.py`` and ``encode_multi.py`` already produce
(``dermlip``, ``dermlip_clinic``, ``biomedclip_derm``, ``biomedclip_clinic``) is read-only input
here and is never rewritten. This script adds exactly the two feature families the phase-1 brief
asks for that do not exist yet:

* **L3 (TTA)** -- horizontal-flip feature averaging. For each of the ``dermlip`` views (derm and
  clinic), the flipped image is encoded (a genuinely new forward pass -- flipping is not a
  post-hoc transform of the cached vector) and averaged with the already-cached un-flipped vector,
  then re-normalised to unit length (the mean of two unit vectors is not itself unit length).
* **L5 (zero-shot text tower)** -- DermLIP is CLIP-style, so its text tower can score class-name
  prompts against the *already-cached* dermoscopic image vectors with zero new image forward
  passes. Verified interactively before writing this script: ``open_clip.get_tokenizer`` and
  ``model.encode_text`` both work fully offline (``HF_HUB_OFFLINE=1``) on the cached DermLIP
  snapshot, so this lever is implemented rather than skipped.

Everything new is written under ``research/evidence/derm7pt_v1/features_phase1/`` -- a directory
``medguard_v2/eval/gates/g6_multiview_stacking.py``'s ``discover_feature_sets()`` never scans
(it is hardcoded to ``research/evidence/derm7pt_v1/features/``). That is deliberate: dropping new
feature sets into the directory G6 auto-discovers would silently change what a future real run of
that frozen gate does. Phase-1 code opts into using these caches explicitly instead.

No file under ``medguard_v2/`` is imported for its *side effects* -- only pure functions/constants
(``ENCODERS``, ``encoder_snapshot``, ``checkpoint_fingerprint``, ``load_cached_features``, ...) are
reused, and nothing there is edited. Phase 1 is experimentation-only.

CLI::

    venv_win/Scripts/python.exe scripts/dev/derm7pt_p1_features.py            # build (cached)
    venv_win/Scripts/python.exe scripts/dev/derm7pt_p1_features.py --force    # ignore caches
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.data.derm7pt import Derm7ptCase, load_cases  # noqa: E402
from medguard_v2.eval.gates.g6_multiview_stacking import discover_feature_sets  # noqa: E402
from medguard_v2.perception import encode as _encode  # noqa: E402
from medguard_v2.perception import encode_multi as _encode_multi  # noqa: E402
from medguard_v2.perception.encode import (  # noqa: E402
    ENCODERS,
    PerceptionEncodeError,
    checkpoint_fingerprint,
    encoder_snapshot,
)

FEATURE_ROOT = REPO / "research" / "evidence" / "derm7pt_v1" / "features"
FEATURE_ROOT_P1 = REPO / "research" / "evidence" / "derm7pt_v1" / "features_phase1"

SEED = 20260810

#: view key -> (base feature-set key to pair with, Derm7ptCase attribute holding the image path)
TTA_VIEWS: Mapping[str, tuple[str, str]] = {
    "dermlip_flip": ("dermlip", "derm_path"),
    "dermlip_clinic_flip": ("dermlip_clinic", "clinic_path"),
}

#: Class-name prompts for the zero-shot text tower (L5). Recorded verbatim in the manifest so the
#: exact wording used is auditable -- prompt phrasing materially changes zero-shot logits.
ZEROSHOT_PROMPT_TEMPLATE = "a dermoscopic image of {text}"
ZEROSHOT_CLASS_TEXT: Mapping[str, str] = {
    "BCC": "basal cell carcinoma",
    "NEV": "a benign melanocytic nevus",
    "MEL": "melanoma",
    "SK": "seborrheic keratosis",
    "MISC": "a miscellaneous benign lesion such as a dermatofibroma or lentigo",
}
#: Canonical column order for the 5-d zero-shot logit vector (matches DERM7PT_CLASS_ORDER).
ZEROSHOT_CLASS_ORDER = ("BCC", "NEV", "MEL", "SK", "MISC")


def _sha256_file(path: Path) -> str:
    return _encode._sha256_file(path)


def _features_digest(case_nums: np.ndarray, features: np.ndarray) -> str:
    return _encode._features_digest(case_nums, features)


def _write_bundle(directory: Path, case_nums: np.ndarray, features: np.ndarray, manifest: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "features.npz"
    temp = target.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, case_num=case_nums, features=features)
    os.replace(temp, target)
    meta = directory / "feature_manifest.json"
    temp_meta = meta.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_meta, meta)


def _load_npz(directory: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    npz = directory / "features.npz"
    meta = directory / "feature_manifest.json"
    if not npz.is_file() or not meta.is_file():
        return None
    manifest = json.loads(meta.read_text(encoding="utf-8"))
    with np.load(npz) as payload:
        case_nums = np.asarray(payload["case_num"], dtype=np.int64)
        features = np.asarray(payload["features"], dtype=np.float32)
    observed = _features_digest(case_nums, features)
    if manifest.get("features_sha256") != observed:
        raise PerceptionEncodeError(f"corrupt phase1 cache {npz}: digest mismatch")
    return case_nums, features, manifest


# --------------------------------------------------------------------------------- L3: TTA flip


def _flip_encoder(encoder_key: str, *, allow_download: bool = False):
    """Mirrors ``encode.load_encoder`` exactly, except the image is horizontally flipped before
    the (unchanged, deterministic Resize/CenterCrop/Normalize) preprocessing pipeline. Verified
    deterministic -- no random crop/augmentation in DermLIP's preprocess Compose -- so flipping
    the raw image first is a faithful horizontal-flip TTA view, not a hack.
    """
    import open_clip
    import torch
    from PIL import Image, ImageOps

    spec = ENCODERS[encoder_key]
    if spec.loader != "open_clip":
        raise PerceptionEncodeError(f"unsupported loader {spec.loader!r} for {encoder_key!r}")

    snapshot = encoder_snapshot(encoder_key, allow_download=allow_download)
    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{spec.model_id}")
    model.eval()

    def encode_flipped(paths: Sequence[Path]) -> np.ndarray:
        batch = torch.stack(
            [preprocess(ImageOps.mirror(Image.open(path).convert("RGB"))) for path in paths]
        )
        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype("float32")

    provenance = {
        "encoder_key": encoder_key,
        "model_id": spec.model_id,
        "resolved_model_id": f"hf-hub:{spec.model_id}",
        "loader": spec.loader,
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "trainable": False,
        "augmentation": "horizontal_flip_before_preprocess (PIL.ImageOps.mirror)",
        "checkpoint": checkpoint_fingerprint(snapshot),
    }
    return encode_flipped, provenance


def build_tta_flip(cases: Sequence[Derm7ptCase], *, force: bool = False, verbose: bool = True) -> dict:
    """Encode the flipped image for each of the two DermLIP views and cache it.

    Returns ``{view_key: (case_nums, features, manifest)}`` for the FLIP-ONLY features (not yet
    averaged with the original -- that happens in :func:`build_tta_averaged`).
    """
    out: dict = {}
    ordered = sorted(cases, key=lambda c: c.case_num)
    keys = np.asarray([c.case_num for c in ordered], dtype=np.int64)

    for flip_key, (base_key, attr) in TTA_VIEWS.items():
        directory = FEATURE_ROOT_P1 / flip_key
        base_manifest_path = FEATURE_ROOT / base_key / "feature_manifest.json"
        if not base_manifest_path.is_file():
            raise PerceptionEncodeError(f"base feature set {base_key!r} is not cached; run encode_multi first")
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        base_digest = base_manifest["image_set_sha256"]

        if not force:
            cached = _load_npz(directory)
            if cached is not None:
                c_nums, feats, manifest = cached
                if (
                    manifest.get("protocol") == "derm7pt-frozen-features-tta-flip-v1"
                    and manifest.get("paired_with_base") == base_key
                    and manifest.get("base_image_set_sha256") == base_digest
                    and manifest.get("n_cases") == len(ordered)
                    and np.array_equal(c_nums, keys)
                ):
                    if verbose:
                        print(f"[{flip_key}] cache hit ({len(c_nums)} cases)")
                    out[flip_key] = (c_nums, feats, manifest)
                    continue

        encode_fn, provenance = _flip_encoder("dermlip")
        paths = [getattr(c, attr) for c in ordered]
        missing = [f"case_num {c.case_num}" for c, p in zip(ordered, paths) if not p.is_file()]
        if missing:
            raise PerceptionEncodeError(f"{flip_key}: {len(missing)} image(s) missing on disk")

        matrix = np.zeros((len(ordered), 512), dtype=np.float32)
        batch_size = 32
        started = time.perf_counter()
        for start in range(0, len(ordered), batch_size):
            chunk_paths = paths[start : start + batch_size]
            block = encode_fn(chunk_paths)
            if not np.isfinite(block).all():
                raise PerceptionEncodeError(f"{flip_key}: non-finite features in batch at {start}")
            norms = np.linalg.norm(block, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise PerceptionEncodeError(f"{flip_key}: flipped features not L2-normalised")
            matrix[start : start + len(chunk_paths)] = block
            if verbose:
                done = start + len(chunk_paths)
                print(f"[{flip_key}] {done}/{len(ordered)} ({time.perf_counter()-started:.1f}s)", flush=True)

        manifest = {
            "protocol": "derm7pt-frozen-features-tta-flip-v1",
            "feature_set_key": flip_key,
            "paired_with_base": base_key,
            "base_image_set_sha256": base_digest,
            "image_view": _encode_multi.FEATURE_SETS[base_key].image_view,
            "n_cases": len(ordered),
            "feature_dim": int(matrix.shape[1]),
            "l2_normalised": True,
            "features_sha256": _features_digest(keys, matrix),
            "note": (
                "horizontal-flip-only encode of the same images as "
                f"features/{base_key}/features.npz; NOT yet averaged with the original -- see "
                f"{flip_key.replace('_flip','_tta')} for the TTA-combined, renormalised vector"
            ),
            "provenance": provenance,
        }
        _write_bundle(directory, keys, matrix, manifest)
        out[flip_key] = (keys, matrix, manifest)
        if verbose:
            print(f"[{flip_key}] wrote {directory}")
    return out


def build_tta_averaged(cases: Sequence[Derm7ptCase], flip_bundles: dict, *, force: bool = False, verbose: bool = True) -> dict:
    """Average each base view's cached vector with its cached flip vector, renormalise, cache."""
    ordered = sorted(cases, key=lambda c: c.case_num)
    keys = np.asarray([c.case_num for c in ordered], dtype=np.int64)
    out: dict = {}

    for flip_key, (base_key, _attr) in TTA_VIEWS.items():
        tta_key = flip_key.replace("_flip", "_tta")
        directory = FEATURE_ROOT_P1 / tta_key

        if base_key == "dermlip":
            base_bundle = _encode.load_cached_features("dermlip")
        else:
            base_bundle = _encode_multi.load_feature_set(base_key)
        if base_bundle is None:
            raise PerceptionEncodeError(f"base feature set {base_key!r} is not cached")
        base_features = base_bundle.matrix_for(keys.tolist())

        flip_keys, flip_features, flip_manifest = flip_bundles[flip_key]
        if not np.array_equal(flip_keys, keys):
            raise PerceptionEncodeError(f"{flip_key}: case_num order mismatch against base")

        if not force:
            cached = _load_npz(directory)
            if cached is not None:
                c_nums, feats, manifest = cached
                if (
                    manifest.get("protocol") == "derm7pt-frozen-features-tta-v1"
                    and manifest.get("base_features_sha256") == base_bundle.manifest.get("features_sha256")
                    and manifest.get("flip_features_sha256") == flip_manifest.get("features_sha256")
                    and np.array_equal(c_nums, keys)
                ):
                    if verbose:
                        print(f"[{tta_key}] cache hit ({len(c_nums)} cases)")
                    out[tta_key] = (c_nums, feats, manifest)
                    continue

        averaged = (base_features.astype(np.float64) + flip_features.astype(np.float64)) / 2.0
        norms = np.linalg.norm(averaged, axis=1, keepdims=True)
        renormalised = (averaged / np.clip(norms, 1e-12, None)).astype(np.float32)
        if not np.isfinite(renormalised).all():
            raise PerceptionEncodeError(f"{tta_key}: non-finite TTA-averaged features")

        manifest = {
            "protocol": "derm7pt-frozen-features-tta-v1",
            "feature_set_key": tta_key,
            "base_feature_set": base_key,
            "flip_feature_set": flip_key,
            "base_features_sha256": base_bundle.manifest.get("features_sha256"),
            "flip_features_sha256": flip_manifest.get("features_sha256"),
            "n_cases": len(ordered),
            "feature_dim": int(renormalised.shape[1]),
            "l2_normalised": True,
            "combination": "L2_normalize((base + flip) / 2)",
            "features_sha256": _features_digest(keys, renormalised),
            "note": "L3 lever: horizontal-flip TTA, feature-vector averaging, renormalised to unit length",
        }
        _write_bundle(directory, keys, renormalised, manifest)
        out[tta_key] = (keys, renormalised, manifest)
        if verbose:
            print(f"[{tta_key}] wrote {directory}")
    return out


# ------------------------------------------------------------------------ L5: zero-shot text tower


def zero_shot_text_tower_available() -> tuple[bool, str]:
    """Feasibility check, run before anything is built (mirrors the brief's instruction to report
    and skip rather than hack around it if the text tower is not reachable offline)."""
    try:
        import open_clip

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        spec = ENCODERS["dermlip"]
        model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:{spec.model_id}")
        if not hasattr(model, "encode_text"):
            return False, "loaded CLIP model has no encode_text method"
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{spec.model_id}")
        import torch

        toks = tokenizer(["melanoma"])
        with torch.inference_mode():
            feats = model.encode_text(toks)
        if not torch.isfinite(feats).all():
            return False, "encode_text produced non-finite output"
        return True, "open_clip.get_tokenizer + model.encode_text both work offline on the cached DermLIP snapshot"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def build_zero_shot_logits(cases: Sequence[Derm7ptCase], *, force: bool = False, verbose: bool = True) -> dict | None:
    ok, reason = zero_shot_text_tower_available()
    if not ok:
        print(f"[dermlip_zeroshot5] SKIPPED: text tower not accessible ({reason})")
        return None

    directory = FEATURE_ROOT_P1 / "dermlip_zeroshot5"
    ordered = sorted(cases, key=lambda c: c.case_num)
    keys = np.asarray([c.case_num for c in ordered], dtype=np.int64)

    image_bundle = _encode.load_cached_features("dermlip")
    if image_bundle is None:
        raise PerceptionEncodeError("dermlip (derm view) is not cached; run encode.py first")
    image_features = image_bundle.matrix_for(keys.tolist())

    if not force:
        cached = _load_npz(directory)
        if cached is not None:
            c_nums, feats, manifest = cached
            if (
                manifest.get("protocol") == "derm7pt-zeroshot-text-logits-v1"
                and manifest.get("image_features_sha256") == image_bundle.manifest.get("features_sha256")
                and np.array_equal(c_nums, keys)
            ):
                if verbose:
                    print(f"[dermlip_zeroshot5] cache hit ({len(c_nums)} cases)")
                return {"case_num": c_nums, "features": feats, "manifest": manifest}

    import open_clip
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    spec = ENCODERS["dermlip"]
    snapshot = encoder_snapshot("dermlip")
    model, _, _preprocess = open_clip.create_model_and_transforms(f"hf-hub:{spec.model_id}")
    model.eval()
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{spec.model_id}")

    prompts = [ZEROSHOT_PROMPT_TEMPLATE.format(text=ZEROSHOT_CLASS_TEXT[c]) for c in ZEROSHOT_CLASS_ORDER]
    tokens = tokenizer(prompts)
    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        has_logit_scale = hasattr(model, "logit_scale")
        scale = float(model.logit_scale.exp().item()) if has_logit_scale else 1.0
    text_features = text_features.cpu().numpy().astype(np.float64)  # (5, 512)

    cosine = image_features.astype(np.float64) @ text_features.T  # (n, 5)
    logits = (cosine * scale).astype(np.float32)
    if not np.isfinite(logits).all():
        raise PerceptionEncodeError("zero-shot logits contain non-finite values")

    manifest = {
        "protocol": "derm7pt-zeroshot-text-logits-v1",
        "feature_set_key": "dermlip_zeroshot5",
        "class_order": list(ZEROSHOT_CLASS_ORDER),
        "prompt_template": ZEROSHOT_PROMPT_TEMPLATE,
        "prompts": prompts,
        "logit_scale_used": scale,
        "logit_scale_source": "model.logit_scale.exp()" if has_logit_scale else "no logit_scale attribute; scale fixed to 1.0",
        "image_features_source": "features/dermlip/features.npz (derm view, already cached; no new image forward pass)",
        "image_features_sha256": image_bundle.manifest.get("features_sha256"),
        "n_cases": len(ordered),
        "feature_dim": 5,
        "l2_normalised": False,
        "note": "cosine(image, text-prompt) x logit_scale, per class; NOT a probability, an auxiliary raw score",
        "features_sha256": _features_digest(keys, logits),
        "tower_note": "text tower only run here; the image tower's cached dermlip vectors are reused unchanged",
    }
    _write_bundle(directory, keys, logits, manifest)
    if verbose:
        print(f"[dermlip_zeroshot5] wrote {directory} (prompts={prompts})")
    return {"case_num": keys, "features": logits, "manifest": manifest}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("existing cached feature sets:", sorted(discover_feature_sets()))
    cases = load_cases()

    flip_bundles = build_tta_flip(cases, force=args.force)
    build_tta_averaged(cases, flip_bundles, force=args.force)
    build_zero_shot_logits(cases, force=args.force)

    print("\nphase1 feature caches now present:")
    for child in sorted(FEATURE_ROOT_P1.iterdir()):
        if (child / "features.npz").is_file():
            print(" ", child.name)


if __name__ == "__main__":
    _main()
