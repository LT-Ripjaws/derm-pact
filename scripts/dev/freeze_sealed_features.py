"""Freeze DermLIP QUERY features for the sealed partition.

Discovered by the sealed pipeline smoke test: the image-exemplar channel encodes the
patient's own images to find similar train exemplars, and it does so through
``FrozenDermLIPFeatureEncoder`` -- a LOOKUP TABLE over the train+calibration feature
cache that explicitly rejects any other split. Sealed query images are not in it, so
exemplar retrieval fails on the first sealed lesion.

The perception freeze already computed these embeddings and discarded them. This script
recomputes and PERSISTS them as a separate, hash-receipted sealed feature artifact, so
the sealed run reads frozen features exactly as the development run does rather than
encoding live mid-evaluation.

Nothing about the model changes: same frozen checkpoint, same deterministic encode path
the perception freeze used and the calibration rehearsal validated.

    python freeze_sealed_features.py --out research/experiments/medguard_h2h_v1/features/dermlip_sealed_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT_MANIFEST = ROOT / "research/manifests/ham10000_split_v2.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "calibration"])
    ap.add_argument("--override-token", default=None)
    args = ap.parse_args()

    import numpy as np
    from medguard.eval.sealed_access import open_sealed_access

    data = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    rows = sorted((r for r in data["records"] if r.get("split") == args.split),
                  key=lambda r: str(r["case_id"]))
    lesions = {str(r["cluster_id"]) for r in rows}
    print(f"[{args.split}] {len(rows)} images over {len(lesions)} lesions")

    grant = None
    if args.split == "test":
        grant = open_sealed_access(
            "sealed_feature_freeze",
            detail=(f"persist DermLIP query features for the sealed partition "
                    f"({len(rows)} images); required by the image-exemplar channel, "
                    f"which reads frozen features and cannot encode sealed images. No "
                    f"labels read, no paid calls, model unchanged."),
            override_token=args.override_token,
            extra={"images": len(rows), "lesions": len(lesions)},
        ).as_dict()
        print(f"[sealed] access logged seq={grant['sequence']} {grant['entry_sha256'][:16]}")

    # Reuse the freezer's encode path so sealed features are produced by exactly the
    # same code that produced the sealed commitments.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_fsp", str(ROOT / "scripts/dev/freeze_sealed_perception.py"))
    fsp = importlib.util.module_from_spec(spec)
    sys.argv = ["freeze_sealed_perception.py"]
    spec.loader.exec_module(fsp)

    features = fsp._encode(rows)
    if features.shape != (len(rows), 512):
        raise SystemExit(f"encoder produced {features.shape}, expected ({len(rows)}, 512)")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit(f"non-unit embeddings: norm range [{norms.min():.4f}, {norms.max():.4f}]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_file = out_dir / "features.npy"
    np.save(feature_file, features.astype("float32"))

    manifest = {
        "schema_version": "medguard-sealed-feature-manifest-v1",
        "model_id": "redlessone/DermLIP_ViT-B-16",
        "split": args.split,
        "n_records": len(rows),
        "feature_dim": 512,
        "feature_file_sha256": _sha256(feature_file),
        "split_manifest_sha256": _sha256(SPLIT_MANIFEST),
        "image_paths": [str(r["image_path"]) for r in rows],
        "image_sha256s": [str(r["image_sha256"]) for r in rows],
        "case_ids": [str(r["case_id"]) for r in rows],
        "truth_fields_present": False,
        "generation_calls": 0,
        "sealed_access_grant": grant,
        "note": ("Query features for the sealed partition. The exemplar ATLAS remains "
                 "train-only; these are only the patient-side query embeddings."),
    }
    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[out] {features.shape} -> {feature_file}")
    print(f"      manifest sha256 {_sha256(manifest_path)[:16]}")


if __name__ == "__main__":
    main()
