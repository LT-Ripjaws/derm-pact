"""Freeze the frozen DermLIP probe's commitments over a held-out partition.

The development perception artifact covers 160 calibration lesions and its receipt
asserts ``test_records_loaded == 0``. The sealed evaluation needs the same artifact for
the 1,495 test lesions, produced by the SAME frozen probe, temperature and actionable
threshold -- nothing is refit here, and nothing may be.

    python freeze_sealed_perception.py --split calibration --out <dir>   # rehearsal
    python freeze_sealed_perception.py --split test       --out <dir>    # SEALED

``--split calibration`` is the dress rehearsal: it exercises the whole path on a
partition the preregistration already designates non-confirmatory, and its DermLIP
features are already cached, so it costs nothing and touches nothing sealed.

``--split test`` opens the one-shot guard under ``sealed_perception_freeze``, logs the
access, and (because test images were never encoded) runs the frozen encoder over
1,991 images. That is offline and unpaid, but it IS the sealed partition, so it happens
once and is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT_MANIFEST = ROOT / "research/manifests/ham10000_split_v2.json"
FEATURE_DIR = ROOT / "research/experiments/derm_perception/v2_tournament/features/dermlip"
PROBE = ROOT / ("research/experiments/derm_perception/v4_nested_lesion_tournament/"
                "results/dermlip/train_only_probes.joblib")
SELECTION = ROOT / ("research/experiments/derm_perception/v4_nested_lesion_tournament/"
                    "results/dermlip/result.json")
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc")


def _locator(path: Path) -> str:
    """Repo-relative when inside the repo, absolute otherwise.

    A rehearsal writes to a scratch directory outside the tree; refusing to record a
    path just because it is external would make the rehearsal artifact unreceiptable.
    """
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_split(split: str) -> List[dict]:
    data = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    rows = [r for r in data["records"] if r.get("split") == split]
    if not rows:
        raise SystemExit(f"no rows for split {split!r}")
    return sorted(rows, key=lambda r: str(r["case_id"]))


def _features_for(rows: List[dict], split: str):
    """Reuse cached DermLIP features when the partition is already encoded.

    train and calibration are in the existing cache; test is not and must be encoded.
    Encoding is deterministic and uses the same frozen checkpoint, so a rehearsal on
    calibration genuinely exercises the same arithmetic the sealed run will.
    """
    import numpy as np

    manifest = json.loads((FEATURE_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    included = set(manifest.get("included_splits") or ())
    if split in included:
        parent_path = Path(str(manifest["development_manifest"]))
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        index = {str(r["case_id"]): i for i, r in enumerate(parent["records"])}
        features = np.load(FEATURE_DIR / "features.npy", mmap_mode="r")
        idx = []
        for row in rows:
            key = str(row["case_id"])
            if key not in index:
                raise SystemExit(f"cached features do not contain {key}")
            idx.append(index[key])
        return np.asarray(features[np.asarray(idx, dtype=int)]), "cached", manifest
    return _encode(rows), "encoded", manifest


def _encode(rows: List[dict]):
    """Run the frozen DermLIP encoder over images that were never cached."""
    import numpy as np
    import torch
    from PIL import Image
    import open_clip

    model_id = "hf-hub:redlessone/DermLIP_ViT-B-16"
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    model.eval()
    out = np.zeros((len(rows), 512), dtype=np.float32)
    batch, positions = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            feats = model.encode_image(torch.stack(batch))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for pos, vec in zip(positions, feats.cpu().numpy()):
            out[pos] = vec
        batch.clear()
        positions.clear()

    for i, row in enumerate(rows):
        image = Image.open(str(row["image_path"])).convert("RGB")
        batch.append(preprocess(image))
        positions.append(i)
        if len(batch) == 32:
            flush()
            print(f"  encoded {i + 1}/{len(rows)}", flush=True)
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["calibration", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override-token", default=None,
                    help="required to re-run a spent sealed purpose")
    args = ap.parse_args()

    import joblib
    import numpy as np
    from medguard.eval.derm_encoder_tournament import _softmax

    rows = _load_split(args.split)
    lesions = sorted({str(r["cluster_id"]) for r in rows})
    print(f"[{args.split}] {len(rows)} images over {len(lesions)} lesions")

    grant = None
    if args.split == "test":
        # The encoder path has never executed: the calibration rehearsal reused cached
        # features. Exercise it on a NON-sealed image first, so a broken encoder is
        # discovered before the one-shot guard is spent rather than after.
        print("[preflight] exercising the encoder on one calibration image "
              "(non-sealed) before opening the guard...")
        probe_rows = _load_split("calibration")[:1]
        probe_features = _encode(probe_rows)
        if probe_features.shape != (1, 512):
            raise SystemExit(f"encoder preflight produced {probe_features.shape}, "
                             f"expected (1, 512); refusing to open sealed access")
        import numpy as _np
        norm = float(_np.linalg.norm(probe_features[0]))
        if not (0.99 <= norm <= 1.01):
            raise SystemExit(f"encoder preflight produced a non-unit embedding "
                             f"(norm {norm:.4f}); refusing to open sealed access")
        print(f"[preflight] OK -- shape {probe_features.shape}, unit norm {norm:.4f}")

        from medguard.eval.sealed_access import open_sealed_access
        grant = open_sealed_access(
            "sealed_perception_freeze",
            detail=(f"freeze frozen-DermLIP commitments over the sealed test partition "
                    f"({len(rows)} images / {len(lesions)} lesions); no refitting, no "
                    f"labels read, no paid calls"),
            override_token=args.override_token,
            extra={"images": len(rows), "lesions": len(lesions)},
        ).as_dict()
        print(f"[sealed] access logged: seq={grant['sequence']} {grant['entry_sha256'][:16]}")

    features, provenance, feature_manifest = _features_for(rows, args.split)
    print(f"[features] {provenance}: {features.shape}")

    # joblib.load unpickles, which executes arbitrary code if the file is untrusted.
    # This one is repo-local and produced by our own frozen training pipeline, and it is
    # hash-pinned in the DEVELOPMENT perception receipt -- so verify it against that pin
    # before unpickling. If the probe artifact ever changes, this fails closed instead of
    # silently scoring the sealed set with a different model than development used.
    dev_receipt = json.loads(
        (ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1/receipt.json")
        .read_text(encoding="utf-8"))
    pinned = next((i["sha256"] for i in dev_receipt["inputs"]
                   if str(i["path"]).replace("\\", "/").endswith("train_only_probes.joblib")),
                  None)
    if pinned is None:
        raise SystemExit("development receipt does not pin the train-only probe")
    actual = _sha256_file(PROBE)
    if actual != pinned:
        raise SystemExit(
            f"probe artifact hash {actual[:16]} does not match the development pin "
            f"{pinned[:16]}; refusing to unpickle. The sealed run must use the SAME "
            f"frozen probe development used."
        )
    probes = joblib.load(PROBE)
    classes = [str(v) for v in probes["diagnosis_classes"]]
    if tuple(classes) != CLASS_ORDER:
        raise SystemExit("probe class order differs from the registered class order")
    diagnostic = probes["diagnosis"]
    order = [list(map(str, diagnostic.classes_)).index(c) for c in classes]
    temperature = float(probes["temperature"])
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    actionable_threshold = float(
        selection["operating_point"]["specificity_90"]["threshold"])

    probabilities = _softmax(diagnostic.decision_function(features)[:, order], temperature)
    actionable_raw = probes["malignant"].decision_function(features)
    actionable = probes["platt"].predict_proba(actionable_raw[:, None])[:, 1]

    by_lesion: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        by_lesion.setdefault(str(row["cluster_id"]), []).append(i)

    out_rows = []
    for cluster_id in lesions:
        idx = by_lesion[cluster_id]
        mean_probs = probabilities[idx].mean(axis=0)
        mean_probs = mean_probs / mean_probs.sum()
        score = float(actionable[idx].mean())
        top = int(np.argmax(mean_probs))
        out_rows.append({
            "actionable_prediction": bool(score >= actionable_threshold),
            "actionable_score": score,
            "calibrated_confidence": float(mean_probs[top]),
            "case_id": f"lesion:{cluster_id}",
            "cluster_id": cluster_id,
            "committed_diagnosis": classes[top],
            "diagnosis_probabilities": [float(p) for p in mean_probs],
            "view_case_ids": [str(rows[i]["case_id"]) for i in idx],
            "view_count": len(idx),
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = out_dir / "predictions.jsonl"
    payload = "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for r in out_rows).encode("utf-8")
    predictions.write_bytes(payload)

    receipt = {
        "schema_version": "medguard-h2h-perception-receipt-v1",
        "protocol_id": "medguard-derm-evidence-contract-h2h-v1",
        "partition": args.split,
        "model_id": "redlessone/DermLIP_ViT-B-16",
        "class_order": list(CLASS_ORDER),
        "diagnosis_temperature": temperature,
        "actionable_threshold": actionable_threshold,
        "nothing_refit": ("probe, temperature and actionable threshold are loaded from "
                          "the frozen train-only artifacts; this script fits nothing"),
        "feature_provenance": provenance,
        "feature_manifest_sha256": _sha256_file(FEATURE_DIR / "feature_manifest.json"),
        "probe_sha256": _sha256_file(PROBE),
        "selection_result_sha256": _sha256_file(SELECTION),
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "lesion_outputs": len(out_rows),
        "source_images": len(rows),
        # The sealed artifact is the mirror image of the development one: the dev
        # receipt asserts test_records_loaded == 0, this asserts it is the partition size.
        "test_records_loaded": len(rows) if args.split == "test" else 0,
        "truth_fields_present": False,
        "encoder_calls": 0 if provenance == "cached" else len(rows),
        "generation_calls": 0,
        "sealed_access_grant": grant,
        "outputs": [{"path": _locator(predictions),
                     "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}],
    }
    receipt_path = out_dir / "receipt.json"
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True,
                               ensure_ascii=False).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    (out_dir / "receipt.json.sha256").write_text(
        hashlib.sha256(receipt_bytes).hexdigest(), encoding="ascii")

    import collections
    counts = collections.Counter(r["committed_diagnosis"] for r in out_rows)
    print(f"[out] {len(out_rows)} lesions -> {predictions}")
    print(f"      committed class counts: {dict(counts)}")
    print(f"      receipt sha256: {hashlib.sha256(receipt_bytes).hexdigest()[:16]}")
    print("      NOTE: no labels were read; this artifact is truth-redacted.")


if __name__ == "__main__":
    main()
