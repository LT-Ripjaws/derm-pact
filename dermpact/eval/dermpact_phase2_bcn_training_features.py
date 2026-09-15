"""Derive the Phase 2 BCN-training feature matrix without opening protected rows."""
from __future__ import annotations

import argparse
import json
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_baselines import sha256_file


ROOT = screen.ROOT
BCN_MANIFEST = ROOT / "research/manifests/bcn20000_external_v1.json"
BCN_SPLIT = ROOT / "research/manifests/bcn20000_lesion_split_v2.json"
BCN_TWO_ARM = ROOT / "research/manifests/dermpact_phase1_bcn_two_arm_v1.json"
SOURCE_DIR = (
    ROOT
    / "research/experiments/derm_perception/v2_tournament"
    / "external_bcn_v1/features/panderm"
)
SOURCE_FEATURES = SOURCE_DIR / "features.npy"
SOURCE_MANIFEST = SOURCE_DIR / "feature_manifest.json"
OUTPUT_DIR = (
    ROOT
    / "research/experiments/dermpact_phase2_2/dataset_composition_v1"
    / "features/panderm/bcn_train"
)
OUTPUT_FEATURES = OUTPUT_DIR / "features.npy"
OUTPUT_MANIFEST = OUTPUT_DIR / "feature_manifest.json"
AUDIT_DIR = (
    ROOT
    / "research/manifests/dermpact_phase2_2/dataset_composition_v1"
)
PROTOCOL_PATH = AUDIT_DIR / "bcn_training_feature_derivation_protocol.json"
RECEIPT_PATH = AUDIT_DIR / "bcn_training_feature_derivation_receipt.json"

EXPECTED_SOURCE_SHAPE = (12356, 512)
EXPECTED_TRAIN_IMAGES = 7405
EXPECTED_TRAIN_LESIONS = 2143
EXPECTED_EXCLUDED = {"validation_calibration": 2441, "final_test": 2510}
CANDIDATE = "panderm"
MODEL_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    return screen._canonical_hash(value)


def _role_for_source_split(value: str) -> str:
    mapping = {
        "train": "train",
        "valid": "validation_calibration",
        "test": "final_test",
    }
    if value not in mapping:
        raise ValueError(f"unknown frozen BCN split: {value!r}")
    return mapping[value]


def _source_and_training_rows() -> tuple[list[dict[str, Any]], list[int]]:
    manifest = _load_json(BCN_MANIFEST)
    split = _load_json(BCN_SPLIT)
    records = list(manifest["records"])
    lesion_split = {
        str(name): str(value)
        for name, value in split["lesion_split"].items()
    }
    if len(records) != EXPECTED_SOURCE_SHAPE[0]:
        raise AssertionError("BCN source record count changed")
    missing = {
        str(row["cluster_id"]) for row in records
    } - set(lesion_split)
    if missing:
        raise AssertionError(f"BCN lesions missing frozen roles: {len(missing)}")
    role_counts = Counter(
        _role_for_source_split(lesion_split[str(row["cluster_id"])])
        for row in records
    )
    if role_counts != Counter(
        {
            "train": EXPECTED_TRAIN_IMAGES,
            **EXPECTED_EXCLUDED,
        }
    ):
        raise AssertionError(f"BCN image role counts changed: {role_counts}")
    positions = [
        index
        for index, row in enumerate(records)
        if lesion_split[str(row["cluster_id"])] == "train"
    ]
    training_rows = [records[index] for index in positions]
    if len(training_rows) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError("BCN training image count changed")
    if len({str(row["cluster_id"]) for row in training_rows}) != EXPECTED_TRAIN_LESIONS:
        raise AssertionError("BCN training lesion count changed")
    labels_by_lesion: dict[str, set[str]] = defaultdict(set)
    for row in training_rows:
        labels_by_lesion[str(row["cluster_id"])].add(str(row["label"]))
    conflicts = [lesion for lesion, labels in labels_by_lesion.items() if len(labels) != 1]
    if conflicts:
        raise AssertionError(f"BCN training lesions have conflicting labels: {conflicts[:3]}")
    return training_rows, positions


def _identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "case_id": str(row["case_id"]),
            "cluster_id": str(row["cluster_id"]),
            "image_sha256": str(row["image_sha256"]),
            "label": str(row["label"]),
        }
        for row in rows
    ]


def build_protocol() -> dict[str, Any]:
    source_manifest = _load_json(SOURCE_MANIFEST)
    if source_manifest["model_key"] != CANDIDATE:
        raise AssertionError("sealed BCN feature candidate changed")
    if source_manifest["model_id"] != MODEL_ID:
        raise AssertionError("sealed BCN feature model changed")
    if source_manifest["external_manifest_sha256"] != sha256_file(BCN_MANIFEST):
        raise AssertionError("sealed BCN features use a different source manifest")
    if source_manifest["feature_file_sha256"] != sha256_file(SOURCE_FEATURES):
        raise AssertionError("sealed BCN feature matrix hash mismatch")
    source = np.load(SOURCE_FEATURES, mmap_mode="r")
    if source.shape != EXPECTED_SOURCE_SHAPE:
        raise AssertionError(f"sealed BCN feature shape changed: {source.shape}")
    rows, positions = _source_and_training_rows()
    protocol = {
        "protocol": "dermpact-phase2-bcn-training-feature-derivation-v1",
        "status": "PRESPECIFIED_BEFORE_TRAINING_ROW_DERIVATION",
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "method": "row-exact copy from sealed historical matrix",
        "permitted_role": "BCN training only",
        "row_contract": {
            "source_rows": EXPECTED_SOURCE_SHAPE[0],
            "selected_training_rows": len(rows),
            "selected_training_lesions": len(
                {str(row["cluster_id"]) for row in rows}
            ),
            "first_selected_source_index": int(positions[0]),
            "last_selected_source_index": int(positions[-1]),
            "selected_source_positions_sha256": _canonical_hash(positions),
            "record_identity_sha256": _canonical_hash(_identity(rows)),
        },
        "hard_scope": {
            "validation_calibration_rows_indexed_or_copied": 0,
            "final_test_rows_indexed_or_copied": 0,
            "predictions_generated": 0,
            "metrics_computed": 0,
            "source_matrix_hashing_is_integrity_only": True,
        },
        "artifacts": {
            "bcn_manifest": {
                "path": str(BCN_MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(BCN_MANIFEST),
            },
            "bcn_split": {
                "path": str(BCN_SPLIT.relative_to(ROOT)),
                "sha256": sha256_file(BCN_SPLIT),
            },
            "bcn_two_arm_contract": {
                "path": str(BCN_TWO_ARM.relative_to(ROOT)),
                "sha256": sha256_file(BCN_TWO_ARM),
            },
            "source_feature_manifest": {
                "path": str(SOURCE_MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(SOURCE_MANIFEST),
            },
            "source_feature_matrix": {
                "path": str(SOURCE_FEATURES.relative_to(ROOT)),
                "sha256": sha256_file(SOURCE_FEATURES),
            },
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    protocol_hash = screen._write_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": protocol_hash}


def _verify_protocol(protocol_sha256: str) -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != protocol_sha256:
        raise AssertionError("BCN training-feature protocol hash mismatch")
    if PROTOCOL_PATH.with_suffix(".json.sha256").read_text(
        encoding="ascii"
    ).strip() != protocol_sha256:
        raise AssertionError("BCN training-feature protocol sidecar mismatch")
    protocol = _load_json(PROTOCOL_PATH)
    for name, artifact in protocol["artifacts"].items():
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"protocol artifact changed: {name}")
    return protocol


def derive(*, protocol_sha256: str) -> dict[str, Any]:
    protocol = _verify_protocol(protocol_sha256)
    rows, positions = _source_and_training_rows()
    expected_identity = protocol["row_contract"]["record_identity_sha256"]
    if _canonical_hash(_identity(rows)) != expected_identity:
        raise AssertionError("BCN training row identity changed after protocol freeze")
    if _canonical_hash(positions) != protocol["row_contract"][
        "selected_source_positions_sha256"
    ]:
        raise AssertionError("BCN training source positions changed after protocol freeze")
    source = np.load(SOURCE_FEATURES, mmap_mode="r")
    features = np.asarray(source[positions], dtype="float32")
    if features.shape != (EXPECTED_TRAIN_IMAGES, EXPECTED_SOURCE_SHAPE[1]):
        raise AssertionError(f"derived BCN training feature shape changed: {features.shape}")
    norms = np.linalg.norm(features, axis=1)
    maximum_norm_error = float(np.max(np.abs(norms - 1.0)))
    if not np.isfinite(features).all() or maximum_norm_error > 2e-5:
        raise AssertionError("derived BCN training features violate unit-norm contract")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FEATURES.with_suffix(".partial.npy")
    np.save(temporary, features)
    os.replace(temporary, OUTPUT_FEATURES)
    label_counts = Counter(str(row["label"]) for row in rows)
    manifest = {
        "protocol": "dermpact-phase2-bcn-training-features-v1",
        "protocol_sha256": protocol_sha256,
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "dataset": "bcn20000",
        "allowed_role": "training only",
        "n_records": len(rows),
        "n_lesions": len({str(row["cluster_id"]) for row in rows}),
        "feature_dim": int(features.shape[1]),
        "record_identity_sha256": expected_identity,
        "selected_source_positions_sha256": protocol["row_contract"][
            "selected_source_positions_sha256"
        ],
        "feature_file_sha256": sha256_file(OUTPUT_FEATURES),
        "maximum_unit_norm_error": maximum_norm_error,
        "label_counts": dict(sorted(label_counts.items())),
        "source": {
            "path": str(SOURCE_FEATURES.relative_to(ROOT)),
            "sha256": sha256_file(SOURCE_FEATURES),
            "selection": "only frozen train-role source indices",
        },
        "excluded_counts": EXPECTED_EXCLUDED,
        "hard_scope_checks": {
            "selected_rows_outside_training_role": 0,
            "validation_calibration_feature_rows_copied": 0,
            "final_test_feature_rows_copied": 0,
            "predictions_generated": 0,
            "metrics_computed": 0,
        },
    }
    manifest_hash = screen._write_json(OUTPUT_MANIFEST, manifest)
    receipt = {
        "protocol": "dermpact-phase2-bcn-training-feature-derivation-receipt-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "output_feature_matrix": {
            "path": str(OUTPUT_FEATURES.relative_to(ROOT)),
            "sha256": manifest["feature_file_sha256"],
        },
        "output_feature_manifest": {
            "path": str(OUTPUT_MANIFEST.relative_to(ROOT)),
            "sha256": manifest_hash,
        },
        "training_images": len(rows),
        "training_lesions": manifest["n_lesions"],
        "protected_rows_copied": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "cpu",
        },
    }
    receipt_hash = screen._write_json(RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": receipt_hash}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args(argv)
    protocol = build_protocol()
    payload = (
        protocol
        if args.protocol_only
        else derive(protocol_sha256=protocol["protocol_sha256"])
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
