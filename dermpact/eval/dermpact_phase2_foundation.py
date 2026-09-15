"""Phase 2.0 environment, checkpoint, and loader acceptance gates."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from .derm_encoders import (
    DEV_MANIFEST,
    FEATURE_ROOT,
    PANDERM_BASE_CHECKPOINT,
    PANDERM_BASE_CHECKPOINT_SHA256,
    PANDERM_BASE_REPO,
    PANDERM_BASE_REPO_COMMIT,
    ROOT,
    _development_records,
    load_candidate_encoder,
)


LOCK_FILE = ROOT / "requirements-p2.lock"
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_0"
PHASE2_AUDIT_DIR = ROOT / "Planning discussions/phase2_audits"
RESOURCE_SNAPSHOT = PHASE2_AUDIT_DIR / "resource_snapshot.json"
PHASE1_OVERLAP_LEDGER = (
    ROOT
    / "research/manifests/dermpact_phase1_v1/dermpact_backbone_overlap_ledger_v1.json"
)
LEGACY_KEYS = ("dermlip", "panderm")
MODULE_PATH = "dermpact.eval.dermpact_phase2_foundation"
LEGACY_EXPECTED = {
    "dermlip": {
        "feature_sha256": "c640f2f934b7d86854d6034609af5714a6cafd1277dc9a420b36331fe2006763",
        "snapshot_revision": "b7d427364122927b0d19faf45cec05eea5469226",
        "weight_sha256": "727b22532b30d027447563d775cbc9b46188ad952c79b0e117eb970efa6bfbfb",
        "feature_dim": 512,
    },
    "panderm": {
        "feature_sha256": "92bedcf642fb19b5fa253e24aab5d92c01e592255e9721d74c897cbce17131cd",
        "snapshot_revision": "b079489efd7ec1f8f9c5553c117ab8d4335f99c3",
        "weight_sha256": "c01315cc8bda77decb7bfaaafaf4668979d065d0ded33c1cb42108cd8d56fedf",
        "feature_dim": 512,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temp.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_temp, sidecar)
    return digest


def _development_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("development manifest is empty")
    count = max(1, min(int(count), length))
    if count == 1:
        return [0]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def _assert_development_rows(records: Sequence[Any], indices: Sequence[int]) -> None:
    allowed = {"train", "calibration"}
    for index in indices:
        record = records[index]
        if record.split not in allowed:
            raise AssertionError(f"non-development split entered Phase 2.0: {record.split}")
        image = Path(record.image_path)
        if sha256_file(image) != record.image_sha256:
            raise AssertionError(f"development image hash mismatch: {record.case_id}")


def audit_legacy_candidate(key: str, *, rows: int = 3) -> dict[str, Any]:
    """Recompute spaced development vectors against a sealed legacy matrix."""
    if key not in LEGACY_KEYS:
        raise ValueError(f"legacy key must be one of {LEGACY_KEYS}, got {key!r}")
    import numpy as np

    records = _development_records(DEV_MANIFEST)
    indices = _development_indices(len(records), rows)
    _assert_development_rows(records, indices)
    reference_path = FEATURE_ROOT / key / "features.npy"
    manifest_path = FEATURE_ROOT / key / "feature_manifest.json"
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = LEGACY_EXPECTED[key]
    reference_sha = sha256_file(reference_path)
    if reference_sha != expected["feature_sha256"]:
        raise AssertionError(f"sealed {key} feature-matrix hash changed")
    if feature_manifest["feature_file_sha256"] != reference_sha:
        raise AssertionError(f"{key} feature manifest does not bind its matrix")

    reference = np.load(reference_path, mmap_mode="r")
    encode, provenance = load_candidate_encoder(key)
    observed = encode([records[index].image_path for index in indices])
    expected_rows = np.asarray(reference[indices], dtype="float32")
    absolute = np.abs(observed - expected_rows)
    cosine = np.sum(observed * expected_rows, axis=1) / (
        np.linalg.norm(observed, axis=1) * np.linalg.norm(expected_rows, axis=1)
    ).clip(min=1e-12)
    weight_files = provenance["checkpoint"]["weight_files"]
    observed_weight_hashes = [item["sha256"] for item in weight_files]
    checks = {
        "shape": list(observed.shape) == [len(indices), expected["feature_dim"]],
        "finite": bool(np.isfinite(observed).all()),
        "max_abs_le_1e-6": float(absolute.max()) <= 1e-6,
        "min_cosine_ge_0_999999": float(cosine.min()) >= 0.999999,
        "snapshot_revision": (
            provenance["checkpoint"]["snapshot_revision"]
            == expected["snapshot_revision"]
        ),
        "weight_sha256": expected["weight_sha256"] in observed_weight_hashes,
        "full_reference_sha256": reference_sha == expected["feature_sha256"],
        "test_records_loaded_zero": feature_manifest["test_records_loaded"] == 0,
    }
    return {
        "protocol": "dermpact-phase2-0-legacy-golden-v1",
        "candidate": key,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "development_manifest": str(DEV_MANIFEST),
        "development_manifest_sha256": sha256_file(DEV_MANIFEST),
        "selected_case_ids": [records[index].case_id for index in indices],
        "selected_splits": [records[index].split for index in indices],
        "reference_matrix": str(reference_path),
        "reference_matrix_sha256": reference_sha,
        "metrics": {
            "max_absolute_error": float(absolute.max()),
            "mean_absolute_error": float(absolute.mean()),
            "minimum_cosine_similarity": float(cosine.min()),
        },
        "checks": checks,
        "provenance": provenance,
    }


def _official_panderm_reference(paths: Sequence[str]):
    """Run the pinned authors' public builder as an independent call path."""
    import numpy as np
    import torch
    from PIL import Image

    classification = PANDERM_BASE_REPO / "classification"
    sys.path.insert(0, str(classification))
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                from models import get_encoder  # type: ignore

                builder_source = Path(sys.modules[get_encoder.__module__].__file__).resolve()
                if not builder_source.is_relative_to(classification.resolve()):
                    raise ImportError(
                        "official PanDerm audit imported an unrelated top-level 'models' "
                        f"package from {builder_source}"
                    )
                args = SimpleNamespace(pretrained_checkpoint=str(PANDERM_BASE_CHECKPOINT))
                model, preprocess = get_encoder(args, "PanDerm_Base_LP")
        model.eval()
        batch = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths])
        with torch.inference_mode():
            features = model(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype("float32")
    finally:
        if sys.path and sys.path[0] == str(classification):
            sys.path.pop(0)


def audit_panderm_base(*, rows: int = 2) -> dict[str, Any]:
    """Validate pure PanDerm loading, repeatability, and official equivalence."""
    import numpy as np

    records = _development_records(DEV_MANIFEST)
    indices = _development_indices(len(records), rows)
    _assert_development_rows(records, indices)
    paths = [records[index].image_path for index in indices]
    encode, provenance = load_candidate_encoder("panderm_base")
    observed_first = encode(paths)
    observed_second = encode(paths)
    official = _official_panderm_reference(paths)
    official_absolute = np.abs(observed_first - official)
    norms = np.linalg.norm(observed_first, axis=1)
    checks = {
        "shape": list(observed_first.shape) == [len(indices), 768],
        "finite": bool(np.isfinite(observed_first).all()),
        "unit_norm_max_error_le_1e-6": float(np.max(np.abs(norms - 1.0))) <= 1e-6,
        "repeatability_max_abs_le_1e-7": (
            float(np.max(np.abs(observed_first - observed_second))) <= 1e-7
        ),
        "official_builder_max_abs_le_1e-6": float(official_absolute.max()) <= 1e-6,
        "checkpoint_sha256": (
            provenance["checkpoint"]["sha256"] == PANDERM_BASE_CHECKPOINT_SHA256
        ),
        "official_source_commit": (
            provenance["official_panderm_commit"] == PANDERM_BASE_REPO_COMMIT
        ),
        "state_dict_gate": provenance["state_dict_gate"]
        == {
            "permitted_missing_keys": ["head.bias", "head.weight"],
            "unexpected_keys": [],
        },
    }
    return {
        "protocol": "dermpact-phase2-0-panderm-base-loader-v1",
        "candidate": "panderm_base",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "development_manifest": str(DEV_MANIFEST),
        "development_manifest_sha256": sha256_file(DEV_MANIFEST),
        "selected_case_ids": [records[index].case_id for index in indices],
        "selected_splits": [records[index].split for index in indices],
        "metrics": {
            "feature_dim": int(observed_first.shape[1]),
            "repeatability_max_absolute_error": float(
                np.max(np.abs(observed_first - observed_second))
            ),
            "official_builder_max_absolute_error": float(official_absolute.max()),
            "official_builder_mean_absolute_error": float(official_absolute.mean()),
            "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
        },
        "checks": checks,
        "provenance": provenance,
    }


def _normalized_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def audit_environment() -> dict[str, Any]:
    lock_lines = [
        line.strip()
        for line in LOCK_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    locked = {
        _normalized_distribution_name(line.split("==", 1)[0]): line.split("==", 1)[1]
        for line in lock_lines
    }
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=False,
        capture_output=True,
        text=True,
    )
    freeze_lines = [
        line.strip()
        for line in pip_freeze.stdout.splitlines()
        if line.strip() and "==" in line
    ]
    installed = {
        _normalized_distribution_name(line.split("==", 1)[0]): line.split("==", 1)[1]
        for line in freeze_lines
    }
    missing = sorted(name for name in locked if name not in installed)
    mismatched = {
        name: {"locked": version, "installed": installed.get(name)}
        for name, version in locked.items()
        if installed.get(name) != version
    }
    extra = sorted(name for name in installed if name not in locked)
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        import psutil

        memory = psutil.virtual_memory()
        resources = {
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "ram_bytes": int(memory.total),
        }
    except ImportError:
        resources = {"collection_error": "psutil unavailable"}
    resource_snapshot = (
        json.loads(RESOURCE_SNAPSHOT.read_text(encoding="utf-8"))
        if RESOURCE_SNAPSHOT.is_file()
        else None
    )
    checks = {
        "lock_exists": LOCK_FILE.is_file(),
        "lock_missing_packages_zero": not missing,
        "lock_version_mismatches_zero": not mismatched,
        "unlocked_installed_packages_zero": not extra,
        "pip_freeze_exit_zero": pip_freeze.returncode == 0,
        "pip_check_exit_zero": pip_check.returncode == 0,
        "cpu_torch_build": importlib.metadata.version("torch").endswith("+cpu"),
        "python_3_12": sys.version_info[:2] == (3, 12),
    }
    return {
        "protocol": "dermpact-phase2-0-environment-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "resources": resources,
        "resource_detector_snapshot": resource_snapshot,
        "known_gpu_supplement": {
            "discrete_gpu": "AMD Radeon RX 5700 XT",
            "reported_adapter_ram_bytes": 8573157376,
            "integrated_gpu": "Intel UHD Graphics 770",
            "note": "The cross-platform resource detector did not enumerate Windows AMD adapters; values were verified from the local Windows adapter registry during Phase 2.0.",
        },
        "lock_file": str(LOCK_FILE),
        "lock_file_sha256": sha256_file(LOCK_FILE),
        "locked_distribution_count": len(locked),
        "installed_distribution_count": len(installed),
        "missing_locked_distributions": missing,
        "version_mismatches": mismatched,
        "unlocked_installed_distributions": extra,
        "pip_check": {
            "returncode": pip_check.returncode,
            "stdout": pip_check.stdout.strip(),
            "stderr": pip_check.stderr.strip(),
        },
        "pip_freeze": {
            "returncode": pip_freeze.returncode,
            "stderr": pip_freeze.stderr.strip(),
            "line_count": len(freeze_lines),
        },
    }


def _write_overlap_ledger_v2(panderm_audit: dict[str, Any]) -> dict[str, Any]:
    source_hash = sha256_file(PHASE1_OVERLAP_LEDGER)
    ledger = json.loads(PHASE1_OVERLAP_LEDGER.read_text(encoding="utf-8"))
    ledger["artifact"] = "dermpact-backbone-overlap-ledger-v2"
    ledger["supersedes"] = {
        "path": str(PHASE1_OVERLAP_LEDGER.relative_to(ROOT)).replace("\\", "/"),
        "sha256": source_hash,
    }
    ledger["phase2_0_update"] = (
        "Pure PanDerm Base was materialized and loader-validated. Its overlap status remains "
        "unresolved, so this update does not authorize a primary generalization claim."
    )
    for candidate in ledger["candidates"]:
        if candidate["candidate_id"] == "panderm_base_vit_b16":
            provenance = panderm_audit["provenance"]
            candidate["checkpoint"] = {
                "availability": "materialized_and_loader_validated",
                "combined_weight_sha256": provenance["checkpoint"]["sha256"],
                "repository_id": "SiyuanYan1/PanDerm",
                "snapshot_revision": provenance["official_panderm_commit"],
                "evidence_path": (
                    "research/manifests/dermpact_phase2_0/"
                    "dermpact_phase2_0_panderm_base_audit_v1.json"
                ),
                "checkpoint_path": str(PANDERM_BASE_CHECKPOINT.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "checkpoint_bytes": provenance["checkpoint"]["bytes"],
                "source_url": provenance["checkpoint"]["source"],
                "model_code_sha256": provenance["model_code_sha256"],
                "license": provenance["license"],
            }
            candidate["role"] = "development_candidate_pending_overlap_clearance"
        elif candidate["candidate_id"] == "dermlip_panderm_pmb256":
            candidate["display_name"] = "DermLIP–PanDerm Base with PubMedBERT-256"
    path = OUTPUT_DIR / "dermpact_backbone_overlap_ledger_v2.json"
    digest = _write_hashed_json(path, ledger)
    return {"path": str(path), "sha256": digest}


def _run_child(kind: str, candidate: str | None = None) -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE_PATH, "--child", kind]
    if candidate:
        command.extend(["--candidate", candidate])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Phase 2.0 child audit failed ({kind}, {candidate}):\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def run_all() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audits = {
        "environment": audit_environment(),
        "dermlip": _run_child("legacy", "dermlip"),
        "dermlip_panderm_hybrid": _run_child("legacy", "panderm"),
        "panderm_base": _run_child("panderm-base"),
    }
    files: dict[str, dict[str, str]] = {}
    filenames = {
        "environment": "dermpact_phase2_0_environment_audit_v1.json",
        "dermlip": "dermpact_phase2_0_dermlip_golden_v1.json",
        "dermlip_panderm_hybrid": "dermpact_phase2_0_dermlip_panderm_golden_v1.json",
        "panderm_base": "dermpact_phase2_0_panderm_base_audit_v1.json",
    }
    for key, audit in audits.items():
        path = OUTPUT_DIR / filenames[key]
        files[key] = {"path": str(path), "sha256": _write_hashed_json(path, audit)}
    ledger = _write_overlap_ledger_v2(audits["panderm_base"])
    all_pass = all(audit["status"] == "PASS" for audit in audits.values())
    receipt = {
        "protocol": "dermpact-phase2-0-foundation-receipt-v1",
        "phase": "2.0",
        "status": "PASS" if all_pass else "FAIL",
        "implementation": {
            "loader": {
                "path": "dermpact/eval/derm_encoders.py",
                "sha256": sha256_file(ROOT / "dermpact/eval/derm_encoders.py"),
            },
            "acceptance_runner": {
                "path": "dermpact/eval/dermpact_phase2_foundation.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "environment_lock": {
                "path": "requirements-p2.lock",
                "sha256": sha256_file(LOCK_FILE),
            },
        },
        "scope_completed": [
            "isolated CPU evaluation environment",
            "exact package lock and dependency consistency gate",
            "cached DermLIP golden-vector compatibility",
            "cached DermLIP–PanDerm hybrid golden-vector compatibility",
            "official pure PanDerm-Base checkpoint materialization and hashing",
            "strict pure PanDerm-Base loader and official-builder equivalence",
            "conservative overlap-ledger provenance update",
        ],
        "explicitly_not_performed": [
            "model training or model selection",
            "PH2 prediction access",
            "HAM10000 final-test prediction access",
            "Derm7pt final-test prediction access",
            "BCN20000 final evaluation",
            "DirectML environment adoption",
        ],
        "audits": files,
        "overlap_ledger": ledger,
        "claim_restriction": (
            "All three foundation candidates remain unresolved for image-level pretraining "
            "overlap and therefore remain ineligible for the primary external-generalization "
            "claim until the separate overlap-promotion rule is satisfied."
        ),
        "next_phase": (
            "Phase 2.1 may reproduce frozen development baselines; it must remain development-only."
        ),
    }
    receipt_path = OUTPUT_DIR / "dermpact_phase2_0_receipt_v1.json"
    receipt_sha = _write_hashed_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("legacy", "panderm-base"))
    parser.add_argument("--candidate", choices=LEGACY_KEYS)
    args = parser.parse_args(argv)
    if args.child == "legacy":
        if not args.candidate:
            parser.error("--candidate is required for --child legacy")
        payload = audit_legacy_candidate(args.candidate)
    elif args.child == "panderm-base":
        payload = audit_panderm_base()
    else:
        payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
