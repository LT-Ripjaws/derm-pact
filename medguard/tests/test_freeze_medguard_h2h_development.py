import hashlib
import json
import tempfile
from pathlib import Path

from medguard.data.freeze_medguard_h2h_development import (
    CLASS_ORDER,
    freeze_development_h2h,
)
from medguard.data.manifest import CaseRecord, file_sha256, freeze_manifest


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fixture(root: Path, *, include_test: bool = False, pinned_spec: bool = True):
    parent = root / "research" / "manifests" / "parent.json"
    records = []
    expected = {}
    for label in CLASS_ORDER:
        expected[label] = 2
        for cluster_index in range(2):
            cluster_id = f"{label}-{cluster_index}"
            view_total = 2 if cluster_index == 0 else 1
            for view_index in range(view_total):
                case_id = f"{cluster_id}-view-{view_index}"
                image = root / "images" / f"{case_id}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(case_id.encode())
                records.append(CaseRecord(
                    case_id=case_id,
                    cluster_id=cluster_id,
                    label=label,
                    modality="dermoscopy",
                    image_path=str(image),
                    source="fixture",
                    split="calibration",
                    image_sha256=file_sha256(image),
                ))
    if include_test:
        image = root / "images" / "sealed.jpg"
        image.write_bytes(b"sealed")
        records.append(CaseRecord(
            case_id="sealed-case", cluster_id="sealed-lesion", label="nevus",
            modality="dermoscopy", image_path=str(image), source="fixture",
            split="test", image_sha256=file_sha256(image),
        ))
    freeze_manifest(records, parent, metadata={"test_records_redacted": 0})
    expected["total"] = 2 * len(CLASS_ORDER)
    spec = root / "research" / "EXPERIMENT_SPEC_V1.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec_payload = {
        "spec_id": "fixture-spec",
        "status": "design_frozen_artifacts_pending",
        "effective_date": "2026-07-16",
        "scope": {"class_order": list(CLASS_ORDER)},
        "frozen_dataset_artifacts": [
            {"path": "research/manifests/parent.json", "sha256": file_sha256(parent)}
        ],
        "development_h2h_manifest_rule": {
            "parent": "research/manifests/parent.json",
            "expected_lesion_counts": expected,
        },
    }
    spec.write_bytes(_json_bytes(spec_payload))
    design_receipt = root / "research" / "EXPERIMENT_SPEC_V1.receipt.json"
    pinned_hash = file_sha256(spec) if pinned_spec else "0" * 64
    design_receipt.write_bytes(_json_bytes({
        "spec_id": "fixture-spec",
        "files": [{"path": "research/EXPERIMENT_SPEC_V1.json",
                   "sha256": pinned_hash}],
    }))
    output = root / "research" / "manifests" / "development.json"
    return spec, design_receipt, output


def test_freezer_is_deterministic_retains_all_views_and_writes_receipt():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec, design_receipt, output = _fixture(root)
        first = freeze_development_h2h(
            spec, design_receipt, output, root=root)
        second = freeze_development_h2h(
            spec, design_receipt, output, root=root)
        assert first == second
        payload = json.loads(output.read_text())
        records = payload["records"]
        assert len({row["cluster_id"] for row in records}) == 14
        assert len(records) == 21
        assert all(row["split"] == "calibration" for row in records)
        assert payload["metadata"]["all_views_retained"] is True
        receipt = output.with_name(output.stem + ".receipt.json")
        sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
        assert file_sha256(receipt) == sidecar.read_text().strip()
        assert first["test_records_loaded"] == 0


def test_freezer_rejects_any_test_record_in_the_parent():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec, design_receipt, output = _fixture(root, include_test=True)
        try:
            freeze_development_h2h(spec, design_receipt, output, root=root)
        except ValueError as exc:
            assert "test record" in str(exc)
        else:
            raise AssertionError("test-bearing parent was accepted")


def test_freezer_rejects_an_unpinned_specification():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec, design_receipt, output = _fixture(root, pinned_spec=False)
        try:
            freeze_development_h2h(spec, design_receipt, output, root=root)
        except ValueError as exc:
            assert "hash is not pinned" in str(exc)
        else:
            raise AssertionError("unbound experiment specification was accepted")
