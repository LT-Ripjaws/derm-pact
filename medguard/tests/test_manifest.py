import tempfile
import json
from pathlib import Path

from medguard.data.builders import build_ham10000_jsonl_split
from medguard.data.manifest import (
    CaseRecord, assign_cluster_splits, audit_manifest, derive_development_manifest,
    freeze_manifest, load_manifest,
)


def _records(n=20):
    return [CaseRecord(f"case{i}", f"lesion{i}", "melanoma" if i % 2 else "nevus",
                       "dermoscopy", f"image{i}.jpg", "HAM10000") for i in range(n)]


def test_manifest_split_is_deterministic_and_cluster_safe():
    a = assign_cluster_splits(_records(), seed=7)
    b = assign_cluster_splits(_records(), seed=7)
    assert [r.split for r in a] == [r.split for r in b]
    assert audit_manifest(a).valid
    assert {r.split for r in a} == {"train", "calibration", "test"}


def test_frozen_manifest_roundtrip_and_hash():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        records = assign_cluster_splits(_records(8), seed=1)
        digest = freeze_manifest(records, path, metadata={"seed": 1})
        assert len(digest) == 64
        assert load_manifest(path) == records


def test_ham_jsonl_split_freezes_before_images_are_staged():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata = root / "metadata.jsonl"
        rows = [
            {"image_id": "i1", "lesion_id": "l1", "class": "melanoma"},
            {"image_id": "i2", "lesion_id": "l1", "class": "melanoma"},
            {"image_id": "i3", "lesion_id": "l2", "class": "nevus"},
            {"image_id": "i4", "lesion_id": "l3", "class": "nevus"},
            {"image_id": "i5", "lesion_id": "l4", "class": "nevus"},
        ]
        metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        records = build_ham10000_jsonl_split(metadata, root / "images", seed=7)
        assert len(records) == 5
        assert len({row.split for row in records if row.cluster_id == "l1"}) == 1
        assert audit_manifest(records, check_paths=False).valid
        assert not audit_manifest(records, check_paths=True).valid


def test_derived_development_manifest_redacts_test_records():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = root / "image.jpg"
        image.write_bytes(b"image")
        records = [
            CaseRecord("train", "lesion-train", "nevus", "dermoscopy", str(image), "HAM", "train"),
            CaseRecord("cal", "lesion-cal", "melanoma", "dermoscopy", str(image), "HAM", "calibration"),
            CaseRecord("test", "lesion-test", "bcc", "dermoscopy", str(image), "HAM", "test"),
        ]
        parent = root / "parent.json"
        freeze_manifest(records, parent)
        output = root / "development.json"
        report = derive_development_manifest(parent, output)
        derived = load_manifest(output)
        assert report["n_test_records_redacted"] == 1
        assert {row.split for row in derived} == {"train", "calibration"}
        assert all(row.case_id != "test" for row in derived)
