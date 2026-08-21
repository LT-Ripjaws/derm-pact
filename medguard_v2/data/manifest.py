"""Byte-reproducible, hash-verified manifest for the Derm7pt release_v0 split.

Derm7pt is CC BY-NC-ND: the images are never copied or embedded.  The manifest
records only metadata and the SHA-256 of each image file, which is enough to
prove that a later run saw the identical pixels without redistributing them.

Reproducibility rules enforced here:

* no timestamps, no hostnames, no absolute paths (image paths are stored
  relative to ``release_v0/images``);
* the case list is sorted by ``case_index``;
* the body is canonicalised with ``sort_keys=True`` and compact separators
  before hashing, and the file itself is written from that same body with a
  fixed indent and a single trailing newline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from medguard_v2.data.derm7pt import (
    DEFAULT_ROOT,
    DIAGNOSIS_GROUP_ORDER,
    Derm7ptCase,
    SEVEN_CRITERIA,
    SPLITS,
    class_counts,
    load_cases,
    split_counts,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "DEFAULT_MANIFEST_PATH",
    "build_manifest",
    "canonical_body_bytes",
    "freeze_manifest",
    "file_sha256",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "derm7pt_v1" / "manifest.json"

MANIFEST_SCHEMA = "derm7pt_manifest_v1"

LICENSE_NOTE = (
    "Derm7pt (Kawahara et al.) is CC BY-NC-ND. This manifest stores metadata and "
    "SHA-256 digests only; no image bytes are reproduced or redistributed."
)

_HASH_CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_body_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(
        body, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")


def _relative_image(path: Path, images_root: Path) -> str:
    return path.relative_to(images_root).as_posix()


def build_manifest(
    cases: Iterable[Derm7ptCase] | None = None, root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    """Build the manifest dict, including its own body digest."""
    root = Path(root)
    case_list = list(cases) if cases is not None else load_cases(root)
    case_list.sort(key=lambda case: case.case_index)
    images_root = (root / "images").resolve()

    digests: dict[Path, str] = {}

    def digest_of(path: Path) -> str:
        if path not in digests:
            digests[path] = file_sha256(path)
        return digests[path]

    records = []
    for case in case_list:
        records.append(
            {
                "case_index": case.case_index,
                "case_num": case.case_num,
                "split": case.split,
                "diagnosis": case.diagnosis,
                "diagnosis_group": case.diagnosis_group,
                "management": case.management,
                "difficulty": case.difficulty,
                "seven_point_score": case.seven_point_score,
                "criteria": {name: case.criteria[name] for name in SEVEN_CRITERIA},
                "clinic_image": {
                    "path": _relative_image(case.clinic_path, images_root),
                    "sha256": digest_of(case.clinic_path),
                },
                "derm_image": {
                    "path": _relative_image(case.derm_path, images_root),
                    "sha256": digest_of(case.derm_path),
                },
                "has_distinct_views": case.has_distinct_views,
            }
        )

    body: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "dataset": "derm7pt_release_v0",
        "license_note": LICENSE_NOTE,
        "seven_criteria": list(SEVEN_CRITERIA),
        "diagnosis_group_order": list(DIAGNOSIS_GROUP_ORDER),
        "n_records": len(records),
        "split_counts": {split: split_counts(case_list)[split] for split in SPLITS},
        "class_counts": {
            group: class_counts(case_list)[group] for group in DIAGNOSIS_GROUP_ORDER
        },
        "n_cases_without_distinct_views": sum(
            1 for record in records if not record["has_distinct_views"]
        ),
        "n_unique_image_files": len(digests),
        "cases": records,
    }

    manifest = dict(body)
    manifest["body_sha256"] = hashlib.sha256(canonical_body_bytes(body)).hexdigest()
    return manifest


def freeze_manifest(
    out_path: Path = DEFAULT_MANIFEST_PATH, root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    """Write the manifest to ``out_path`` byte-reproducibly; return the manifest."""
    manifest = build_manifest(root=root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return manifest


def _main() -> None:
    manifest = freeze_manifest()
    print(f"wrote {DEFAULT_MANIFEST_PATH}")
    print(f"body_sha256: {manifest['body_sha256']}")
    print(f"file_sha256: {file_sha256(DEFAULT_MANIFEST_PATH)}")
    print(f"n_records:   {manifest['n_records']}")
    print("split_counts:", manifest["split_counts"])
    print("class_counts:", manifest["class_counts"])


if __name__ == "__main__":
    _main()
