"""Deterministic, cluster-safe dataset manifests and leakage audits."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPLITS = ("train", "calibration", "test")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    cluster_id: str
    label: str
    modality: str
    image_path: str
    source: str
    split: str = ""
    study_id: str = ""
    patient_id: str = ""
    image_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = (self.case_id, self.cluster_id, self.label, self.modality,
                    self.image_path, self.source)
        if any(not str(x).strip() for x in required):
            raise ValueError(f"manifest record has missing required fields: {self.case_id!r}")
        if self.split and self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}")


@dataclass
class LeakageAudit:
    valid: bool
    n_records: int
    n_clusters: int
    split_counts: Dict[str, int]
    duplicate_case_ids: List[str]
    cross_split_clusters: Dict[str, List[str]]
    label_conflict_clusters: Dict[str, List[str]]
    missing_paths: List[str]

    def require_valid(self) -> "LeakageAudit":
        if not self.valid:
            raise ValueError(
                "manifest leakage audit failed: "
                f"duplicate_cases={len(self.duplicate_case_ids)}, "
                f"cross_split_clusters={len(self.cross_split_clusters)}, "
                f"label_conflicts={len(self.label_conflict_clusters)}, "
                f"missing_paths={len(self.missing_paths)}"
            )
        return self


def _stable_rank(seed: int, label: str, cluster_id: str) -> str:
    return hashlib.sha256(f"{seed}|{label}|{cluster_id}".encode("utf-8")).hexdigest()


def _split_counts(n: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 0, 1)
    raw = [n * float(r) for r in ratios]
    counts = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda j: raw[j] - counts[j], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    # For evaluability, every label with >=3 clusters appears in every split.
    for target in (1, 2):
        if counts[target] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            counts[donor] -= 1
            counts[target] += 1
    return tuple(counts)  # type: ignore[return-value]


def assign_cluster_splits(
    records: Sequence[CaseRecord], *, seed: int = 20260712,
    ratios: Sequence[float] = (0.70, 0.10, 0.20),
) -> List[CaseRecord]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9 or any(r < 0 for r in ratios):
        raise ValueError("split ratios must be three nonnegative values summing to one")
    clusters: Dict[str, List[CaseRecord]] = defaultdict(list)
    for record in records:
        record.validate()
        clusters[record.cluster_id].append(record)
    labels: Dict[str, str] = {}
    for cluster_id, group in clusters.items():
        unique = {r.label for r in group}
        if len(unique) != 1:
            raise ValueError(f"cluster {cluster_id!r} has conflicting labels: {sorted(unique)}")
        labels[cluster_id] = next(iter(unique))
    by_label: Dict[str, List[str]] = defaultdict(list)
    for cluster_id, label in labels.items():
        by_label[label].append(cluster_id)
    assignments: Dict[str, str] = {}
    for label, cluster_ids in sorted(by_label.items()):
        ordered = sorted(cluster_ids, key=lambda c: _stable_rank(seed, label, c))
        n_train, n_cal, _ = _split_counts(len(ordered), ratios)
        for i, cluster_id in enumerate(ordered):
            assignments[cluster_id] = ("train" if i < n_train else
                                       "calibration" if i < n_train + n_cal else "test")
    return [CaseRecord(**{**asdict(r), "split": assignments[r.cluster_id]}) for r in records]


def audit_manifest(records: Sequence[CaseRecord], *, check_paths: bool = False) -> LeakageAudit:
    case_counts = Counter(r.case_id for r in records)
    duplicates = sorted(k for k, v in case_counts.items() if v > 1)
    split_by_cluster: Dict[str, set] = defaultdict(set)
    labels_by_cluster: Dict[str, set] = defaultdict(set)
    missing: List[str] = []
    for record in records:
        record.validate()
        split_by_cluster[record.cluster_id].add(record.split)
        labels_by_cluster[record.cluster_id].add(record.label)
        if check_paths and not Path(record.image_path).is_file():
            missing.append(record.image_path)
    cross = {k: sorted(v) for k, v in split_by_cluster.items() if len(v) > 1}
    conflicts = {k: sorted(v) for k, v in labels_by_cluster.items() if len(v) > 1}
    counts = Counter(r.split for r in records)
    valid = not duplicates and not cross and not conflicts and not missing
    return LeakageAudit(valid, len(records), len(split_by_cluster), dict(counts),
                        duplicates, cross, conflicts, sorted(set(missing)))


def _canonical_payload(records: Sequence[CaseRecord], metadata: Optional[Mapping[str, Any]]) -> bytes:
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "records": [asdict(r) for r in sorted(records, key=lambda x: x.case_id)],
    }
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n").encode("utf-8")


def freeze_manifest(
    records: Sequence[CaseRecord], path: os.PathLike[str] | str, *,
    metadata: Optional[Mapping[str, Any]] = None, check_paths: bool = False,
) -> str:
    audit_manifest(records, check_paths=check_paths).require_valid()
    payload = _canonical_payload(records, metadata)
    digest = hashlib.sha256(payload).hexdigest()
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if target.exists() and target.read_bytes() != payload:
        raise FileExistsError(f"refusing to mutate frozen manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, target)
    if sidecar.exists() and sidecar.read_text(encoding="ascii").strip() != digest:
        raise FileExistsError(f"manifest hash sidecar conflicts with content: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(digest + "\n", encoding="ascii")
    return digest


def load_manifest(path: os.PathLike[str] | str, *, verify_hash: bool = True) -> List[CaseRecord]:
    target = Path(path)
    payload = target.read_bytes()
    if verify_hash:
        sidecar = target.with_suffix(target.suffix + ".sha256")
        expected = sidecar.read_text(encoding="ascii").strip()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise ValueError(f"manifest hash mismatch: expected {expected}, observed {observed}")
    data = json.loads(payload)
    records = [CaseRecord(**r) for r in data.get("records", [])]
    audit_manifest(records).require_valid()
    return records


def derive_development_manifest(
    parent_path: os.PathLike[str] | str, output_path: os.PathLike[str] | str,
) -> Dict[str, Any]:
    """Freeze a train+calibration view without exposing test identities."""
    parent = Path(parent_path)
    records = load_manifest(parent)
    development = [record for record in records if record.split in {"train", "calibration"}]
    redacted = len(records) - len(development)
    digest = freeze_manifest(
        development, output_path,
        metadata={
            "derivation": "train_and_calibration_only",
            "parent_manifest_sha256": file_sha256(parent),
            "test_records_redacted": redacted,
        },
    )
    return {
        "n_parent_records": len(records),
        "n_development_records": len(development),
        "n_test_records_redacted": redacted,
        "output_sha256": digest,
    }


def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
