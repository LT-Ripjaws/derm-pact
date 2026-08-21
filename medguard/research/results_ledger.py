"""Machine-checkable ledger of genuine MedGuard findings and experiment decisions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


STATUSES = {"verified", "provisional", "failed", "abandoned", "superseded"}
PAPER_ROLES = {
    "headline-candidate", "supporting-result", "ablation", "limitation",
    "data-method", "development-only", "not-admissible",
}
DECISIONS = {"keep", "kill", "demote", "redo", "freeze", "pending"}
_IGNORED_PARTS = {".git", "__pycache__"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(root: Path, relative_path: str) -> str:
    """Hash a file or a directory as sorted relative-path/hash pairs."""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or any(
        part.startswith(".env") for part in relative.parts
    ):
        raise ValueError(f"unsafe artifact path {relative_path!r}")
    target = root.joinpath(*relative.parts)
    if target.is_file():
        return file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    rows: list[str] = []
    for path in sorted(candidate for candidate in target.rglob("*") if candidate.is_file()):
        rel = path.relative_to(target)
        if any(part in _IGNORED_PARTS or part.startswith(".env") for part in rel.parts):
            continue
        if path.suffix == ".pyc":
            continue
        rows.append(f"{rel.as_posix()}\t{file_sha256(path)}")
    if not rows:
        raise ValueError(f"artifact directory is empty after exclusions: {target}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    kind: str


@dataclass
class ResultRecord:
    result_id: str
    date: str
    status: str
    evidence_level: str
    claim: str
    protocol: str
    modality: list[str]
    dataset_split: str
    metrics: dict[str, Any]
    decision: str
    paper_role: str
    limitations: list[str]
    code_paths: list[str]
    artifacts: list[ArtifactRef]
    notes: str = ""
    supersedes: list[str] = field(default_factory=list)

    def validate(
        self, *, root: Path | None = None,
        intentionally_removed_code_paths: frozenset[str] = frozenset(),
    ) -> list[str]:
        errors: list[str] = []
        if not self.result_id or not self.claim or not self.protocol:
            errors.append("result_id, claim, and protocol are required")
        if self.status not in STATUSES:
            errors.append(f"unknown status {self.status!r}")
        if self.evidence_level not in {"verified", "inferred", "guess"}:
            errors.append(f"unknown evidence level {self.evidence_level!r}")
        if self.paper_role not in PAPER_ROLES:
            errors.append(f"unknown paper role {self.paper_role!r}")
        if self.decision not in DECISIONS:
            errors.append(f"unknown decision {self.decision!r}")
        if self.status == "verified" and (not self.artifacts or not self.code_paths):
            errors.append("verified result requires artifacts and code paths")
        if self.status in {"failed", "abandoned"} and self.decision == "keep":
            errors.append("failed/abandoned result cannot have keep decision")
        for value in [*self.code_paths, *(artifact.path for artifact in self.artifacts)]:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or any(
                part.startswith(".env") for part in path.parts
            ):
                errors.append(f"unsafe path {value!r}")
        if root is not None:
            for code_path in self.code_paths:
                if code_path in intentionally_removed_code_paths:
                    continue
                if not root.joinpath(*PurePosixPath(code_path).parts).is_file():
                    errors.append(f"missing code path {code_path!r}")
            for artifact in self.artifacts:
                try:
                    observed = artifact_sha256(root, artifact.path)
                except (FileNotFoundError, ValueError) as error:
                    errors.append(str(error))
                    continue
                if observed != artifact.sha256:
                    errors.append(
                        f"artifact hash mismatch {artifact.path!r}: "
                        f"{observed} != {artifact.sha256}"
                    )
        return errors


class ResultsLedger:
    def __init__(
        self, records: Sequence[ResultRecord], *, updated: str = "",
        code_retirement: Mapping[str, Any] | None = None,
    ):
        self.records = list(records)
        self.updated = updated
        self.code_retirement = dict(code_retirement or {})

    @property
    def intentionally_removed_code_paths(self) -> frozenset[str]:
        return frozenset(str(path) for path in self.code_retirement.get("code_paths", []))

    @classmethod
    def load(cls, path: Path) -> "ResultsLedger":
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for value in raw.get("records", []):
            row = dict(value)
            row["artifacts"] = [ArtifactRef(**item) for item in row.get("artifacts", [])]
            records.append(ResultRecord(**row))
        return cls(
            records, updated=str(raw.get("updated", "")),
            code_retirement=raw.get("code_retirement", {}),
        )

    def validate(self, *, root: Path | None = None) -> dict[str, list[str]]:
        retired = self.intentionally_removed_code_paths
        errors = {
            record.result_id: record.validate(
                root=root, intentionally_removed_code_paths=retired)
            for record in self.records
        }
        ids = [record.result_id for record in self.records]
        if len(ids) != len(set(ids)):
            errors.setdefault("__ledger__", []).append("duplicate result_id")
        known = set(ids)
        for record in self.records:
            missing = sorted(set(record.supersedes) - known)
            if missing:
                errors.setdefault(record.result_id, []).append(
                    f"unknown superseded result ids: {missing}"
                )
        referenced = {path for record in self.records for path in record.code_paths}
        for path in sorted(retired):
            parsed = PurePosixPath(path)
            if parsed.is_absolute() or ".." in parsed.parts or any(
                part.startswith(".env") for part in parsed.parts
            ):
                errors.setdefault("__code_retirement__", []).append(
                    f"unsafe retired code path {path!r}")
            elif path not in referenced:
                errors.setdefault("__code_retirement__", []).append(
                    f"retired code path is not referenced by a result: {path!r}")
            elif root is not None and root.joinpath(*parsed.parts).exists():
                errors.setdefault("__code_retirement__", []).append(
                    f"intentionally removed code path still exists: {path!r}")
        return {key: value for key, value in errors.items() if value}

    def require_valid(self, *, root: Path | None = None) -> "ResultsLedger":
        errors = self.validate(root=root)
        if errors:
            raise ValueError(f"invalid results ledger: {errors}")
        return self


def audit_ledger(path: Path, *, root: Path) -> dict[str, Any]:
    ledger = ResultsLedger.load(path)
    errors = ledger.validate(root=root)
    return {
        "valid": not errors,
        "updated": ledger.updated,
        "n_records": len(ledger.records),
        "status_counts": {
            status: sum(record.status == status for record in ledger.records)
            for status in sorted(STATUSES)
        },
        "paper_role_counts": {
            role: sum(record.paper_role == role for record in ledger.records)
            for role in sorted(PAPER_ROLES)
        },
        "intentionally_removed_code_paths": len(ledger.intentionally_removed_code_paths),
        "code_retirement": ledger.code_retirement,
        "errors": errors,
    }
