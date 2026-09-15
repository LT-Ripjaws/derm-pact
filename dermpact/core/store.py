"""WP-A2: file-backed case store."""
from __future__ import annotations

import json
import os
import uuid
import re
import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from dermpact.core.episode import EpisodeEvent
from dermpact.core.record_v2 import StructuredCaseRecordV2, verify_chain as verify_chain_v2


_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class StoreErrorV2(ValueError):
    """Base class for every typed store rejection."""


class WrongCaseError(StoreErrorV2):
    """A command named, or a payload was bound to, a different case than the one addressed."""


class StaleVersionError(StoreErrorV2):
    """An ``append_record`` call's ``expected_record_version``/parent hash does not match the current head."""


class ChainVerificationError(StoreErrorV2):
    """The on-disk record chain for a case fails independent verification."""


class HeadMismatchError(StoreErrorV2):
    """``head.json`` does not point at the latest persisted record version."""


@dataclass(frozen=True)
class CaseState:
    case_id: str
    records: Tuple[StructuredCaseRecordV2, ...]
    head: Dict[str, Any]
    ledger_events: Tuple[Dict[str, Any], ...]
    episode_events: Tuple[EpisodeEvent, ...]
    xai_receipt_ids: Tuple[str, ...]

    @property
    def current_record(self) -> StructuredCaseRecordV2:
        return self.records[-1]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}{uuid.uuid4().hex[:8]}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    """Append one line atomically: read the existing content (if any), append the new line, and replace the whole file via temp+``os.replace`` -- an in-place ``open(..., "a")`` is not guaranteed atomic across processes on every filesystem this prototype might run on, and the log is small enough (per-case, JSON-lines) that a full rewrite per append is cheap."""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps(dict(obj), sort_keys=True, ensure_ascii=False)
    new_text = existing + (line + "\n")
    _atomic_write_text(path, new_text)


def _read_jsonl(path: Path) -> Tuple[Dict[str, Any], ...]:
    if not path.exists():
        return ()
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return tuple(out)


@contextmanager
def _exclusive_file_lock(path: Path, *, timeout_seconds: float = 30.0):
    """Serialize a case mutation across threads and Python processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        handle = path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            deadline = time.monotonic() + timeout_seconds
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise StoreErrorV2(f"timed out acquiring case lock {path.name!r}")
                        time.sleep(0.02)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise StoreErrorV2(f"timed out acquiring case lock {path.name!r}")
                        time.sleep(0.02)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class CaseStore:
    """File-backed store for one ``state_dir``."""

    def __init__(
        self, state_dir: Any, *, catalogue, engine, calibrators: Optional[Mapping[str, Any]] = None,
        bundle: Optional[Any] = None,
    ):
        self.state_dir = Path(state_dir)
        self._catalogue = catalogue
        self._engine = engine
        self._calibrators = calibrators
        self._bundle = bundle
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------

    def _case_dir(self, case_id: str) -> Path:
        return self.state_dir / "cases" / case_id

    def _records_dir(self, case_id: str) -> Path:
        return self._case_dir(case_id) / "records"

    def _ledger_path(self, case_id: str) -> Path:
        return self._case_dir(case_id) / "ledger_events.jsonl"

    def _episode_path(self, case_id: str) -> Path:
        return self._case_dir(case_id) / "episode_events.jsonl"

    def _xai_dir(self, case_id: str) -> Path:
        return self._case_dir(case_id) / "xai"

    def _head_path(self, case_id: str) -> Path:
        return self._case_dir(case_id) / "head.json"

    def _lock_path(self, case_id: str) -> Path:
        digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
        return self.state_dir / ".locks" / f"{digest}.lock"

    def _read_head(self, case_id: str) -> Dict[str, Any]:
        p = self._head_path(case_id)
        if not p.exists():
            raise StoreErrorV2(f"no head.json for case {case_id!r}")
        return json.loads(p.read_text(encoding="utf-8"))

    # -- case lifecycle -----------------------------------------------------

    def create_case(self, record: StructuredCaseRecordV2) -> CaseState:
        case_id = record.identity.case_id
        with _exclusive_file_lock(self._lock_path(case_id)):
            if self._case_dir(case_id).exists():
                raise StoreErrorV2(f"case {case_id!r} already exists")
            if record.identity.record_version != 1 or record.identity.parent_record_sha256 is not None:
                raise StoreErrorV2("create_case requires a fresh version-1 record with no parent")
            _atomic_write_json(self._records_dir(case_id) / "v1.json", record.to_dict())
            head = {"current_record_sha256": record.record_sha256, "record_version": 1}
            _atomic_write_json(self._head_path(case_id), head)
        return CaseState(case_id=case_id, records=(record,), head=head, ledger_events=(), episode_events=(), xai_receipt_ids=())

    def append_record(
        self, case_id: str, record: StructuredCaseRecordV2, *, expected_record_version: Optional[int] = None,
    ) -> CaseState:
        if record.identity.case_id != case_id:
            raise WrongCaseError(f"record is for case {record.identity.case_id!r}, not {case_id!r}")
        with _exclusive_file_lock(self._lock_path(case_id)):
            head = self._read_head(case_id)
            if expected_record_version is not None and expected_record_version != head["record_version"]:
                raise StaleVersionError(
                    f"expected_record_version={expected_record_version} but head is at {head['record_version']}"
                )
            if record.identity.record_version != head["record_version"] + 1:
                raise StaleVersionError(
                    f"record_version {record.identity.record_version} does not extend head {head['record_version']}"
                )
            if record.identity.parent_record_sha256 != head["current_record_sha256"]:
                raise StaleVersionError("parent_record_sha256 does not match the current head (concurrent write)")
            _atomic_write_json(self._records_dir(case_id) / f"v{record.identity.record_version}.json", record.to_dict())
            new_head = {"current_record_sha256": record.record_sha256, "record_version": record.identity.record_version}
            _atomic_write_json(self._head_path(case_id), new_head)
        return self.load_case(case_id)

    def load_case(self, case_id: str) -> CaseState:
        d = self._case_dir(case_id)
        if not d.exists():
            raise StoreErrorV2(f"no such case {case_id!r}")
        records_dir = self._records_dir(case_id)
        versions = sorted(int(p.stem[1:]) for p in records_dir.glob("v*.json")) if records_dir.exists() else []
        records = []
        for v in versions:
            payload = json.loads((records_dir / f"v{v}.json").read_text(encoding="utf-8"))
            rec = StructuredCaseRecordV2.from_dict(payload)
            if rec.identity.case_id != case_id:
                raise WrongCaseError(f"records/v{v}.json is bound to case {rec.identity.case_id!r}, not {case_id!r}")
            if rec.identity.record_version != v:
                raise StoreErrorV2(f"records/v{v}.json has record_version={rec.identity.record_version}")
            records.append(rec)
        records_t = tuple(records)

        for rec in records_t:
            if not rec.verify_sha256():
                raise ChainVerificationError(
                    f"records/v{rec.identity.record_version}.json for case {case_id!r} does not reproduce its own record_sha256"
                )

        if records_t:
            verdict = verify_chain_v2(
                records_t, catalogue=self._catalogue, engine=self._engine, calibrators=self._calibrators, bundle=self._bundle,
            )
            if not verdict.passed:
                raise ChainVerificationError(f"case {case_id!r} chain verification failed: {verdict.failures}")

        head = self._read_head(case_id)
        if records_t:
            latest = records_t[-1]
            if head["current_record_sha256"] != latest.record_sha256 or head["record_version"] != latest.identity.record_version:
                raise HeadMismatchError(f"head.json does not match the latest persisted record for case {case_id!r}")

        ledger_events = _read_jsonl(self._ledger_path(case_id))
        episode_events = tuple(EpisodeEvent.from_dict(e) for e in _read_jsonl(self._episode_path(case_id)))
        xai_dir = self._xai_dir(case_id)
        xai_ids = tuple(sorted(p.stem for p in xai_dir.glob("*.json"))) if xai_dir.exists() else ()

        return CaseState(
            case_id=case_id, records=records_t, head=head, ledger_events=ledger_events,
            episode_events=episode_events, xai_receipt_ids=xai_ids,
        )

    def list_cases(self) -> Tuple[str, ...]:
        base = self.state_dir / "cases"
        if not base.exists():
            return ()
        return tuple(sorted(p.name for p in base.iterdir() if p.is_dir()))

    # -- ledger / episode / xai event append ---------------------------------

    def append_ledger_event(self, case_id: str, event: Mapping[str, Any]) -> None:
        event_case_id = event.get("case_id")
        if event_case_id is not None and event_case_id != case_id:
            raise WrongCaseError(f"ledger event is bound to case {event_case_id!r}, not {case_id!r}")
        with _exclusive_file_lock(self._lock_path(case_id)):
            _append_jsonl(self._ledger_path(case_id), event)

    def append_episode_event(self, case_id: str, event: Any) -> None:
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        if payload.get("case_id") != case_id:
            raise WrongCaseError(f"episode event is bound to case {payload.get('case_id')!r}, not {case_id!r}")
        with _exclusive_file_lock(self._lock_path(case_id)):
            prior_payloads = _read_jsonl(self._episode_path(case_id))
            prior_events = tuple(EpisodeEvent.from_dict(item) for item in prior_payloads)
            incoming = EpisodeEvent.from_dict(payload)
            for prior, prior_payload in zip(prior_events, prior_payloads):
                if incoming.event_id == prior.event_id or incoming.idempotency_key == prior.idempotency_key:
                    if payload == prior_payload:
                        return
                    raise StoreErrorV2("episode event id/idempotency key conflicts with an existing event")
            expected_previous = prior_events[-1].event_sha256 if prior_events else None
            if incoming.previous_event_sha256 != expected_previous:
                raise StaleVersionError("episode event does not extend the current content-hash chain")
            _append_jsonl(self._episode_path(case_id), payload)

    def save_xai_receipt(self, case_id: str, receipt_dict: Mapping[str, Any]) -> None:
        rid = receipt_dict.get("receipt_id")
        if not isinstance(rid, str) or re.fullmatch(r"[0-9a-f]{64}", rid) is None:
            raise StoreErrorV2("xai receipt id must be a 64-character lowercase sha256")
        from dermpact.core.xai import verify_receipt_content
        verify_receipt_content(receipt_dict)
        with _exclusive_file_lock(self._lock_path(case_id)):
            path = self._xai_dir(case_id) / f"{rid}.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                verify_receipt_content(existing)
                if existing != dict(receipt_dict):
                    raise StoreErrorV2("refusing to replace an immutable XAI receipt with different content")
                return
            _atomic_write_json(path, dict(receipt_dict))

    def load_xai_receipts(self, case_id: str) -> Tuple[Dict[str, Any], ...]:
        d = self._xai_dir(case_id)
        if not d.exists():
            return ()
        from dermpact.core.xai import verify_receipt_content
        receipts = tuple(json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json")))
        for receipt in receipts:
            verify_receipt_content(receipt)
        return receipts


__all__ = [
    "CaseState",
    "CaseStore",
    "ChainVerificationError",
    "HeadMismatchError",
    "StaleVersionError",
    "StoreErrorV2",
    "WrongCaseError",
]
