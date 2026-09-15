"""WP-A2: persistent, event-sourced review episode."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from dermpact.core import versions as V
from dermpact.core.record_v2 import CLINICIAN_ASSERTION_FACTS, StructuredCaseRecordV2

EPISODE_VERSION = V.EPISODE_VERSION
EVENT_SCHEMA_VERSION = "dermpact-review-event-v2"

#: Every event type this module produces/consumes.
EVENT_TYPES: FrozenSet[str] = frozenset({
    "review_requested", "awaiting_review", "awaiting_information", "review_acknowledged",
    "information_received", "recomputed", "resolved", "unresolved", "outcome_confirmed",
    "follow_up_date_recorded", "comparison_reminder_unsolicited",
})

ALLOWED_TRANSITIONS: Mapping[str, FrozenSet[str]] = {
    "review_requested": frozenset({"not_started", "resolved", "unresolved", "recomputed", "information_received", "acknowledged"}),
    "awaiting_review": frozenset({"requested"}),
    "awaiting_information": frozenset({"not_started", "requested", "awaiting_review", "acknowledged", "information_received", "recomputed", "resolved", "unresolved"}),
    "review_acknowledged": frozenset({"requested", "awaiting_review"}),
    "information_received": frozenset({"requested", "awaiting_review", "awaiting_information", "acknowledged", "resolved", "unresolved", "recomputed"}),
    "recomputed": frozenset({"not_started", "requested", "awaiting_review", "awaiting_information", "acknowledged", "information_received", "resolved", "unresolved", "recomputed"}),
    "resolved": frozenset({"requested", "awaiting_review", "acknowledged", "information_received", "recomputed", "unresolved"}),
    "unresolved": frozenset({"requested", "awaiting_review", "awaiting_information", "acknowledged", "information_received", "recomputed"}),
    "outcome_confirmed": frozenset({"resolved"}),
    "follow_up_date_recorded": frozenset({"requested", "awaiting_review", "acknowledged", "information_received", "recomputed", "resolved", "unresolved"}),
    "comparison_reminder_unsolicited": frozenset({"not_started", "requested", "awaiting_review", "awaiting_information", "acknowledged", "information_received", "recomputed", "resolved", "unresolved"}),
}

#: The only permitted ``source`` values for ``record_follow_up_date`` (never ``"agent"``).
ALLOWED_FOLLOW_UP_SOURCES: FrozenSet[str] = frozenset({"clinician", "rule_source"})

#: The fixed ``what_remains`` action vocabulary.
WHAT_REMAINS_ACTIONS: FrozenSet[str] = frozenset({
    "await_clinician_review", "acquire_dermoscopic_image", "review_image_or_reacquire",
    "confirm_or_cancel_pending_proposal", "record_outcome_or_close", "none",
})  # plus the parameterised "supply_fact:<fact>"


class EpisodeErrorV2(ValueError):
    """An episode primitive was called with structurally invalid arguments."""


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()




@dataclass(frozen=True)
class EpisodeEvent:
    schema_version: str
    event_id: str
    event_type: str
    idempotency_key: str
    actor: str
    session_id: str
    case_id: str
    record_version_at_event: int
    at: str
    previous_event_sha256: Optional[str]
    event_sha256: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version, "event_id": self.event_id, "event_type": self.event_type,
            "idempotency_key": self.idempotency_key, "actor": self.actor, "session_id": self.session_id,
            "case_id": self.case_id, "record_version_at_event": self.record_version_at_event, "at": self.at,
            "previous_event_sha256": self.previous_event_sha256, "payload": dict(self.payload),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.unsigned_dict(), "event_sha256": self.event_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeEvent":
        event = cls(
            schema_version=str(payload["schema_version"]),
            event_id=str(payload["event_id"]), event_type=str(payload["event_type"]),
            idempotency_key=str(payload["idempotency_key"]), actor=str(payload["actor"]),
            session_id=str(payload["session_id"]), case_id=str(payload["case_id"]),
            record_version_at_event=int(payload["record_version_at_event"]), at=str(payload["at"]),
            previous_event_sha256=(str(payload["previous_event_sha256"]) if payload.get("previous_event_sha256") else None),
            event_sha256=str(payload["event_sha256"]),
            payload=dict(payload.get("payload", {})),
        )
        if event.schema_version != EVENT_SCHEMA_VERSION:
            raise EpisodeErrorV2("unknown episode event schema")
        if event.event_type not in EVENT_TYPES:
            raise EpisodeErrorV2(f"unknown episode event type {event.event_type!r}")
        expected = hashlib.sha256(json.dumps(event.unsigned_dict(), sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False).encode("utf-8")).hexdigest()
        if expected != event.event_sha256:
            raise EpisodeErrorV2("episode event content hash mismatch")
        return event


def _make_event(
    *, episode_id: str, case_id: str, event_type: str, idempotency_key: str, actor: str, session_id: str,
    at: str, current_record_version: int, payload: Mapping[str, Any], previous_event_sha256: Optional[str],
) -> EpisodeEvent:
    event_id = _sha256_hex(f"{episode_id}|{case_id}|{event_type}|{idempotency_key}")
    event = EpisodeEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        event_id=event_id, event_type=event_type, idempotency_key=str(idempotency_key), actor=str(actor),
        session_id=str(session_id), case_id=case_id, record_version_at_event=int(current_record_version),
        at=str(at), previous_event_sha256=previous_event_sha256, event_sha256="", payload=dict(payload),
    )
    event_sha = hashlib.sha256(json.dumps(event.unsigned_dict(), sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode("utf-8")).hexdigest()
    return EpisodeEvent(**{**event.__dict__, "event_sha256": event_sha})


@dataclass(frozen=True)
class EpisodeCommandResult:
    status: str  # "ok" | "rejected"
    event: Optional[EpisodeEvent]
    state: "EpisodeState"
    is_new: bool = True
    reason: Optional[str] = None
    detail: str = ""




@dataclass(frozen=True)
class EpisodeState:
    episode_id: str
    case_id: str
    status: str
    latest_record_version: int
    pending_awaiting_fact: Optional[Tuple[str, bool]]  # (fact, unavailable)
    unavailable_facts: FrozenSet[str]
    outcome_confirmed: bool
    outcome: Optional[Dict[str, Any]]
    follow_up_date: Optional[str]
    follow_up_source: Optional[str]
    events: Tuple[EpisodeEvent, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id, "case_id": self.case_id, "status": self.status,
            "latest_record_version": self.latest_record_version,
            "pending_awaiting_fact": list(self.pending_awaiting_fact) if self.pending_awaiting_fact else None,
            "unavailable_facts": sorted(self.unavailable_facts), "outcome_confirmed": self.outcome_confirmed,
            "outcome": dict(self.outcome) if self.outcome is not None else None,
            "follow_up_date": self.follow_up_date, "follow_up_source": self.follow_up_source,
            "n_events": len(self.events),
        }


def derive_state(episode_id: str, case_id: str, events: Sequence[EpisodeEvent]) -> EpisodeState:
    """Pure fold: the episode's entire state is a function of its event log."""
    status = "not_started"
    latest_record_version = 0
    pending_awaiting_fact: Optional[Tuple[str, bool]] = None
    unavailable: set = set()
    outcome_confirmed = False
    outcome: Optional[Dict[str, Any]] = None
    follow_up_date: Optional[str] = None
    follow_up_source: Optional[str] = None

    previous_hash = None
    previous_at: Optional[datetime] = None
    previous_record_version = 0
    event_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    for ev in events:
        ev = EpisodeEvent.from_dict(ev.to_dict())
        if ev.case_id != case_id:
            raise EpisodeErrorV2(f"episode event belongs to {ev.case_id!r}, not {case_id!r}")
        if ev.previous_event_sha256 != previous_hash:
            raise EpisodeErrorV2("episode event chain is reordered, deleted, or spliced")
        if ev.event_id in event_ids or ev.idempotency_key in idempotency_keys:
            raise EpisodeErrorV2("duplicate episode event/idempotency key")
        try:
            current_at = datetime.fromisoformat(ev.at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EpisodeErrorV2("invalid episode event timestamp") from exc
        if previous_at is not None and current_at < previous_at:
            raise EpisodeErrorV2("episode event timestamps are out of order")
        if ev.record_version_at_event < previous_record_version:
            raise EpisodeErrorV2("episode event record versions move backwards")
        if status not in ALLOWED_TRANSITIONS[ev.event_type]:
            raise EpisodeErrorV2(f"invalid episode transition: {status} -> {ev.event_type}")
        previous_hash, previous_at, previous_record_version = ev.event_sha256, current_at, ev.record_version_at_event
        event_ids.add(ev.event_id); idempotency_keys.add(ev.idempotency_key)
        latest_record_version = max(latest_record_version, ev.record_version_at_event)
        t = ev.event_type
        if t == "review_requested":
            status = "requested"
            pending_awaiting_fact = None
        elif t == "awaiting_review":
            status = "awaiting_review"
            pending_awaiting_fact = None
        elif t == "awaiting_information":
            fact = str(ev.payload.get("fact"))
            unavail = bool(ev.payload.get("unavailable", False))
            status = "awaiting_information"
            pending_awaiting_fact = (fact, unavail)
            if unavail:
                unavailable.add(fact)
        elif t == "review_acknowledged":
            status = "acknowledged"
            pending_awaiting_fact = None
        elif t == "information_received":
            fact = ev.payload.get("fact")
            if fact is not None:
                unavailable.discard(str(fact))
            pending_awaiting_fact = None
            status = "new_information" if status == "resolved" else "information_received"
        elif t == "recomputed":
            status = "recomputed"
            pending_awaiting_fact = None
        elif t == "resolved":
            status = "resolved"
            pending_awaiting_fact = None
        elif t == "unresolved":
            status = "unresolved"
            pending_awaiting_fact = None
        elif t == "outcome_confirmed":
            outcome_confirmed = True
            outcome = dict(ev.payload.get("outcome", {}))
        elif t == "follow_up_date_recorded":
            follow_up_date = ev.payload.get("date")
            follow_up_source = ev.payload.get("source")
        elif t == "comparison_reminder_unsolicited":
            pass  # budget receipt only; it never changes clinical/review status

    return EpisodeState(
        episode_id=episode_id, case_id=case_id, status=status, latest_record_version=latest_record_version,
        pending_awaiting_fact=pending_awaiting_fact, unavailable_facts=frozenset(unavailable),
        outcome_confirmed=outcome_confirmed, outcome=outcome, follow_up_date=follow_up_date,
        follow_up_source=follow_up_source, events=tuple(events),
    )


def empty_state(episode_id: str, case_id: str) -> EpisodeState:
    return derive_state(episode_id, case_id, ())




def _existing(state: EpisodeState, idempotency_key: str) -> Optional[EpisodeEvent]:
    for ev in state.events:
        if ev.idempotency_key == idempotency_key:
            return ev
    return None


def _rejected(state: EpisodeState, reason: str, detail: str) -> EpisodeCommandResult:
    return EpisodeCommandResult(status="rejected", event=None, state=state, is_new=False, reason=reason, detail=detail)


def _apply(
    state: EpisodeState, *, event_type: str, idempotency_key: str, actor: str, session_id: str, at: str,
    current_record_version: int, expected_record_version: Optional[int], payload: Mapping[str, Any],
) -> EpisodeCommandResult:
    dup = _existing(state, idempotency_key)
    if dup is not None:
        if dup.event_type != event_type or dup.actor != str(actor) or dup.session_id != str(session_id) or dup.payload != dict(payload):
            return _rejected(state, "idempotency_conflict", "idempotency key was already used for a different command")
        return EpisodeCommandResult(status="ok", event=dup, state=state, is_new=False)
    if expected_record_version is not None and expected_record_version != current_record_version:
        return _rejected(state, "stale_record_version",
                          f"expected_record_version={expected_record_version} but the case is at {current_record_version}")
    if event_type not in EVENT_TYPES or state.status not in ALLOWED_TRANSITIONS[event_type]:
        return _rejected(state, "invalid_transition", f"cannot apply {event_type!r} while episode status is {state.status!r}")
    try:
        at_value = datetime.fromisoformat(str(at).replace("Z", "+00:00"))
        if state.events:
            last = datetime.fromisoformat(state.events[-1].at.replace("Z", "+00:00"))
            if at_value < last:
                return _rejected(state, "timestamp_out_of_order", "event timestamp precedes the current episode head")
    except ValueError:
        return _rejected(state, "timestamp_invalid", "event timestamp must be ISO-8601")
    event = _make_event(
        episode_id=state.episode_id, case_id=state.case_id, event_type=event_type, idempotency_key=idempotency_key,
        actor=actor, session_id=session_id, at=at, current_record_version=current_record_version, payload=payload,
        previous_event_sha256=(state.events[-1].event_sha256 if state.events else None),
    )
    new_state = derive_state(state.episode_id, state.case_id, state.events + (event,))
    return EpisodeCommandResult(status="ok", event=event, state=new_state, is_new=True)


def request_review(
    state: EpisodeState, *, reason: str, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    return _apply(state, event_type="review_requested", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload={"reason": reason})


def mark_awaiting_review(
    state: EpisodeState, *, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    return _apply(state, event_type="awaiting_review", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload={})


def mark_awaiting_information(
    state: EpisodeState, *, fact: str, unavailable: bool, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    if fact not in CLINICIAN_ASSERTION_FACTS:
        return _rejected(state, "unknown_fact", f"{fact!r} is not one of the nine amendable clinician facts")
    return _apply(state, event_type="awaiting_information", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload={"fact": fact, "unavailable": bool(unavailable)})


def acknowledge_review(
    state: EpisodeState, *, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    return _apply(state, event_type="review_acknowledged", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload={})


def record_information(
    state: EpisodeState, *, kind: str, ref: str, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None, fact: Optional[str] = None,
) -> EpisodeCommandResult:
    """``kind`` is one of ``amendment`` / ``new_image`` / ``source_change`` / ``memory_revocation``; ``ref`` names the triggering artefact (amendment id, image sha, ...)."""
    payload: Dict[str, Any] = {"kind": kind, "ref": ref}
    if fact is not None:
        payload["fact"] = fact
    return _apply(state, event_type="information_received", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload=payload)


def record_recomputed(
    state: EpisodeState, *, record_version: int, actor: str, session_id: str, idempotency_key: str, at: str,
    expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    dup = _existing(state, idempotency_key)
    if dup is not None:
        return EpisodeCommandResult(status="ok", event=dup, state=state, is_new=False)
    if record_version <= state.latest_record_version:
        return _rejected(state, "record_version_not_advanced",
                          f"record_version={record_version} does not advance past {state.latest_record_version}")
    return _apply(state, event_type="recomputed", idempotency_key=idempotency_key, actor=actor, session_id=session_id,
                  at=at, current_record_version=record_version, expected_record_version=expected_record_version, payload={})


def resolve(
    state: EpisodeState, *, reason: str, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    return _apply(state, event_type="resolved", idempotency_key=idempotency_key, actor=actor, session_id=session_id,
                  at=at, current_record_version=current_record_version, expected_record_version=expected_record_version,
                  payload={"reason": reason})


def mark_unresolved(
    state: EpisodeState, *, reason: str, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    return _apply(state, event_type="unresolved", idempotency_key=idempotency_key, actor=actor, session_id=session_id,
                  at=at, current_record_version=current_record_version, expected_record_version=expected_record_version,
                  payload={"reason": reason})


def confirm_outcome(
    state: EpisodeState, *, outcome: Mapping[str, Any], evidence_ref: str, actor: str, session_id: str,
    idempotency_key: str, at: str, current_record_version: int, expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    if not evidence_ref:
        return _rejected(state, "evidence_ref_required", "confirm_outcome requires a non-empty evidence_ref")
    payload = {"outcome": dict(outcome), "evidence_ref": evidence_ref}
    return _apply(state, event_type="outcome_confirmed", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload=payload)


def record_follow_up_date(
    state: EpisodeState, *, date: str, actor: str, session_id: str, idempotency_key: str, at: str,
    current_record_version: int, source: str = "clinician", expected_record_version: Optional[int] = None,
) -> EpisodeCommandResult:
    if source not in ALLOWED_FOLLOW_UP_SOURCES:
        return _rejected(state, "invalid_follow_up_source",
                          f"source={source!r} is not permitted; a follow-up date must come from the clinician or a rule's own source text")
    payload = {"date": date, "source": source}
    return _apply(state, event_type="follow_up_date_recorded", idempotency_key=idempotency_key, actor=actor,
                  session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=expected_record_version, payload=payload)


def record_unsolicited_memory_reminder(
    state: EpisodeState, *, comparison_receipt_sha256: str, actor: str, session_id: str,
    idempotency_key: str, at: str, current_record_version: int,
) -> EpisodeCommandResult:
    if not comparison_receipt_sha256:
        return _rejected(state, "comparison_receipt_required", "memory reminder requires a comparison receipt hash")
    return _apply(state, event_type="comparison_reminder_unsolicited", idempotency_key=idempotency_key,
                  actor=actor, session_id=session_id, at=at, current_record_version=current_record_version,
                  expected_record_version=None, payload={"comparison_receipt_sha256": comparison_receipt_sha256})




@dataclass(frozen=True)
class WhatRemains:
    action: str
    sentence: str


def what_remains(
    state: EpisodeState, record: StructuredCaseRecordV2, *, pending_proposal: Optional[Mapping[str, Any]] = None,
) -> WhatRemains:
    """Exactly one next permitted action, in priority order:"""
    if pending_proposal is not None:
        return WhatRemains("confirm_or_cancel_pending_proposal",
                            "A proposed amendment is awaiting clinician confirmation or cancellation.")

    if state.pending_awaiting_fact is not None:
        fact, unavailable = state.pending_awaiting_fact
        if not unavailable:
            return WhatRemains(f"supply_fact:{fact}", f"Awaiting the clinician's answer for {fact}.")

    if record.image_status.state == "unavailable_no_image":
        return WhatRemains("acquire_dermoscopic_image", "No usable dermoscopic image; acquire one to proceed with image-based assessment.")
    if record.image_status.state == "limited_review_flagged":
        return WhatRemains("review_image_or_reacquire", "The image is limited/review-flagged; review it or reacquire a usable one.")

    if state.status in ("requested", "awaiting_review"):
        return WhatRemains("await_clinician_review", "Awaiting clinician review of this case.")
    if state.status == "resolved" and not state.outcome_confirmed:
        return WhatRemains("record_outcome_or_close", "The episode is resolved; record a confirmed outcome or close it.")

    return WhatRemains("none", "Nothing is currently outstanding for this episode.")


__all__ = [
    "ALLOWED_FOLLOW_UP_SOURCES",
    "EPISODE_VERSION",
    "EVENT_SCHEMA_VERSION",
    "ALLOWED_TRANSITIONS",
    "EVENT_TYPES",
    "EpisodeCommandResult",
    "EpisodeErrorV2",
    "EpisodeEvent",
    "EpisodeState",
    "WHAT_REMAINS_ACTIONS",
    "WhatRemains",
    "acknowledge_review",
    "confirm_outcome",
    "derive_state",
    "empty_state",
    "mark_awaiting_information",
    "mark_awaiting_review",
    "mark_unresolved",
    "record_follow_up_date",
    "record_information",
    "record_unsolicited_memory_reminder",
    "record_recomputed",
    "request_review",
    "resolve",
    "what_remains",
]
