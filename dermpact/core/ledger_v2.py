"""WP-A2: confirmation-gated amendment ledger v2."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.guidelines.rules import canonical_json
from dermpact.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES, INDETERMINATE

from dermpact.core.gate_v2 import VerificationGateV2
from dermpact.core.record_v2 import (
    AmendmentDeltaV2,
    AmendmentEntryV2,
    CLINICIAN_ASSERTION_FACTS,
    StructuredCaseRecordV2,
    current_value,
    effective_dermoscopic_present,
    finalize_with_amendments,
    verify_chain as verify_chain_v2,
)
from dermpact.core.record_v2 import _diff_consequences as _diff_consequences_v2

AMENDMENT_KINDS: Tuple[str, ...] = (
    "criterion_value", "clinician_fact", "clinician_fact_retraction", "reliability_override",
)
DEFAULT_TTL_S = 600


class AmendmentErrorV2(ValueError):
    """A ledger v2 primitive was called with structurally invalid arguments."""


class _TimestampError(ValueError):
    """Raised by ``_parse_iso``; callers translate this into a typed rejection receipt."""


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, normalising to a timezone-AWARE UTC datetime (see :mod:`dermpact.record.amendments` for the rationale: a naive input is treated as UTC so two datetimes are never compared naive-vs-aware)."""
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise _TimestampError(f"malformed timestamp {value!r}: {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt




@dataclass(frozen=True)
class AmendmentProposal:
    kind: str
    field: str
    value: Any = None


@dataclass(frozen=True)
class ProposalReceipt:
    status: str  # "proposed" | "rejected"
    case_id: str
    record_sha256: str
    kind: str
    field: str
    before: Any = None
    after: Any = None
    actor: Optional[str] = None
    session_id: Optional[str] = None
    proposal_id: Optional[str] = None
    reason: Optional[str] = None
    detail: str = ""
    next_step_changed: Optional[bool] = None
    consequence_preview: Optional[Dict[str, Any]] = None
    consequence_preview_sha256: Optional[str] = None
    clinician_assertions_change: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ConfirmationToken:
    status: str  # "issued" | "rejected"
    proposal_id: Optional[str] = None
    token: Optional[str] = None
    token_sha256: Optional[str] = None
    case_id: Optional[str] = None
    record_sha256: Optional[str] = None
    consequence_preview_sha256: Optional[str] = None
    requires_acknowledgement: Optional[bool] = None
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None
    session_id: Optional[str] = None
    reason: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class CommitReceipt:
    status: str  # "committed" | "rejected"
    reason: Optional[str] = None
    detail: str = ""
    record: Optional[StructuredCaseRecordV2] = None
    entry: Optional[AmendmentEntryV2] = None


@dataclass(frozen=True)
class CancellationReceipt:
    status: str  # "cancelled" | "rejected"
    proposal_id: str
    reason: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class _PendingProposal:
    proposal_id: str
    case_id: str
    record_sha256: str
    delta: AmendmentDeltaV2
    actor: str
    session_id: str
    proposed_at: str
    consequence_preview: Dict[str, Any]
    consequence_preview_sha256: str
    next_step_changed: bool


@dataclass
class _TokenRecord:
    proposal_id: str
    case_id: str
    record_sha256: str
    consequence_preview_sha256: str
    requires_ack: bool
    actor: str
    session_id: str
    issued_at: str
    expires_at: str
    used: bool = False


def _reject_proposal(record: StructuredCaseRecordV2, kind: str, fld: str, reason: str, detail: str) -> ProposalReceipt:
    return ProposalReceipt(
        status="rejected", case_id=record.identity.case_id, record_sha256=record.record_sha256,
        kind=kind, field=fld, reason=reason, detail=detail,
    )


def _validate_proposal(record: StructuredCaseRecordV2, proposal: AmendmentProposal) -> Optional[ProposalReceipt]:
    kind, fld, value = proposal.kind, proposal.field, proposal.value
    if kind not in AMENDMENT_KINDS:
        return _reject_proposal(record, kind, fld, "kind_not_allowed", f"unknown amendment kind {kind!r}")

    if kind == "criterion_value":
        if fld not in CRITERION_NAMES:
            return _reject_proposal(record, kind, fld, "kind_not_allowed", f"{fld!r} is not a checklist criterion")
        if not effective_dermoscopic_present(record.inputs.input_mask["dermoscopic"], record.reliability.dermoscopic_present_override):
            return _reject_proposal(record, kind, fld, "perception_unavailable",
                                     "no usable dermoscopic image; criterion values cannot be amended")
        allowed = set(CRITERIA_BY_NAME[fld].vocabulary) | {INDETERMINATE}
        if value not in allowed:
            return _reject_proposal(record, kind, fld, "value_not_allowed", f"{value!r} not in {sorted(allowed)}")
    elif kind == "clinician_fact":
        if fld not in CLINICIAN_ASSERTION_FACTS:
            return _reject_proposal(record, kind, fld, "kind_not_allowed", f"{fld!r} is not an amendable clinician fact")
        if value is None:
            return _reject_proposal(record, kind, fld, "value_not_allowed",
                                     "clinician_fact cannot assert None; use clinician_fact_retraction to retract")
        if fld == "patient.total_nevus_count":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return _reject_proposal(record, kind, fld, "value_not_allowed",
                                         "patient.total_nevus_count must be a non-negative int")
        elif not isinstance(value, bool):
            return _reject_proposal(record, kind, fld, "value_not_allowed", f"{fld} must be a bool")
    elif kind == "clinician_fact_retraction":
        if fld not in CLINICIAN_ASSERTION_FACTS:
            return _reject_proposal(record, kind, fld, "kind_not_allowed", f"{fld!r} is not an amendable clinician fact")
        if value is not None:
            return _reject_proposal(record, kind, fld, "value_not_allowed",
                                     "clinician_fact_retraction ignores any supplied value; it always retracts to unknown")
        if current_value(record, fld) is None:
            return _reject_proposal(record, kind, fld, "no_change", f"{fld} is already unasserted")
        return None  # no-op check below does not apply (before != after=None is guaranteed above)
    elif kind == "reliability_override":
        if fld != "image.dermoscopic_present":
            return _reject_proposal(record, kind, fld, "kind_not_allowed",
                                     "reliability_override may only assert image.dermoscopic_present")
        if value is not False:
            return _reject_proposal(record, kind, fld, "value_not_allowed",
                                     "reliability_override may only set image.dermoscopic_present to False "
                                     "(IQA/OOD pass states are Phase-3 outputs, never a clinician assertion)")
    else:  # pragma: no cover - guarded by AMENDMENT_KINDS membership above
        return _reject_proposal(record, kind, fld, "kind_not_allowed", f"unhandled kind {kind!r}")

    if current_value(record, fld) == value:
        return _reject_proposal(record, kind, fld, "no_change", f"{fld} is already {value!r}")
    return None


class AmendmentLedgerV2:
    """Host-owned, confirmation-gated amendment ledger v2 for one running process."""

    def __init__(
        self, *, catalogue, engine, calibrators: Optional[Mapping[str, Any]] = None, bundle: Optional[Any] = None,
        ledger_nonce: Optional[str] = None, default_ttl_s: int = DEFAULT_TTL_S,
    ):
        self._catalogue = catalogue
        self._engine = engine
        self._calibrators = calibrators
        self._bundle = bundle
        self._gate = VerificationGateV2(catalogue, engine, calibrators=calibrators, bundle=bundle)
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingProposal] = {}
        self._tokens: Dict[str, _TokenRecord] = {}
        self._superseded: set = set()
        self._events: List[Dict[str, Any]] = []
        self._nonce = ledger_nonce or uuid.uuid4().hex
        self._counter = 0
        self._default_ttl_s = int(default_ttl_s)

    # -- event log / restart --------------------------------------------

    @property
    def history(self) -> Tuple[Dict[str, Any], ...]:
        with self._lock:
            return tuple(copy.deepcopy(e) for e in self._events)

    def to_events(self) -> Tuple[Dict[str, Any], ...]:
        """Alias of :attr:`history`: the full append-only event log, granular enough for :meth:`replay` to reconstruct pending proposals, issued tokens and superseded record hashes."""
        return self.history

    @classmethod
    def replay(
        cls, events: Sequence[Mapping[str, Any]], *, catalogue, engine,
        calibrators: Optional[Mapping[str, Any]] = None, bundle: Optional[Any] = None,
        ledger_nonce: Optional[str] = None, default_ttl_s: int = DEFAULT_TTL_S,
    ) -> "AmendmentLedgerV2":
        ledger = cls(catalogue=catalogue, engine=engine, calibrators=calibrators, bundle=bundle,
                     ledger_nonce=ledger_nonce, default_ttl_s=default_ttl_s)
        with ledger._lock:
            for ev in events:
                ledger._replay_one_locked(dict(ev))
        return ledger

    def _replay_one_locked(self, ev: Dict[str, Any]) -> None:
        etype = ev.get("type")
        self._events.append(copy.deepcopy(ev))
        if etype == "proposed":
            delta = AmendmentDeltaV2(kind=ev["kind"], field=ev["field"], before=ev["before"], after=ev["after"],
                                     actor=ev["actor"], session_id=ev["session_id"])
            self._pending[ev["proposal_id"]] = _PendingProposal(
                proposal_id=ev["proposal_id"], case_id=ev["case_id"], record_sha256=ev["record_sha256"], delta=delta,
                actor=ev["actor"], session_id=ev["session_id"], proposed_at=ev["proposed_at"],
                consequence_preview=copy.deepcopy(ev["consequence_preview"]),
                consequence_preview_sha256=ev["consequence_preview_sha256"],
                next_step_changed=bool(ev["next_step_changed"]),
            )
        elif etype == "issued":
            self._tokens[ev["token"]] = _TokenRecord(
                proposal_id=ev["proposal_id"], case_id=ev["case_id"], record_sha256=ev["record_sha256"],
                consequence_preview_sha256=ev["consequence_preview_sha256"], requires_ack=bool(ev["requires_acknowledgement"]),
                actor=ev["actor"], session_id=ev["session_id"], issued_at=ev["issued_at"], expires_at=ev["expires_at"],
                used=False,
            )
        elif etype == "committed":
            self._superseded.add(ev["from_record_sha256"])
            self._pending.pop(ev["proposal_id"], None)
            token = ev.get("token")
            if token and token in self._tokens:
                self._tokens[token].used = True
        elif etype in ("cancelled", "expired"):
            self._pending.pop(ev["proposal_id"], None)
            token = ev.get("token")
            if token and token in self._tokens:
                self._tokens[token].used = True

    def _expire_pending_locked(self, now: str, *, exclude_token: Optional[str] = None) -> None:
        try:
            now_dt = _parse_iso(now)
        except _TimestampError:
            return
        for token, rec in list(self._tokens.items()):
            if rec.used or token == exclude_token:
                continue
            try:
                expires_dt = _parse_iso(rec.expires_at)
            except _TimestampError:
                continue
            if now_dt > expires_dt:
                rec.used = True
                self._events.append({
                    "type": "expired", "proposal_id": rec.proposal_id, "case_id": rec.case_id,
                    "token": token, "token_sha256": _sha256_hex(token), "checked_at": now, "expires_at": rec.expires_at,
                })
                self._pending.pop(rec.proposal_id, None)

    def expire_pending(self, now: str) -> None:
        with self._lock:
            self._expire_pending_locked(now)

    # -- 1. propose -----------------------------------------------------

    def propose(
        self, record: StructuredCaseRecordV2, proposal: AmendmentProposal, *, actor: str, session_id: str,
        proposed_at: str, statement: str = "",
    ) -> ProposalReceipt:
        proposal = AmendmentProposal(kind=str(proposal.kind), field=str(proposal.field), value=copy.deepcopy(proposal.value))
        with self._lock:
            self._expire_pending_locked(proposed_at)
            rejected = _validate_proposal(record, proposal)
            if rejected is not None:
                return dataclasses.replace(rejected, actor=str(actor), session_id=str(session_id))

            fld = proposal.field
            before = current_value(record, fld)
            after = None if proposal.kind == "clinician_fact_retraction" else proposal.value
            delta = AmendmentDeltaV2(kind=proposal.kind, field=fld, before=before, after=after,
                                     actor=str(actor), session_id=str(session_id))

            proposal_id = _canonical_sha256({
                "kind": proposal.kind, "field": fld, "value": after,
                "record_sha256": record.record_sha256, "actor": str(actor), "session_id": str(session_id),
            })

            preview_record = record.with_amendment(
                delta=delta, catalogue=self._catalogue, engine=self._engine, calibrators=self._calibrators,
                created_at=proposed_at, bundle=self._bundle,
            )
            diff = dict(_diff_consequences_v2(record, preview_record))
            diff_sha = _canonical_sha256(diff)
            assertions_change = (
                {"field": fld, "before": before, "after": after} if fld in CLINICIAN_ASSERTION_FACTS else None
            )

            self._pending[proposal_id] = _PendingProposal(
                proposal_id=proposal_id, case_id=record.identity.case_id, record_sha256=record.record_sha256,
                delta=delta, actor=str(actor), session_id=str(session_id), proposed_at=str(proposed_at),
                consequence_preview=copy.deepcopy(diff), consequence_preview_sha256=diff_sha,
                next_step_changed=bool(diff["next_step_changed"]),
            )
            self._events.append({
                "type": "proposed", "proposal_id": proposal_id, "case_id": record.identity.case_id,
                "record_sha256": record.record_sha256, "kind": proposal.kind, "field": fld, "before": before, "after": after,
                "actor": str(actor), "session_id": str(session_id), "proposed_at": str(proposed_at), "statement": str(statement),
                "consequence_preview": copy.deepcopy(diff), "consequence_preview_sha256": diff_sha,
                "next_step_changed": bool(diff["next_step_changed"]), "clinician_assertions_change": assertions_change,
            })
            return ProposalReceipt(
                status="proposed", case_id=record.identity.case_id, record_sha256=record.record_sha256,
                kind=proposal.kind, field=fld, before=before, after=after, actor=str(actor), session_id=str(session_id),
                proposal_id=proposal_id, next_step_changed=bool(diff["next_step_changed"]),
                consequence_preview=copy.deepcopy(diff), consequence_preview_sha256=diff_sha,
                clinician_assertions_change=assertions_change,
            )

    # -- 2. issue_confirmation_token -------------------------------------

    def issue_confirmation_token(
        self, proposal_id: str, *, issued_at: str, session_id: str, ttl_s: Optional[int] = None,
    ) -> ConfirmationToken:
        with self._lock:
            try:
                issued_dt = _parse_iso(issued_at)
            except _TimestampError as exc:
                return ConfirmationToken(status="rejected", reason="timestamp_invalid", detail=str(exc))
            pending = self._pending.get(proposal_id)
            if pending is None:
                return ConfirmationToken(status="rejected", reason="proposal_not_found",
                                          detail=f"no pending proposal {proposal_id!r}")
            if str(session_id) != pending.session_id:
                return ConfirmationToken(status="rejected", reason="session_mismatch",
                                          detail="token requested from a different session than the one that proposed the amendment")
            effective_ttl = self._default_ttl_s if ttl_s is None else int(ttl_s)
            self._counter += 1
            token = _sha256_hex(
                f"{pending.case_id}|{pending.record_sha256}|{pending.proposal_id}|"
                f"{pending.consequence_preview_sha256}|{self._nonce}|{self._counter}"
            )
            expires_dt = issued_dt + timedelta(seconds=float(effective_ttl))
            expires_at = expires_dt.isoformat()
            self._tokens[token] = _TokenRecord(
                proposal_id=proposal_id, case_id=pending.case_id, record_sha256=pending.record_sha256,
                consequence_preview_sha256=pending.consequence_preview_sha256, requires_ack=pending.next_step_changed,
                actor=pending.actor, session_id=pending.session_id, issued_at=str(issued_at), expires_at=expires_at,
            )
            self._events.append({
                "type": "issued", "proposal_id": proposal_id, "case_id": pending.case_id, "record_sha256": pending.record_sha256,
                "token": token, "token_sha256": _sha256_hex(token), "consequence_preview_sha256": pending.consequence_preview_sha256,
                "requires_acknowledgement": pending.next_step_changed, "actor": pending.actor, "session_id": pending.session_id,
                "issued_at": str(issued_at), "expires_at": expires_at,
            })
            return ConfirmationToken(
                status="issued", proposal_id=proposal_id, token=token, token_sha256=_sha256_hex(token),
                case_id=pending.case_id, record_sha256=pending.record_sha256,
                consequence_preview_sha256=pending.consequence_preview_sha256,
                requires_acknowledgement=pending.next_step_changed, issued_at=str(issued_at), expires_at=expires_at,
                session_id=pending.session_id,
            )

    # -- 3. commit ---------------------------------------------------------

    def commit(
        self, record: StructuredCaseRecordV2, token: Optional[str], *, now: str, actor: str, session_id: str,
        acknowledged_consequence_sha256: Optional[str] = None,
    ) -> CommitReceipt:
        with self._lock:
            if not token:
                return CommitReceipt(status="rejected", reason="token_missing", detail="no token was presented")
            try:
                now_dt = _parse_iso(now)
            except _TimestampError as exc:
                return CommitReceipt(status="rejected", reason="timestamp_invalid", detail=str(exc))
            self._expire_pending_locked(now, exclude_token=token)
            rec = self._tokens.get(token)
            if rec is None:
                return CommitReceipt(status="rejected", reason="token_unknown", detail="token not recognised")
            if rec.used:
                return CommitReceipt(status="rejected", reason="token_used", detail="token already consumed")
            try:
                expires_dt = _parse_iso(rec.expires_at)
            except _TimestampError as exc:
                return CommitReceipt(status="rejected", reason="timestamp_invalid", detail=str(exc))
            if now_dt > expires_dt:
                rec.used = True
                self._events.append({
                    "type": "expired", "proposal_id": rec.proposal_id, "case_id": rec.case_id,
                    "token": token, "token_sha256": _sha256_hex(token), "checked_at": now, "expires_at": rec.expires_at,
                })
                self._pending.pop(rec.proposal_id, None)
                return CommitReceipt(status="rejected", reason="token_expired", detail=f"token expired at {rec.expires_at}")
            if rec.case_id != record.identity.case_id:
                return CommitReceipt(status="rejected", reason="token_case_mismatch",
                                      detail=f"token is bound to case {rec.case_id!r}, not {record.identity.case_id!r}")
            if str(actor) != rec.actor:
                return CommitReceipt(status="rejected", reason="actor_mismatch",
                                      detail="commit actor does not match the actor who proposed the amendment")
            if str(session_id) != rec.session_id:
                return CommitReceipt(status="rejected", reason="session_mismatch",
                                      detail="commit session does not match the session that proposed the amendment")
            if rec.record_sha256 != record.record_sha256:
                return CommitReceipt(status="rejected", reason="stale_record_version",
                                      detail="token is bound to a record hash that no longer matches (concurrent amendment)")
            if rec.record_sha256 in self._superseded:
                return CommitReceipt(status="rejected", reason="stale_record_version",
                                      detail="this record version was already amended by another committed change")
            if rec.requires_ack:
                if acknowledged_consequence_sha256 is None:
                    return CommitReceipt(status="rejected", reason="acknowledgement_required",
                                          detail="this amendment changes the recommended next step and must be acknowledged")
                if acknowledged_consequence_sha256 != rec.consequence_preview_sha256:
                    return CommitReceipt(status="rejected", reason="acknowledgement_mismatch",
                                          detail="acknowledged_consequence_sha256 does not match the previewed consequence")

            pending = self._pending.get(rec.proposal_id)
            if pending is None:
                return CommitReceipt(status="rejected", reason="proposal_not_found",
                                      detail="the underlying proposal is no longer pending")

            before_gate = self._gate.verify(record, expected_case_id=record.identity.case_id, now=now_dt.date(), lineage=[record])
            if not before_gate.passed and any(f.code in ("integrity", "entity_mismatch", "calculation") for f in before_gate.failures):
                return CommitReceipt(status="rejected", reason="source_record_invalid",
                                      detail=f"record fails verification before commit: {[f.code for f in before_gate.failures]}")

            rebuilt = record.with_amendment(
                delta=pending.delta, catalogue=self._catalogue, engine=self._engine, calibrators=self._calibrators,
                created_at=now, bundle=self._bundle,
            )
            fresh_diff = dict(_diff_consequences_v2(record, rebuilt))
            fresh_sha = _canonical_sha256(fresh_diff)
            if fresh_sha != rec.consequence_preview_sha256:
                return CommitReceipt(status="rejected", reason="consequence_drift",
                                      detail="recomputing the amendment at commit time no longer matches the preview")

            entry = AmendmentEntryV2(
                amendment_id=pending.proposal_id, kind=pending.delta.kind, field=pending.delta.field,
                before=pending.delta.before, after=pending.delta.after, actor=pending.actor, session_id=pending.session_id,
                proposed_at=pending.proposed_at, committed_at=str(now), proposal_sha256=pending.proposal_id,
                token_sha256=_sha256_hex(token), consequence_preview_sha256=pending.consequence_preview_sha256,
                recomputed_consequences=fresh_diff,
            )
            final = finalize_with_amendments(rebuilt, tuple(record.amendments) + (entry,))

            after_gate = self._gate.verify(final, expected_case_id=final.identity.case_id, now=now_dt.date(), lineage=[record, final])
            if not after_gate.passed and any(f.code in ("integrity", "entity_mismatch", "calculation", "amendment_consistency") for f in after_gate.failures):
                return CommitReceipt(status="rejected", reason="result_record_invalid",
                                      detail=f"resulting record fails verification: {[f.code for f in after_gate.failures]}")

            rec.used = True
            self._superseded.add(record.record_sha256)
            self._pending.pop(pending.proposal_id, None)
            self._events.append({
                "type": "committed", "amendment_id": entry.amendment_id, "proposal_id": pending.proposal_id,
                "case_id": final.identity.case_id, "from_record_sha256": record.record_sha256,
                "to_record_sha256": final.record_sha256, "token": token, "token_sha256": _sha256_hex(token),
                "actor": str(actor), "session_id": str(session_id), "committed_at": str(now),
            })
            return CommitReceipt(status="committed", record=final, entry=entry)

    # -- cancel ------------------------------------------------------------

    def cancel(self, proposal_id: str, *, session_id: str, now: str) -> CancellationReceipt:
        with self._lock:
            try:
                _parse_iso(now)
            except _TimestampError as exc:
                return CancellationReceipt(status="rejected", proposal_id=proposal_id, reason="timestamp_invalid", detail=str(exc))
            self._expire_pending_locked(now)
            pending = self._pending.get(proposal_id)
            if pending is None:
                return CancellationReceipt(status="rejected", proposal_id=proposal_id, reason="proposal_not_found",
                                            detail=f"no pending proposal {proposal_id!r}")
            if str(session_id) != pending.session_id:
                return CancellationReceipt(status="rejected", proposal_id=proposal_id, reason="session_mismatch",
                                            detail="cancel session does not match the session that proposed the amendment")
            cancelled_token: Optional[str] = None
            for tok, rec in self._tokens.items():
                if rec.proposal_id == proposal_id and not rec.used:
                    rec.used = True
                    cancelled_token = tok
            self._pending.pop(proposal_id, None)
            self._events.append({
                "type": "cancelled", "proposal_id": proposal_id, "case_id": pending.case_id,
                "token": cancelled_token, "session_id": str(session_id), "cancelled_at": str(now),
            })
            return CancellationReceipt(status="cancelled", proposal_id=proposal_id)

    # -- verify_chain --------------------------------------------------------

    def verify_chain(self, records: Sequence[StructuredCaseRecordV2]):
        return verify_chain_v2(records, catalogue=self._catalogue, engine=self._engine,
                               calibrators=self._calibrators, bundle=self._bundle)


__all__ = [
    "AMENDMENT_KINDS",
    "AmendmentErrorV2",
    "AmendmentLedgerV2",
    "AmendmentProposal",
    "CancellationReceipt",
    "CommitReceipt",
    "ConfirmationToken",
    "DEFAULT_TTL_S",
    "ProposalReceipt",
]
