"""Phase 5: append-only amendment ledger with confirmation-gated commit."""
from __future__ import annotations

import copy
import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.guidelines.engine import ApplicabilityEngine
from dermpact.guidelines.rules import RuleCatalogue, canonical_json
from dermpact.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES, INDETERMINATE

from dermpact.record.case_record import (
    AmendmentDelta,
    AmendmentEntry,
    CLINICIAN_FIELD_TO_ATTR,
    ChainVerdict,
    StructuredCaseRecord,
    _diff_consequences,
    current_value,
    effective_dermoscopic_present,
    finalize_with_amendments,
    verify_chain,
)
from dermpact.record.gate import VerificationGate

AMENDMENT_KINDS: Tuple[str, ...] = ("criterion_value", "clinician_fact", "reliability_override")


class AmendmentError(ValueError):
    """A ledger primitive was called with structurally invalid arguments."""


class _TimestampError(ValueError):
    """Raised by ``_parse_iso`` on a malformed timestamp; callers translate this into a typed rejection receipt rather than letting it propagate raw."""


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, normalising to a timezone-AWARE UTC datetime."""
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
    value: Any


@dataclass(frozen=True)
class ProposalReceipt:
    status: str  # "proposed" | "rejected"
    case_id: str
    record_sha256: str
    kind: str
    field: str
    before: Any = None
    after: Any = None
    proposal_id: Optional[str] = None
    reason: Optional[str] = None
    detail: str = ""
    next_step_changed: Optional[bool] = None
    consequence_preview: Optional[Dict[str, Any]] = None
    consequence_preview_sha256: Optional[str] = None


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
    reason: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class CommitReceipt:
    status: str  # "committed" | "rejected"
    reason: Optional[str] = None
    detail: str = ""
    record: Optional[StructuredCaseRecord] = None
    entry: Optional[AmendmentEntry] = None


@dataclass(frozen=True)
class _PendingProposal:
    proposal_id: str
    case_id: str
    record_sha256: str
    delta: AmendmentDelta
    actor: str
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
    issued_at: str
    expires_at: str
    used: bool = False


def _reject_proposal(record: StructuredCaseRecord, kind: str, fld: str, reason: str, detail: str) -> ProposalReceipt:
    return ProposalReceipt(
        status="rejected", case_id=record.identity.case_id, record_sha256=record.record_sha256,
        kind=kind, field=fld, reason=reason, detail=detail,
    )


def _validate_proposal(record: StructuredCaseRecord, proposal: AmendmentProposal) -> Optional[ProposalReceipt]:
    kind, fld, value = proposal.kind, proposal.field, proposal.value
    if kind not in AMENDMENT_KINDS:
        return _reject_proposal(record, kind, fld, "kind_not_allowed", f"unknown amendment kind {kind!r}")

    if kind == "criterion_value":
        if fld not in CRITERION_NAMES:
            return _reject_proposal(record, kind, fld, "kind_not_allowed", f"{fld!r} is not a checklist criterion")
        if not effective_dermoscopic_present(record):
            return _reject_proposal(record, kind, fld, "perception_unavailable",
                                     "no usable dermoscopic image; criterion values cannot be amended")
        allowed = set(CRITERIA_BY_NAME[fld].vocabulary) | {INDETERMINATE}
        if value not in allowed:
            return _reject_proposal(record, kind, fld, "value_not_allowed",
                                     f"{value!r} not in {sorted(allowed)}")
    elif kind == "clinician_fact":
        if fld not in CLINICIAN_FIELD_TO_ATTR:
            return _reject_proposal(record, kind, fld, "kind_not_allowed",
                                     f"{fld!r} is not an amendable clinician fact")
        if fld == "patient.total_nevus_count":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return _reject_proposal(record, kind, fld, "value_not_allowed",
                                         "patient.total_nevus_count must be a non-negative int")
        elif not isinstance(value, bool):
            return _reject_proposal(record, kind, fld, "value_not_allowed", f"{fld} must be a bool")
    elif kind == "reliability_override":
        if fld != "image.dermoscopic_present":
            return _reject_proposal(record, kind, fld, "kind_not_allowed",
                                     "reliability_override may only assert image.dermoscopic_present")
        if value is not False:
            return _reject_proposal(record, kind, fld, "value_not_allowed",
                                     "reliability_override may only set image.dermoscopic_present to False "
                                     "(IQA/OOD pass states are Phase-3 outputs, never a clinician assertion)")
    else:
        return _reject_proposal(record, kind, fld, "kind_not_allowed", f"unhandled kind {kind!r}")  # pragma: no cover

    if current_value(record, fld) == value:
        return _reject_proposal(record, kind, fld, "no_change", f"{fld} is already {value!r}")
    return None


class AmendmentLedger:
    """Host-owned, confirmation-gated amendment ledger for one running process."""

    def __init__(self, *, catalogue: RuleCatalogue, engine: ApplicabilityEngine, ledger_nonce: Optional[str] = None):
        self._catalogue = catalogue
        self._engine = engine
        self._gate = VerificationGate(catalogue, engine)
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingProposal] = {}
        self._tokens: Dict[str, _TokenRecord] = {}
        self._superseded: set = set()
        self._history: List[Dict[str, Any]] = []
        self._nonce = ledger_nonce or uuid.uuid4().hex
        self._counter = 0

    @property
    def history(self) -> Tuple[Dict[str, Any], ...]:
        with self._lock:
            return tuple(copy.deepcopy(h) for h in self._history)

    def _expire_pending_locked(self, now: str, *, exclude_token: Optional[str] = None) -> None:
        """Sweep every not-yet-used token whose TTL has passed as of ``now``, recording each as ``"expired"`` in history (brief #2: "a proposal that expires without a commit attempt must appear in history as expired")."""
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
                self._history.append({
                    "type": "expired", "proposal_id": rec.proposal_id, "token_sha256": _sha256_hex(token),
                    "checked_at": now, "expires_at": rec.expires_at,
                })
                self._pending.pop(rec.proposal_id, None)

    def expire_pending(self, now: str) -> None:
        """Public, self-locking form of the sweep above -- for standalone use (tests, or a caller that wants to expire abandoned proposals without also proposing/committing something)."""
        with self._lock:
            self._expire_pending_locked(now)

    # -- 1. propose ---------------------------------------------------

    def propose(
        self, record: StructuredCaseRecord, proposal: AmendmentProposal, *, actor: str, proposed_at: str,
    ) -> ProposalReceipt:
        proposal = AmendmentProposal(kind=str(proposal.kind), field=str(proposal.field), value=copy.deepcopy(proposal.value))
        with self._lock:
            self._expire_pending_locked(proposed_at)
            rejected = _validate_proposal(record, proposal)
            if rejected is not None:
                return rejected

            before = current_value(record, proposal.field)
            delta = AmendmentDelta(kind=proposal.kind, field=proposal.field, before=before, after=proposal.value)

            proposal_id = _canonical_sha256({
                "kind": proposal.kind, "field": proposal.field, "value": proposal.value,
                "record_sha256": record.record_sha256, "actor": actor,
            })

            preview_record = record.with_amendment(
                delta=delta, catalogue=self._catalogue, engine=self._engine, created_at=proposed_at,
            )
            diff = _diff_consequences(record, preview_record)
            diff_sha = _canonical_sha256(diff)

            self._pending[proposal_id] = _PendingProposal(
                proposal_id=proposal_id, case_id=record.identity.case_id, record_sha256=record.record_sha256,
                delta=delta, actor=str(actor), proposed_at=str(proposed_at),
                consequence_preview=copy.deepcopy(diff), consequence_preview_sha256=diff_sha,
                next_step_changed=bool(diff["next_step_changed"]),
            )
            return ProposalReceipt(
                status="proposed", case_id=record.identity.case_id, record_sha256=record.record_sha256,
                kind=proposal.kind, field=proposal.field, before=before, after=proposal.value,
                proposal_id=proposal_id, next_step_changed=bool(diff["next_step_changed"]),
                consequence_preview=copy.deepcopy(diff), consequence_preview_sha256=diff_sha,
            )

    # -- 2. issue_confirmation_token -----------------------------------

    def issue_confirmation_token(self, proposal_id: str, *, issued_at: str, ttl_s: int) -> ConfirmationToken:
        with self._lock:
            try:
                issued_dt = _parse_iso(issued_at)
            except _TimestampError as exc:
                return ConfirmationToken(status="rejected", reason="timestamp_invalid", detail=str(exc))
            pending = self._pending.get(proposal_id)
            if pending is None:
                return ConfirmationToken(status="rejected", reason="proposal_not_found",
                                          detail=f"no pending proposal {proposal_id!r}")
            self._counter += 1
            token = _sha256_hex(
                f"{pending.case_id}|{pending.record_sha256}|{pending.proposal_id}|"
                f"{pending.consequence_preview_sha256}|{self._nonce}|{self._counter}"
            )
            expires_dt = issued_dt + timedelta(seconds=float(ttl_s))
            expires_at = expires_dt.isoformat()
            self._tokens[token] = _TokenRecord(
                proposal_id=proposal_id, case_id=pending.case_id, record_sha256=pending.record_sha256,
                consequence_preview_sha256=pending.consequence_preview_sha256,
                requires_ack=pending.next_step_changed, issued_at=str(issued_at), expires_at=expires_at,
            )
            return ConfirmationToken(
                status="issued", proposal_id=proposal_id, token=token, token_sha256=_sha256_hex(token),
                case_id=pending.case_id, record_sha256=pending.record_sha256,
                consequence_preview_sha256=pending.consequence_preview_sha256,
                requires_acknowledgement=pending.next_step_changed,
                issued_at=str(issued_at), expires_at=expires_at,
            )

    # -- 3. commit -------------------------------------------------------

    def commit(
        self, record: StructuredCaseRecord, token: Optional[str], *, now: str,
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
                self._history.append({
                    "type": "expired", "proposal_id": rec.proposal_id, "token_sha256": _sha256_hex(token),
                    "checked_at": now, "expires_at": rec.expires_at,
                })
                self._pending.pop(rec.proposal_id, None)
                return CommitReceipt(status="rejected", reason="token_expired", detail=f"token expired at {rec.expires_at}")
            if rec.case_id != record.identity.case_id:
                return CommitReceipt(status="rejected", reason="token_case_mismatch",
                                      detail=f"token is bound to case {rec.case_id!r}, not {record.identity.case_id!r}")
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
                delta=pending.delta, catalogue=self._catalogue, engine=self._engine, created_at=now,
            )
            fresh_diff = _diff_consequences(record, rebuilt)
            fresh_sha = _canonical_sha256(fresh_diff)
            if fresh_sha != rec.consequence_preview_sha256:
                return CommitReceipt(status="rejected", reason="consequence_drift",
                                      detail="recomputing the amendment at commit time no longer matches the preview")

            entry = AmendmentEntry(
                amendment_id=pending.proposal_id, kind=pending.delta.kind, field=pending.delta.field,
                before=pending.delta.before, after=pending.delta.after, actor=pending.actor,
                proposed_at=pending.proposed_at, committed_at=str(now),
                proposal_sha256=pending.proposal_id, token_sha256=_sha256_hex(token),
                consequence_preview_sha256=pending.consequence_preview_sha256,
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
            self._history.append({
                "type": "committed", "amendment_id": entry.amendment_id, "case_id": final.identity.case_id,
                "from_record_sha256": record.record_sha256, "to_record_sha256": final.record_sha256,
                "committed_at": str(now),
            })
            return CommitReceipt(status="committed", record=final, entry=entry)

    # -- 4. verify_chain ---------------------------------------------------

    def verify_chain(self, records: Sequence[StructuredCaseRecord]) -> ChainVerdict:
        return verify_chain(records, catalogue=self._catalogue, engine=self._engine)


__all__ = [
    "AMENDMENT_KINDS",
    "AmendmentError",
    "AmendmentLedger",
    "AmendmentProposal",
    "ChainVerdict",
    "CommitReceipt",
    "ConfirmationToken",
    "ProposalReceipt",
    "verify_chain",
]

