"""WP-D: allowlisted development-case inventory + tagged synthetic fixtures."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from dermpact.reliability.policy import ReliabilityDecision
from dermpact.core.episode import empty_state, mark_awaiting_review, request_review
from dermpact.core.ledger_v2 import AmendmentLedgerV2, AmendmentProposal
from dermpact.core.record_v2 import StructuredCaseRecordV2, build_record_from_prediction
from dermpact.core.prototype.state import now_iso

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY_DIR = ROOT / "research/experiments/dermpact_phase6/demo_inventory_v1"

BASE_FIXTURE_CASE_ID = "derm7pt-case-22"

FIXTURE_NO_IMAGE = "fixture-no-image"
FIXTURE_IMAGE_QUALITY_FLAG = "fixture-image-quality-flag"
FIXTURE_MEMORY_NO_MATCH = "fixture-memory-no-match"
FIXTURE_AWAITING_REVIEW_RESUME = "fixture-awaiting-review-resume"
FIXTURE_REVOKED_MEMORY_EPISODE = "fixture-revoked-memory-episode"
FIXTURE_CLINICAL_ABSENT = "fixture-clinical-absent"

FIXTURE_CASE_IDS: Tuple[str, ...] = (
    FIXTURE_NO_IMAGE, FIXTURE_IMAGE_QUALITY_FLAG, FIXTURE_MEMORY_NO_MATCH,
    FIXTURE_AWAITING_REVIEW_RESUME, FIXTURE_REVOKED_MEMORY_EPISODE, FIXTURE_CLINICAL_ABSENT,
)

FIXTURE_LABELS: Dict[str, str] = {
    FIXTURE_NO_IMAGE: "no usable dermoscopic image (pre-committed reliability_override by fixture-builder)",
    FIXTURE_IMAGE_QUALITY_FLAG: "image-quality review_required (real bundle threshold copied)",
    FIXTURE_MEMORY_NO_MATCH: "comparison-memory no-match (embedding replaced by an orthogonal vector)",
    FIXTURE_AWAITING_REVIEW_RESUME: "review episode pre-populated as awaiting_review",
    FIXTURE_REVOKED_MEMORY_EPISODE: "comparison against a memory bank overlay with one revoked episode",
    FIXTURE_CLINICAL_ABSENT: (
        "no clinical photo (clinical_availability=not_supplied) -- WP-D addition: all 50 real "
        "reserved cases have a real clinical photo, so brief item 6's absent-photo path needs this fixture"
    ),
}

SKIPPED_FIXTURES: Dict[str, str] = {
    "fixture_member_deferred": (
        "not built: derm7pt-case-284 (and 4 other real cases) already has selective_abstention=deferred"
    ),
}


class UnknownCaseError(ValueError):
    """Raised for a case_id that is neither a demo_subset case nor a known fixture."""


@dataclass(frozen=True)
class InventoryManifest:
    demo_subset: Tuple[Dict[str, str], ...]
    rows: Tuple[Dict[str, Any], ...]
    n_cases: int
    generated_at: str

    def row(self, case_id: str) -> Optional[Dict[str, Any]]:
        for r in self.rows:
            if r["case_id"] == case_id:
                return r
        return None

    def demo_case_ids(self) -> Tuple[str, ...]:
        return tuple(d["case_id"] for d in self.demo_subset)


def load_manifest(inventory_dir: Path = DEFAULT_INVENTORY_DIR) -> InventoryManifest:
    payload = json.loads((Path(inventory_dir) / "inventory_manifest.json").read_text(encoding="utf-8"))
    return InventoryManifest(
        demo_subset=tuple(dict(d) for d in payload["demo_subset"]),
        rows=tuple(dict(r) for r in payload["rows"]),
        n_cases=int(payload["n_cases"]), generated_at=str(payload.get("generated_at", "")),
    )


def load_prediction_payload(inventory_dir: Path, case_id: str) -> Dict[str, Any]:
    path = Path(inventory_dir) / "predictions" / f"{case_id}.json"
    if not path.is_file():
        raise UnknownCaseError(f"no cached inventory prediction for {case_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_paths(meta: Mapping[str, Any]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if meta.get("dermoscopic_relative_path"):
        paths["dermoscopic"] = str(meta["dermoscopic_relative_path"])
    if meta.get("clinical_relative_path"):
        paths["clinical"] = str(meta["clinical_relative_path"])
    return paths


def build_record_from_inventory(
    *, inventory_dir: Path, case_id: str, catalogue: Any, engine: Any, bundle: Any,
) -> StructuredCaseRecordV2:
    """Build the version-1 record for a real ``demo_subset`` case from WP-X's already-computed cached prediction."""
    payload = load_prediction_payload(inventory_dir, case_id)
    decision = ReliabilityDecision(**payload["decision"])
    return build_record_from_prediction(
        case_id, payload["prediction"], decision, bundle, {}, observed_facts={},
        evaluation_date=date.today(), created_at=now_iso(), source_label="development_allowlisted_case",
        relative_paths=_relative_paths(payload.get("case_meta", {})), catalogue=catalogue, engine=engine,
    )


def _base_payload(inventory_dir: Path) -> Dict[str, Any]:
    return load_prediction_payload(inventory_dir, BASE_FIXTURE_CASE_ID)


def build_fixture_image_quality_flag(*, inventory_dir: Path, catalogue: Any, engine: Any, bundle: Any) -> StructuredCaseRecordV2:
    payload = _base_payload(inventory_dir)
    decision_dict = dict(payload["decision"])
    threshold = (bundle.quality or {}).get("threshold")
    decision_dict["iqa_state"] = "review_required"
    decision_dict["disposition"] = "review"
    decision_dict["warnings"] = tuple(decision_dict.get("warnings", ())) + (
        {
            "code": "image_quality_review_required",
            "reason": f"synthetic fixture: synthetic-corruption probability exceeds the real bundle threshold {threshold}",
            "recovery_action": "review_image_or_reacquire",
            "remaining_usable_output": "perception outputs remain displayed with the limitation noted",
        },
    )
    decision = ReliabilityDecision(**decision_dict)
    return build_record_from_prediction(
        FIXTURE_IMAGE_QUALITY_FLAG, payload["prediction"], decision, bundle, {}, observed_facts={},
        evaluation_date=date.today(), created_at=now_iso(), source_label="synthetic_workflow_fixture",
        relative_paths={}, catalogue=catalogue, engine=engine,
    )


def build_fixture_memory_no_match(*, inventory_dir: Path, catalogue: Any, engine: Any, bundle: Any) -> StructuredCaseRecordV2:
    payload = _base_payload(inventory_dir)
    decision = ReliabilityDecision(**payload["decision"])
    return build_record_from_prediction(
        FIXTURE_MEMORY_NO_MATCH, payload["prediction"], decision, bundle, {}, observed_facts={},
        evaluation_date=date.today(), created_at=now_iso(), source_label="synthetic_workflow_fixture",
        relative_paths={}, catalogue=catalogue, engine=engine,
    )


def build_fixture_revoked_memory_episode(*, inventory_dir: Path, catalogue: Any, engine: Any, bundle: Any) -> StructuredCaseRecordV2:
    payload = _base_payload(inventory_dir)
    decision = ReliabilityDecision(**payload["decision"])
    return build_record_from_prediction(
        FIXTURE_REVOKED_MEMORY_EPISODE, payload["prediction"], decision, bundle, {}, observed_facts={},
        evaluation_date=date.today(), created_at=now_iso(), source_label="synthetic_workflow_fixture",
        relative_paths={}, catalogue=catalogue, engine=engine,
    )


def build_fixture_clinical_absent(*, inventory_dir: Path, catalogue: Any, engine: Any, bundle: Any) -> StructuredCaseRecordV2:
    payload = _base_payload(inventory_dir)
    decision = ReliabilityDecision(**payload["decision"])
    prediction = dict(payload["prediction"])
    prediction["clinical_present"] = False
    prediction["clinical_receipt"] = None
    meta = payload.get("case_meta", {})
    relative_paths = {"dermoscopic": meta["dermoscopic_relative_path"]} if meta.get("dermoscopic_relative_path") else {}
    return build_record_from_prediction(
        FIXTURE_CLINICAL_ABSENT, prediction, decision, bundle, {}, observed_facts={},
        evaluation_date=date.today(), created_at=now_iso(), source_label="synthetic_workflow_fixture",
        relative_paths=relative_paths, catalogue=catalogue, engine=engine,
    )


def build_fixture_no_image_versions(
    *, inventory_dir: Path, catalogue: Any, engine: Any, bundle: Any,
) -> Tuple[StructuredCaseRecordV2, StructuredCaseRecordV2, Tuple[Dict[str, Any], ...]]:
    """Returns ``(v1_real_image, v2_after_override, ledger_events)``: v1 is a normal fixture-tagged record built from the real base case's prediction; v2 is v1 with a pre-committed ``reliability_override`` amendment (actor="fixture-builder") setting ``image.dermoscopic_present=False``, so the "no image" state is visible in the fixture's own amendment history."""
    payload = _base_payload(inventory_dir)
    decision = ReliabilityDecision(**payload["decision"])
    v1 = build_record_from_prediction(
        FIXTURE_NO_IMAGE, payload["prediction"], decision, bundle, {}, observed_facts={},
        evaluation_date=date.today(), created_at=now_iso(), source_label="synthetic_workflow_fixture",
        relative_paths={}, catalogue=catalogue, engine=engine,
    )
    ledger = AmendmentLedgerV2(catalogue=catalogue, engine=engine, calibrators=bundle.calibration, bundle=bundle)
    now = now_iso()
    actor, session_id = "fixture-builder", "fixture-builder-session"
    proposal = ledger.propose(
        v1, AmendmentProposal(kind="reliability_override", field="image.dermoscopic_present", value=False),
        actor=actor, session_id=session_id, proposed_at=now,
        statement="synthetic fixture: no usable dermoscopic image",
    )
    if proposal.status != "proposed":
        raise RuntimeError(f"fixture_no_image bootstrap proposal rejected: {proposal.reason} {proposal.detail}")
    token = ledger.issue_confirmation_token(proposal.proposal_id, issued_at=now, session_id=session_id)
    if token.status != "issued":
        raise RuntimeError(f"fixture_no_image bootstrap token rejected: {token.reason} {token.detail}")
    ack = proposal.consequence_preview_sha256 if token.requires_acknowledgement else None
    commit = ledger.commit(v1, token.token, now=now, actor=actor, session_id=session_id, acknowledged_consequence_sha256=ack)
    if commit.status != "committed":
        raise RuntimeError(f"fixture_no_image bootstrap commit rejected: {commit.reason} {commit.detail}")
    return v1, commit.record, ledger.to_events()


def build_fixture_awaiting_review_resume_events(case_id: str) -> Tuple[Any, ...]:
    """Two pure episode commands (``review_requested`` -> ``awaiting_review``) folded from an empty state, ready for the host to append/persist."""
    episode_id = f"episode-{case_id}"
    actor, session_id, at = "fixture-builder", "fixture-builder-session", now_iso()
    st0 = empty_state(episode_id, case_id)
    r1 = request_review(
        st0, reason="synthetic fixture: pre-populated awaiting review", actor=actor, session_id=session_id,
        idempotency_key=f"{case_id}-fixture-review-requested", at=at, current_record_version=1,
    )
    r2 = mark_awaiting_review(
        r1.state, actor=actor, session_id=session_id, idempotency_key=f"{case_id}-fixture-awaiting-review",
        at=at, current_record_version=1,
    )
    return (r1.event, r2.event)


__all__ = [
    "BASE_FIXTURE_CASE_ID", "DEFAULT_INVENTORY_DIR", "FIXTURE_AWAITING_REVIEW_RESUME", "FIXTURE_CASE_IDS",
    "FIXTURE_CLINICAL_ABSENT", "FIXTURE_IMAGE_QUALITY_FLAG", "FIXTURE_LABELS", "FIXTURE_MEMORY_NO_MATCH",
    "FIXTURE_NO_IMAGE", "FIXTURE_REVOKED_MEMORY_EPISODE", "InventoryManifest", "SKIPPED_FIXTURES", "UnknownCaseError",
    "build_fixture_awaiting_review_resume_events", "build_fixture_clinical_absent", "build_fixture_image_quality_flag",
    "build_fixture_memory_no_match", "build_fixture_no_image_versions", "build_fixture_revoked_memory_episode",
    "build_record_from_inventory", "load_manifest", "load_prediction_payload",
]
