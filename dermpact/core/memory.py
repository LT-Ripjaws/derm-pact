"""Comparison-oriented Verified Case Memory v2 (DermPact Phase 6, WP-M)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dermpact.core.versions import MEMORY_EMBEDDING_MODEL, MEMORY_PROTOCOL_VERSION
from dermpact.record.case_record import round_floats

ROOT = Path(__file__).resolve().parents[2]


DATA_ROLE_DEVELOPMENT = "development"
DATA_ROLE_EVALUATION_QUERY = "evaluation_query_only"
DATA_ROLE_DEPLOYMENT_QUERY = "deployment_query_only"
ALLOWED_QUERY_ROLES = frozenset({DATA_ROLE_DEVELOPMENT, DATA_ROLE_EVALUATION_QUERY, DATA_ROLE_DEPLOYMENT_QUERY})
MODALITY_DERMOSCOPIC = "dermoscopic"
CONSENT_DEVELOPMENT = "research_development_use"
DERM7PT_IDENTITY_NOTE = "Derm7pt provides no patient identifiers"
CASE_LEVEL_IDENTITY_NOTE = (
    "case-level only; patient-level exclusion not verifiable (Derm7pt has no patient identifiers)"
)

ELIGIBILITY_STATES = frozenset({"eligible", "pending", "revoked", "superseded"})
IDENTITY_POLICIES = frozenset({"require_patient_identity", "case_level_research_only"})
OUTCOME_GRADES = frozenset(
    {
        "dataset_reference_label",
        "pathology_confirmed",
        "expert_consensus",
        "clinician_confirmed_single",
        "unknown",
    }
)
MODEL_RECORD_GRADE = "model_oof_single_member"

CRITERIA = (
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
)
COMPARISON_FIELDS = ("site", "sex", "elevation") + CRITERIA


COMPARISON_MELANOMA = "melanoma"
COMPARISON_NEVUS = "nevus"
COMPARISON_BCC = "basal_cell_carcinoma"
COMPARISON_SK = "seborrheic_keratosis"
COMPARISON_MISC = "miscellaneous"
COMPARISON_UNMAPPED = "unmapped"
COMPARISON_DIAGNOSES = frozenset(
    {COMPARISON_MELANOMA, COMPARISON_NEVUS, COMPARISON_BCC, COMPARISON_SK, COMPARISON_MISC}
)

#: Derm7pt harmonized5 labels (dataset-side outcome.diagnosis values).
_HARMONIZED5_TO_COMPARISON: Dict[str, str] = {
    "MEL": COMPARISON_MELANOMA,
    "NEV": COMPARISON_NEVUS,
    "BCC": COMPARISON_BCC,
    "SK": COMPARISON_SK,
    "MISC": COMPARISON_MISC,
}

_GLOBAL_CLASSES_TO_COMPARISON: Dict[str, str] = {
    "melanoma": COMPARISON_MELANOMA,
    "nevus": COMPARISON_NEVUS,
    "basal_cell_carcinoma": COMPARISON_BCC,
    "squamous_cell_carcinoma": COMPARISON_MISC,
    "actinic_keratosis_or_intraepithelial_carcinoma": COMPARISON_MISC,
    "seborrheic_keratosis_or_benign_keratosis": COMPARISON_SK,
    "dermatofibroma": COMPARISON_MISC,
    "vascular_lesion": COMPARISON_MISC,
    "other": COMPARISON_UNMAPPED,
}

DIAGNOSIS_HARMONISATION_TABLE: Dict[str, str] = {
    **_HARMONIZED5_TO_COMPARISON,
    **_GLOBAL_CLASSES_TO_COMPARISON,
}


def harmonise_diagnosis(label: Optional[str]) -> str:
    """Map a Derm7pt harmonized5 label or a model GLOBAL_CLASSES label onto the shared comparison vocabulary."""

    if label is None:
        return COMPARISON_UNMAPPED
    return DIAGNOSIS_HARMONISATION_TABLE.get(str(label), COMPARISON_UNMAPPED)


STATUS_OK = "ok"
STATUS_NO_MATCH = "no_match"
STATUS_EMPTY_BANK = "empty_bank"
STATUS_DECLINED_DATA_ROLE = "declined_prohibited_data_role"
STATUS_DECLINED_EMBEDDING = "declined_embedding_version_mismatch"
STATUS_DECLINED_IDENTITY = "declined_identity_unverifiable"
STATUS_DECLINED_POLICY_BANK_MISMATCH = "declined_policy_bank_mismatch"

EXCLUDED_SELF = "excluded_self_case"
EXCLUDED_SAME_PATIENT = "excluded_same_patient"
EXCLUDED_DATA_ROLE = "excluded_prohibited_data_role"
EXCLUDED_DUPLICATE_IMAGE = "excluded_duplicate_image"
EXCLUDED_INSUFFICIENT_EVIDENCE = "excluded_insufficient_evidence"

COUNTEREXAMPLE_AVAILABLE = "available"
COUNTEREXAMPLE_NONE_ABOVE_THRESHOLD = "none_above_threshold"
COUNTEREXAMPLE_ALL_ONE_CLASS = "all_candidates_one_class"

UNSOLICITED_REMINDER_KIND = "comparison_reminder_unsolicited"

DEFAULT_BANK_DIR = ROOT / "research/experiments/dermpact_phase6/memory_bank_dev_v1"
SENTINEL_FEATURES = ROOT / "research/experiments/dermpact_phase3/image_audit_v1/sentinel_features.npz"
STAGE2_OOF = ROOT / "research/experiments/dermpact_phase2_5/stage2_cefix_v1/stage2_oof.npz"
FRIEND_CUDA_MANIFESTS = ROOT / "friend-cuda/research/manifests"




def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        round_floats(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verify_sidecar(path: Path) -> str:
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = digest_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, got {actual}")
    return actual


def _atomic_write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    digest = digest_file(path)
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    body = json.dumps(round_floats(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _atomic_write_text(path, body)




@dataclass(frozen=True)
class MemoryEpisode:
    """One historical, host-governed comparison episode."""

    episode_id: str
    case_id: str
    lesion_id: str
    patient_id: Optional[str]
    identity_note: Optional[str]
    data_role: str
    image_sha256: str
    embedding_model: str
    embedding_sha256: str
    embedding: Tuple[float, ...]
    modality: str
    site: Optional[str]
    sex: Optional[str]
    elevation: Optional[str]
    documented_criterion_values: Mapping[str, Optional[str]]
    original_model_record: Mapping[str, Any]
    confirmed_amendments: Tuple[Mapping[str, Any], ...]
    outcome: Mapping[str, Any]
    field_provenance: Mapping[str, Mapping[str, str]]
    event_time: str
    author: str
    session_id: Optional[str]
    consent_access: str
    eligibility: str
    supersedes: Optional[str]
    revoked_reason: Optional[str]
    record_sha256: str = ""

    # -- serialisation -----------------------------------------------------

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "protocol": MEMORY_PROTOCOL_VERSION,
            "episode_id": self.episode_id,
            "case_id": self.case_id,
            "lesion_id": self.lesion_id,
            "patient_id": self.patient_id,
            "identity_note": self.identity_note,
            "data_role": self.data_role,
            "image_sha256": self.image_sha256,
            "embedding_model": self.embedding_model,
            "embedding_sha256": self.embedding_sha256,
            "embedding": list(self.embedding),
            "modality": self.modality,
            "site": self.site,
            "sex": self.sex,
            "elevation": self.elevation,
            "documented_criterion_values": dict(self.documented_criterion_values),
            "original_model_record": dict(self.original_model_record),
            "confirmed_amendments": [dict(a) for a in self.confirmed_amendments],
            "outcome": dict(self.outcome),
            "field_provenance": {k: dict(v) for k, v in self.field_provenance.items()},
            "event_time": self.event_time,
            "author": self.author,
            "session_id": self.session_id,
            "consent_access": self.consent_access,
            "eligibility": self.eligibility,
            "supersedes": self.supersedes,
            "revoked_reason": self.revoked_reason,
        }

    def to_dict(self) -> Dict[str, Any]:
        value = round_floats(self.unsigned_dict())
        value["record_sha256"] = _hash(value)
        return value

    @staticmethod
    def _embedding_sha256(embedding: Sequence[float]) -> str:
        return _hash({"embedding": list(embedding)})

    @staticmethod
    def _derive_episode_id(identity: Mapping[str, Any]) -> str:
        return f"vcm2-{_hash(identity)[:20]}"

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        embedding: Sequence[float],
        embedding_model: str,
        image_sha256: str,
        outcome: Mapping[str, Any],
        author: str,
        lesion_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        identity_note: Optional[str] = None,
        data_role: str = DATA_ROLE_DEVELOPMENT,
        modality: str = MODALITY_DERMOSCOPIC,
        site: Optional[str] = None,
        sex: Optional[str] = None,
        elevation: Optional[str] = None,
        documented_criterion_values: Optional[Mapping[str, Optional[str]]] = None,
        original_model_record: Optional[Mapping[str, Any]] = None,
        confirmed_amendments: Sequence[Mapping[str, Any]] = (),
        field_provenance: Optional[Mapping[str, Mapping[str, str]]] = None,
        event_time: Optional[str] = None,
        session_id: Optional[str] = None,
        consent_access: str = CONSENT_DEVELOPMENT,
        eligibility: str = "eligible",
        supersedes: Optional[str] = None,
        revoked_reason: Optional[str] = None,
    ) -> "MemoryEpisode":
        case_id = str(case_id).strip()
        if not case_id:
            raise ValueError("case_id is required")
        lesion_id = str(lesion_id).strip() if lesion_id else case_id
        if patient_id is not None:
            patient_id = str(patient_id).strip() or None
        data_role = str(data_role).strip()
        if not data_role:
            raise ValueError("data_role is required")
        if eligibility not in ELIGIBILITY_STATES:
            raise ValueError(f"unknown eligibility {eligibility!r}")
        author = str(author).strip()
        if not author:
            raise ValueError("author is required")
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a non-empty finite vector")
        embedding_tuple = tuple(float(v) for v in vector)
        outcome = dict(outcome)
        grade = outcome.get("evidence_grade")
        if grade not in OUTCOME_GRADES:
            raise ValueError(f"invalid outcome evidence_grade: {grade!r}")
        embedding_sha256 = cls._embedding_sha256(round_floats(list(embedding_tuple)))
        timestamp = event_time or _utc_now()
        identity = {
            "case_id": case_id,
            "lesion_id": lesion_id,
            "patient_id": patient_id,
            "image_sha256": str(image_sha256),
            "embedding_sha256": embedding_sha256,
            "outcome": outcome,
            "event_time": timestamp,
            "author": author,
            "session_id": session_id,
            "supersedes": supersedes,
        }
        episode_id = cls._derive_episode_id(identity)
        if identity_note is None and patient_id is None:
            identity_note = DERM7PT_IDENTITY_NOTE
        episode = cls(
            episode_id=episode_id,
            case_id=case_id,
            lesion_id=lesion_id,
            patient_id=patient_id,
            identity_note=identity_note,
            data_role=data_role,
            image_sha256=str(image_sha256),
            embedding_model=str(embedding_model),
            embedding_sha256=embedding_sha256,
            embedding=embedding_tuple,
            modality=modality,
            site=site,
            sex=sex,
            elevation=elevation,
            documented_criterion_values=dict(documented_criterion_values or {}),
            original_model_record=dict(original_model_record or {}),
            confirmed_amendments=tuple(dict(a) for a in confirmed_amendments),
            outcome=outcome,
            field_provenance=dict(field_provenance or {}),
            event_time=timestamp,
            author=author,
            session_id=session_id,
            consent_access=consent_access,
            eligibility=eligibility,
            supersedes=supersedes,
            revoked_reason=revoked_reason,
        )
        record_sha256 = episode.to_dict()["record_sha256"]
        return cls(**{**episode.__dict__, "record_sha256": record_sha256})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryEpisode":
        payload = dict(value)
        recorded_hash = str(payload.pop("record_sha256", ""))
        if payload.get("protocol") != MEMORY_PROTOCOL_VERSION:
            raise ValueError("memory episode protocol mismatch")
        if _hash(payload) != recorded_hash:
            raise ValueError("memory episode hash mismatch")
        payload.pop("protocol")
        embedding = tuple(float(v) for v in payload.pop("embedding"))
        confirmed_amendments = tuple(dict(a) for a in payload.pop("confirmed_amendments"))
        field_provenance = {k: dict(v) for k, v in payload.pop("field_provenance").items()}
        episode = cls(
            embedding=embedding,
            confirmed_amendments=confirmed_amendments,
            field_provenance=field_provenance,
            record_sha256=recorded_hash,
            **payload,
        )
        expected_embedding_sha256 = cls._embedding_sha256(round_floats(list(embedding)))
        if expected_embedding_sha256 != episode.embedding_sha256:
            raise ValueError("memory episode embedding_sha256 does not match its embedding")
        expected_identity = {
            "case_id": episode.case_id,
            "lesion_id": episode.lesion_id,
            "patient_id": episode.patient_id,
            "image_sha256": episode.image_sha256,
            "embedding_sha256": episode.embedding_sha256,
            "outcome": dict(episode.outcome),
            "event_time": episode.event_time,
            "author": episode.author,
            "session_id": episode.session_id,
            "supersedes": episode.supersedes,
        }
        if cls._derive_episode_id(expected_identity) != episode.episode_id:
            raise ValueError("memory episode identity does not match its episode_id")
        if episode.eligibility not in ELIGIBILITY_STATES:
            raise ValueError(f"unknown eligibility {episode.eligibility!r}")
        if episode.outcome.get("evidence_grade") not in OUTCOME_GRADES:
            raise ValueError(f"invalid outcome evidence_grade: {episode.outcome.get('evidence_grade')!r}")
        return episode




@dataclass(frozen=True)
class MemoryBank:
    """An immutable, hash-bound collection of memory episodes."""

    episodes: Tuple[MemoryEpisode, ...]
    embedding_model: str
    bank_sha256: str

    def __len__(self) -> int:
        return len(self.episodes)

    def by_id(self) -> Dict[str, MemoryEpisode]:
        return {episode.episode_id: episode for episode in self.episodes}

    def get(self, episode_id: str) -> Optional[MemoryEpisode]:
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def superseded_ids(self) -> frozenset:
        return frozenset(e.supersedes for e in self.episodes if e.supersedes)

    def effective(self) -> Tuple[MemoryEpisode, ...]:
        """Current heads: every episode that nothing else supersedes."""

        superseded = self.superseded_ids()
        return tuple(e for e in self.episodes if e.episode_id not in superseded)

    @classmethod
    def from_episodes(
        cls, episodes: Sequence[MemoryEpisode], *, embedding_model: str
    ) -> "MemoryBank":
        ordered = tuple(sorted(episodes, key=lambda e: e.episode_id))
        ids = [e.episode_id for e in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("memory bank episode ids must be unique")
        models = {e.embedding_model for e in ordered}
        if models and models != {embedding_model}:
            raise ValueError("memory bank embedding_model is inconsistent with its episodes")
        bank_sha256 = _hash(
            {
                "protocol": MEMORY_PROTOCOL_VERSION,
                "embedding_model": embedding_model,
                "episode_hashes": {e.episode_id: e.record_sha256 for e in ordered},
            }
        )
        return cls(episodes=ordered, embedding_model=embedding_model, bank_sha256=bank_sha256)

    def to_jsonl_lines(self) -> List[str]:
        return [json.dumps(e.to_dict(), sort_keys=True) for e in self.episodes]

    def write(
        self,
        out_dir: Path,
        *,
        source: Mapping[str, Any],
        n_embedded: int,
        n_development_cases: int,
        threshold_derivation: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        bank_path = out_dir / "bank.jsonl"
        lines = self.to_jsonl_lines()
        text = ("\n".join(lines) + "\n") if lines else ""
        bank_jsonl_sha256 = _atomic_write_text(bank_path, text)
        manifest = {
            "protocol": MEMORY_PROTOCOL_VERSION,
            "bank_sha256": self.bank_sha256,
            "bank_jsonl_sha256": bank_jsonl_sha256,
            "embedding_model": self.embedding_model,
            "n_episodes": len(self.episodes),
            "n_embedded": n_embedded,
            "n_development_cases": n_development_cases,
            "built_at": _utc_now(),
            "source": dict(source),
            "threshold_derivation": dict(threshold_derivation) if threshold_derivation else None,
        }
        _write_json(out_dir / "manifest.json", manifest)
        return manifest

    @classmethod
    def load(cls, out_dir: Path) -> "MemoryBank":
        out_dir = Path(out_dir)
        bank_path = out_dir / "bank.jsonl"
        manifest_path = out_dir / "manifest.json"
        _verify_sidecar(bank_path)
        _verify_sidecar(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes: List[MemoryEpisode] = []
        with bank_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    episodes.append(MemoryEpisode.from_dict(json.loads(line)))
                except Exception as exc:  # noqa: BLE001 - re-raised with location
                    raise ValueError(f"{bank_path}:{line_number}: invalid memory episode") from exc
        bank = cls.from_episodes(episodes, embedding_model=manifest["embedding_model"])
        if bank.bank_sha256 != manifest["bank_sha256"]:
            raise ValueError("memory bank hash does not match its manifest")
        return bank




@dataclass(frozen=True)
class ComparisonPolicy:
    """Retrieval policy constants."""

    k: int = 3
    min_similarity: float = 0.75
    counterexample_min_similarity: float = 0.70
    identity_policy: str = "case_level_research_only"
    bank_sha256: Optional[str] = None
    threshold_derivation: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        if not 0.0 <= self.counterexample_min_similarity <= 1.0:
            raise ValueError("counterexample_min_similarity must be in [0, 1]")
        if self.identity_policy not in IDENTITY_POLICIES:
            raise ValueError(f"unknown identity_policy {self.identity_policy!r}")
        if self.bank_sha256 is not None and self.threshold_derivation is None:
            raise ValueError(
                "a policy frozen to a bank_sha256 must carry the threshold_derivation "
                "it was computed from"
            )

    @classmethod
    def from_manifest(cls, path: Path) -> "ComparisonPolicy":
        """Load the frozen, data-derived policy from a bank's manifest.json."""

        path = Path(path)
        manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
        _verify_sidecar(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        derivation = manifest.get("threshold_derivation")
        if not derivation:
            raise ValueError(f"{manifest_path} has no threshold_derivation to freeze a policy from")
        return cls(
            min_similarity=float(derivation["min_similarity"]),
            counterexample_min_similarity=float(derivation["counterexample_min_similarity"]),
            bank_sha256=str(manifest["bank_sha256"]),
            threshold_derivation=dict(derivation),
        )


@dataclass(frozen=True)
class ComparisonResult:
    """The output of ``compare``. No diagnosis distribution, vote, or blend."""

    status: str
    matches: Tuple[Mapping[str, Any], ...]
    n_candidates: int
    counterexample: Optional[Mapping[str, Any]]
    counterexample_status: str
    exclusions: Tuple[Mapping[str, Any], ...]
    reason: Optional[str]
    dropped_after_revocation: Tuple[str, ...]
    receipt: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "matches": [dict(m) for m in self.matches],
            "n_candidates": self.n_candidates,
            "counterexample": dict(self.counterexample) if self.counterexample else None,
            "counterexample_status": self.counterexample_status,
            "exclusions": [dict(e) for e in self.exclusions],
            "reason": self.reason,
            "dropped_after_revocation": list(self.dropped_after_revocation),
            "receipt": dict(self.receipt),
        }




def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / denom)


def _field_value(source: Mapping[str, Any], name: str) -> Optional[Any]:
    if name in CRITERIA:
        return (source.get("criterion_values") or {}).get(name)
    return source.get(name)


def _field_bundle(
    query: Mapping[str, Any], episode: MemoryEpisode
) -> Tuple[List[str], List[str], List[str]]:
    matched: List[str] = []
    different: List[str] = []
    unknown: List[str] = []
    for name in COMPARISON_FIELDS:
        q_value = _field_value(query, name)
        e_value = (
            episode.documented_criterion_values.get(name)
            if name in CRITERIA
            else getattr(episode, name)
        )
        if q_value is None or e_value is None:
            unknown.append(name)
        elif q_value == e_value:
            matched.append(name)
        else:
            different.append(name)
    return matched, different, unknown


def _episode_diagnosis(episode: MemoryEpisode) -> Optional[str]:
    return episode.outcome.get("diagnosis")


def _diagnosis_view(query_provisional_class: Optional[str], episode_diagnosis_raw: Optional[str]) -> Dict[str, Any]:
    """Raw + harmonised diagnosis for both sides, and a match status computed only in the shared comparison vocabulary."""

    query_harmonised = harmonise_diagnosis(query_provisional_class)
    episode_harmonised = harmonise_diagnosis(episode_diagnosis_raw)
    if query_harmonised == COMPARISON_UNMAPPED or episode_harmonised == COMPARISON_UNMAPPED:
        status = "unknown"
    elif query_harmonised == episode_harmonised:
        status = "match"
    else:
        status = "different"
    return {
        "query_raw": query_provisional_class,
        "query_harmonised": query_harmonised,
        "episode_raw": episode_diagnosis_raw,
        "episode_harmonised": episode_harmonised,
        "status": status,
    }


def _match_view(similarity: float, episode: MemoryEpisode, query: Mapping[str, Any]) -> Dict[str, Any]:
    matched_fields, different_fields, unknown_fields = _field_bundle(query, episode)
    return {
        "episode_id": episode.episode_id,
        "case_id": episode.case_id,
        "similarity": round(similarity, 6),
        "matched_fields": matched_fields,
        "different_fields": different_fields,
        "unknown_fields": unknown_fields,
        "diagnosis": _diagnosis_view(query.get("provisional_class"), _episode_diagnosis(episode)),
        "outcome": dict(episode.outcome),
        "original_model_record": dict(episode.original_model_record),
        "confirmed_amendments": [dict(a) for a in episode.confirmed_amendments],
        "field_provenance": {k: dict(v) for k, v in episode.field_provenance.items()},
    }


def compare(query: Mapping[str, Any], bank: MemoryBank, policy: ComparisonPolicy) -> ComparisonResult:
    """Host-computed comparison retrieval. Never a diagnosis fusion."""

    query_case_id = query.get("case_id")
    if not query_case_id:
        raise ValueError("query.case_id is required")

    query_embedding_sha256 = _hash(
        {"embedding": round_floats([float(v) for v in np.asarray(query.get("embedding", []), dtype=np.float64)])}
    )
    receipt: Dict[str, Any] = {
        "protocol_version": MEMORY_PROTOCOL_VERSION,
        "bank_sha256": bank.bank_sha256,
        "policy": {
            "k": policy.k,
            "min_similarity": policy.min_similarity,
            "counterexample_min_similarity": policy.counterexample_min_similarity,
            "identity_policy": policy.identity_policy,
        },
        "threshold_derivation": dict(policy.threshold_derivation) if policy.threshold_derivation else None,
        "query_case_id": str(query_case_id),
        "query_embedding_sha256": query_embedding_sha256,
        "generated_at": _utc_now(),
        "identity_exclusion": None,
    }

    def _declined(status: str, reason: str) -> ComparisonResult:
        return ComparisonResult(
            status=status,
            matches=(),
            n_candidates=0,
            counterexample=None,
            counterexample_status=COUNTEREXAMPLE_NONE_ABOVE_THRESHOLD,
            exclusions=(),
            reason=reason,
            dropped_after_revocation=(),
            receipt=dict(receipt),
        )

    query_data_role = query.get("data_role")
    if query_data_role not in ALLOWED_QUERY_ROLES:
        return _declined(
            STATUS_DECLINED_DATA_ROLE,
            f"query data_role {query_data_role!r} is not permitted for comparison memory",
        )

    if len(bank) == 0:
        return _declined(STATUS_EMPTY_BANK, "comparison bank has no episodes")

    if policy.bank_sha256 is not None and policy.bank_sha256 != bank.bank_sha256:
        return _declined(
            STATUS_DECLINED_POLICY_BANK_MISMATCH,
            f"policy is frozen to bank_sha256 {policy.bank_sha256!r} but the supplied bank is "
            f"{bank.bank_sha256!r}",
        )

    query_embedding_model = query.get("embedding_model", bank.embedding_model)
    if query_embedding_model != bank.embedding_model:
        return _declined(
            STATUS_DECLINED_EMBEDDING,
            f"query embedding_model {query_embedding_model!r} does not match bank "
            f"{bank.embedding_model!r}",
        )

    identity = query.get("identity") or {}
    query_patient_id = identity.get("patient_id")
    query_lesion_id = identity.get("lesion_id") or query_case_id

    if policy.identity_policy == "require_patient_identity":
        bank_has_unknown_patient = any(e.patient_id is None for e in bank.effective())
        if query_patient_id is None or bank_has_unknown_patient:
            return _declined(
                STATUS_DECLINED_IDENTITY,
                "patient identity is not established on the query or on eligible historical "
                "episodes; same-patient exclusion cannot be verified under require_patient_identity",
            )
        receipt["identity_exclusion"] = None
    else:
        receipt["identity_exclusion"] = CASE_LEVEL_IDENTITY_NOTE

    query_vector = np.asarray(query.get("embedding"), dtype=np.float64).reshape(-1)
    if query_vector.size == 0 or not np.isfinite(query_vector).all():
        raise ValueError("query.embedding must be a non-empty finite vector")
    query_image_sha256 = query.get("image_sha256")

    exclusions: List[Dict[str, Any]] = []
    raw_candidates: List[Tuple[float, MemoryEpisode]] = []
    for episode in bank.effective():
        reason_code: Optional[str] = None
        if episode.case_id == query_case_id or episode.lesion_id == query_lesion_id:
            reason_code = EXCLUDED_SELF
        elif (
            query_patient_id is not None
            and episode.patient_id is not None
            and episode.patient_id == query_patient_id
        ):
            reason_code = EXCLUDED_SAME_PATIENT
        elif episode.data_role != DATA_ROLE_DEVELOPMENT:
            reason_code = EXCLUDED_DATA_ROLE
        elif query_image_sha256 is not None and episode.image_sha256 == query_image_sha256:
            reason_code = EXCLUDED_DUPLICATE_IMAGE
        elif episode.eligibility != "eligible" or episode.outcome.get("evidence_grade") == "unknown":
            reason_code = EXCLUDED_INSUFFICIENT_EVIDENCE
        if reason_code:
            exclusions.append(
                {"episode_id": episode.episode_id, "case_id": episode.case_id, "reason": reason_code}
            )
            continue
        similarity = _cosine(query_vector, np.asarray(episode.embedding, dtype=np.float64))
        raw_candidates.append((similarity, episode))

    raw_candidates.sort(key=lambda item: (-item[0], item[1].episode_id))

    seen_images: set[str] = set()
    candidates: List[Tuple[float, MemoryEpisode]] = []
    for similarity, episode in raw_candidates:
        if episode.image_sha256 in seen_images:
            exclusions.append(
                {
                    "episode_id": episode.episode_id,
                    "case_id": episode.case_id,
                    "reason": EXCLUDED_DUPLICATE_IMAGE,
                }
            )
            continue
        seen_images.add(episode.image_sha256)
        candidates.append((similarity, episode))

    matched_candidates = [(sim, ep) for sim, ep in candidates if sim >= policy.min_similarity]
    top = matched_candidates[: policy.k]
    matches = [_match_view(similarity, episode, query) for similarity, episode in top]

    top_match_diagnosis = _episode_diagnosis(top[0][1]) if top else None
    top_match_diagnosis_harmonised = (
        harmonise_diagnosis(top_match_diagnosis) if top_match_diagnosis is not None else None
    )
    query_provisional_class = query.get("provisional_class")
    query_diagnosis_harmonised = harmonise_diagnosis(query_provisional_class)

    counter_pool = [
        (sim, ep)
        for sim, ep in candidates
        if sim >= policy.counterexample_min_similarity
        and harmonise_diagnosis(_episode_diagnosis(ep)) != COMPARISON_UNMAPPED
    ]
    if query_diagnosis_harmonised == COMPARISON_UNMAPPED:
        discordant: List[Tuple[float, MemoryEpisode]] = []
    else:
        discordant = [
            (sim, ep)
            for sim, ep in counter_pool
            if harmonise_diagnosis(_episode_diagnosis(ep)) != query_diagnosis_harmonised
            and harmonise_diagnosis(_episode_diagnosis(ep)) != top_match_diagnosis_harmonised
        ]
    if discordant:
        sim, episode = discordant[0]
        counterexample = _match_view(sim, episode, query)
        counterexample_status = COUNTEREXAMPLE_AVAILABLE
    else:
        counterexample = None
        counterexample_status = (
            COUNTEREXAMPLE_ALL_ONE_CLASS if counter_pool else COUNTEREXAMPLE_NONE_ABOVE_THRESHOLD
        )

    status = STATUS_OK if matches else STATUS_NO_MATCH
    reason = None if matches else "no eligible historical episode cleared the frozen similarity threshold"

    return ComparisonResult(
        status=status,
        matches=tuple(matches),
        n_candidates=len(candidates),
        counterexample=counterexample,
        counterexample_status=counterexample_status,
        exclusions=tuple(exclusions),
        reason=reason,
        dropped_after_revocation=(),
        receipt=dict(receipt),
    )


def refresh(result: ComparisonResult, bank: MemoryBank) -> ComparisonResult:
    """Recheck a cached comparison against the bank's *current* eligibility."""

    by_id = bank.by_id()
    effective_ids = {episode.episode_id for episode in bank.effective()}

    def _still_eligible(episode_id: str) -> bool:
        current = by_id.get(episode_id)
        return current is not None and current.eligibility == "eligible" and episode_id in effective_ids

    dropped: List[str] = []
    kept_matches = []
    for match in result.matches:
        if _still_eligible(match["episode_id"]):
            kept_matches.append(match)
        else:
            dropped.append(match["episode_id"])

    counterexample = result.counterexample
    counterexample_status = result.counterexample_status
    if counterexample is not None and not _still_eligible(counterexample["episode_id"]):
        dropped.append(counterexample["episode_id"])
        counterexample = None
        if counterexample_status == COUNTEREXAMPLE_AVAILABLE:
            counterexample_status = COUNTEREXAMPLE_NONE_ABOVE_THRESHOLD

    status = result.status
    if status == STATUS_OK and not kept_matches:
        status = STATUS_NO_MATCH

    return ComparisonResult(
        status=status,
        matches=tuple(kept_matches),
        n_candidates=result.n_candidates,
        counterexample=counterexample,
        counterexample_status=counterexample_status,
        exclusions=result.exclusions,
        reason=result.reason,
        dropped_after_revocation=tuple(dict.fromkeys((*result.dropped_after_revocation, *dropped))),
        receipt=dict(result.receipt),
    )


def reminder_allowed(episode_events: Sequence[Mapping[str, Any]]) -> bool:
    """At most one unsolicited comparison reminder per episode."""

    return not any(
        event.get("event_type", event.get("kind")) == UNSOLICITED_REMINDER_KIND
        for event in episode_events
    )



MIN_SIMILARITY_FLOOR = 0.50
COUNTEREXAMPLE_MARGIN = 0.05
THRESHOLD_DERIVATION_METHOD = (
    "leave-one-out nearest-neighbour cosine similarity over every eligible bank episode's "
    "embedding; min_similarity = 25th percentile of that distribution rounded to 3 dp "
    f"(hard floor {MIN_SIMILARITY_FLOOR}); counterexample_min_similarity = "
    f"min_similarity - {COUNTEREXAMPLE_MARGIN} (same floor)"
)


def _leave_one_out_nearest_neighbour_cosine(vectors: np.ndarray) -> np.ndarray:
    """Row-wise max cosine similarity to any *other* row (self excluded)."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1e-12)
    normalised = vectors / norms
    similarities = normalised @ normalised.T
    np.fill_diagonal(similarities, -np.inf)
    return similarities.max(axis=1)


def derive_similarity_thresholds(
    embeddings: Sequence[Sequence[float]], *, date: Optional[str] = None
) -> Dict[str, Any]:
    """Derive the frozen ``min_similarity``/``counterexample_min_similarity`` values from the leave-one-out nearest-neighbour cosine distribution over ``embeddings`` (one row per eligible bank episode)."""

    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("threshold derivation requires at least two eligible episodes")
    nn_cosine = _leave_one_out_nearest_neighbour_cosine(vectors)
    p10, p25, p50, p90 = (float(np.percentile(nn_cosine, q)) for q in (10, 25, 50, 90))
    min_similarity = round(max(MIN_SIMILARITY_FLOOR, p25), 3)
    counterexample_min_similarity = round(
        max(MIN_SIMILARITY_FLOOR, min_similarity - COUNTEREXAMPLE_MARGIN), 3
    )
    return {
        "method": THRESHOLD_DERIVATION_METHOD,
        "n": int(vectors.shape[0]),
        "p10": round(p10, 6),
        "p25": round(p25, 6),
        "p50": round(p50, 6),
        "p90": round(p90, 6),
        "min_similarity": min_similarity,
        "counterexample_min_similarity": counterexample_min_similarity,
        "date": date or _utc_now(),
    }




def build_development_bank(*, out_dir: Path = DEFAULT_BANK_DIR) -> Dict[str, Any]:
    """Build the frozen development comparison bank from the 616-case frame."""

    from dermpact.eval import dermpact_phase2_4b_inputs as inputs_mod
    from dermpact.eval import dermpact_phase2_baselines as baselines
    from dermpact.eval.dermpact_phase4_benchmark import (
        GLOBAL_CLASSES,
        criterion_values_from_logits,
        criterion_values_from_raw_labels,
    )

    sentinel_sha256 = _verify_sidecar(SENTINEL_FEATURES)
    features = np.load(SENTINEL_FEATURES, allow_pickle=False)
    fids = features["ids"].astype(str)
    datasets = features["datasets"].astype(str)
    derm_mask = datasets == "Derm7pt"
    derm_ids = fids[derm_mask]
    derm_features = features["features"][derm_mask]
    embedding_by_id = {cid: derm_features[i] for i, cid in enumerate(derm_ids)}

    stage2_sha256 = _verify_sidecar(STAGE2_OOF)
    stage2 = np.load(STAGE2_OOF, allow_pickle=False)

    data = inputs_mod.resolve_development_inputs(
        inventory_path=FRIEND_CUDA_MANIFESTS
        / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        fold_manifest_path=FRIEND_CUDA_MANIFESTS
        / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        clinical_photo_audit_path=FRIEND_CUDA_MANIFESTS
        / "dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        structured_fact_support_path=FRIEND_CUDA_MANIFESTS
        / "dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
        require_files=False,
    )
    if len(data.cases) != 616:
        raise AssertionError("expected 616 locked development cases")

    records = baselines._derm7pt_records()
    raw_by_id = {str(r["group_id"]): r for r in records}
    reference_source = (
        "Derm7pt published label (Kawahara et al.); not clinician approval in this system"
    )

    episodes: List[MemoryEpisode] = []
    n_embedded = 0
    for index, case in enumerate(data.cases):
        cid = case.group_id
        if cid not in embedding_by_id:
            continue
        n_embedded += 1
        row = raw_by_id[cid]
        documented = criterion_values_from_raw_labels(row["criteria_raw"])
        diag_logits = stage2["derm_diagnosis_logits"][index]
        criteria_logits = stage2["derm_criteria_logits"][index]
        melanoma_logit = float(stage2["derm_melanoma_logits"][index])
        provisional_class = GLOBAL_CLASSES[int(np.argmax(diag_logits))]
        original_model_record = {
            "provisional_class": provisional_class,
            "provisional_class_harmonised": harmonise_diagnosis(provisional_class),
            "criterion_values": criterion_values_from_logits(criteria_logits),
            "melanoma_probability": 1.0 / (1.0 + math.exp(-melanoma_logit)),
            "grade": MODEL_RECORD_GRADE,
            "source": "Phase 2.5 stage-2 out-of-fold single-member logits; not a clinical confirmation",
        }
        outcome = {
            "diagnosis": row["diagnosis"]["harmonized5"],
            "evidence_grade": "dataset_reference_label",
            "source": reference_source,
            "date": None,
        }
        field_provenance = {
            "outcome.diagnosis": {"grade": "dataset_reference_label", "source": reference_source}
        }
        field_provenance.update(
            {
                f"criteria.{name}": {"grade": "dataset_reference_label", "source": reference_source}
                for name in CRITERIA
            }
        )
        episode = MemoryEpisode.create(
            case_id=cid,
            lesion_id=cid,
            patient_id=None,
            identity_note=DERM7PT_IDENTITY_NOTE,
            data_role=DATA_ROLE_DEVELOPMENT,
            image_sha256=case.dermoscopic_sha256,
            embedding_model=MEMORY_EMBEDDING_MODEL,
            embedding=embedding_by_id[cid],
            modality=MODALITY_DERMOSCOPIC,
            site=case.facts.site,
            sex=case.facts.sex,
            elevation=case.facts.elevation,
            documented_criterion_values=documented,
            original_model_record=original_model_record,
            confirmed_amendments=(),
            outcome=outcome,
            field_provenance=field_provenance,
            event_time="2026-09-06T00:00:00Z",
            author="dermpact_phase6_memory_bank_builder",
            session_id=None,
            consent_access=CONSENT_DEVELOPMENT,
            eligibility="eligible",
        )
        episodes.append(episode)

    bank = MemoryBank.from_episodes(episodes, embedding_model=MEMORY_EMBEDDING_MODEL)
    eligible_embeddings = [e.embedding for e in bank.effective()]
    threshold_derivation = derive_similarity_thresholds(eligible_embeddings)
    manifest = bank.write(
        out_dir,
        source={
            "sentinel_features_path": str(SENTINEL_FEATURES.relative_to(ROOT)).replace("\\", "/"),
            "sentinel_features_sha256": sentinel_sha256,
            "stage2_oof_path": str(STAGE2_OOF.relative_to(ROOT)).replace("\\", "/"),
            "stage2_oof_sha256": stage2_sha256,
            "inventory_sha256": data.inventory_sha256,
            "fold_manifest_sha256": data.fold_manifest_sha256,
        },
        n_embedded=n_embedded,
        n_development_cases=len(data.cases),
        threshold_derivation=threshold_derivation,
    )
    return manifest


def _episode_to_query(episode: MemoryEpisode) -> Dict[str, Any]:
    """Build a self-check query from a bank episode's own recorded facts."""

    return {
        "case_id": episode.case_id,
        "embedding": list(episode.embedding),
        "embedding_model": episode.embedding_model,
        "data_role": episode.data_role,
        "image_sha256": episode.image_sha256,
        "site": episode.site,
        "sex": episode.sex,
        "elevation": episode.elevation,
        "criterion_values": dict(episode.documented_criterion_values),
        "provisional_class": episode.outcome.get("diagnosis"),
        "identity": {"patient_id": episode.patient_id, "lesion_id": episode.lesion_id},
    }


def self_check(*, bank_dir: Path = DEFAULT_BANK_DIR, case_id: str = "derm7pt-case-22") -> Dict[str, Any]:
    """Load the bank, verify hashes, and run one compare() with itself excluded."""

    bank = MemoryBank.load(bank_dir)
    policy = ComparisonPolicy.from_manifest(bank_dir)
    target = next((e for e in bank.episodes if e.case_id == case_id), None)
    if target is None:
        raise SystemExit(f"case {case_id!r} not found in loaded bank {bank_dir}")
    query = _episode_to_query(target)
    result = compare(query, bank, policy).to_dict()
    return {
        "bank_sha256": bank.bank_sha256,
        "n_episodes": len(bank),
        "query_case_id": case_id,
        "min_similarity": policy.min_similarity,
        "counterexample_min_similarity": policy.counterexample_min_similarity,
        "status": result["status"],
        "n_matches": len(result["matches"]),
        "counterexample_status": result["counterexample_status"],
        "identity_exclusion": result["receipt"].get("identity_exclusion"),
    }




def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dermpact.core.memory")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build-dev-bank", help="build the 616-case development bank")
    build_parser.add_argument("--out-dir", type=Path, default=DEFAULT_BANK_DIR)

    check_parser = sub.add_parser("self-check", help="load the bank and run one compare()")
    check_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    check_parser.add_argument("--case-id", default="derm7pt-case-22")

    args = parser.parse_args(argv)
    if args.command == "build-dev-bank":
        manifest = build_development_bank(out_dir=args.out_dir)
        print(
            json.dumps(
                {
                    "bank_sha256": manifest["bank_sha256"],
                    "n_episodes": manifest["n_episodes"],
                    "n_embedded": manifest["n_embedded"],
                    "n_development_cases": manifest["n_development_cases"],
                    "threshold_derivation": manifest["threshold_derivation"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "self-check":
        summary = self_check(bank_dir=args.bank_dir, case_id=args.case_id)
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
