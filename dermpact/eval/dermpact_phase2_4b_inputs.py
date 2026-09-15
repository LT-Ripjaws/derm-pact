"""Development-only input contract for DermPact Phase 2.4B."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DERM7PT_INVENTORY = (
    ROOT
    / "research/manifests/dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json"
)
DEVELOPMENT_FOLDS = (
    ROOT
    / "research/manifests/dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json"
)
DERM7PT_IMAGES = ROOT / "release_v0/images"
CLINICAL_PHOTO_AUDIT = (
    ROOT
    / "research/manifests/dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json"
)
STRUCTURED_FACT_SUPPORT = (
    ROOT
    / "research/manifests/dermpact_phase2_3/structured_facts_v1/structured_fact_support.json"
)

N_CASES = 616
N_AVAILABLE_CLINICAL = 609
N_UNAVAILABLE_CLINICAL = 7
FOLD_COUNTS = {0: 127, 1: 123, 2: 122, 3: 122, 4: 122}
SHARED_PREFIX_BLOCKS = 10

SourceSplit = Literal["train", "valid"]
Sex = Literal["female", "male"]
Site = Literal[
    "head_neck",
    "upper_extremity",
    "lower_extremity",
    "acral",
    "anterior_torso",
    "posterior_torso",
    "trunk_other",
    "genital_or_mucosal",
]
Elevation = Literal["flat", "palpable", "nodular"]
ClinicalUnavailableReason = Literal["dermoscopy_reused", "missing"]

SITE_MAP: Mapping[str, Site] = {
    "head neck": "head_neck",
    "upper limbs": "upper_extremity",
    "lower limbs": "lower_extremity",
    "acral": "acral",
    "abdomen": "anterior_torso",
    "chest": "anterior_torso",
    "back": "posterior_torso",
    "buttocks": "trunk_other",
    "genital areas": "genital_or_mucosal",
}
SEX_VALUES = frozenset({"female", "male"})
ELEVATION_VALUES = frozenset({"flat", "palpable", "nodular"})
MISSING_VALUES = frozenset({"", "unknown", "not-reported", "not available", "nan", "none", "-1"})


@dataclass(frozen=True)
class FactAvailability:
    """One Boolean per allowed fact; availability is never a model feature."""

    age: bool
    sex: bool
    site: bool
    elevation: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool]:
        return (self.age, self.sex, self.site, self.elevation)


@dataclass(frozen=True)
class PrediagnosticFacts:
    """Typed observed facts with ``None`` for unknown/unrecorded values."""

    age_years: float | None
    sex: Sex | None
    site: Site | None
    elevation: Elevation | None
    available: FactAvailability


@dataclass(frozen=True)
class Phase24BCaseInput:
    """One aligned Phase 2.4B case; an unavailable clinical path is absent."""

    case_index: int
    case_num: int
    group_id: str
    fold: int
    source_split: SourceSplit
    dermoscopic_path: Path
    dermoscopic_relative_path: str
    dermoscopic_sha256: str
    clinical_path: Path | None
    clinical_relative_path: str | None
    clinical_sha256: str | None
    clinical_available: bool
    clinical_unavailable_reason: ClinicalUnavailableReason | None
    facts: PrediagnosticFacts


@dataclass(frozen=True)
class Phase24BInputSet:
    cases: tuple[Phase24BCaseInput, ...]
    inventory_path: Path
    inventory_sha256: str
    fold_manifest_path: Path
    fold_manifest_sha256: str
    clinical_photo_audit_path: Path
    clinical_photo_audit_sha256: str
    structured_fact_support_path: Path
    structured_fact_support_sha256: str
    images_root: Path
    hard_scope: Mapping[str, int]

    @property
    def clinical_available_mask(self) -> tuple[bool, ...]:
        return tuple(case.clinical_available for case in self.cases)


@dataclass(frozen=True)
class Block10CacheConfig:
    """All encoder choices that can change block-10 prefix tokens."""

    model_id: str
    model_revision: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    hidden_size: int
    tokens_per_image: int
    dtype: str = "float32"
    token_layout: str = "sequence_hidden"
    shared_prefix_blocks: int = SHARED_PREFIX_BLOCKS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def parse_prediagnostic_facts(record: Mapping[str, Any]) -> PrediagnosticFacts:
    """Parse only Derm7pt's allowed prediagnostic fields, without imputation."""
    raw_source = record.get("structured_facts", {})
    if not isinstance(raw_source, Mapping):
        raise ValueError("structured_facts must be an object")
    allowed = {"sex", "location", "elevation"}
    unexpected = set(raw_source) - allowed
    if unexpected:
        raise ValueError(f"unexpected prediagnostic fact fields: {sorted(unexpected)}")

    raw_sex = _clean(raw_source.get("sex"))
    raw_site = _clean(raw_source.get("location"))
    raw_elevation = _clean(raw_source.get("elevation"))

    sex: Sex | None
    if raw_sex in MISSING_VALUES:
        sex = None
    elif raw_sex in SEX_VALUES:
        sex = raw_sex  # type: ignore[assignment]
    else:
        raise ValueError(f"unmapped recorded sex: {raw_source.get('sex')!r}")

    site: Site | None
    if raw_site in MISSING_VALUES:
        site = None
    elif raw_site in SITE_MAP:
        site = SITE_MAP[raw_site]
    else:
        raise ValueError(f"unmapped recorded location: {raw_source.get('location')!r}")

    elevation: Elevation | None
    if raw_elevation in MISSING_VALUES:
        elevation = None
    elif raw_elevation in ELEVATION_VALUES:
        elevation = raw_elevation  # type: ignore[assignment]
    else:
        raise ValueError(
            f"unmapped recorded elevation: {raw_source.get('elevation')!r}"
        )

    available = FactAvailability(
        age=False,
        sex=sex is not None,
        site=site is not None,
        elevation=elevation is not None,
    )
    return PrediagnosticFacts(
        age_years=None,
        sex=sex,
        site=site,
        elevation=elevation,
        available=available,
    )


def _safe_image_path(images_root: Path, raw_relative: Any) -> tuple[Path, str]:
    relative = str(raw_relative or "").strip().replace("\\", "/")
    if not relative:
        raise ValueError("empty image path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe image path: {relative!r}")
    root = Path(images_root).resolve()
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path escapes dataset root: {relative!r}") from exc
    return path, pure.as_posix()


def resolve_case_input(
    record: Mapping[str, Any],
    *,
    fold: int,
    images_root: Path = DERM7PT_IMAGES,
    require_files: bool = True,
) -> Phase24BCaseInput:
    """Resolve one already-authorized development record, fail-closed."""
    if record.get("phase1_role") != "development":
        raise ValueError("only a Phase 1 development record may be resolved")
    if record.get("source_split") not in {"train", "valid"}:
        raise ValueError("official Derm7pt test row is not an allowed input")
    recorded_fold = record.get("development_fold")
    if not isinstance(recorded_fold, int) or recorded_fold != fold or fold not in range(5):
        raise ValueError("record and locked-ledger fold assignments do not align")

    modalities = record.get("modalities")
    if not isinstance(modalities, Mapping):
        raise ValueError("modalities must be an object")
    derm = modalities.get("dermoscopic")
    clinical = modalities.get("clinical")
    if not isinstance(derm, Mapping) or not isinstance(clinical, Mapping):
        raise ValueError("both dermoscopic and clinical modality records are required")
    if derm.get("available") is not True:
        raise ValueError("Derm7pt dermoscopic input unexpectedly unavailable")

    derm_path, derm_relative = _safe_image_path(images_root, derm.get("path"))
    if require_files and not derm_path.is_file():
        raise FileNotFoundError(derm_path)
    derm_sha256 = str(derm.get("sha256") or "")
    if len(derm_sha256) != 64:
        raise ValueError("invalid dermoscopic SHA-256")

    clinical_path: Path | None = None
    clinical_relative: str | None = None
    clinical_sha256: str | None = None
    unavailable_reason: ClinicalUnavailableReason | None = None
    try:
        candidate_path, candidate_relative = _safe_image_path(
            images_root, clinical.get("path")
        )
    except ValueError:
        candidate_path = None
        candidate_relative = None

    candidate_sha = str(clinical.get("sha256") or "") or None
    reused = bool(
        clinical.get("distinct_from_dermoscopic") is False
        or clinical.get("duplicate_of") == "dermoscopic"
        or (candidate_relative is not None and candidate_relative == derm_relative)
        or (candidate_sha is not None and candidate_sha == derm_sha256)
    )
    missing = bool(
        clinical.get("available") is not True
        or candidate_path is None
        or (require_files and not candidate_path.is_file())
    )
    if reused:
        unavailable_reason = "dermoscopy_reused"
    elif missing:
        unavailable_reason = "missing"
    else:
        if candidate_sha is None or len(candidate_sha) != 64:
            raise ValueError("invalid clinical SHA-256")
        clinical_path = candidate_path
        clinical_relative = candidate_relative
        clinical_sha256 = candidate_sha

    return Phase24BCaseInput(
        case_index=int(record["case_index"]),
        case_num=int(record["case_num"]),
        group_id=str(record["group_id"]),
        fold=fold,
        source_split=record["source_split"],  # type: ignore[arg-type]
        dermoscopic_path=derm_path,
        dermoscopic_relative_path=derm_relative,
        dermoscopic_sha256=derm_sha256,
        clinical_path=clinical_path,
        clinical_relative_path=clinical_relative,
        clinical_sha256=clinical_sha256,
        clinical_available=clinical_path is not None,
        clinical_unavailable_reason=unavailable_reason,
        facts=parse_prediagnostic_facts(record),
    )


def resolve_development_inputs(
    *,
    inventory_path: Path = DERM7PT_INVENTORY,
    fold_manifest_path: Path = DEVELOPMENT_FOLDS,
    clinical_photo_audit_path: Path = CLINICAL_PHOTO_AUDIT,
    structured_fact_support_path: Path = STRUCTURED_FACT_SUPPORT,
    images_root: Path = DERM7PT_IMAGES,
    require_files: bool = True,
) -> Phase24BInputSet:
    """Resolve exactly the locked 616 development groups in inventory order."""
    inventory_path = Path(inventory_path)
    fold_manifest_path = Path(fold_manifest_path)
    clinical_photo_audit_path = Path(clinical_photo_audit_path)
    structured_fact_support_path = Path(structured_fact_support_path)
    inventory_sha256 = sha256_file(inventory_path)
    fold_manifest_sha256 = sha256_file(fold_manifest_path)
    clinical_photo_audit_sha256 = sha256_file(clinical_photo_audit_path)
    structured_fact_support_sha256 = sha256_file(structured_fact_support_path)
    folds_payload = _load_object(fold_manifest_path)
    derm_ledger = folds_payload.get("derm7pt")
    if not isinstance(derm_ledger, Mapping):
        raise ValueError("fold ledger lacks a Derm7pt object")
    if derm_ledger.get("source_inventory_sha256") != inventory_sha256:
        raise ValueError("fold ledger is not pinned to this Phase 1 inventory")
    fold_rows = derm_ledger.get("folds")
    if not isinstance(fold_rows, Sequence) or isinstance(fold_rows, (str, bytes)):
        raise ValueError("Derm7pt fold ledger rows are invalid")
    fold_map: dict[str, tuple[int, str]] = {}
    for item in fold_rows:
        if not isinstance(item, Mapping):
            raise ValueError("Derm7pt fold ledger row is invalid")
        group_id = str(item.get("group_id") or "")
        fold = item.get("fold")
        label = str(item.get("harmonized5") or "")
        if not group_id or not isinstance(fold, int) or fold not in range(5):
            raise ValueError("invalid Derm7pt development fold identity")
        if group_id in fold_map:
            raise ValueError(f"duplicate group in fold ledger: {group_id}")
        fold_map[group_id] = (fold, label)
    if len(fold_map) != N_CASES:
        raise ValueError(f"expected {N_CASES} locked development groups")

    inventory = _load_object(inventory_path)
    clinical_audit = _load_object(clinical_photo_audit_path)
    fact_support = _load_object(structured_fact_support_path)
    clinical_contract = clinical_audit.get("row_contract")
    if not isinstance(clinical_contract, Mapping):
        raise ValueError("Phase 2.3 clinical-photo audit lacks a row contract")
    derm_fact_support = fact_support.get("datasets", {}).get("derm", {})
    expected_fact_support = {
        "age": (0, N_CASES),
        "sex": (N_CASES, 0),
        "site": (N_CASES, 0),
        "elevation": (N_CASES, 0),
    }
    if derm_fact_support.get("rows") != N_CASES:
        raise ValueError("Phase 2.3 Derm7pt structured-fact row count changed")
    for field, (available, missing) in expected_fact_support.items():
        support = derm_fact_support.get("fields", {}).get(field, {})
        if (support.get("available"), support.get("missing")) != (available, missing):
            raise ValueError(f"Phase 2.3 Derm7pt {field} support changed")
    inventory_rows = inventory.get("records")
    if not isinstance(inventory_rows, Sequence) or isinstance(
        inventory_rows, (str, bytes)
    ):
        raise ValueError("Phase 1 inventory records are invalid")

    cases: list[Phase24BCaseInput] = []
    phase23_identity: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in inventory_rows:
        if not isinstance(record, Mapping):
            raise ValueError("Phase 1 inventory row is invalid")
        group_id = str(record.get("group_id") or "")
        locked = fold_map.get(group_id)
        if locked is None:
            continue
        if group_id in seen:
            raise ValueError(f"duplicate authorized inventory group: {group_id}")
        fold, locked_label = locked
        diagnosis = record.get("diagnosis")
        if not isinstance(diagnosis, Mapping) or diagnosis.get("harmonized5") != locked_label:
            raise ValueError(f"fold-label drift for {group_id}")
        clinical = record.get("modalities", {}).get("clinical", {})
        if not isinstance(clinical, Mapping):
            raise ValueError(f"invalid clinical modality record for {group_id}")
        phase23_identity.append(
            {
                "group_id": group_id,
                "clinical_sha256": str(clinical.get("sha256") or ""),
                "distinct_clinical_available": bool(
                    clinical.get("available")
                    and clinical.get("distinct_from_dermoscopic")
                ),
                "development_fold": fold,
            }
        )
        cases.append(
            resolve_case_input(
                record, fold=fold, images_root=images_root, require_files=require_files
            )
        )
        seen.add(group_id)

    if seen != set(fold_map) or len(cases) != N_CASES:
        raise ValueError("Phase 1 inventory does not cover every locked development group")
    if len({case.group_id for case in cases}) != N_CASES:
        raise ValueError("development group identity is not one-to-one")
    if Counter(case.fold for case in cases) != FOLD_COUNTS:
        raise ValueError("locked Derm7pt fold counts changed")
    available = sum(case.clinical_available for case in cases)
    if available != N_AVAILABLE_CLINICAL:
        raise ValueError(f"expected {N_AVAILABLE_CLINICAL} available clinical photos")
    if len(cases) - available != N_UNAVAILABLE_CLINICAL:
        raise ValueError(f"expected {N_UNAVAILABLE_CLINICAL} unavailable clinical photos")
    if clinical_contract.get("n_cases") != N_CASES:
        raise ValueError("Phase 2.3 clinical-photo audit case count changed")
    if clinical_contract.get("n_distinct_clinical_photos") != available:
        raise ValueError("Phase 2.3 clinical-photo availability count changed")
    if clinical_contract.get("n_non_distinct_views_masked") != len(cases) - available:
        raise ValueError("Phase 2.3 clinical-photo unavailable count changed")
    if clinical_contract.get("record_identity_sha256") != canonical_hash(phase23_identity):
        raise ValueError("Phase 2.3 clinical-photo row identity changed")
    if clinical_contract.get("availability_sha256") != canonical_hash(
        [int(case.clinical_available) for case in cases]
    ):
        raise ValueError("Phase 2.3 clinical-photo availability mask changed")

    hard_scope = {
        "derm7pt_development_rows_resolved": N_CASES,
        "derm7pt_official_test_rows_loaded": 0,
        "ham_calibration_or_final_rows_loaded": 0,
        "bcn_validation_or_final_rows_loaded": 0,
        "ph2_rows_loaded": 0,
        "clinical_images_opened": 0,
        "models_loaded": 0,
        "prefix_tokens_extracted": 0,
        "training_steps": 0,
        "predictions_generated": 0,
        "metrics_computed": 0,
    }
    return Phase24BInputSet(
        cases=tuple(cases),
        inventory_path=inventory_path.resolve(),
        inventory_sha256=inventory_sha256,
        fold_manifest_path=fold_manifest_path.resolve(),
        fold_manifest_sha256=fold_manifest_sha256,
        clinical_photo_audit_path=clinical_photo_audit_path.resolve(),
        clinical_photo_audit_sha256=clinical_photo_audit_sha256,
        structured_fact_support_path=structured_fact_support_path.resolve(),
        structured_fact_support_sha256=structured_fact_support_sha256,
        images_root=Path(images_root).resolve(),
        hard_scope=hard_scope,
    )


def _case_identity(case: Phase24BCaseInput) -> dict[str, Any]:
    return {
        "case_index": case.case_index,
        "case_num": case.case_num,
        "group_id": case.group_id,
        "fold": case.fold,
        "source_split": case.source_split,
    }


def _fact_identity(case: Phase24BCaseInput) -> dict[str, Any]:
    facts = case.facts
    return {
        "group_id": case.group_id,
        "age_years": facts.age_years,
        "sex": facts.sex,
        "site": facts.site,
        "elevation": facts.elevation,
        "available": facts.available.as_tuple(),
    }


def build_block10_cache_manifest_spec(
    inputs: Phase24BInputSet, config: Block10CacheConfig
) -> dict[str, Any]:
    """Return a deterministic manifest specification; perform no extraction."""
    if config.shared_prefix_blocks != SHARED_PREFIX_BLOCKS:
        raise ValueError("Phase 2.4B requires exactly ten immutable shared blocks")
    if config.hidden_size <= 0 or config.tokens_per_image <= 0:
        raise ValueError("cache tensor dimensions must be positive")
    for name, value in {
        "checkpoint_sha256": config.checkpoint_sha256,
        "preprocessing_sha256": config.preprocessing_sha256,
    }.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    cases = inputs.cases
    if len(cases) != N_CASES:
        raise ValueError("cache spec requires the complete 616-case input set")
    availability = [int(case.clinical_available) for case in cases]
    row_identity = [_case_identity(case) for case in cases]
    derm_paths = [
        {
            "group_id": case.group_id,
            "path": case.dermoscopic_relative_path,
            "sha256": case.dermoscopic_sha256,
        }
        for case in cases
    ]
    clinical_paths = [
        {
            "group_id": case.group_id,
            "path": case.clinical_relative_path,
            "sha256": case.clinical_sha256,
            "available": case.clinical_available,
            "unavailable_reason": case.clinical_unavailable_reason,
        }
        for case in cases
    ]
    facts = [_fact_identity(case) for case in cases]
    manifest: dict[str, Any] = {
        "protocol": "dermpact-phase2-4b-block10-prefix-cache-manifest-spec-v1",
        "status": "SPECIFICATION_ONLY_NO_EXTRACTION",
        "allowed_role": "Derm7pt 616-case development only",
        "encoder_identity": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "checkpoint_sha256": config.checkpoint_sha256,
            "preprocessing_sha256": config.preprocessing_sha256,
            "shared_prefix_blocks": config.shared_prefix_blocks,
            "cache_boundary": "output tokens after immutable shared block 10",
            "dtype": config.dtype,
            "token_layout": config.token_layout,
            "hidden_size": config.hidden_size,
            "tokens_per_image": config.tokens_per_image,
        },
        "source_artifacts": {
            "phase1_inventory": {
                "path": inputs.inventory_path.relative_to(ROOT).as_posix(),
                "sha256": inputs.inventory_sha256,
            },
            "phase2_1_fold_ledger": {
                "path": inputs.fold_manifest_path.relative_to(ROOT).as_posix(),
                "sha256": inputs.fold_manifest_sha256,
            },
            "phase2_3_clinical_photo_audit": {
                "path": inputs.clinical_photo_audit_path.relative_to(ROOT).as_posix(),
                "sha256": inputs.clinical_photo_audit_sha256,
            },
            "phase2_3_structured_fact_support": {
                "path": inputs.structured_fact_support_path.relative_to(ROOT).as_posix(),
                "sha256": inputs.structured_fact_support_sha256,
            },
        },
        "row_contract": {
            "n_cases": len(cases),
            "n_groups": len({case.group_id for case in cases}),
            "n_available_clinical": sum(availability),
            "n_unavailable_clinical": len(cases) - sum(availability),
            "fold_counts": {str(key): value for key, value in sorted(FOLD_COUNTS.items())},
            "row_identity_sha256": canonical_hash(row_identity),
            "fold_identity_sha256": canonical_hash(
                [{"group_id": case.group_id, "fold": case.fold} for case in cases]
            ),
            "clinical_availability_sha256": canonical_hash(availability),
        },
        "path_contract": {
            "dermoscopic_paths_sha256": canonical_hash(derm_paths),
            "clinical_paths_sha256": canonical_hash(clinical_paths),
            "paired_paths_sha256": canonical_hash(
                [
                    {"dermoscopic": derm, "clinical": clinical}
                    for derm, clinical in zip(derm_paths, clinical_paths)
                ]
            ),
            "available_paths_exist": all(
                case.dermoscopic_path.is_file()
                and (case.clinical_path is None or case.clinical_path.is_file())
                for case in cases
            ),
            "unavailable_clinical_path_policy": "None; never encode or substitute dermoscopy",
        },
        "fact_contract": {
            "fields": ["age", "sex", "site", "elevation"],
            "values_sha256": canonical_hash(facts),
            "missing_policy": "None plus false mask; never impute",
            "age_source_status": "unavailable for Derm7pt",
        },
        "cache_files": {
            "dermoscopic": "dermoscopic_block10_prefix_tokens.npy",
            "clinical": "clinical_block10_prefix_tokens.npy",
            "clinical_available": "clinical_available.npy",
            "row_identity": "row_identity.json",
        },
        "hard_scope": dict(inputs.hard_scope),
    }
    manifest["cache_identity_sha256"] = canonical_hash(manifest)
    return manifest


__all__ = [
    "Block10CacheConfig",
    "FactAvailability",
    "Phase24BCaseInput",
    "Phase24BInputSet",
    "PrediagnosticFacts",
    "build_block10_cache_manifest_spec",
    "canonical_hash",
    "parse_prediagnostic_facts",
    "resolve_case_input",
    "resolve_development_inputs",
]
