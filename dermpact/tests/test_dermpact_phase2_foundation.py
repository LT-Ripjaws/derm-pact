import hashlib
import json
from pathlib import Path

from dermpact.eval.derm_encoders import (
    CANDIDATES,
    PANDERM_BASE_CHECKPOINT_SHA256,
    PANDERM_BASE_REPO_COMMIT,
)
from dermpact.eval.dermpact_phase2_foundation import OUTPUT_DIR


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def test_pure_panderm_is_not_the_historical_hybrid() -> None:
    pure = CANDIDATES["panderm_base"]
    hybrid = CANDIDATES["panderm"]
    assert pure.loader == "panderm_base"
    assert pure.model_id == "SiyuanYan1/PanDerm:PanDerm_Base"
    assert hybrid.model_id == "redlessone/DermLIP_PanDerm-base-w-PubMed-256"


def test_phase2_0_receipt_and_bound_artifacts_pass() -> None:
    receipt_path = OUTPUT_DIR / "dermpact_phase2_0_receipt_v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["phase"] == "2.0"
    for implementation in receipt["implementation"].values():
        path = OUTPUT_DIR.parents[2] / implementation["path"]
        assert path.is_file()
        assert _sha256(path) == implementation["sha256"]
    for artifact in receipt["audits"].values():
        path = Path(artifact["path"])
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_phase2_0_never_claims_final_or_external_evaluation() -> None:
    receipt = _load("dermpact_phase2_0_receipt_v1.json")
    excluded = set(receipt["explicitly_not_performed"])
    assert "PH2 prediction access" in excluded
    assert "HAM10000 final-test prediction access" in excluded
    assert "Derm7pt final-test prediction access" in excluded
    assert "BCN20000 final evaluation" in excluded


def test_panderm_checkpoint_loader_and_official_equivalence_are_pinned() -> None:
    audit = _load("dermpact_phase2_0_panderm_base_audit_v1.json")
    assert audit["status"] == "PASS"
    assert audit["provenance"]["checkpoint"]["sha256"] == PANDERM_BASE_CHECKPOINT_SHA256
    assert audit["provenance"]["official_panderm_commit"] == PANDERM_BASE_REPO_COMMIT
    assert audit["metrics"]["official_builder_max_absolute_error"] <= 1e-6
    assert audit["metrics"]["repeatability_max_absolute_error"] <= 1e-7
    assert set(audit["selected_splits"]) <= {"train", "calibration"}


def test_overlap_restriction_survives_checkpoint_materialization() -> None:
    ledger = _load("dermpact_backbone_overlap_ledger_v2.json")
    pure = next(
        item for item in ledger["candidates"]
        if item["candidate_id"] == "panderm_base_vit_b16"
    )
    assert pure["checkpoint"]["availability"] == "materialized_and_loader_validated"
    assert pure["primary_claim_allowed"] is False
    assert all(
        target["status"] == "unresolved" and target["primary_claim_allowed"] is False
        for target in pure["per_target"].values()
    )


def test_environment_lock_has_no_drift() -> None:
    audit = _load("dermpact_phase2_0_environment_audit_v1.json")
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["missing_locked_distributions"] == []
    assert audit["version_mismatches"] == {}
    assert audit["unlocked_installed_distributions"] == []
