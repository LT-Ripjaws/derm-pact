"""Resume-identity seam for Phase 2.4A-T2 epoch-20."""
import pytest

from dermpact.eval import dermpact_phase2_4a_t2_epoch20 as epoch20


def test_resume_identity_binds_all_upstream_and_candidate_inputs():
    payload = {
        "format": epoch20.RESUME_SCHEMA,
        "protocol_sha256": "protocol",
        "seed": 20260827,
        "fold": 3,
        "source_checkpoint_sha256": "source",
        "t1_ledger_sha256": epoch20.T1_LEDGER_SHA256,
        "t1_ranking_sha256": epoch20.T1_RANKING_SHA256,
        "candidate_id": "c05",
        "candidate_config_sha256": "config",
    }
    epoch20.validate_resume_identity(
        payload, protocol_sha256="protocol", fold=3,
        source_checkpoint_sha256="source",
        t1_ledger_sha256=epoch20.T1_LEDGER_SHA256,
        t1_ranking_sha256=epoch20.T1_RANKING_SHA256,
        candidate_id="c05", candidate_config_sha256="config",
    )
    payload["t1_ranking_sha256"] = "wrong"
    with pytest.raises(AssertionError, match="resume identity mismatch"):
        epoch20.validate_resume_identity(
            payload, protocol_sha256="protocol", fold=3,
            source_checkpoint_sha256="source",
            t1_ledger_sha256=epoch20.T1_LEDGER_SHA256,
            t1_ranking_sha256=epoch20.T1_RANKING_SHA256,
            candidate_id="c05", candidate_config_sha256="config",
        )
