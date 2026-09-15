"""Resume-identity seam for Phase 2.4A-T2-30 finalist refits."""
import pytest

from dermpact.eval import dermpact_phase2_4a_t2_epoch30 as epoch30


def test_resume_identity_binds_all_upstream_and_finalist_inputs():
    payload = {
        "format": epoch30.RESUME_SCHEMA,
        "protocol_sha256": "protocol",
        "seed": 20260827,
        "fold": 3,
        "source_checkpoint_sha256": "source",
        "t2_epoch20_selection_sha256": epoch30.T2_EPOCH20_SELECTION_SHA256,
        "t2_epoch20_ranking_sha256": epoch30.T2_EPOCH20_RANKING_SHA256,
        "candidate_id": "c10",
        "candidate_config_sha256": "config",
    }
    epoch30.validate_resume_identity(
        payload, protocol_sha256="protocol", fold=3,
        source_checkpoint_sha256="source",
        t2_epoch20_selection_sha256=epoch30.T2_EPOCH20_SELECTION_SHA256,
        t2_epoch20_ranking_sha256=epoch30.T2_EPOCH20_RANKING_SHA256,
        candidate_id="c10", candidate_config_sha256="config",
    )
    payload["t2_epoch20_ranking_sha256"] = "wrong"
    with pytest.raises(AssertionError, match="resume identity mismatch"):
        epoch30.validate_resume_identity(
            payload, protocol_sha256="protocol", fold=3,
            source_checkpoint_sha256="source",
            t2_epoch20_selection_sha256=epoch30.T2_EPOCH20_SELECTION_SHA256,
            t2_epoch20_ranking_sha256=epoch30.T2_EPOCH20_RANKING_SHA256,
            candidate_id="c10", candidate_config_sha256="config",
        )
