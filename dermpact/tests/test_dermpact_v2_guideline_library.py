from __future__ import annotations

import json

from dermpact.core.eval import guideline_library_release as release


def test_layered_guideline_release_restores_66_without_executable_conflation(tmp_path):
    manifest = release.build_release(tmp_path)
    assert manifest["source"]["hash_verified"] is True
    assert manifest["layers"]["retrieval_library"]["n_records"] == 66
    assert manifest["layers"]["reviewed_v2_next_step_catalogue"]["n_records"] == 14
    assert manifest["layers"]["reviewed_v2_next_step_catalogue"]["status_counts"]["executable"] == 5
    rows = [json.loads(line) for line in (tmp_path / "public_metadata_catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 66
    assert all("text" not in row for row in rows)
    assert all(row["runtime_layer"] == "retrieval_only_not_executable" for row in rows)
