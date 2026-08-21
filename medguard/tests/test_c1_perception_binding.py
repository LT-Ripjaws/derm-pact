from __future__ import annotations

from medguard.evidence import load_development_perception_bindings


def test_repository_perception_adapter_verifies_all_160_frozen_commitments():
    bindings = load_development_perception_bindings()
    assert len(bindings) == 160
    assert all(binding.artifact_sha256 for binding in bindings.values())
    assert all(len(binding.view_image_sha256s) >= 1 for binding in bindings.values())
    assert all(len(binding.findings) == 1 for binding in bindings.values())


def test_repository_perception_adapter_carries_no_morphology_claims():
    bindings = load_development_perception_bindings()
    assert {
        finding.finding_type.value
        for binding in bindings.values()
        for finding in binding.findings
    } == {"diagnostic_commitment"}
