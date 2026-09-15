from __future__ import annotations

import torch

from dermpact.eval import dermpact_phase2_4c_r_safe_student as safe
from dermpact.eval import dermpact_phase2_4c_r_teacher_feature_only as teacher


def test_privileged_teacher_keeps_label_slot_and_student_export_is_clean() -> None:
    evidence = safe.load_cache()
    arrays = safe.fold_arrays(evidence, 0)
    bundle, _ = teacher.build_bundle(arrays)
    indices = arrays.train_indices[:32]
    features = torch.from_numpy(evidence.stream_features[indices])
    logits = torch.from_numpy(evidence.first_pass_logits[indices])
    uncertainty = torch.from_numpy(evidence.uncertainty[indices])
    present = torch.from_numpy(evidence.stream_present[indices])
    targets = teacher._targets(evidence, indices)
    result, total, _, _ = teacher._loss(
        bundle, features, logits, uncertainty, present, targets, 0.0
    )
    assert torch.equal(result.teacher_conditioning_probabilities, targets)
    assert torch.isfinite(total)
    deployable = bundle.deployable_student()
    assert all(
        "teacher" not in key and "label" not in key
        for key in deployable.state_dict()
    )
