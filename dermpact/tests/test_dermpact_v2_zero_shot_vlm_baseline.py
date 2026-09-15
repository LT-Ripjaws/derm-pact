from __future__ import annotations

import numpy as np

from dermpact.core.eval import zero_shot_vlm_baseline as vlm


def test_metric_probability_contract():
    targets = np.asarray([0, 1, 2, 3, 4] * 2)
    probability = np.full((10, 5), 0.025)
    probability[np.arange(10), targets] = 0.9
    metrics = vlm._metrics(probability, targets)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_ovr_auroc"] == 1.0


def test_global_to_five_mapping_covers_development_vocabulary():
    assert vlm.GLOBAL_TO_FIVE["vascular_lesion"] == "MISC"
    assert vlm.GLOBAL_TO_FIVE["dermatofibroma"] == "MISC"
    assert vlm.GLOBAL_TO_FIVE["seborrheic_keratosis_or_benign_keratosis"] == "SK"
