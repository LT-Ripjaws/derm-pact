"""Diagnostic label vocabulary for the v2 evaluation toolkit.

``medguard_v2.eval.metrics`` was ported verbatim from v1 and carries the
package-relative import ``from ..config import MALIGNANT_LABELS``.  This module
supplies that symbol so the ported metrics resolve inside ``medguard_v2``
instead of reaching back into the runtime-locked v1 package.

Only the label vocabulary is ported here.  The v1 module also holds gateway
credentials, model ids and evidence-bundle paths; none of that is needed by the
evaluation toolkit and porting it would prejudge v2's runtime configuration.
The three constants below are copied verbatim from ``medguard/config.py`` and
``medguard_v2.tests.test_metrics_parity`` asserts they stay identical to v1.
"""
from __future__ import annotations

DIAGNOSIS_LABELS = ["melanoma", "nevus", "bcc", "akiec", "bkl", "df", "vasc"]
MALIGNANT_LABELS = {"melanoma", "bcc", "akiec"}
LABEL_FULL_NAMES = {
    "melanoma": "melanoma", "nevus": "melanocytic nevus",
    "bcc": "basal cell carcinoma", "akiec": "actinic keratosis",
    "bkl": "benign keratosis", "df": "dermatofibroma", "vasc": "vascular lesion",
}
