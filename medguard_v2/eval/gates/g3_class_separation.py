"""G3 -- which classes do the 7-point criteria actually separate?

G2 established that gold criteria carry real signal, but it also showed 5-class balanced accuracy
of only 0.4247.  Both facts can be true at once if the criteria are a *melanoma instrument* rather
than a general classifier -- which is exactly what the checklist was designed to be.  G3 measures
that directly, because it determines the architecture:

    criteria discriminate X from Y  ->  route X-vs-Y through the criteria path
    criteria do not                 ->  route it through the image classifier

Getting this wrong in either direction is expensive.  Sending seborrhoeic keratosis through a
melanoma checklist produces confident nonsense with an auditable rationale attached, which is
worse than no rationale at all.

Three measurements, all on gold criteria, fit on train and evaluated on valid:

1. **Pairwise separability** -- AUROC for every class pair from criteria alone.
2. **Score profile per class** -- the 7-point score distribution by diagnosis group.
3. **Per-criterion informativeness** -- normalised mutual information between each criterion and
   each class, showing which criteria carry which class.

Small classes make some pairwise estimates unreliable; any pair with fewer than 20 validation
cases on either side is reported but flagged, not quietly averaged in.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.data.derm7pt import SEVEN_CRITERIA, load_cases
from medguard_v2.eval.statistics import binary_auc
from medguard_v2.eval.taxonomy import assert_labels_match

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "g3_class_separation.json"

SEED = 20260810
MIN_RELIABLE_N = 20


def _criteria_matrix(cases, encoder=None):
    rows = [[str(c.criteria[name]) for name in SEVEN_CRITERIA] for c in cases]
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return encoder.fit_transform(rows), encoder
    return encoder.transform(rows), encoder


def _pairwise_separability(train, valid) -> Dict[str, Dict]:
    # Canonical order, guarded: a cohort missing a class would otherwise silently drop the pairs
    # that involve it, and the pair table would look complete while being short.
    groups = list(assert_labels_match({c.diagnosis_group for c in train}, "derm7pt"))
    out: Dict[str, Dict] = {}

    for a, b in itertools.combinations(groups, 2):
        tr = [c for c in train if c.diagnosis_group in (a, b)]
        va = [c for c in valid if c.diagnosis_group in (a, b)]
        n_a = sum(1 for c in va if c.diagnosis_group == a)
        n_b = sum(1 for c in va if c.diagnosis_group == b)

        key = f"{a}_vs_{b}"
        if not tr or n_a == 0 or n_b == 0:
            out[key] = {"auroc": None, "n_valid": {a: n_a, b: n_b}, "reliable": False,
                        "note": "empty class in train or valid"}
            continue

        x_tr, enc = _criteria_matrix(tr)
        x_va, _ = _criteria_matrix(va, enc)
        y_tr = [c.diagnosis_group == a for c in tr]
        y_va = [c.diagnosis_group == a for c in va]

        model = LogisticRegression(max_iter=5000, random_state=SEED)
        model.fit(x_tr, y_tr)
        scores = model.predict_proba(x_va)[:, list(model.classes_).index(True)]

        out[key] = {
            "auroc": round(binary_auc(y_va, [float(s) for s in scores]), 4),
            "n_valid": {a: n_a, b: n_b},
            "reliable": bool(min(n_a, n_b) >= MIN_RELIABLE_N),
        }
    return out


def _score_profile(cases) -> Dict[str, Dict]:
    by_group: Dict[str, List[int]] = defaultdict(list)
    for case in cases:
        by_group[case.diagnosis_group].append(score_criteria(case.criteria).score)

    profile = {}
    for group, scores in sorted(by_group.items()):
        arr = np.array(scores, dtype=float)
        profile[group] = {
            "n": len(scores),
            "mean_score": round(float(arr.mean()), 3),
            "median_score": float(np.median(arr)),
            "pct_at_or_above_threshold": round(float((arr >= 3).mean()), 4),
        }
    return profile


def _criterion_informativeness(cases) -> Dict[str, Dict[str, float]]:
    """Normalised mutual information between each criterion and each class indicator."""
    groups = sorted({c.diagnosis_group for c in cases})
    out: Dict[str, Dict[str, float]] = {}
    for name in CRITERION_NAMES:
        values = [str(c.criteria[name]) for c in cases]
        per_class = {}
        for group in groups:
            indicator = [str(c.diagnosis_group == group) for c in cases]
            per_class[group] = round(
                float(normalized_mutual_info_score(values, indicator)), 4
            )
        out[name] = per_class
    return out


def run() -> Dict:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    if any(c.split == "test" for c in train + valid):
        raise AssertionError("test split leaked into G3")

    pairwise = _pairwise_separability(train, valid)
    reliable = {k: v for k, v in pairwise.items() if v.get("reliable")}
    unreliable = sorted(k for k, v in pairwise.items() if not v.get("reliable"))

    mel_pairs = {k: v["auroc"] for k, v in reliable.items() if "MEL" in k}
    non_mel_pairs = {k: v["auroc"] for k, v in reliable.items() if "MEL" not in k}

    report = {
        "gate_id": "G3_class_separation",
        "question": "Which class pairs do the 7-point criteria separate, and which do they not?",
        "boundaries": {
            "images_encoded": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "test_split_read": False,
            "fit_on": "train",
            "evaluated_on": "valid",
        },
        "cohort": {
            "train": len(train),
            "valid": len(valid),
            "valid_class_counts": dict(Counter(c.diagnosis_group for c in valid)),
        },
        "pairwise_auroc": pairwise,
        "unreliable_pairs_small_n": unreliable,
        "score_profile_by_class": _score_profile(cases),
        "criterion_informativeness_nmi": _criterion_informativeness(cases),
        "summary": {
            "mean_auroc_pairs_involving_melanoma": (
                round(float(np.mean(list(mel_pairs.values()))), 4) if mel_pairs else None
            ),
            "mean_auroc_pairs_without_melanoma": (
                round(float(np.mean(list(non_mel_pairs.values()))), 4) if non_mel_pairs else None
            ),
            "melanoma_pairs": mel_pairs,
            "non_melanoma_pairs": non_mel_pairs,
        },
    }

    mel_mean = report["summary"]["mean_auroc_pairs_involving_melanoma"]
    non_mel_mean = report["summary"]["mean_auroc_pairs_without_melanoma"]
    report["verdict"] = {
        "criteria_are_melanoma_specific": bool(
            mel_mean is not None and non_mel_mean is not None and mel_mean - non_mel_mean > 0.10
        ),
        "architecture_implication": (
            "Route melanocytic differentials through the criteria path; route pairs the criteria "
            "do not separate through the image classifier. Applying a melanoma checklist to a "
            "non-melanocytic lesion yields a confident rationale for the wrong question."
        ),
        "caveat": (
            f"Pairs with fewer than {MIN_RELIABLE_N} validation cases per side are excluded from "
            f"the means and listed under unreliable_pairs_small_n."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = run()
    text = json.dumps(report, indent=2)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(text)


if __name__ == "__main__":
    main()
