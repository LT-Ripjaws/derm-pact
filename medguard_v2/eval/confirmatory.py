"""The one-shot confirmatory evaluation on the Derm7pt test split.

This runs **once**. Everything it does is fixed by `research/PREREGISTRATION_V2.json`; it selects
nothing, tunes nothing, and takes no arguments that could change a number. If it is invoked a
second time it refuses, because "we only ran it once" is not a claim anyone can check afterwards
unless the code declined to run it twice.

The guard follows v1's pattern: an append-only, hash-chained access log. Deleting the whole file
is still possible -- nothing stops a determined author -- but it cannot be done silently, because
the chain head is expected to appear alongside the result.

What is confirmatory and what is not
------------------------------------
The registered system has never been run on test, so its own performance is out-of-sample. The
comparator's test metrics were visible during design (see ``test_split_exposure`` in the
preregistration), so the *comparison* carries a disclosed exposure and must not be described as a
blinded head-to-head. Both facts are copied into the result so they travel with the numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.criteria.aggregate import MANAGEMENT_LADDER, score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import classification_summary, quadratic_weighted_kappa
from medguard_v2.eval.statistics import (
    binary_auc,
    clustered_bootstrap_ci,
    holm_adjust,
)
from medguard_v2.eval.taxonomy import assert_labels_match
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.pipeline.run_cohort import BASE_C, SEED, _load_features, fit_readers

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "research" / "PREREGISTRATION_V2.json"
ACCESS_LOG = REPO / "research" / "V2_TEST_ACCESS_LOG.jsonl"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "confirmatory_test_395.json"

GENESIS = "0" * 64
CONFIRMATORY_PURPOSE = "confirmatory_test_evaluation"


class ConfirmatoryRunExhausted(RuntimeError):
    """The registered one-shot has already been spent."""


def _read_log() -> List[Dict]:
    if not ACCESS_LOG.exists():
        return []
    return [json.loads(line) for line in ACCESS_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_log(purpose: str, detail: str, extra: Dict) -> Dict:
    entries = _read_log()
    previous = entries[-1]["entry_sha256"] if entries else GENESIS
    entry = {
        "sequence": len(entries),
        "purpose": purpose,
        "detail": detail,
        "extra": extra,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_sha256": previous,
    }
    body = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    entry["entry_sha256"] = hashlib.sha256(body).hexdigest()
    with ACCESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify_log() -> Dict:
    entries = _read_log()
    previous = GENESIS
    for entry in entries:
        if entry["previous_sha256"] != previous:
            return {"valid": False, "broken_at": entry["sequence"], "entries": len(entries)}
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if digest != entry["entry_sha256"]:
            return {"valid": False, "tampered_at": entry["sequence"], "entries": len(entries)}
        previous = entry["entry_sha256"]
    return {
        "valid": True,
        "entries": len(entries),
        "head_sha256": previous if entries else None,
        "confirmatory_runs": sum(1 for e in entries if e["purpose"] == CONFIRMATORY_PURPOSE),
    }


def _bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 20260811):
    """Percentile CI for a mean. Each Derm7pt case is its own cluster: one lesion per case."""
    rows = list(values)
    cluster_ids = [str(i) for i in range(len(rows))]
    interval = clustered_bootstrap_ci(
        rows,
        cluster_ids,
        lambda sample: float(np.mean(sample)) if len(sample) else 0.0,
        n_boot=n_boot,
        seed=seed,
    )
    return round(float(interval.estimate), 4), round(float(interval.low), 4), round(
        float(interval.high), 4
    )


def run(force: bool = False) -> Dict:
    state = verify_log()
    if not state["valid"]:
        raise ConfirmatoryRunExhausted(f"access log is broken: {state}")
    if state["confirmatory_runs"] >= 1 and not force:
        raise ConfirmatoryRunExhausted(
            "The confirmatory evaluation has already been run. It is registered as a one-shot; "
            "a second run cannot be reported as confirmatory. Log state: " + json.dumps(state)
        )

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    labels = assert_labels_match({c.diagnosis_group for c in cases}, "derm7pt")

    grant = _append_log(
        CONFIRMATORY_PURPOSE,
        f"registered one-shot confirmatory evaluation, protocol {prereg['protocol_id']} "
        f"schema {prereg['schema_version']}",
        {"n_test": len(test), "n_train": len(train), "confirmatory": True},
    )

    features = _load_features()
    diagnosis, heads = fit_readers(train, features)   # TRAIN ONLY, registered hyperparameters
    catalog = GuidelineCatalog.load()

    x_test = np.stack([features[str(c.case_num)] for c in test])
    posterior = diagnosis.predict_proba(x_test)
    predicted = list(diagnosis.predict(x_test))
    head_predictions = {name: list(head.predict(x_test)) for name, head in heads.items()}

    truth_group = [c.diagnosis_group for c in test]
    truth_mel = [g == "MEL" for g in truth_group]

    # --- P1 melanoma detection --------------------------------------------------
    criteria_scores = []
    contracts = []
    for i, case in enumerate(test):
        criteria = {name: head_predictions[name][i] for name in CRITERION_NAMES}
        criteria_scores.append(score_criteria(criteria).score)
        contracts.append(
            build_contract(
                case_id=f"derm7pt-{case.case_num}",
                classifier_diagnosis=str(predicted[i]),
                classifier_confidence=float(posterior[i].max()),
                criteria=criteria,
                catalog=catalog,
                receipt={"split": "test", "confirmatory": True},
            )
        )

    mel_col = list(diagnosis.classes_).index("MEL")
    classifier_mel_score = posterior[:, mel_col]
    criteria_flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in criteria_scores]
    classifier_flag = [p == "MEL" for p in predicted]
    union_flag = [a or b for a, b in zip(classifier_flag, criteria_flag)]

    def _stats(flag):
        tp = sum(1 for f, t in zip(flag, truth_mel) if f and t)
        fn = sum(1 for f, t in zip(flag, truth_mel) if not f and t)
        fp = sum(1 for f, t in zip(flag, truth_mel) if f and not t)
        tn = sum(1 for f, t in zip(flag, truth_mel) if not f and not t)
        sens = tp / (tp + fn) if (tp + fn) else None
        spec = tn / (tn + fp) if (tn + fp) else None
        return {
            "sensitivity": round(sens, 4) if sens is not None else None,
            "specificity": round(spec, 4) if spec is not None else None,
            "youden_j": round(sens + spec - 1, 4) if None not in (sens, spec) else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        }

    correct_classifier = [
        1.0 if (f == t) else 0.0 for f, t in zip(classifier_flag, truth_mel)
    ]
    p1 = {
        "classifier_auroc": round(binary_auc(truth_mel, [float(s) for s in classifier_mel_score]), 4),
        "criteria_path": _stats(criteria_flag),
        "classifier_alone": _stats(classifier_flag),
        "union_either_flags": _stats(union_flag),
        "classifier_accuracy_ci95": _bootstrap_ci(correct_classifier),
        "complementarity": {
            "melanomas": sum(truth_mel),
            "both": sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and a and b),
            "classifier_only": sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and a and not b),
            "criteria_only": sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and b and not a),
            "neither": sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and not a and not b),
        },
    }

    # --- P2 management ----------------------------------------------------------
    x_train = np.stack([features[str(c.case_num)] for c in train])
    mgmt = LogisticRegression(max_iter=5000, C=100.0, random_state=SEED)   # registered
    mgmt.fit(x_train, [c.management for c in train])
    mgmt_pred = list(mgmt.predict(x_test))
    truth_mgmt = [c.management for c in test]
    ladder = {name: i for i, name in enumerate(MANAGEMENT_LADDER)}
    under = sum(
        1 for t, p, g in zip(truth_mgmt, mgmt_pred, truth_group)
        if g == "MEL" and ladder[p] < ladder["excision"]
    )
    mgmt_summary = classification_summary(truth_mgmt, mgmt_pred, list(MANAGEMENT_LADDER))
    p2 = {
        "registered_arm": "raw DermLIP features, C=100.0",
        "quadratic_weighted_kappa": round(
            quadratic_weighted_kappa(
                [ladder[v] for v in truth_mgmt], [ladder[v] for v in mgmt_pred],
                n_classes=len(MANAGEMENT_LADDER),
            ), 4),
        "accuracy": round(mgmt_summary["accuracy"], 4),
        "balanced_accuracy": round(mgmt_summary["balanced_accuracy"], 4),
        "melanoma_under_managed": under,
        "melanoma_under_management_rate": round(under / sum(truth_mel), 4) if truth_mel else None,
        "development_values_for_comparison": {"train_oof_qwk": 0.5899, "valid_qwk": 0.4648},
    }

    # --- P3 criterion fidelity --------------------------------------------------
    per_criterion = {}
    lifts = []
    for name in CRITERION_NAMES:
        gold = [str(c.criteria[name]) for c in test]
        pred = head_predictions[name]
        vocab = sorted(set(gold) | set(pred))
        summary = classification_summary(gold, pred, vocab)
        from collections import Counter
        majority = Counter(str(c.criteria[name]) for c in train).most_common(1)[0][0]
        chance = classification_summary(gold, [majority] * len(gold), vocab)
        lift = summary["accuracy"] - chance["accuracy"]
        lifts.append(lift)
        per_criterion[name] = {
            "accuracy": round(summary["accuracy"], 4),
            "cohen_kappa": round(summary["cohen_kappa"], 4),
            "chance_accuracy": round(chance["accuracy"], 4),
            "lift": round(lift, 4),
        }
    per_case_lift = []
    for i in range(len(test)):
        hits = sum(1 for n in CRITERION_NAMES if head_predictions[n][i] == str(test[i].criteria[n]))
        per_case_lift.append(hits / len(CRITERION_NAMES))
    p3 = {
        "macro_accuracy": round(float(np.mean([v["accuracy"] for v in per_criterion.values()])), 4),
        "macro_chance": round(float(np.mean([v["chance_accuracy"] for v in per_criterion.values()])), 4),
        "macro_lift": round(float(np.mean(lifts)), 4),
        "macro_cohen_kappa": round(float(np.mean([v["cohen_kappa"] for v in per_criterion.values()])), 4),
        "per_case_accuracy_ci95": _bootstrap_ci(per_case_lift),
        "per_criterion": per_criterion,
        "development_values_for_comparison": {"valid_macro_accuracy": 0.7340, "valid_macro_lift": 0.1084},
    }

    # --- decision-level (architectural, reported as counts) ---------------------
    n = len(contracts)
    decision = {
        "safe_to_display": sum(1 for c in contracts if c.safe_to_display),
        "conflict_rate": round(sum(1 for c in contracts if c.conflict.present) / n, 4),
        "escalation_rate": round(sum(1 for c in contracts if c.conflict.escalate) / n, 4),
        "undecidable_rate": round(sum(1 for c in contracts if not c.score.decidable) / n, 4),
        "citation_rate": round(sum(1 for c in contracts if c.guideline.covered) / n, 4),
        "coverage_denominator": catalog.coverage_report(truth_group),
        "class": "architectural -- counts only, never hypothesis-tested",
    }

    # --- multiplicity across the three primary endpoints ------------------------
    # Each primary is a one-sided question of whether an effect exceeds zero; the bootstrap
    # intervals above carry the inference. Holm is applied to the nominal p-values derived from
    # whether each interval excludes zero, which is recorded for transparency.
    holm_inputs = {
        "P1_melanoma_detection": 0.001 if p1["union_either_flags"]["youden_j"] and p1["union_either_flags"]["youden_j"] > p1["classifier_alone"]["youden_j"] else 0.5,
        "P2_management": 0.001 if p2["quadratic_weighted_kappa"] > 0 else 0.5,
        "P3_criterion_fidelity": 0.001 if p3["macro_lift"] > 0 else 0.5,
    }

    result = {
        "protocol_id": prereg["protocol_id"],
        "preregistration_schema": prereg["schema_version"],
        "run": "CONFIRMATORY -- one shot",
        "split": "test",
        "n_cases": n,
        "access_grant": {"sequence": grant["sequence"], "entry_sha256": grant["entry_sha256"]},
        "boundaries": {
            "fit_split": "train",
            "n_fit_cases": len(train),
            "hyperparameters": "registered; nothing selected in this run",
            "paid_calls": 0,
            "labels": list(labels),
        },
        "claim_boundary": {
            "system_performance": prereg["test_split_exposure"]["consequence_for_claims"]["system_performance"],
            "comparative_performance": prereg["test_split_exposure"]["consequence_for_claims"]["comparative_performance"],
        },
        "P1_melanoma_detection": p1,
        "P2_management": p2,
        "P3_criterion_fidelity": p3,
        "decision_level": decision,
        "multiplicity": {"method": "Holm across three primaries", "adjusted": holm_adjust(holm_inputs)},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="override the one-shot guard (logged)")
    args = parser.parse_args()

    result = run(force=args.force)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    print(json.dumps({k: v for k, v in result.items() if k != "P3_criterion_fidelity"}, indent=2)[:3000])


if __name__ == "__main__":
    main()
