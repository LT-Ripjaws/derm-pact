"""Finalize all paper metrics: update stale artifacts, extract figure data,
and produce a single paper_metrics_summary.json with every reportable number.

Run once after all experiments (E1-E8) are complete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

D7 = REPO / "research/evidence/derm7pt_v1"
BCN = REPO / "research/evidence/bcn_v1"
HAM = REPO / "research/evidence/ham_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def compute_combined_faithfulness(ragas: dict) -> dict:
    """Deduplicated combined faithfulness: every non-out-of-scope answer counts once.

    Bug this replaces: (n_contract*1.0 + n_scored*faith) / (n_contract+n_scored)
    assumes contract-grounded and RAGAS-scored rows are disjoint sets. They are not:
    some override-kind rows both searched guidelines (so run_ragas scored them) AND
    are labeled grounding="contract" (so they're also counted in n_contract_grounded).
    That double-counts those rows.

    Fix: read the ragas per-sample records directly. RAGAS-scored rows (non-null
    faithfulness) contribute their own judged score; contract-grounded rows NOT
    already RAGAS-scored contribute 1.0. Every non-out-of-scope answer is counted
    exactly once.
    """
    per_sample = ragas["per_sample"]
    scored = [r for r in per_sample if r.get("faithfulness") is not None]
    overlap_n = sum(1 for r in scored if r.get("grounding") == "contract")
    n_contract = ragas["n_contract_grounded"]
    remaining_contract_n = n_contract - overlap_n
    combined_n = len(scored) + remaining_contract_n
    combined_sum = sum(r["faithfulness"] for r in scored) + remaining_contract_n * 1.0
    combined_faith = combined_sum / combined_n if combined_n else 0.0

    n_non_out_of_scope = ragas["n_rows_total"] - ragas["n_out_of_scope"]
    assert combined_n <= n_non_out_of_scope, (
        f"combined_n ({combined_n}) exceeds non-out-of-scope answer rows "
        f"({n_non_out_of_scope}) — dedup logic is double-counting somewhere."
    )

    return {
        "combined_faithfulness": combined_faith,
        "combined_n": combined_n,
        "n_ragas_scored": len(scored),
        "n_ragas_scored_and_contract": overlap_n,
        "n_contract_only_remaining": remaining_contract_n,
    }


# ── 1. Update comparison_table.json with current RAGAS numbers ────────

def _promoted_recipe_block(recipe: dict, dataset: str, taxonomy: str = "five_group") -> dict:
    """One HAM/BCN promoted-recipe row for the summary, from promoted_recipe_ham_bcn.json.

    ADOPTED numbers are the promoted recipe at ARGMAX (metadata + train-CV C): the
    Derm7pt-tuned per-class operating point does NOT transfer to HAM/BCN -- it collapses
    melanoma recall there (e.g. BCN 0.50 -> 0.29) -- so it is reported only as a disclosed
    non-transfer, never adopted. Post-hoc evaluations; registered one-shot numbers stand.
    """
    row = next(
        r for r in recipe["summary_table"]
        if r["dataset"] == dataset and r["taxonomy"] == taxonomy
    )
    mel_argmax, mel_op = row["mel_recall_b_argmax_vs_op"]
    return {
        "post_hoc": True,
        "adopted_arm": "promoted recipe at argmax (metadata + train-CV C; NO operating point)",
        "accuracy": row["b_argmax_accuracy"],
        "accuracy_delta_vs_baseline": row["accuracy_delta_b_argmax_minus_baseline"],
        "mel_auroc": row["b_mel_auroc"],
        "mel_auroc_delta_vs_baseline": row["auroc_delta_b_minus_baseline"],
        "minority_recall_baseline": row["minority_recall_baseline"],
        "minority_recall_at_op": row["minority_recall_b_op"],
        "operating_point_non_transfer": {
            "mel_recall_argmax": mel_argmax,
            "mel_recall_at_op": mel_op,
            "note": "The Derm7pt-tuned SK/MISC downweights collapse melanoma recall on this "
                    "dataset; the operating point is a Derm7pt-specific policy and is NOT "
                    "adopted here (boundary finding, consistent with G8/abstention-shift).",
        },
    }


def update_comparison_table():
    print("=== 1. Updating comparison_table.json ===")
    ct = load(D7 / "comparison_table.json")
    ragas_v3 = load(D7 / "agent_ragas_metrics_v5_4.json")
    agent_v3 = load(D7 / "agent_eval_report_v5_4.json")

    faith = ragas_v3["faithfulness"]
    n_scored = ragas_v3["n_scored_faithfulness"]
    combined = compute_combined_faithfulness(ragas_v3)
    prompt_version = agent_v3.get("prompt_version", "unknown")
    n_assessment = ragas_v3["scope_distribution"]["assessment"]
    n_scope_total = sum(ragas_v3["scope_distribution"].values())

    for row in ct["table_a_head_to_head"]["rows"]:
        m = row["metric"]
        if m == "Faithfulness (combined: exact + RAGAS)":
            row["medguard_v2"] = round(combined["combined_faithfulness"], 4)
            row["medguard_v2_n"] = combined["combined_n"]
            row["note"] = (
                f"Deduplicated so every non-out-of-scope answer counts once: "
                f"{combined['n_ragas_scored']} RAGAS-scored answers at {faith} "
                f"({combined['n_ragas_scored_and_contract']} of those are also "
                f"contract-grounded and counted only via their RAGAS score, not twice) "
                f"plus {combined['n_contract_only_remaining']} additional contract-only "
                f"answers verified exactly (1.0); n={combined['combined_n']}. "
                f"Prompt version: {prompt_version}."
            )
        elif m == "Faithfulness (RAGAS-only, guideline subset)":
            row["medguard_v2"] = faith
            row["medguard_v2_n"] = n_scored
            row["note"] = (
                f"RAGAS faithfulness on guideline+mixed answers. "
                f"Assessment-scoped ({n_assessment}/{n_scope_total}): "
                f"{ragas_v3['faithfulness_detail']['assessment_mean']}. "
                f"Judge noise floor: 0.0489 (measured)."
            )
        elif m == "Counterfactual accuracy":
            ep = agent_v3["endpoints"]["counterfactual_accuracy"]
            n = ep["n"]
            correct = sum(
                1 for r in agent_v3["rows"]
                if r.get("counterfactual") and r["counterfactual"].get("correct")
            )
            row["medguard_v2"] = f"{correct}/{n}"
        elif m == "Total test cases evaluated":
            row["medguard_v2"] = 4197
            row["note"] = "Derm7pt 395 + BCN 712 + HAM E3 1595 + HAM sealed 1495, with bootstrap CIs."

    # Update agent VQA count
    if "vqa_framing" in ct:
        ct["vqa_framing"]["our_vqa_counts"]["agent_vqa"]["n_cases"] = agent_v3["n_cases"]
        ct["vqa_framing"]["our_vqa_counts"]["agent_vqa"]["n_questions_approx"] = agent_v3["n_questions"]

    ct["_updated"] = f"{prompt_version} numbers applied by finalize_paper_metrics.py"
    save(D7 / "comparison_table.json", ct)


# ── 2. Update efficiency_analysis.json with current agent stats ───────

def update_efficiency_analysis():
    print("=== 2. Updating efficiency_analysis.json ===")
    eff = load(D7 / "efficiency_analysis.json")
    agent_v3 = load(D7 / "agent_eval_report_v5_4.json")
    ragas_v3 = load(D7 / "agent_ragas_metrics_v5_4.json")

    usage = agent_v3.get("usage", {})
    n_cases = agent_v3["n_cases"]
    n_questions = agent_v3["n_questions"]
    paid = usage.get("calls", 0)
    prompt_version = agent_v3.get("prompt_version", "unknown")

    eff["agent_evaluation_cost"] = {
        "n_cases": n_cases,
        "n_questions": n_questions,
        "paid_calls": paid,
        "cache_hits": usage.get("cache_hits", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_s": usage.get("latency_s", 0),
        "calls_per_case": round(paid / max(n_cases, 1), 2),
        "prompt_version": prompt_version,
    }

    ragas_cache = ragas_v3.get("cache_stats", {})
    eff["ragas_evaluation_cost"] = {
        "n_scorable": ragas_v3["n_guideline_scorable"],
        "judge_calls_cached": ragas_cache.get("hits", 0),
        "judge_calls_fresh": ragas_cache.get("misses", 0),
        "judge_model": ragas_v3["judge_model"],
    }

    # Honest cost reporting: the eval runner does not record token counts
    # (prompt_tokens/completion_tokens are 0 above), so an earlier ~$0.01/call
    # heuristic was used to estimate spend. Measured token-based pricing
    # ($0.10/1M prompt + $0.40/1M completion; a comparable 270-call verifier
    # study cost $0.068 total) shows that heuristic overstates cost by ~40x.
    # Report call counts as the primary figures and give only one honest
    # aggregate dollar estimate, not fabricated per-item dollars.
    eff["total_pipeline_cost"].pop("agent_eval_101_cases", None)
    eff["total_pipeline_cost"]["agent_eval"] = (
        f"{paid} calls ({n_cases} cases, {n_questions} questions)"
    )
    eff["total_pipeline_cost"]["ragas_judge"] = (
        f"{ragas_cache.get('misses', 0)} fresh judge calls, "
        f"{ragas_cache.get('hits', 0)} cached"
    )
    eff["total_pipeline_cost"]["note"] = (
        "<$2 total (token-based estimate; the per-call heuristic used previously "
        "overstates ~40x). Measured pricing: $0.10/1M prompt + $0.40/1M completion "
        "tokens; a comparable 270-call verifier study cost $0.068 total. No frontier "
        "model spend on this axis. The only prior frontier spend was G1c's one-off "
        "model sweep ($5.79)."
    )

    eff["_updated"] = f"{prompt_version} agent+RAGAS stats applied by finalize_paper_metrics.py"
    save(D7 / "efficiency_analysis.json", eff)


# ── 3. Extract figure data files ──────────────────────────────────────

def extract_roc_curve_data():
    print("=== 3a. Extracting ROC curve data ===")
    tmc = load(D7 / "test_metrics_complete.json")
    ptva = load(D7 / "promoted_two_view_ablation.json")
    roc = tmc["roc_curve"]
    stacked = tmc["stacked_ensemble"]
    proc = ptva["roc_curve"]
    parms = ptva["arms"]

    out = {
        "description": "ROC curve data for Derm7pt test (n=395). PROMOTED perception is the "
                       "system the paper describes; base (pre-promotion) curves kept for "
                       "comparison and explicitly labeled.",
        "promoted_classifier_roc": {
            "n_points": len(proc),
            "points": proc,
            "auroc": ptva["melanoma_auroc"]["point_estimate"],
            "auroc_ci": [
                ptva["melanoma_auroc"]["ci95"]["ci95_low"],
                ptva["melanoma_auroc"]["ci95"]["ci95_high"],
            ],
        },
        "promoted_union_operating_point": {
            "sensitivity": parms["union_rule"]["sensitivity"],
            "specificity": parms["union_rule"]["specificity"],
            "note": "Union is binary (OR rule), plotted as a single point on the ROC.",
        },
        "promoted_clinical_thresholds": ptva["clinical_operating_points"],
        "classifier_roc": {
            "label": "BASE (pre-promotion) classifier -- reference only",
            "n_points": len(roc),
            "points": roc,
            "auroc": tmc["melanoma_auroc"]["point_estimate"],
            "auroc_ci": [
                tmc["melanoma_auroc"]["ci95"]["ci95_low"],
                tmc["melanoma_auroc"]["ci95"]["ci95_high"],
            ],
        },
        "stacked_roc": {
            "label": "BASE-arm stacked ensemble (null result) -- reference only",
            "auroc": stacked["auroc"],
            "delong_vs_classifier": stacked["auroc_vs_classifier_delong"],
        },
        "union_operating_point": {
            "label": "BASE (pre-promotion) union -- reference only",
            "sensitivity": tmc.get("operating_point_bootstrap", {}).get("union", {}).get("sensitivity", 0.7426),
            "specificity": tmc.get("operating_point_bootstrap", {}).get("union", {}).get("specificity", 0.8503),
            "note": "Union is binary (OR rule), plotted as a single point on the ROC.",
        },
        "clinical_thresholds": tmc["clinical_operating_points"],
    }
    save(D7 / "roc_curve_data.json", out)


def extract_complementarity_venn():
    print("=== 3b. Extracting complementarity Venn data ===")
    tmc = load(D7 / "test_metrics_complete.json")
    ptva = load(D7 / "promoted_two_view_ablation.json")
    venn = ptva["complementarity_venn"]
    odds = ptva["complementarity_odds_ratio"]
    base_venn = tmc["complementarity_venn"]

    out = {
        "description": "Melanoma detection complementarity on Derm7pt test (n=101 melanomas), "
                       "PROMOTED perception. Base (pre-promotion) counts kept for comparison.",
        "venn_counts": venn,
        "odds_ratio": odds,
        "interpretation": (
            f"Classifier catches {venn['classifier_only']} melanomas the criteria miss; "
            f"criteria catch {venn['criteria_only']} the classifier misses. "
            f"Union sensitivity: {(venn['both'] + venn['classifier_only'] + venn['criteria_only'])/venn['melanomas']:.4f}."
        ),
        "base_arm_reference": {
            "label": "PRE-PROMOTION arms -- reference only",
            "venn_counts": base_venn,
            "odds_ratio": tmc["complementarity_odds_ratio"],
        },
    }
    save(D7 / "complementarity_venn_data.json", out)


def extract_confusion_matrices():
    print("=== 3c. Extracting confusion matrices ===")
    tmc = load(D7 / "test_metrics_complete.json")
    bcn = load(BCN / "bcn_test_evaluation.json")
    ham = load(HAM / "ham_evaluation.json")

    out = {
        "description": "Confusion matrices across all 3 datasets",
        "derm7pt_test": {
            "n": 395,
            "labels": tmc["classification_5class"]["labels"],
            "matrix": tmc["classification_5class"]["confusion_matrix"],
            "row_normalized": tmc["classification_5class"]["row_normalized_confusion_matrix"],
        },
        "bcn_test_in_domain": {
            "n": 712,
            "labels": bcn["arm1_in_domain_classifier"]["confusion_matrix_labels"],
            "matrix": bcn["arm1_in_domain_classifier"]["confusion_matrix"],
        },
        "bcn_test_cross_centre": {
            "n": 712,
            "labels": bcn["arm1_in_domain_classifier"]["confusion_matrix_labels"],
            "matrix": bcn["arm2_cross_centre_classifier"]["confusion_matrix"],
        },
        "ham_e3_5group": {
            "n": 1595,
            "labels": ham["five_group_taxonomy"]["test_metrics"]["labels"],
            "matrix": ham["five_group_taxonomy"]["test_metrics"]["confusion_matrix"],
        },
    }

    # Management confusion matrix
    mgmt = tmc.get("management", {})
    if mgmt:
        out["derm7pt_management"] = {
            "n": 395,
            "labels": mgmt["labels"],
            "matrix": mgmt["confusion_matrix"],
            "row_normalized": mgmt["row_normalized_confusion_matrix"],
        }

    save(D7 / "confusion_matrices.json", out)


def extract_case_study():
    print("=== 3d. Extracting case study data ===")
    agent = load(D7 / "agent_eval_report_v5_4.json")

    # Find a good conflict+override case
    best = None
    for row in agent["rows"]:
        if row.get("kind") == "override" and row.get("tool_calls"):
            overrides = [tc for tc in row["tool_calls"] if tc.get("tool") == "override_criterion"]
            if overrides:
                best = row
                break

    if not best:
        print("  WARNING: no suitable case study found")
        return

    # Find the corresponding 'why' and 'guideline' answers for same case
    case_id = best["case_id"]
    related = [r for r in agent["rows"] if r["case_id"] == case_id]

    empty_kinds = [r["kind"] for r in related if not (r.get("answer") or "").strip()]
    if empty_kinds:
        print(f"  WARNING: case {case_id} has empty answer text for kinds: {empty_kinds}")

    out = {
        "description": f"Case study: {case_id} — conflict, override, verdict change",
        "case_id": case_id,
        "answers": [
            {
                "kind": r["kind"],
                "question": r["question"],
                "answer": r.get("answer", ""),
                "tool_calls": [
                    {"tool": tc["tool"], "args": tc.get("args", {})}
                    for tc in r.get("tool_calls", [])
                ],
                "turns": r.get("turns", 0),
            }
            for r in related
        ],
    }
    print(f"  case study case_id: {case_id} ({len(related)} related rows, "
          f"all non-empty: {not empty_kinds})")
    save(D7 / "case_study_figure_data.json", out)


# ── 4. Comprehensive paper metrics summary ────────────────────────────

def build_paper_metrics_summary():
    print("=== 4. Building paper_metrics_summary.json ===")

    tmc = load(D7 / "test_metrics_complete.json")
    bcn = load(BCN / "bcn_test_evaluation.json")
    ham = load(HAM / "ham_evaluation.json")
    abl = load(D7 / "ablation_table.json")
    eff = load(D7 / "efficiency_analysis.json")
    agent = load(D7 / "agent_eval_report_v5_4.json")
    ragas = load(D7 / "agent_ragas_metrics_v5_4.json")
    v3meta = load(REPO / "research/evidence/v3_metadata_evaluation.json")
    ct = load(D7 / "comparison_table.json")
    # PROMOTED perception artifacts -- the system the paper describes (run_cohort(promoted=True)).
    ptva = load(D7 / "promoted_two_view_ablation.json")
    pms = load(D7 / "promoted_management_and_score.json")
    pcr = load(D7 / "promoted_contracts_routing.json")
    recipe = load(D7 / "promoted_recipe_ham_bcn.json")
    brief = load(D7 / "case_brief_measurement.json")

    ham5 = ham["five_group_taxonomy"]["test_metrics"]
    ham7 = ham["seven_class_taxonomy"]["test_metrics"]
    bcn1 = bcn["arm1_in_domain_classifier"]
    bcn2 = bcn["arm2_cross_centre_classifier"]
    cls5 = tmc["classification_5class"]
    mel = tmc["melanoma_auroc"]
    abl_a = abl["table_a_perception_ablation"]["arms"]
    mgmt = tmc["management"]
    cal = tmc["calibration"]
    parms = ptva["arms"]
    pmel = ptva["melanoma_auroc"]
    pmgmt = pms["management"]

    n_contract = ragas["n_contract_grounded"]
    n_scored = ragas["n_scored_faithfulness"]
    combined = compute_combined_faithfulness(ragas)

    summary = {
        "protocol": "paper_metrics_summary_v5",
        "description": (
            "Every reportable number for the MedGuard v2 paper, organized by section. "
            "Derm7pt perception / ablation / management / contracts sections describe the "
            "PROMOTED perception (run_cohort(promoted=True): diagnosis dermlip+metadata14 "
            "unbalanced C=100 + SK/MISC operating point; criteria dermlip_tta+metadata14 "
            "lr_balanced, argmax). Pre-promotion numbers are retained ONLY under explicitly "
            "labeled base_arm_reference keys."
        ),

        # ── Section: Datasets ──
        "datasets": {
            "derm7pt": {
                "train": 413, "valid": 203, "test": 395, "total": 1011,
                "note": "Seven-point checklist annotations available",
            },
            "bcn20000": {
                "train_lesions": 2143, "valid_lesions": 715, "test_lesions": 712,
                "test_images": 2510,
                "note": "Image-to-lesion pooling via mean DermLIP features",
            },
            "ham10000": {
                "e3_train": 6429, "e3_test": 1595,
                "sealed_test": 1991,
                "note": "E3: 80/20 lesion-level split from train+cal. Sealed: separate frozen partition.",
            },
            "total_test_cases": 4197,
            "total_test_breakdown": "Derm7pt 395 + BCN 712 + HAM-E3 1595 + HAM-sealed 1495",
        },

        # ── Section: Perception — Derm7pt test (n=395), PROMOTED system ──
        "perception_derm7pt": {
            "n": 395,
            "system": (
                "PROMOTED perception (promoted_two_view_ablation.json / "
                "promoted_management_and_score.json); pre-promotion values only under "
                "base_arm_reference"
            ),
            "classifier_5class": {
                "argmax": {
                    k: ptva["diagnosis_5class"]["argmax"][k]
                    for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                              "mcc", "cohen_kappa", "per_class")
                },
                "operating_point": {
                    "policy": ptva["diagnosis_5class"]["operating_point"]["policy"],
                    **{
                        k: ptva["diagnosis_5class"]["operating_point"][k]
                        for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                                  "mcc", "cohen_kappa", "per_class")
                    },
                },
                "base_arm_reference": {
                    "label": "PRE-PROMOTION classifier (dermlip 512-d, C=10)",
                    "accuracy": cls5["accuracy"],
                    "balanced_accuracy": cls5["balanced_accuracy"],
                    "macro_f1": cls5["macro_f1"],
                },
            },
            "melanoma_auroc": {
                "point": pmel["point_estimate"],
                "ci95_low": pmel["ci95"]["ci95_low"],
                "ci95_high": pmel["ci95"]["ci95_high"],
                "base_arm_reference": {
                    "point": mel["point_estimate"],
                    "ci95_low": mel["ci95"]["ci95_low"],
                    "ci95_high": mel["ci95"]["ci95_high"],
                },
                "note": "Promoted AUROC preserved (not traded away): >= the 0.895 base by design.",
            },
            "operating_points": ptva["clinical_operating_points"],
            "operating_point_bootstrap": {
                "classifier": parms["classifier_alone"],
                "criteria": parms["criteria_alone"],
                "union": parms["union_rule"],
            },
            "criteria_aggregate": {
                **pms["criteria_aggregate"]["promoted"],
                "base_arm_reference": pms["criteria_aggregate"]["base_arm_reference"],
            },
            "per_criterion_summary": pms["per_criterion_summary"]["promoted"],
            "per_criterion_summary_base_arm_reference": pms["per_criterion_summary"][
                "base_arm_reference"],
            "seven_point_score": {
                **{k: v for k, v in pms["seven_point_score_agreement"]["promoted"].items()},
                "base_arm_reference": pms["seven_point_score_agreement"]["base_arm_reference"],
                "honesty_note": pms["seven_point_score_agreement"]["honesty_note"],
            },
            "calibration": {
                "ece_uncalibrated": ptva["calibration"]["ece_uncalibrated"],
                "aurc": ptva["calibration"]["aurc"],
                "note": ptva["calibration"]["honesty_note"],
                "base_arm_reference": {
                    "ece_calibrated": cal["ece"],
                    "aurc": cal["aurc"],
                    "label": "PRE-PROMOTION head with temperature calibration",
                },
            },
        },

        # ── Section: Perception — BCN test (n=712 lesions) ──
        "perception_bcn": {
            "n": 712,
            "in_domain": {
                "accuracy": bcn1["accuracy"],
                "balanced_accuracy": bcn1["balanced_accuracy"],
                "macro_f1": bcn1["macro_f1"],
                "mcc": bcn1["mcc"],
                "cohen_kappa": bcn1["cohen_kappa"],
                "mel_auroc": bcn1["melanoma"]["auroc"]["estimate"],
                "mel_auroc_ci": [bcn1["melanoma"]["auroc"]["ci95_low"], bcn1["melanoma"]["auroc"]["ci95_high"]],
                "ece": bcn1["calibration"]["ece"],
                "aurc": bcn1["calibration"]["aurc"],
                "sens_at_spec90": bcn1["melanoma"]["sensitivity_at_specificity"]["0.90"]["sensitivity"],
                "sens_at_spec95": bcn1["melanoma"]["sensitivity_at_specificity"]["0.95"]["sensitivity"],
            },
            "cross_centre": {
                "accuracy": bcn2["accuracy"],
                "balanced_accuracy": bcn2["balanced_accuracy"],
                "macro_f1": bcn2["macro_f1"],
                "mcc": bcn2["mcc"],
                "mel_auroc": bcn2.get("melanoma", {}).get("auroc", {}).get("estimate"),
            },
            "transfer_delta": {
                "accuracy_drop": round(bcn1["accuracy"] - bcn2["accuracy"], 4),
                "note": "Cross-centre transfer boundary: in-domain trained BCN classifier vs Derm7pt-trained.",
            },
            "promoted_recipe_post_hoc": _promoted_recipe_block(recipe, "bcn20000_test"),
        },

        # ── Section: Perception — HAM10000 ──
        "perception_ham": {
            "e3_split": {
                "n_test": 1595,
                "five_group": {
                    "accuracy": ham5["accuracy"],
                    "balanced_accuracy": ham5["balanced_accuracy"],
                    "macro_f1": ham5["macro_f1"],
                    "mcc": ham5["mcc"],
                    "cohen_kappa": ham5["cohen_kappa"],
                    "mel_auroc": ham5["melanoma_auroc"]["point_estimate"],
                    "mel_auroc_ci": [ham5["melanoma_auroc"]["ci_low"], ham5["melanoma_auroc"]["ci_high"]],
                    "ece": ham5["calibration"]["ece"],
                    "aurc": ham5["calibration"]["aurc"],
                    "sens_at_spec90": ham5["sensitivity_at_specificity"]["0.90"]["sensitivity"],
                    "sens_at_spec95": ham5["sensitivity_at_specificity"]["0.95"]["sensitivity"],
                },
                "seven_class": {
                    "accuracy": ham7["accuracy"],
                    "balanced_accuracy": ham7["balanced_accuracy"],
                    "macro_f1": ham7["macro_f1"],
                    "mcc": ham7["mcc"],
                    "cohen_kappa": ham7["cohen_kappa"],
                },
            },
            "sealed_test": {
                "n_test": 1495,
                "note": "From comparison_table.json; evaluated in prior session. These remain the "
                        "REGISTERED one-shot confirmatory numbers.",
                "accuracy_5group": 0.8783,
                "mel_auroc": 0.9444,
                "mel_auroc_ci": [0.9263, 0.9600],
                "ece": 0.0169,
            },
            "promoted_recipe_post_hoc": {
                "e3_five_group": _promoted_recipe_block(recipe, "ham10000_e3", "five_group"),
                "e3_seven_class": _promoted_recipe_block(recipe, "ham10000_e3", "seven_class"),
                "sealed_five_group": _promoted_recipe_block(recipe, "ham10000_sealed", "five_group"),
            },
            "dermagent_comparison": ham.get("dermagent_comparison", {}),
        },

        # ── Section: Melanoma detection arms (ablation), PROMOTED system ──
        "ablation_perception": {
            "system": "PROMOTED arms (promoted_two_view_ablation.json)",
            "derm7pt_test_n395": {
                "classifier_alone": {
                    "sensitivity": parms["classifier_alone"]["sensitivity"],
                    "specificity": parms["classifier_alone"]["specificity"],
                    "youden_j": parms["classifier_alone"]["youden_j"],
                    "auroc": parms["classifier_alone"]["melanoma_auroc"],
                },
                "criteria_alone": {
                    "sensitivity": parms["criteria_alone"]["sensitivity"],
                    "specificity": parms["criteria_alone"]["specificity"],
                    "youden_j": parms["criteria_alone"]["youden_j"],
                },
                "union_rule": {
                    "sensitivity": parms["union_rule"]["sensitivity"],
                    "specificity": parms["union_rule"]["specificity"],
                    "youden_j": parms["union_rule"]["youden_j"],
                },
                "union_sensitivity_gain": ptva["union_sensitivity_gain"][
                    "gain_over_classifier"],
                "union_sensitivity_gain_ci95": [
                    ptva["union_sensitivity_gain"]["paired_bootstrap_ci95"]["ci95_low"],
                    ptva["union_sensitivity_gain"]["paired_bootstrap_ci95"]["ci95_high"],
                ],
                "union_sensitivity_delta_vs_base_union": ptva[
                    "union_sensitivity_delta_vs_base_union"],
                "stacked_ensemble_base_arm_reference": {
                    "label": "BASE-arm stacked ensemble (null result; no promoted stacking arm exists)",
                    "auroc": tmc["stacked_ensemble"]["auroc"],
                    "delong_p": tmc["stacked_ensemble"]["auroc_vs_classifier_delong"]["p_value"],
                },
            },
            "complementarity": ptva["complementarity_venn"],
            "complementarity_odds_ratio": ptva["complementarity_odds_ratio"],
            "mcnemar": ptva["mcnemar_classifier_vs_union"],
            "mcnemar_honesty_note": (
                "On BINARY accuracy the promoted union is significantly WORSE than the promoted "
                "classifier alone (the specificity cost of the sensitive checklist); the union is "
                "accepted on melanoma sensitivity, and the disagreement is surfaced for human "
                "adjudication rather than auto-acted on. In the base system this McNemar was "
                "null (16/15, p=1.0)."
            ),
            "disclosed_specificity_cost": ptva["disclosed_specificity_cost"],
            "base_arm_reference": {
                "label": "PRE-PROMOTION arms (test_metrics_complete.json / ablation_table.json)",
                "classifier_alone": {
                    "sensitivity": abl_a["classifier_alone"]["test_n395"]["sensitivity"],
                    "specificity": abl_a["classifier_alone"]["test_n395"]["specificity"],
                    "youden_j": abl_a["classifier_alone"]["test_n395"]["youden_j"],
                    "auroc": abl_a["classifier_alone"]["test_n395"]["melanoma_auroc"],
                },
                "criteria_alone": {
                    "sensitivity": abl_a["criteria_path_alone"]["test_n395"]["sensitivity"],
                    "specificity": abl_a["criteria_path_alone"]["test_n395"]["specificity"],
                    "youden_j": abl_a["criteria_path_alone"]["test_n395"]["youden_j"],
                },
                "union_rule": {
                    "sensitivity": abl_a["union_rule"]["test_n395"]["sensitivity"],
                    "specificity": abl_a["union_rule"]["test_n395"]["specificity"],
                    "youden_j": abl_a["union_rule"]["test_n395"]["youden_j"],
                },
                "union_sensitivity_gain": round(
                    abl_a["union_rule"]["test_n395"]["sensitivity"]
                    - abl_a["classifier_alone"]["test_n395"]["sensitivity"], 4
                ),
                "complementarity": tmc["complementarity_venn"],
                "complementarity_odds_ratio": tmc["complementarity_odds_ratio"],
                "mcnemar": tmc["mcnemar_classifier_vs_union"],
            },
        },

        # ── Section: Management, PROMOTED view head ──
        "management": {
            "n": 395,
            "system": pmgmt["promoted"]["registered_arm"],
            "accuracy": pmgmt["promoted"]["accuracy"],
            "balanced_accuracy": pmgmt["promoted"]["balanced_accuracy"],
            "macro_f1": pmgmt["promoted"]["macro_f1"],
            "qwk": pmgmt["promoted"]["quadratic_weighted_kappa"],
            "qwk_ci": [
                pmgmt["promoted"]["quadratic_weighted_kappa_ci95"]["ci95_low"],
                pmgmt["promoted"]["quadratic_weighted_kappa_ci95"]["ci95_high"],
            ],
            "labels": pmgmt["promoted"]["labels"],
            "per_class": pmgmt["promoted"]["per_class"],
            "honesty_note": (
                "Promoted-view management improves accuracy/balanced-accuracy/macro-F1 but its "
                "ordinal QWK (0.544) is slightly BELOW the base arm's 0.5735; the promoted QWK "
                "CI covers the base point estimate, so the dip is not resolvable at n=395. "
                "Dual-reported, not hidden."
            ),
            "base_arm_reference": {
                "label": "PRE-PROMOTION registered P2 arm (raw dermlip 512-d, C=100)",
                "accuracy": mgmt["accuracy"],
                "balanced_accuracy": mgmt["balanced_accuracy"],
                "macro_f1": mgmt["macro_f1"],
                "qwk": mgmt["quadratic_weighted_kappa"],
                "qwk_ci": [
                    mgmt["quadratic_weighted_kappa_ci95"]["ci95_low"],
                    mgmt["quadratic_weighted_kappa_ci95"]["ci95_high"],
                ],
            },
            "checklist_implied_secondary": {
                "accuracy": pmgmt["checklist_implied_secondary"]["accuracy"],
                "qwk": pmgmt["checklist_implied_secondary"]["quadratic_weighted_kappa"],
                "note": pmgmt["checklist_implied_secondary"]["structural_note"],
            },
        },

        # ── Section: Contract & Router, PROMOTED system (valid n=203) ──
        "contracts_and_routing": {
            "system": (
                "PROMOTED contracts (cdss_contracts_valid_promoted.json, valid n=203, recomputed "
                "+ cross-checked in promoted_contracts_routing.json). The base references "
                "include the pre-promotion TEST-split rates; split denominators differ and are "
                "stated per entry."
            ),
            "contract_safe_rate_derm7pt": pcr["decision_behaviour"]["safe_to_display_rate"],
            "contract_n_derm7pt": pcr["n_cases"],
            "contract_safe_rate_sealed": 1.0,
            "contract_n_sealed": 1495,
            "conflict_rate": pcr["decision_behaviour"]["conflict_rate"],
            "escalation_rate": pcr["decision_behaviour"]["escalation_rate"],
            "undecidable_rate": pcr["decision_behaviour"]["undecidable_rate"],
            "undecidable_note": pcr["abstention"]["note"],
            "citation_rate": pcr["guideline_citation"]["citation_rate"],
            "router_state_distribution": pcr["router"]["state_distribution"],
            "retrieval_rate": pcr["router"]["case_level_retrieval_rate"],
            "retrieval_avoided": pcr["router"]["case_level_retrieval_avoided"],
            "retrieval_savings_call_level": pcr["router"]["call_level_hypothetical_savings"],
            "retrieval_note": (
                "Router classification, not measured gating: binding is a free local "
                "lookup that ran for every case. " + pcr["router"]["call_vs_case_note"]
            ),
            "criterion_abstention_rate": pcr["abstention"]["criterion_abstention_rate"],
            "criterion_abstention_by_construction": True,
            "base_arm_reference": pcr["base_arm_reference"],
        },

        # ── Section: Agent evaluation ──
        "agent_eval": {
            "prompt_version": agent.get("prompt_version", "unknown"),
            "model": agent["model"],
            "n_cases": agent["n_cases"],
            "n_questions": agent["n_questions"],
            "endpoints": agent["endpoints"],
            "verification_loop": {
                "first_draft_failure_rate": agent.get("verification", {}).get("first_draft_failure_rate"),
                "revision_reasons": agent.get("verification", {}).get("revision_reasons"),
                "retrieval_gating": {
                    "served_total": agent.get("retrieval_gating", {}).get("served_total"),
                    "skipped_total": agent.get("retrieval_gating", {}).get("skipped_total"),
                },
            },
            # Generation under deterministic verification, measured on the real model
            # (case_brief_measurement.json): the same host gate that audits agent answers
            # gates one-shot generated briefs; unverified prose is never displayed.
            "case_briefs": {
                "n": brief["n"],
                "verified_rate": brief["verified_rate"],
                "verified_first_try_rate": brief["verified_first_try_rate"],
                "verified_after_revision_rate": brief["verified_after_revision_rate"],
                "fallback_rate": brief["fallback_rate"],
                "note": brief["guarantee_note"],
            },
        },

        # ── Section: RAGAS faithfulness ──
        "ragas": {
            "prompt_version": agent.get("prompt_version", "unknown"),
            "judge_model": ragas["judge_model"],
            "overall_faithfulness": ragas["faithfulness"],
            "answer_relevancy": ragas["answer_relevancy"],
            "combined_faithfulness": round(combined["combined_faithfulness"], 4),
            "combined_n": combined["combined_n"],
            "combined_note": (
                "Deduplicated so every non-out-of-scope answer counts once: "
                f"{combined['n_ragas_scored']} RAGAS-scored answers "
                f"({combined['n_ragas_scored_and_contract']} of those also "
                "contract-grounded, counted only via their RAGAS score, not twice) "
                f"plus {combined['n_contract_only_remaining']} additional "
                "contract-only answers at 1.0."
            ),
            "contract_grounded_n": n_contract,
            "guideline_scored_n": n_scored,
            "by_grounding": ragas["faithfulness_by_grounding"],
            "scope_distribution": ragas["scope_distribution"],
            "assessment_scoped_mean": ragas["faithfulness_detail"]["assessment_mean"],
            "judge_noise_floor": 0.0489,
            "v2_comparison": {
                "v2_faithfulness": 0.6781,
                "current_faithfulness": ragas["faithfulness"],
                "v2_guideline": 0.8121,
                "current_guideline": ragas["faithfulness_by_grounding"]["guideline"]["mean"],
                "v2_mixed": 0.0972,
                "current_mixed": ragas["faithfulness_by_grounding"]["mixed"]["mean"],
            },
        },

        # ── Section: Metadata augmentation (v2 vs v3) ──
        "metadata_augmentation": {
            "summary": v3meta["summary"],
            "note": "v2 = visual-only (512-d DermLIP), v3 = visual + clinical metadata. Delta small on BCN/HAM, +1.8% on Derm7pt.",
        },

        # ── Section: Efficiency ──
        "efficiency": {
            "feature_model": "DermLIP ViT-B/16, frozen, 512-d",
            "classifier": "LogisticRegression, 5-class",
            "criterion_heads": "7× LogisticRegression",
            "llm_calls_per_case": 0,
            # Bug fix: this used to be assigned calls_per_case (paid calls / n_cases, 4.98 for
            # v5.2) -- a per-CASE figure mislabelled as per-QUESTION. The honest per-question
            # figure is (paid calls + cache hits) / n_questions from the same
            # agent_evaluation_cost block (~2.67 for v5.2): cache hits are free replays but still
            # real turns/tool-calls in the conversation, so they belong in a "calls per question"
            # count same as build_efficiency_analysis.py's own dermagent_comparison figure does.
            "llm_calls_per_question": (
                round(
                    (eff["agent_evaluation_cost"]["paid_calls"]
                     + eff["agent_evaluation_cost"]["cache_hits"])
                    / eff["agent_evaluation_cost"]["n_questions"],
                    2,
                )
                if eff["agent_evaluation_cost"]["n_questions"]
                else None
            ),
            "hypothetical_retrieval_savings": eff["router_savings"]["retrieval_avoided"],
            "hypothetical_retrieval_savings_call_level": eff["router_savings"][
                "call_level_hypothetical_savings"],
            "retrieval_savings_note": (
                "Router classification only (PROMOTED contracts, valid n=203); binding is a "
                "free local lookup that ran for every case. The saving is what gating would "
                "avoid where retrieval is remote or paid. Base arm avoided 0.70; the promoted "
                "checklist disagrees with the classifier more often, so it retrieves more."
            ),
            "agent_model": "qwen/qwen3-vl-32b-instruct",
        },

        # ── Section: Head-to-head vs DermAgent ──
        "dermagent_comparison": {
            "our_ham_7class_acc": ham7["accuracy"],
            "our_ham_7class_n": 1595,
            "dermagent_ham_acc": 0.6183,
            "dermagent_ham_n": 642,
            "accuracy_gap": round(ham7["accuracy"] - 0.6183, 4),
            "our_criterion_f1_binary": pms["criteria_aggregate"]["promoted"][
                "binary_concept_f1_macro"],
            "our_criterion_f1_multiclass": pms["criteria_aggregate"]["promoted"][
                "f1_macro_across_criteria"],
            "our_criterion_f1_binary_base_arm_reference": tmc["criteria_aggregate"][
                "binary_concept_f1_macro"],
            "our_criterion_n": 395,
            "dermagent_criterion_f1": 0.6506,
            "dermagent_criterion_n": 100,
            "our_total_test_cases": 4197,
            "dermagent_total_test_cases": 1542,
            "our_citation_resolvability": 1.0,
            "dermagent_citation_resolvability": None,
            "our_faithfulness": ragas["faithfulness"],
            "dermagent_faithfulness": None,
        },

        # ── Section: Negative results ──
        "negative_results": {
            "g7_cross_dataset_transfer": {
                "description": "Derm7pt-trained classifier on HAM valid: AUROC drops",
                "derm7pt_auroc_base_arm": mel["point_estimate"],
                "note": "Cross-dataset transfer boundary (contribution, not limitation). The "
                        "transfer experiment used the BASE (pre-promotion) probe (AUROC 0.895 "
                        "in-domain); the promoted in-domain probe is 0.9151 (see "
                        "perception_derm7pt).",
            },
            "g8_criteria_cross_centre": {
                "description": "Derm7pt criteria heads on BCN: complementarity collapses",
                "in_domain_complementarity": None,
                "cross_centre_complementarity": 0.019,
            },
            "stacked_ensemble_null": {
                "description": "Stacked ensemble does not beat classifier (DeLong p=0.5935)",
                "delong_p": tmc["stacked_ensemble"]["auroc_vs_classifier_delong"]["p_value"],
            },
        },

        # ── Section: What we cannot claim ──
        "limitations": [
            "No human evaluation (acknowledged; listed as future work)",
            "Stacked ensemble does not beat classifier (p=0.5935)",
            "Cross-centre generalizability limited (3 negatives)",
            "Criteria path only available for Derm7pt (annotation bottleneck: 413 cases is world supply)",
            "RAGAS judge is nondeterministic at temp 0 (noise floor 0.0489, demonstrated FP at p=0.023)",
        ],
    }

    save(D7 / "paper_metrics_summary.json", summary)
    return summary


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Finalizing all paper metrics...\n")
    update_comparison_table()
    print()
    update_efficiency_analysis()
    print()
    extract_roc_curve_data()
    extract_complementarity_venn()
    extract_confusion_matrices()
    extract_case_study()
    print()
    summary = build_paper_metrics_summary()

    # Quick sanity print
    print("\n=== Key numbers for abstract (PROMOTED perception) ===")
    s = summary
    print(f"  Total test cases: {s['datasets']['total_test_cases']}")
    c5 = s["perception_derm7pt"]["classifier_5class"]
    print(f"  Derm7pt accuracy: argmax {c5['argmax']['accuracy']} / "
          f"op {c5['operating_point']['accuracy']} (base {c5['base_arm_reference']['accuracy']})")
    print(f"  Derm7pt MEL AUROC: {s['perception_derm7pt']['melanoma_auroc']['point']} "
          f"[{s['perception_derm7pt']['melanoma_auroc']['ci95_low']:.3f}, "
          f"{s['perception_derm7pt']['melanoma_auroc']['ci95_high']:.3f}] (base 0.895)")
    print(f"  HAM 7-class accuracy: {s['perception_ham']['e3_split']['seven_class']['accuracy']} (vs DermAgent 0.6183)")
    print(f"  BCN in-domain accuracy: {s['perception_bcn']['in_domain']['accuracy']}")
    ab = s["ablation_perception"]["derm7pt_test_n395"]
    print(f"  Union sens/spec: {ab['union_rule']['sensitivity']}/{ab['union_rule']['specificity']}; "
          f"gain over classifier +{ab['union_sensitivity_gain']} "
          f"CI {ab['union_sensitivity_gain_ci95']}")
    print(f"  Criteria binary F1: {s['perception_derm7pt']['criteria_aggregate']['binary_concept_f1_macro']} "
          f"(base {s['perception_derm7pt']['criteria_aggregate']['base_arm_reference']['binary_concept_f1_macro']})")
    print(f"  Management QWK: {s['management']['qwk']} [{s['management']['qwk_ci'][0]:.3f}, "
          f"{s['management']['qwk_ci'][1]:.3f}] (base {s['management']['base_arm_reference']['qwk']})")
    print(f"  Conflict rate (promoted valid): {s['contracts_and_routing']['conflict_rate']}")
    print(f"  Retrieval avoided (promoted): {s['contracts_and_routing']['retrieval_avoided']}")
    print(f"  Citation resolvability: {s['dermagent_comparison']['our_citation_resolvability']}")
    print(f"  Faithfulness (RAGAS): {s['ragas']['overall_faithfulness']}")
    print(f"  Faithfulness (combined): {s['ragas']['combined_faithfulness']}")
    print(f"  Answer relevancy: {s['ragas']['answer_relevancy']}")
    print(f"  Counterfactual: {s['agent_eval']['endpoints']['counterfactual_accuracy']}")
    print(f"  Contracts safe: {s['contracts_and_routing']['contract_safe_rate_derm7pt']}")
    print(f"  ECE (Derm7pt, promoted, uncalibrated): "
          f"{s['perception_derm7pt']['calibration']['ece_uncalibrated']}")
    print(f"  ECE (HAM sealed): {s['perception_ham']['sealed_test']['ece']}")
    print(f"  Prompt version: {s['agent_eval']['prompt_version']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
