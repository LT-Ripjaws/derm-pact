"""Prospective task/reference contract. No candidate answers are inputs."""
from dermpact.core.agent.redesign.evidence import probability_percent, comparison_counts

VERSION = "dermpact-read-contract-v6"
QUESTIONS = {
    "evidence_report": "For this lesion, report the model's predicted diagnosis, the separate melanoma estimate as a percentage with two decimal places, and the two strongest predicted dermoscopic findings. Distinguish all these model outputs from confirmed clinical observations.",
    "reliability_explanation": "Was this lesion selectively referred for clinician review? Explain the recorded referral triggers, or state that none are recorded. Identify any active unfamiliarity advisory and distinguish its role; if none is active, say so.",
    "referral_and_comparison": "Was this lesion selectively referred for clinician review, and what recorded trigger explains that status? Report standard similarity matches and any eligible contrasting example, noting overlap if both exist. If historical examples are available, state their label provenance and limitations. Can these examples change the stored referral decision?",
}


def reference(snapshot, family):
    p, r, c = (snapshot[k] for k in ("perception", "reliability", "comparison"))
    if family == "evidence_report":
        top = sorted(p["criterion_positive_probabilities"], key=lambda k: (-p["criterion_positive_probabilities"][k], k))[:2]
        return (f"For this lesion, the model predicts {p['diagnosis'].replace('_', ' ')}. "
                f"The separate melanoma estimate is {probability_percent(p['melanoma_probability'])}%. "
                + " ".join(f"{name.replace('_', ' ')} is one of the two strongest predicted dermoscopic findings." for name in top)
                + " The diagnosis, melanoma estimate and findings are model outputs, not confirmed clinical observations.")
    if r["disposition"] == "review":
        parts = ["This lesion was selectively referred for clinician review."]
        parts.extend("A recorded reason for that referral was: " + w["reason"].rstrip(".") + "." for w in r["warnings"])
    elif r["disposition"] == "display":
        parts = ["This lesion was not selectively referred for clinician review."]
        if not r["warnings"]:
            parts.append("No referral trigger was recorded.")
    else:
        parts = ["The selective-referral status is unknown."]
    if family == "reliability_explanation":
        parts.append("An unfamiliarity advisory is active; it provides information only and cannot trigger referral." if r["ood_state"] == "flagged" else
                     "No unfamiliarity advisory is active." if r["ood_state"] == "not_flagged" else
                     "The unfamiliarity-advisory status is unknown.")
    elif family == "referral_and_comparison":
        counts = comparison_counts(c)
        n = counts["standard_matches"]
        parts.append(f"There {'is 1 standard similarity match' if n == 1 else f'are {n} standard similarity matches'}.")
        parts.append("An eligible contrasting example is available." if counts["contrasting_available"] else "No eligible contrasting example is available.")
        if n and counts["contrasting_available"]:
            parts.append("That contrasting example is already among the standard matches." if counts["contrasting_overlaps_standard"]
                         else "That contrasting example is distinct from the standard matches.")
        if counts["distinct_examples"]:
            parts.append("Historical labels are dataset references, not independent clinical confirmation of this lesion's diagnosis.")
        parts.append("Historical comparisons provide context only and cannot change the stored referral decision through this read-only workflow.")
    else:
        raise ValueError("unknown task family")
    return " ".join(parts)
