"""Source-derived provenance completion; same questions as v6, no answer inputs."""
from .read_contract_v6 import QUESTIONS, reference as prior_reference

VERSION = "dermpact-read-contract-v7-full-provenance"


def reference(snapshot, family):
    text = prior_reference(snapshot, family)
    if family != "referral_and_comparison":
        return text
    comparison = snapshot["comparison"]
    examples = comparison["matches"] + ([comparison["counterexample"]] if comparison.get("counterexample") else [])
    sources = sorted({e["outcome"]["source"].rstrip(".") for e in examples})
    if sources:
        text += " Historical label provenance: " + "; ".join(sources) + ". These historical labels describe other cases."
    return text
