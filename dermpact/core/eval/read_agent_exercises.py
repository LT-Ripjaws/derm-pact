"""Drive adversarial engineering fixtures through the complete new read graph."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dermpact.core.agent.redesign.evidence import EvidenceError
from dermpact.core.agent.redesign.provider import GenerationError, JsonReply
from dermpact.core.agent.redesign.verification import SemanticVerifier
from dermpact.core.agent.redesign.workflow import ReadAgent


class FixtureSource:
    def __init__(self, *, stale=False, wrong_receipt=False):
        self.version = 1
        self.stale = stale
        self.wrong_receipt = wrong_receipt
        self.reads = 0

    def identity(self):
        return {"case_id": "synthetic-pilot-case", "record_version": self.version,
                "record_sha256": "fixture-record-" + str(self.version), "source_label": "synthetic_workflow_fixture"}

    def tools(self):
        return {"read_reliability": "read reliability", "retrieve_case_comparisons": "read comparisons"}

    def read(self, tool, args):
        self.reads += 1
        if tool not in self.tools():
            raise EvidenceError("tool_not_permitted")
        if args.get("case_id", self.identity()["case_id"]) != self.identity()["case_id"]:
            raise EvidenceError("wrong_case_or_version_requested")
        receipt = {**self.identity(), "tool": tool, "ok": True, "error": None,
                   "data": {"disposition": "review", "trigger": "member_uncertainty", "risk": 0.92,
                            "threshold": 0.88, "ood_advisory": True, "comparison_changes_prediction": False}}
        if self.wrong_receipt:
            receipt["case_id"] = "wrong-case"
        if self.stale:
            self.version += 1
        return receipt


class Commands:
    def __init__(self, commands):
        self.commands = copy.deepcopy(commands)
        self.index = 0

    def complete(self, system, messages):
        if self.index >= len(self.commands):
            raise GenerationError("script_exhausted")
        value = self.commands[self.index]
        self.index += 1
        if isinstance(value, str):
            raise GenerationError(value)
        return JsonReply(value, {"observed_model": "scripted_engineering_fixture"})


class Verdicts:
    def __init__(self, values=None):
        self.values = list(values or [True])
        self.index = 0

    def verify(self, question, draft, receipts):
        value = self.values[min(self.index, len(self.values)-1)]
        self.index += 1
        return {"ok": value, "issues": [] if value else ["unsupported_field_attribution"],
                "qualification": "scripted_semantic_outcome"}


READ = {"kind": "read", "tool": "read_reliability", "args": {}}
ANSWER = {"kind": "answer", "statements": [{"text": "The case was referred for recorded uncertainty.",
    "references": [{"fact_id": "e1.f6"}]}]}


def run_exercises():
    wrong_value = copy.deepcopy(ANSWER)
    wrong_value["statements"][0]["references"][0]["fact_id"] = "e1.f9999"
    invented = copy.deepcopy(ANSWER)
    invented["statements"][0]["references"][0]["fact_id"] = "e999.f1"
    no_claims = {"kind": "answer", "statements": []}
    scenarios = [
        ("valid_read", FixtureSource(), [READ, ANSWER], Verdicts(), "answered", None),
        ("unknown_fact_repaired", FixtureSource(), [READ, wrong_value, ANSWER], Verdicts(), "answered_after_repair", None),
        ("unsupported_semantics_repaired", FixtureSource(), [READ, ANSWER, ANSWER], Verdicts([False, True]), "answered_after_repair", None),
        ("invented_source_not_repaired", FixtureSource(), [READ, invented, invented], Verdicts(), "safe_fallback", None),
        ("unsupported_semantics_not_repaired", FixtureSource(), [READ, ANSWER, ANSWER], Verdicts([False, False]), "safe_fallback", None),
        ("stale_record", FixtureSource(stale=True), [READ], Verdicts(), "failed", "record_changed_during_turn"),
        ("wrong_receipt_case", FixtureSource(wrong_receipt=True), [READ], Verdicts(), "failed", "receipt_binding_mismatch"),
        ("write_tool_denied", FixtureSource(), [{"kind": "read", "tool": "commit_amendment", "args": {}}], Verdicts(), "failed", "tool_not_permitted"),
        ("wrong_requested_case", FixtureSource(), [{**READ, "args": {"case_id": "other-case"}}], Verdicts(), "failed", "wrong_case_or_version_requested"),
        ("duplicate_tool", FixtureSource(), [READ, READ], Verdicts(), "failed", "duplicate_read"),
        ("provider_failure", FixtureSource(), ["gateway_http_403"], Verdicts(), "failed", "gateway_http_403"),
        ("claimless_answer", FixtureSource(), [READ, no_claims], Verdicts(), "failed", "invalid_generation_schema"),
        ("answer_without_evidence", FixtureSource(), [ANSWER, ANSWER], Verdicts(), "safe_fallback", None),
    ]
    malformed_verdict = {"claims": [], "addresses_question": True, "missing_question_parts": [], "scope_violations": []}
    scenarios.append(("missing_semantic_verdict", FixtureSource(), [READ, ANSWER],
                      SemanticVerifier(Commands([malformed_verdict])), "failed", "invalid_semantic_verdict"))
    results = []
    for name, source, commands, verifier, expected, expected_error in scenarios:
        result = ReadAgent(source, Commands(commands), verifier).run("Why was this case referred?")
        assert result["outcome"] == expected, (name, result["outcome"], expected)
        assert result["failure"] == expected_error, (name, result["failure"], expected_error)
        if name == "write_tool_denied":
            assert source.reads == 0, "forbidden tool reached dispatcher"
        if result["outcome"] == "safe_fallback":
            assert "verified explanation could not be produced" in result["final_text"]
        results.append({"scenario": name, "outcome": result["outcome"], "failure": result["failure"],
                        "repairs": result["repairs"], "trace": result["trace"]})
        print(name, result["outcome"])
    return results


if __name__ == "__main__":
    results = run_exercises()
    out = Path("research/experiments/dermpact_agent_redesign/engineering_exercises_v1.json")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scope": "scripted synthetic engineering exercises, not model performance",
                               "results": results}, indent=2) + "\n", encoding="utf-8")
