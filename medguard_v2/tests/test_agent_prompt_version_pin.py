"""Guard: the clinician-agent module may not change silently under a stable PROMPT_VERSION.

Why this exists (2026-08-16, decision (b) in agent_eval_report_v5_4.PROVENANCE.md): after the v5.4 seal the
agent module changed (step 9 in explain_reasoning, 7f94d9b9) while PROMPT_VERSION stayed "clinician-agent-v5.4",
so 54 sealed "why" answers stopped being byte-replayable from HEAD and nothing caught it -- the eval cache keys on
PROMPT_VERSION, which did not move. This test pins the module's sha256 next to PROMPT_VERSION so the next edit is a
deliberate act: either bump PROMPT_VERSION (and update BOTH pins here, and re-run the eval), or, if the edit is
provably behaviour-neutral for the eval path, update the hash pin alone and say why in the commit message.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_MODULE = ROOT / "medguard_v2" / "agent" / "clinician_agent.py"
SEALED_REPORT = ROOT / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_v5_4.json"
PROVENANCE = ROOT / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_v5_4.PROVENANCE.md"

PINNED_PROMPT_VERSION = "clinician-agent-v5.4"
# sha256 of medguard_v2/agent/clinician_agent.py at the pin (HEAD 46d51ab3, 2026-08-16).
PINNED_AGENT_MODULE_SHA256 = "96b961ef50917cd617b556dea6b6ab641622d9ca5b8c6476d404fee02e0a6252"
PINNED_SEAL_COMMIT = "3aeacead"


class AgentPromptVersionPin(unittest.TestCase):
    def test_prompt_version_matches_the_sealed_report(self):
        from medguard_v2.agent.clinician_agent import PROMPT_VERSION
        self.assertEqual(PROMPT_VERSION, PINNED_PROMPT_VERSION)
        report = json.loads(SEALED_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["prompt_version"], PINNED_PROMPT_VERSION)

    def test_agent_module_is_pinned_under_this_prompt_version(self):
        digest = hashlib.sha256(AGENT_MODULE.read_bytes()).hexdigest()
        self.assertEqual(
            digest, PINNED_AGENT_MODULE_SHA256,
            "medguard_v2/agent/clinician_agent.py changed while PROMPT_VERSION is still "
            f"{PINNED_PROMPT_VERSION!r}. Behavioural change => bump PROMPT_VERSION, update both pins in this "
            "test and re-run the agent eval; behaviour-neutral change => update PINNED_AGENT_MODULE_SHA256 "
            "and justify it in the commit. See agent_eval_report_v5_4.PROVENANCE.md.",
        )

    def test_provenance_sidecar_pins_the_seal_commit(self):
        text = PROVENANCE.read_text(encoding="utf-8")
        self.assertIn(PINNED_SEAL_COMMIT, text)
        self.assertIn(PINNED_PROMPT_VERSION, text)
        self.assertIn(PINNED_AGENT_MODULE_SHA256, text)


if __name__ == "__main__":
    unittest.main()
