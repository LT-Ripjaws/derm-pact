# Provenance pin — `agent_eval_report_v5_4.json` (clinician-agent-v5.4) — decision (b), 2026-08-16

**Record of truth.** `agent_eval_report_v5_4.json` (203 cases / 591 questions, prompt version `clinician-agent-v5.4`)
was produced at commit **`3aeacead2bbb26aaf28e3d0934f18081ebe29e2a`** (2026-08-14 20:03:12 +06, "fix(v2): reconcile
fabrication metric with in-loop citation check; v5.4 promoted eval"); the sealed message cache
`research/evidence/derm7pt_v1/agent_cache/` (7,886 files, git-tracked) was written the same evening (newest mtime
2026-08-14 19:51). These numbers are **the paper's agent results and are not to be regenerated.**

## Why this pin exists
After the seal, `medguard_v2/agent/clinician_agent.py` changed in three commits — `7f94d9b9` (2026-08-14 23:45,
adds "step 9", the on-demand case-based explanation, to `explain_reasoning()`), `c4f2a285` (dense-RAG control arm +
walled-off `literature_lookup`), `abd7a998` (override-as-annotation engine) — while `PROMPT_VERSION` stayed
`clinician-agent-v5.4` (set in `cac3f740`, before the seal). The eval cache keys on `PROMPT_VERSION`, so this drift is
invisible to the cache. Empirical impact (`agent_efficiency_v1.json → part2_offline_accounting`, replayed read-only from
`agent_cache/`): **537 / 591 questions replay byte-exact from HEAD; the 54 "why" questions do not** — their second-turn
tool output (`explain_reasoning`) now carries step 9, so their sealed answers correspond to the pre-step-9 tool output.
No other endpoint (counterfactual n=185, refusal n=203, absence n=100, citations, briefs) is affected.

## Decision (user, 2026-08-16): option (b) — keep v5.4 as the record, pin the commit
Not re-run (option a), not made opt-in (option c). Consequences and obligations:
1. **Manuscript reproducibility statement** must say: *"Agent results (v5.4) were produced at commit 3aeacead; the
   case-based-explanation step added afterwards (7f94d9b9) changes only the tool output for the 54 'why' questions,
   whose sealed answers therefore reflect the earlier tool output; exact replay requires that commit."*
2. **Replay instructions** (bit-exact for all 591): check out the pinned commit in a separate worktree — e.g.
   `git worktree add ../medguard-v54 3aeacead` — run under `venv_win`, and replay against the tracked `agent_cache/`
   (`scripts/dev/run_agent_eval.py --cohort report --promoted --limit 203`; all calls are cache hits, zero paid).
   ⚠ Before removing any worktree, scan for junctions/reparse points (memory `derm7pt-dataset-backup-and-worktree-incident`).
3. **Guard against recurrence:** `medguard_v2/tests/test_agent_prompt_version_pin.py` pins the sha256 of
   `medguard_v2/agent/clinician_agent.py` at HEAD together with `PROMPT_VERSION`; any future edit to the agent module
   fails the suite until `PROMPT_VERSION` is bumped (or the edit is consciously re-pinned as behaviour-neutral).
4. Related agent-dependent artifacts and the commits that produced them: `agent_ragas_metrics_v5_4.json` → `7bebc471`;
   `trajectory_trust_eval.json` → `7f94d9b9` (run WITH step 9 — consistent with its own code state);
   `case_brief_measurement.json` → `bdb848ff`. Each is the record for its own commit; none is regenerated.

## Hashes at pin time (HEAD `46d51ab3`, 2026-08-16)
- `medguard_v2/agent/clinician_agent.py` sha256 `96b961ef50917cd617b556dea6b6ab641622d9ca5b8c6476d404fee02e0a6252`
- `agent_cache/` manifest digest (sorted `size path` list, sha256[:16]) `0cbd59ca2300c0db` — 7,886 files
