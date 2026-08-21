"""Diagnose why A2 (graph) and A3 (remediation) look identical to M.

Reads the offline ablation report and checks, per lesion:
- M vs A2 evidence_document_ids  -> does the graph change WHICH records enter?
- M vs A2 guideline_citations     -> does it change WHICH guidelines are cited?
- M remediation trajectory        -> does remediation ever fire at all?
"""
import json
from pathlib import Path

REPORT = Path(__file__).parent / "offline_ablation.json"
r = json.loads(REPORT.read_text(encoding="utf-8"))

ev_diff = cite_diff = 0
remediation_fired = 0
remediation_steps_total = 0
a2_ev_diff_lesions = []
for case in r["cases"]:
    arms = case["arms"]
    m = arms.get("M", {})
    a2 = arms.get("A2", {})
    m_ev = m.get("evidence_document_ids", [])
    a2_ev = a2.get("evidence_document_ids", [])
    m_cite = sorted(m.get("guideline_citations", []))
    a2_cite = sorted(a2.get("guideline_citations", []))
    if m_ev != a2_ev:
        ev_diff += 1
        a2_ev_diff_lesions.append(case["cluster_id"])
    if m_cite != a2_cite:
        cite_diff += 1
    traj = m.get("trajectory", [])
    if traj:
        remediation_fired += 1
        remediation_steps_total += len(traj)

n = len(r["cases"])
print(f"lesions: {n}")
print(f"\nA2 (graph off) vs M:")
print(f"  evidence_document_ids differ on {ev_diff}/{n} lesions")
print(f"  guideline_citations differ on   {cite_diff}/{n} lesions")
if a2_ev_diff_lesions:
    print(f"  examples: {a2_ev_diff_lesions[:8]}")

print(f"\nM remediation:")
print(f"  lesions where a remediation step fired: {remediation_fired}/{n}")
print(f"  total remediation steps across all lesions: {remediation_steps_total}")

# Also: what do M's remediation trajectories look like on a couple lesions?
shown = 0
for case in r["cases"]:
    traj = case["arms"].get("M", {}).get("trajectory", [])
    rem = case["arms"].get("M", {}).get("remediation_applied", [])
    if traj or rem:
        print(f"  {case['cluster_id']}: trajectory={traj} remediation_applied={rem}")
        shown += 1
        if shown >= 5:
            break
if shown == 0:
    print("  (no remediation trajectory on any lesion — the loop is vacuous on this dev set)")

# A1 sanity: is the host contract even built (citations present) though it abstains?
a1 = r["cases"][0]["arms"].get("A1", {})
print(f"\nA1 first lesion: final_action={a1.get('final_action')} "
      f"n_evidence={len(a1.get('evidence_document_ids', []))} "
      f"first_pass_failures={a1.get('verification_first_pass_failures')}")
