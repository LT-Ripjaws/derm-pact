"""Build the CALIBER-style ablation matrix table from the dev matrix run.

Reads dev_ablation_matrix.json (audit cases -> coverage/actions) and
.scored.json (per-arm endpoints + RAGAS sample_rows), computes copy_reference_ratio
per arm from the RAGAS sample rows, and prints the ablation table + a markdown block.
"""
import json, sys
from pathlib import Path
PROJECT_ROOT = Path("E:/Studies/Research/MedGuard-RAG-Journal")
sys.path.insert(0, str(PROJECT_ROOT))
from medguard.eval.metrics import copy_reference_ratio

SP = Path(__file__).parent
report = json.loads((SP / "dev_ablation_matrix.json").read_text(encoding="utf-8"))
scored = json.loads((SP / "dev_ablation_matrix.scored.json").read_text(encoding="utf-8"))
per_arm = scored["per_arm"]
arms = report["arms"]

# coverage / actions per arm from the audit cases
from collections import Counter
actions = {a: Counter() for a in arms}
gen_errors = {a: 0 for a in arms}
for case in report["cases"]:
    for a in arms:
        row = case["arms"].get(a, {})
        fa = row.get("final_action")
        actions[a][fa] += 1
        if row.get("generation_error"):
            gen_errors[a] += 1


def copy_ratio_for(arm):
    rows = per_arm.get(arm, {}).get("ragas_supporting", {}).get("sample_rows", []) or []
    ratios = []
    for r in rows:
        ans = r.get("response", "")
        ctx = [{"text": c} for c in r.get("retrieved_contexts", [])]
        ratios.append(copy_reference_ratio(ans, ctx))
    return round(sum(ratios) / len(ratios), 4) if ratios else None


def g(arm, *path, default=None):
    d = per_arm.get(arm, {})
    for k in path:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d not in ({}, None) else default


print(f"\n{'arm':>4} {'PASS':>7} {'ans-acc':>7} {'complete':>8} {'cite':>10} "
      f"{'faith':>6} {'copy':>6} {'genErr':>6} {'ttok':>9}")
print("=" * 78)
rows_md = []
for a in arms:
    blk = per_arm.get(a, {})
    n = sum(actions[a].values())
    passed = actions[a].get("PASS", 0) + actions[a].get("ANSWER_UNGATED", 0)
    prov = blk.get("provenance_citation_integrity", {}) or {}
    proc = blk.get("process_contract_completion", {}) or {}
    sel = blk.get("selective", {}) or {}
    rag = blk.get("ragas_supporting", {}) or {}
    faith = None
    sc = rag.get("scores")
    if isinstance(sc, dict) and isinstance(sc.get("faithfulness"), dict):
        faith = sc["faithfulness"].get("mean")
    cite = prov.get("integrity_rate")
    cite_cov = prov.get("cases_with_citations")
    complete = proc.get("contract_complete_rate") if proc.get("applicable") else None
    ansacc = sel.get("answered_accuracy")
    copyr = copy_ratio_for(a)
    ttok = blk.get("efficiency", {}).get("total_tokens")
    def f(v, w=6, p=3):
        return (f"{v:.{p}f}" if isinstance(v, (int, float)) else str(v)).rjust(w)
    print(f"{a:>4} {str(passed)+'/'+str(n):>7} {f(ansacc,7)} {f(complete,8)} "
          f"{str(cite)+'@'+str(cite_cov):>10} {f(faith)} {f(copyr)} {gen_errors[a]:>6} {str(ttok):>9}")
    rows_md.append((a, f"{passed}/{n}", ansacc, complete, cite, cite_cov, faith, copyr, ttok))

# paired diffs (M vs B0, M vs A2)
print("\nPaired diagnostic-accuracy differences:")
for pd in scored.get("paired_differences", []):
    print(f"  {pd['arm_a']} - {pd['arm_b']}: est={pd.get('estimate')} "
          f"[{pd.get('low')}, {pd.get('high')}] n={pd.get('n_paired')}")

print("\nactions per arm:", {a: dict(actions[a]) for a in arms})
