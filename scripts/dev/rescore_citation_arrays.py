"""Re-score stored agent-eval answers against the two citation fixes, at zero API cost.

Two questions, both answered from already-stored rows plus a fresh catalog load -- no agent calls.

1. ARRAY-RESOLUTION AUDIT (does citation_resolvability 1.0 survive the stricter check?).
   The red-team pass (research/evidence/derm7pt_v1/redteam_v1.json) proved that the IN-LOOP host
   checks never validated the structured ``citations`` array, so a fabricated record_id placed only
   there was invisible during generation. ``verification.citation_array_checks`` now closes that.
   Here we independently re-resolve the recorded citations for every answer that carries any, and
   check whether the reported resolvability was masking an unresolved array citation.

   Note on data: the report rows store the SCORED citation dict (cited / resolved / intact /
   unresolvable_ids), not the raw id list. That is sufficient to settle the question conclusively:
   ``eval.agent_eval.score_citations`` computes ``resolved = [i for i in ids if i in known]`` and
   ``unresolvable_ids = [i for i in ids if i not in known]`` over the SAME catalog membership this
   script's ``has_record`` uses -- so a fabricated id can never be miscounted as resolved (it is
   not in ``known``); it must land in ``unresolvable_ids``. We therefore (a) re-resolve every
   recorded unresolvable id against a fresh catalog to confirm it genuinely does not resolve (and
   catch any recorded-unresolvable that in fact DOES, which would be a scorer bug), (b) confirm the
   arithmetic ``resolved + len(unresolvable_ids) == cited`` for every row so no cited id was
   dropped from accounting, and (c) recompute the aggregate resolvability and compare it to the
   report's own endpoint. score_citations itself is left untouched.

2. UNCITED-SPECIFICS RECONCILIATION (does the false-positive fix reduce the flagged-answer count?).
   ``uncited_specifics_check`` used to flag any answer that quoted a guideline specific with an
   EMPTY structured array, even when a REAL inline record id already traced the quote -- 14 (v5.2)
   / 15 (v5.3) answers, with the v5.3 second-revision loop failing to converge on 11. The check now
   reconciles both channels (inline OR structured array). Here we recompute, from each stored final
   answer's own text, how many answers the OLD check flagged vs how many the NEW check flags, and
   name the false positives cleared. This part is fully independent: it resolves inline ids from the
   raw answer text.

Output: research/evidence/derm7pt_v1/citation_array_audit.json
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.verification import (  # noqa: E402
    ADVICE_SPECIFICS,
    _CITATION_TOKEN,
    uncited_specifics_check,
)
from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
SOURCE_TAGS = ("v5_2", "v5_3")  # v5_3 audited only if present (a paid run may still be writing it)
OUT = EVIDENCE / "citation_array_audit.json"


def _load_report(tag: str):
    path = EVIDENCE / f"agent_eval_report_{tag}.json"
    if not path.exists():
        return path, None, "absent"
    try:
        return path, json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, OSError) as exc:
        # A concurrent paid eval may be mid-write; skip rather than crash or half-read.
        return path, None, f"unloadable ({type(exc).__name__}: {exc})"


def _resolvable_inline_ids(catalog: GuidelineCatalog, text: str):
    """Inline citation-shaped tokens (same tokenizer/lowercasing as deterministic_checks) that
    resolve in the catalog."""
    return sorted(t for t in set(_CITATION_TOKEN.findall((text or "").lower())) if catalog.has_record(t))


def _array_resolution_audit(catalog: GuidelineCatalog, rows):
    cited_rows = [r for r in rows if r.get("citations", {}).get("cited", 0) > 0]
    offending = []
    scorer_disagreements = []
    arithmetic_inconsistent = []
    sum_resolved = sum_cited = 0
    for r in cited_rows:
        c = r["citations"]
        cited = int(c.get("cited", 0))
        resolved = int(c.get("resolved", 0))
        unresolvable = list(c.get("unresolvable_ids", []) or [])
        sum_resolved += resolved
        sum_cited += cited
        # (a) independently re-resolve each recorded unresolvable id.
        actually_resolves = [i for i in unresolvable if catalog.has_record(i)]
        if actually_resolves:
            scorer_disagreements.append(
                {"case_id": r["case_id"], "kind": r["kind"], "recorded_unresolvable_but_resolves": actually_resolves}
            )
        # (b) arithmetic accounts for every cited id.
        if resolved + len(unresolvable) != cited:
            arithmetic_inconsistent.append(
                {"case_id": r["case_id"], "kind": r["kind"], "cited": cited,
                 "resolved": resolved, "n_unresolvable": len(unresolvable)}
            )
        if unresolvable:
            offending.append(
                {"case_id": r["case_id"], "kind": r["kind"], "unresolvable_ids": unresolvable}
            )
    recomputed = round(sum_resolved / sum_cited, 4) if sum_cited else None
    return {
        "answers_with_citations": len(cited_rows),
        "answers_with_unresolved_array_citation": len(offending),
        "offending_case_ids": offending,
        "arithmetic_inconsistent_rows": arithmetic_inconsistent,
        "recorded_unresolvable_that_actually_resolve": scorer_disagreements,
        "recomputed_resolvability": recomputed,
        "resolvability_holds": len(offending) == 0
        and not arithmetic_inconsistent
        and not scorer_disagreements
        and (recomputed == 1.0 if recomputed is not None else True),
    }


def _uncited_specifics_reconciliation(catalog: GuidelineCatalog, rows):
    answers = [r for r in rows if r.get("answer_kind") == "answer"]
    old_flagged = new_flagged = 0
    cleared = []
    still_flagged = []
    for r in answers:
        text = r.get("answer", "") or ""
        cited = int(r.get("citations", {}).get("cited", 0))
        hits = [m for m in ADVICE_SPECIFICS if m in text.lower()]
        if not hits:
            continue
        # OLD in-loop check: fired iff a specific was quoted with an EMPTY structured array.
        # NEW check reconciles channels: fires only if no resolvable id in EITHER channel. For
        # cited>0 rows both pass (the structured array resolves -- resolvability 1.0, audited
        # above), so the whole delta lives in cited==0 rows, where we can resolve inline ids from
        # the raw text with full fidelity.
        if cited == 0:
            old = True
            new = bool(uncited_specifics_check(text, [], catalog))
        else:
            old = new = False
        old_flagged += int(old)
        new_flagged += int(new)
        if old and not new:
            cleared.append({
                "case_id": r["case_id"], "kind": r["kind"], "specifics": hits,
                "resolvable_inline_ids": _resolvable_inline_ids(catalog, text),
                "revisions": r.get("verification", {}).get("revisions"),
            })
        elif new:
            still_flagged.append({"case_id": r["case_id"], "kind": r["kind"], "specifics": hits})
    return {
        "answers": len(answers),
        "old_check_flagged": old_flagged,
        "new_check_flagged": new_flagged,
        "false_positives_cleared": old_flagged - new_flagged,
        "cleared_case_ids": cleared,
        "still_flagged_case_ids": still_flagged,
    }


def main() -> None:
    catalog = GuidelineCatalog.load()
    sources = {}
    for tag in SOURCE_TAGS:
        path, report, status = _load_report(tag)
        if report is None:
            sources[tag] = {"source_report": path.name, "status": status}
            continue
        rows = report.get("rows", [])
        sources[tag] = {
            "source_report": path.name,
            "status": "loaded",
            "report_prompt_version": report.get("prompt_version"),
            "reported_citation_resolvability": report.get("endpoints", {}).get("citation_resolvability"),
            "array_resolution_audit": _array_resolution_audit(catalog, rows),
            "uncited_specifics_reconciliation": _uncited_specifics_reconciliation(catalog, rows),
        }

    loaded = {t: s for t, s in sources.items() if s.get("status") == "loaded"}
    out = {
        "protocol": "citation-array-audit-and-uncited-specifics-reconciliation-rescore",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "catalog_n_records": len(catalog),
        "note": (
            "Zero API cost: re-scores stored rows against a fresh catalog. Part 1 audits whether "
            "the reported citation_resolvability survives independent array re-resolution (see the "
            "module docstring for why the scored dict is conclusive without the raw id list). Part "
            "2 measures the uncited-specifics false-positive reduction from channel reconciliation. "
            "score_citations and score_fabrication metric definitions are untouched."
        ),
        "headline": {
            "citation_resolvability_holds": {
                t: s["array_resolution_audit"]["resolvability_holds"] for t, s in loaded.items()
            },
            "unresolved_array_citations_found": {
                t: s["array_resolution_audit"]["answers_with_unresolved_array_citation"]
                for t, s in loaded.items()
            },
            "uncited_specifics_false_positives_cleared": {
                t: s["uncited_specifics_reconciliation"]["false_positives_cleared"]
                for t, s in loaded.items()
            },
        },
        "sources": sources,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    for tag, s in sources.items():
        if s.get("status") != "loaded":
            print(f"{tag}: {s.get('status')}")
            continue
        a = s["array_resolution_audit"]
        u = s["uncited_specifics_reconciliation"]
        print(
            f"{tag}: resolvability_holds={a['resolvability_holds']} "
            f"unresolved_array={a['answers_with_unresolved_array_citation']}/"
            f"{a['answers_with_citations']} recomputed={a['recomputed_resolvability']} | "
            f"uncited_specifics old={u['old_check_flagged']} new={u['new_check_flagged']} "
            f"cleared={u['false_positives_cleared']}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
