"""Controlled comparison: verified keyed retrieval vs dense chunk retrieval.

For the SAME clinician guideline-questions used in the v5.4 agent eval
(``research/evidence/derm7pt_v1/agent_eval_report_v5_4.json``), this measures what a standard
dense medical RAG gives up relative to the provenance-verified keyed path. Everything is
deterministic and, with a local embedder, involves ZERO paid calls.

The keyed path is the incumbent: for a case committed to a condition, ``GuidelineCatalog.bind``
returns whole recommendation records, each with a resolvable id that re-verifies its content hash
-- citation-resolvability 1.0 (reported in the v5.4 artifact). The dense arm re-chunks the SAME
source text, embeds it, and returns top-k passages by cosine similarity. We measure, per case:

* dense citation-resolvability -- a dense system emits chunk ids, which are not catalog record
  ids, so its resolvability against the hash-pinned catalog is 0 by construction. A best-case
  "attributed" resolvability (attribute each chunk to its majority-overlap record, then require
  that record's hash to re-verify against the chunk text) shows the CEILING: reachable only in
  record-aligned chunking, i.e. only by reconstructing the curated boundaries the keyed corpus
  already provides.
* retrieval precision / answer-coverage -- does the dense top-k actually contain the record the
  keyed path serves (exact hit), how much of that record's text it recovers (token coverage), how
  often it retrieves a passage from the WRONG condition (contamination the keyed path structurally
  cannot make).
* absence integrity on uncovered cases -- where the keyed path correctly reports "no guideline
  exists" (naevi, SK, misc), the dense arm has no absence mode: it always returns malignancy
  passages, which is a safety-relevant degradation.

Faithfulness (RAGAS) is deliberately NOT scored here: doing so would require GENERATING new agent
answers on dense contexts (paid agent calls) and paid judge calls. The task sanctions the
deterministic route, and the resolvability + precision + coverage gap is decisive on its own; the
verified path's faithfulness already lives in the RAGAS artifact.

Usage (from repo root, using the offline-capable venv)::

    venv_win/Scripts/python.exe scripts/dev/build_dense_rag_control.py \
        --embedders mpnet,dermlip --modes mechanical,record_aligned
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.evidence.dense_retrieval import (  # noqa: E402
    ChunkConfig,
    DenseGuidelineIndex,
    DermLIPTextEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    load_source_records,
)
from medguard_v2.evidence.guideline_binding import (  # noqa: E402
    GROUP_TO_CONDITION,
    MANAGEMENT_QUESTION_TYPES,
    GuidelineCatalog,
)

V54_REPORT = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v5_4.json"
PROMOTED_CONTRACTS = REPO / "research/evidence/derm7pt_v1/cdss_contracts_valid_promoted.json"
OUT_JSON = REPO / "research/evidence/derm7pt_v1/dense_rag_control.json"
INDEX_ROOT = REPO / "research/evidence/derm7pt_v1/dense_index"

GROUP_DISPLAY = {
    "BCC": "Basal cell carcinoma",
    "MEL": "Melanoma",
    "NEV": "Melanocytic naevus",
    "SK": "Seborrhoeic keratosis",
    "MISC": "Skin lesion",
}

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _embedder(key: str):
    if key == "mpnet":
        return SentenceTransformerEmbedder("sentence-transformers/all-mpnet-base-v2")
    if key == "dermlip":
        return DermLIPTextEmbedder()
    if key == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {key!r}")


def _load_case_groups() -> Dict[str, str]:
    """case_id -> classifier diagnosis group (BCC/MEL/NEV/SK/MISC), from the promoted contracts."""
    raw = json.loads(PROMOTED_CONTRACTS.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else (
        list(raw.values()) if all(isinstance(v, dict) and "case_id" in v for v in raw.values())
        else next(v for v in raw.values() if isinstance(v, list))
    )
    return {e["case_id"]: str(e["classifier"]["diagnosis"]) for e in entries}


def _guideline_questions() -> List[Dict[str, Any]]:
    report = json.loads(V54_REPORT.read_text(encoding="utf-8"))
    return [r for r in report.get("rows", []) if r.get("kind") == "guideline"]


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _score_covered(
    index: DenseGuidelineIndex,
    catalog: GuidelineCatalog,
    record_meta: Dict[str, Dict[str, str]],
    catalog_hashes: set,
    cases: List[Tuple[str, str]],   # (case_id, group)
    k: int,
) -> Dict[str, Any]:
    """Per-covered-case metrics at retrieval depth k, then averaged."""
    per_case: List[Dict[str, Any]] = []
    for case_id, group in cases:
        condition = GROUP_TO_CONDITION[group]
        served = catalog.bind(group).recommendations
        gold_ids = {r.record_id for r in served}
        gold_text_tokens = {r.record_id: _tokens(r.text) for r in served}

        query = f"{GROUP_DISPLAY[group]}. What does the guideline corpus say I should do for this lesion?"
        hits = index.retrieve_detailed(query, k=k)

        chunk_conditions = []
        attributed_reverify = []
        retrieved_text_tokens: set = set()
        retrieved_majority_ids = set()
        for r in hits:
            maj = r.chunk.majority_record_id
            cond = record_meta.get(maj, {}).get("condition") if maj else None
            chunk_conditions.append(cond)
            retrieved_majority_ids.add(maj)
            retrieved_text_tokens |= _tokens(r.chunk.text)
            # best-case attributed resolvability: the chunk's majority record id resolves AND the
            # chunk text re-verifies that record's content hash.
            attributed_reverify.append(
                1.0 if (maj is not None and catalog.has_record(maj)
                        and r.chunk.content_sha256 in catalog_hashes
                        and r.chunk.content_sha256 == record_meta[maj]["sha"]) else 0.0
            )

        precision_condition = _mean([1.0 if c == condition else 0.0 for c in chunk_conditions])
        cross_condition = _mean([
            1.0 if (c is not None and c != condition) else 0.0 for c in chunk_conditions
        ])
        keyed_recall = round(
            len(gold_ids & retrieved_majority_ids) / len(gold_ids), 4
        ) if gold_ids else None
        coverage_text = _mean([
            len(gold_text_tokens[g] & retrieved_text_tokens) / max(1, len(gold_text_tokens[g]))
            for g in gold_ids
        ]) if gold_ids else None

        per_case.append({
            "case_id": case_id,
            "group": group,
            "n_gold": len(gold_ids),
            "dense_citation_resolvability": 0.0,
            "attributed_resolvability": _mean(attributed_reverify),
            "precision_condition": precision_condition,
            "cross_condition_rate": cross_condition,
            "keyed_recall": keyed_recall,
            "answer_coverage_text": coverage_text,
        })

    return {
        "n": len(per_case),
        "k": k,
        "dense_citation_resolvability": 0.0,
        "attributed_resolvability": _mean([c["attributed_resolvability"] for c in per_case]),
        "precision_condition": _mean([c["precision_condition"] for c in per_case]),
        "cross_condition_rate": _mean([c["cross_condition_rate"] for c in per_case]),
        "keyed_recall": _mean([c["keyed_recall"] for c in per_case]),
        "answer_coverage_text": _mean([c["answer_coverage_text"] for c in per_case]),
    }


def _score_uncovered(
    index: DenseGuidelineIndex,
    record_meta: Dict[str, Dict[str, str]],
    cases: List[Tuple[str, str]],
    k: int,
) -> Dict[str, Any]:
    """Absence integrity: the keyed path reports 'no guideline' for these; the dense arm cannot."""
    top1_sims: List[float] = []
    malignant_top1: List[float] = []
    any_malignant: List[float] = []
    for case_id, group in cases:
        query = f"{GROUP_DISPLAY[group]}. What does the guideline corpus say I should do for this lesion?"
        hits = index.retrieve_detailed(query, k=k)
        if not hits:
            continue
        top1_sims.append(hits[0].score)
        top1_cond = record_meta.get(hits[0].chunk.majority_record_id, {}).get("condition")
        malignant_top1.append(1.0 if top1_cond in ("melanoma", "basal_cell_carcinoma") else 0.0)
        conds = {record_meta.get(h.chunk.majority_record_id, {}).get("condition") for h in hits}
        any_malignant.append(
            1.0 if conds & {"melanoma", "basal_cell_carcinoma"} else 0.0
        )
    return {
        "n": len(cases),
        "k": k,
        "dense_returns_chunks_rate": 1.0 if cases else None,
        "keyed_path_behaviour": "reports absence (no guideline exists); citation array empty",
        "mean_top1_similarity": _mean(top1_sims),
        "top1_is_malignancy_text_rate": _mean(malignant_top1),
        "any_malignancy_text_in_topk_rate": _mean(any_malignant),
    }


def run(embedders: List[str], modes: List[str], ks: List[int]) -> Dict[str, Any]:
    catalog = GuidelineCatalog.load()
    records = load_source_records()
    record_meta = {
        r.record_id: {"condition": r.condition, "sha": r.content_sha256} for r in records
    }
    catalog_hashes = {m["sha"] for m in record_meta.values()}

    case_groups = _load_case_groups()
    guideline_rows = _guideline_questions()

    covered_cases: List[Tuple[str, str]] = []
    uncovered_cases: List[Tuple[str, str]] = []
    missing = 0
    for row in guideline_rows:
        cid = row["case_id"]
        group = case_groups.get(cid)
        if group is None:
            missing += 1
            continue
        (covered_cases if GROUP_TO_CONDITION.get(group) else uncovered_cases).append((cid, group))

    results: Dict[str, Any] = {}
    for ek in embedders:
        emb = _embedder(ek)
        results[ek] = {}
        for mode in modes:
            config = ChunkConfig(mode=mode)
            index = DenseGuidelineIndex.build(emb, config=config, records=records)
            index.save(INDEX_ROOT / ek / mode)
            block: Dict[str, Any] = {
                "index": {
                    "version_hash": index.version_hash,
                    "n_chunks": len(index.chunks),
                    "dim": int(index.vectors.shape[1]),
                    "embedder": index.embedder_identity,
                    "chunk_config": config.as_dict(),
                },
                "covered": {str(k): _score_covered(
                    index, catalog, record_meta, catalog_hashes, covered_cases, k) for k in ks},
                "uncovered": {str(k): _score_uncovered(
                    index, record_meta, uncovered_cases, k) for k in ks},
            }
            results[ek][mode] = block
            print(f"[{ek}/{mode}] chunks={len(index.chunks)} "
                  f"covered@3 keyed_recall={block['covered']['3']['keyed_recall']} "
                  f"cross_cond={block['covered']['3']['cross_condition_rate']} "
                  f"attr_resolv={block['covered']['3']['attributed_resolvability']}")

    report = {
        "protocol": "medguard-v2-dense-rag-control-v1",
        "question_source": V54_REPORT.name,
        "n_guideline_questions": len(guideline_rows),
        "n_covered_cases": len(covered_cases),
        "n_uncovered_cases": len(uncovered_cases),
        "n_cases_unmapped": missing,
        "verified_keyed_path": {
            "citation_resolvability": 1.0,
            "source": "agent_eval_report_v5_4.json (endpoints.citation_resolvability)",
            "note": "Every served item is a whole recommendation record whose id resolves and "
                    "whose content hash re-verifies; structurally condition-scoped, so no "
                    "cross-condition retrieval is possible and absence is reported explicitly.",
        },
        "dense_query_policy": (
            "The dense arm is given the committed condition (the same information the keyed path "
            "keys on) as natural-language text plus the clinician's question, then retrieves "
            "top-k chunks by cosine similarity."
        ),
        "faithfulness_scored": False,
        "faithfulness_note": (
            "RAGAS faithfulness not scored: it would require generating new agent answers on "
            "dense contexts (paid agent calls) plus paid judge calls. The resolvability + "
            "precision + coverage gap is reported deterministically instead, per task guidance."
        ),
        "retrieval_depths": ks,
        "results": results,
        "paid_calls": 0,
        "embedders_note": (
            "mpnet = all-mpnet-base-v2 (canonical retrieval-tuned dense model, primary arm chosen "
            "so the comparison is not a strawman); dermlip = the in-domain DermLIP text tower "
            "(robustness row, same frozen model the perception layer reads)."
        ),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default="mpnet,dermlip",
                    help="comma list of mpnet,dermlip,hashing")
    ap.add_argument("--modes", default="mechanical,record_aligned",
                    help="comma list of mechanical,record_aligned")
    ap.add_argument("--ks", default="3,5", help="comma list of retrieval depths")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    report = run(embedders, modes, ks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"covered={report['n_covered_cases']} uncovered={report['n_uncovered_cases']} "
          f"unmapped={report['n_cases_unmapped']} paid_calls={report['paid_calls']}")


if __name__ == "__main__":
    main()
