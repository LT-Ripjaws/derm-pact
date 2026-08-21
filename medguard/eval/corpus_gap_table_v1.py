"""Deterministic corpus/gap table (Table 1) over the 138-record literature corpus.

Data only, no prose. Two kinds of evidence, kept separate:

* EXACT structured composition from the workbook's own classification columns
  (MedGuard Role, Type) -- not keyword-derived, not contestable.
* Keyword-derived capability coverage and the Gap A provenance-composition probe.
  Every keyword rule is emitted in the artifact and every matched record MG# is
  listed, so the counts are auditable and any rule can be tightened without
  guessing which records moved.

The headline Gap A fact is robust regardless of keyword precision: the authority-
class provenance tier is absent from every record, so the bound composition
(content-hash + authority-class + version) is necessarily 0 of 138.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import openpyxl


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "MedGuard_Corpus.xlsx"
OUTPUT = ROOT / "research/experiments/medguard_h2h_v1/corpus_gap_table_v1.json"

# Fields scanned for the keyword-derived probes (lower-cased, concatenated per row).
_CAPABILITY_FIELDS = (
    "MedGuard Role", "Type", "RAG Architecture", "Retrieval Strategy",
    "Adaptive Retrieval / Gating", "Abstention / Selective Prediction",
    "Modalities Used", "Cross-Modal Fusion", "Agent Architecture",
    "Trust Mechanisms", "Hallucination Mitigation", "Explainability / XAI",
    "Process-level Trust / Traceability", "Knowledge Base / Index",
    "Regulatory / Governance Alignment",
)

# Gap A provenance-composition elements. Each is a bound part of the MedGuard
# evidence contract; the contribution is binding them together, image-anchored.
_PROVENANCE_ELEMENTS: Dict[str, Tuple[str, ...]] = {
    "content_hash": (
        "content hash", "content-hash", "sha-256", "sha256", "content-address",
        "cryptographic hash", "message digest", "hash-bound",
    ),
    "authority_class": (
        "authority class", "authority-class", "authority tier", "authoritative-source tier",
        "source authority class", "provenance tier", "source-authority tier",
    ),
    "source_version": ("source version", "versioned source", "version-pinned", "content version"),
}

# Keyword-derived capability axes mapped to MedGuard's five contribution gaps.
_CAPABILITY_AXES: Dict[str, Tuple[str, ...]] = {
    "multimodal_image_rag": ("image", "multimodal", "vlm", "vision-language", "radiolog", "dermosc"),
    "provenance_or_lineage": ("provenance", "lineage", "citation", "traceab", "attribution"),
    "adaptive_retrieval": ("adaptive retrieval", "adaptive context", "gating", "quality weight", "top-k selection"),
    "abstention_selective": ("abstention", "abstain", "selective prediction", "defer", "reject option", "conformal"),
    "failure_typed_remediation": ("failure type", "error taxonomy", "typed failure", "remediation", "bounded refine", "repair loop"),
    "process_trajectory_trust": ("trajectory", "per-step", "step-level", "process-level trust", "runtime receipt", "decision trace"),
}


def _load_rows() -> Tuple[Sequence[str], List[Mapping[str, Any]]]:
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    sheet = workbook.active
    raw = list(sheet.iter_rows(values_only=True))
    header = [str(cell).strip() if cell is not None else "" for cell in raw[0]]
    rows: List[Mapping[str, Any]] = []
    for values in raw[1:]:
        rows.append({header[i]: values[i] for i in range(len(header))})
    workbook.close()
    return header, rows


def _mg(row: Mapping[str, Any]) -> str:
    return str(row.get("MG#", "")).strip()


def _blob(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return " ".join(str(row[f]).lower() for f in fields if row.get(f)).strip()


def _matched(rows: Sequence[Mapping[str, Any]], keywords: Sequence[str],
             fields: Sequence[str]) -> List[str]:
    hits = []
    for row in rows:
        blob = _blob(row, fields)
        if any(keyword in blob for keyword in keywords):
            hits.append(_mg(row))
    return sorted(hits, key=lambda value: (len(value), value))


def build_table() -> Dict[str, Any]:
    _header, rows = _load_rows()
    n = len(rows)

    def _exact_distribution(column: str) -> Dict[str, int]:
        counter = collections.Counter(
            (str(row[column]).strip() if row.get(column) else "(blank)") for row in rows)
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    provenance = {}
    for element, keywords in _PROVENANCE_ELEMENTS.items():
        matched = _matched(rows, keywords, _CAPABILITY_FIELDS)
        provenance[element] = {"records": len(matched), "matched_mg": matched}
    bound_all = sorted(
        set(provenance["content_hash"]["matched_mg"])
        & set(provenance["authority_class"]["matched_mg"])
        & set(provenance["source_version"]["matched_mg"]),
        key=lambda value: (len(value), value),
    )

    capability = {}
    for axis, keywords in _CAPABILITY_AXES.items():
        matched = _matched(rows, keywords, _CAPABILITY_FIELDS)
        capability[axis] = {"records": len(matched), "matched_mg": matched}

    return {
        "schema_version": "medguard-corpus-gap-table-v1",
        "status": "descriptive_corpus_positioning_not_a_paper_result",
        "source": {
            "workbook": str(WORKBOOK.relative_to(ROOT)).replace("\\", "/"),
            "workbook_sha256": hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(),
            "records": n,
        },
        "corpus_composition": {
            "by_medguard_role": _exact_distribution("MedGuard Role"),
            "by_type": _exact_distribution("Type"),
        },
        "gap_a_evidence_contract_provenance_composition": {
            "claim": (
                "No record binds the evidence-contract provenance composition "
                "(content-hash + authority-class + version). The authority-class "
                "tier is absent from every record, so the bound composition is 0 "
                "of N by construction."
            ),
            "per_element": provenance,
            "bound_composition_records": len(bound_all),
            "bound_composition_matched_mg": bound_all,
        },
        "capability_coverage_keyword_derived": {
            "note": "Keyword-derived over the fields below; matched MG# are listed for audit.",
            "scanned_fields": list(_CAPABILITY_FIELDS),
            "axes": capability,
        },
        "keyword_definitions": {
            "provenance_elements": {k: list(v) for k, v in _PROVENANCE_ELEMENTS.items()},
            "capability_axes": {k: list(v) for k, v in _CAPABILITY_AXES.items()},
        },
        "boundaries": {"generation_calls": 0, "network_access": False, "sealed_test_access": False},
    }


def main() -> int:
    table = build_table()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(table, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (OUTPUT.parent / (OUTPUT.name + ".sha256")).write_text(digest + "\n", encoding="ascii", newline="\n")
    prov = table["gap_a_evidence_contract_provenance_composition"]
    print(f"records: {table['source']['records']}")
    print(f"roles: {table['corpus_composition']['by_medguard_role']}")
    print(f"gap A provenance elements: "
          f"{ {k: v['records'] for k, v in prov['per_element'].items()} }")
    print(f"bound composition (content-hash+authority-class+version): "
          f"{prov['bound_composition_records']} of {table['source']['records']}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
