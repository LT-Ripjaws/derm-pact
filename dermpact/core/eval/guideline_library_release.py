"""Freeze the 66-record guideline library as a layered Phase 6 release."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dermpact.guidelines.reviewed_v2 import ReviewedApplicabilityEngine
from dermpact.pipeline.guideline_retriever import FrozenGuidelineRetriever

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "dermpact/data/corpus/derm_guideline_recommendations_v1.jsonl"
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase6/guideline_library_v1"
SCHEMA_VERSION = "dermpact-phase6-layered-guideline-library-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.write_text(body, encoding="utf-8", newline="\n")
    partial.replace(path)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_release(out_dir: Path = OUTPUT_DIR) -> Mapping[str, Any]:
    expected = Path(str(CORPUS) + ".sha256").read_text(encoding="ascii").strip()
    retriever = FrozenGuidelineRetriever.load(CORPUS, expected_sha256=expected)
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(retriever.records):
        raise RuntimeError("guideline library row count changed during verified load")

    public_rows = []
    for row in rows:
        public_rows.append({
            "schema_version": SCHEMA_VERSION,
            "document_id": row["document_id"], "source_document_id": row["source_document_id"],
            "title": row["title"], "issuer": row["issuer"], "condition": row["condition"],
            "source_id": row["source_id"], "citation_id": row["citation_id"],
            "source_version": row["source_version"], "section": row["section"],
            "article_page": row["article_page"], "pdf_page": row["pdf_page"],
            "provenance_uri": row["provenance_uri"], "license": row["license"],
            "distribution_status": row["distribution_status"],
            "eligible_claim_types": row["eligible_claim_types"], "question_types": row["question_types"],
            "content_sha256": row["content_sha256"], "record_sha256": row["record_sha256"],
            "source_pdf_sha256": row["source_pdf_sha256"],
            "runtime_layer": "retrieval_only_not_executable",
            "public_payload": "metadata_only_text_omitted",
        })

    out_dir = Path(out_dir)
    public_path = out_dir / "public_metadata_catalog.jsonl"
    _atomic_text(public_path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in public_rows))

    executable = ReviewedApplicabilityEngine().catalogue.records
    status_counts = collections.Counter(rule.status.value for rule in executable)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": CORPUS.relative_to(ROOT).as_posix(), "sha256": expected,
            "hash_verified": _sha256(CORPUS) == expected, "internal_text_rows": len(rows),
        },
        "layers": {
            "retrieval_library": {
                "n_records": len(rows),
                "conditions": dict(sorted(collections.Counter(r["condition"] for r in rows).items())),
                "issuers": dict(sorted(collections.Counter(r["issuer"] for r in rows).items())),
                "distribution_status": dict(sorted(collections.Counter(r["distribution_status"] for r in rows).items())),
                "decision_authority": "retrieval_only_not_executable",
            },
            "reviewed_v2_next_step_catalogue": {
                "n_records": len(executable), "status_counts": dict(sorted(status_counts.items())),
                "decision_authority": "sole_executable_next_step_layer",
            },
        },
        "safety_contract": {
            "patient_diagnosis_from_library": False,
            "next_step_mutation_from_library": False,
            "management_treatment_staging_follow_up_exposed_to_current_agent": False,
            "jurisdiction_fallback": False,
            "public_export_contains_recommendation_text": False,
        },
        "public_catalog": {
            "path": _portable_path(public_path), "sha256": _sha256(public_path),
            "n_records": len(public_rows),
        },
    }
    manifest_path = out_dir / "manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _atomic_text(Path(str(manifest_path) + ".sha256"), _sha256(manifest_path) + "\n")
    result_path = out_dir / "RESULT.md"
    _atomic_text(result_path, (
        "# Layered guideline library release\n\n"
        f"- Internal hash-verified retrieval records: **{len(rows)}**.\n"
        f"- Reviewed-v2 records: **{len(executable)}**; executable: **{status_counts.get('executable', 0)}**.\n"
        "- The 66 records are restored to the agent as condition-scoped, source-exact retrieval evidence.\n"
        "- They are not converted into patient facts or executable next steps.\n"
        "- The public catalogue is metadata-only; recommendation text remains governed by each source licence.\n"
    ))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_release(args.out_dir)
    print(json.dumps({"status": "PASS", "n_retrieval": manifest["layers"]["retrieval_library"]["n_records"],
                      "n_executable": manifest["layers"]["reviewed_v2_next_step_catalogue"]["status_counts"].get("executable", 0),
                      "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
