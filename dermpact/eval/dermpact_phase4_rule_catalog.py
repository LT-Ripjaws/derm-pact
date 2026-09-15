"""Phase 4.1-4.2: build the source-bound EADO melanoma Part 1 rule catalogue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from dermpact.guidelines.rules import (
    Action,
    Predicate,
    RuleCatalogue,
    RuleRecord,
    RuleStatus,
    SourceBinding,
    Supersession,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_TEXT = ROOT / "research/manifests/dermpact_phase4/eado_part1_source_text_v1/eado_melanoma_part1_2024_pages.json"
CATALOGUE_PATH = ROOT / "dermpact/data/corpus/eado_melanoma_part1_rules_v1.jsonl"
MANIFEST_DIR = ROOT / "research/manifests/dermpact_phase4/rule_catalog_v1"

CATALOGUE_ID = "eado-mel-p1-2024"
CATALOGUE_VERSION = "v1"
DOCUMENT_ID = "eado:pmid:39700658"
DOI = "10.1016/j.ejca.2024.115152"
TITLE = "European consensus-based interdisciplinary guideline for melanoma. Part 1: Diagnostics - Update 2024"
ISSUER = "EADO/EDF/EORTC"
AUTHORITY = "clinical_practice_guideline"
JURISDICTION = "EU"
GUIDELINE_VERSION = "update-2024-published-2025"
VALID_FROM = "2024-11-28"   # available online (page 1)
VALID_UNTIL = "2026-12-31"  # "This guideline is valid until the end of 2026" (page 2)
CHECKED_ON = "2026-09-01"
DERM7PT_INSTRUMENT = "derm7pt_dermoscopic_7point_argenziano"

#: Display normalisation for the extracted text (U+FFFD marks a dropped 'ti' ligature).
GLYPH_NORMALISATION = {"�": "ti", "ﬁ": "fi", "ﬂ": "fl", "–": "-"}


def normalise_display(text: str) -> str:
    for k, v in GLYPH_NORMALISATION.items():
        text = text.replace(k, v)
    return " ".join(text.split())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    with tmp.open("r+b") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    digest = _sha256_file(path)
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest




def _spec(
    *,
    rid: str,
    status: RuleStatus,
    scope: str,
    rec_no: Optional[int],
    title: str,
    page: int,
    section: str,
    start: str,
    end: str,
    instrument: Optional[str],
    population: str,
    prerequisites: Optional[Mapping[str, Any]],
    exclusions: Optional[Mapping[str, Any]],
    required: Sequence[str],
    action: Optional[Mapping[str, Any]],
    strength: Optional[str],
    evidence: Optional[str],
    consensus: Optional[str],
    basis: str,
    rationale: str,
    points: Sequence[str],
) -> Dict[str, Any]:
    return dict(
        rid=rid, status=status, scope=scope, rec_no=rec_no, title=title, page=page, section=section,
        start=start, end=end, instrument=instrument, population=population, prerequisites=prerequisites,
        exclusions=exclusions, required=list(required), action=action, strength=strength, evidence=evidence,
        consensus=consensus, basis=basis, rationale=rationale, points=list(points),
    )


REVIEWER_A_ENCODING: List[Dict[str, Any]] = [
    _spec(
        rid=f"{CATALOGUE_ID}:R3", status=RuleStatus.EXECUTABLE, scope="lesion_next_step", rec_no=3,
        title="Clinical diagnosis: histopathological confirmation of suspected melanoma",
        page=8, section="7.1 Clinical and dermatoscopic diagnosis",
        start="Recommendation 3.", end="Consensus rate: 100% (19/19)",
        instrument=DERM7PT_INSTRUMENT,
        population="Patients with a skin lesion under dermoscopic assessment in which melanoma is clinically suspected.",
        prerequisites={"all_of": [
            {"fact_is": ["image.dermoscopic_present", True]},
            {"fact_is": ["lesion.melanoma_suspected", True]},
        ]},
        exclusions=None,
        required=["image.dermoscopic_present", "lesion.melanoma_suspected"],
        action={"code": "histopathological_confirmation",
                "text": "Recommended diagnostic next step: obtain histopathological confirmation "
                        "(biopsy/excision according to the applicable guideline and local practice).",
                "conditional_on": []},
        strength="GCP", evidence=None, consensus="100% (19/19)", basis="consensus based statement",
        rationale="'shall be confirmed by histopathology' is a mandatory next step once melanoma is "
                  "clinically suspected. Suspicion is a clinician judgement; the engine consumes the fact "
                  "lesion.melanoma_suspected, which is the clinician assertion when known, otherwise the "
                  "frozen derivation (provisional class melanoma OR decidable checklist meeting the threshold).",
        points=["Is treating a decidable 7-point score >= 3 or a provisional melanoma class as 'clinically "
                "suspected' acceptable as the default derivation when no clinician assertion exists?",
                "Undecidable checklist without assertion -> applicability_unknown (one question)."],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R4", status=RuleStatus.EXECUTABLE, scope="lesion_assessment", rec_no=4,
        title="Dermatoscopic diagnosis: dermatoscopy for pigmented and non-pigmented lesions",
        page=8, section="7.1 Clinical and dermatoscopic diagnosis",
        start="Recommendation 4 [72", end="Consensus rate: 100% (19/19)",
        instrument=None,
        population="Any pigmented or non-pigmented skin lesion under assessment.",
        prerequisites={"fact_is": ["lesion.under_dermoscopic_assessment", True]},
        exclusions=None,
        required=["lesion.under_dermoscopic_assessment"],
        action={"code": "dermatoscopic_assessment",
                "text": "Dermatoscopy shall be used for the assessment of this lesion; an interpretable "
                        "dermoscopic image is required (reacquire if image quality fails).",
                "conditional_on": []},
        strength="A", evidence="1b", consensus="100% (19/19)", basis="evidence based recommendation",
        rationale="Universal within the intended use (one dermoscopic image per case). Encoded so the "
                  "record states the indication and so a failed/missing image surfaces as reacquisition "
                  "rather than silence.",
        points=["Second sentence of the box lists literature-basis notes only; no additional condition encoded."],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R5", status=RuleStatus.EXECUTABLE, scope="patient_surveillance", rec_no=5,
        title="Whole-body photography with sequential examinations in high-risk patients",
        page=9, section="7 Diagnostic approach",
        start="Recommendation 5 [81]", end="Consensus rate: 100% (19/19)",
        instrument=None,
        population="Patients in a known melanoma high-risk group.",
        prerequisites={"fact_is": ["patient.high_risk_status", True]},
        exclusions=None,
        required=["patient.high_risk_status"],
        action={"code": "whole_body_photography_surveillance",
                "text": "Whole-body photography with sequential examinations should be used for the early "
                        "detection of melanoma in this high-risk patient.",
                "conditional_on": []},
        strength="B", evidence="2b", consensus="100% (19/19)", basis="evidence based recommendation",
        rationale="'high-risk patients' is not closed-world defined by the box; page 8 lists genetic "
                  "predisposition, personal melanoma history, high total nevus count 'etc.'. The engine "
                  "therefore requires an explicit clinician assertion and never infers high-risk from "
                  "partial components.",
        points=["Scope is patient-level surveillance, adjacent to the lesion-level intended use; "
                "adjudicator may demote to prose_only if judged outside Phase 0.1 scope."],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R6", status=RuleStatus.EXECUTABLE, scope="patient_surveillance", rec_no=6,
        title="Sequential digital dermatoscopy in high-risk patients with a high total nevus count",
        page=9, section="7 Diagnostic approach",
        start="Recommendation 6 [82]", end="Consensus rate: 100% (19/19)",
        instrument=None,
        population="High-risk patients with a high total nevus count.",
        prerequisites={"all_of": [
            {"fact_is": ["patient.high_risk_status", True]},
            {"fact_is": ["patient.high_total_nevus_count", True]},
        ]},
        exclusions=None,
        required=["patient.high_risk_status", "patient.high_total_nevus_count"],
        action={"code": "sequential_digital_dermatoscopy",
                "text": "Sequential digital dermatoscopy should be used for this high-risk patient with a "
                        "high total nevus count.",
                "conditional_on": []},
        strength="B", evidence="2b", consensus="100% (19/19)", basis="evidence based recommendation",
        rationale="Two conjunctive prerequisites read directly from the box. 'High total nevus count' is "
                  "operationalised from the guideline's own risk-factor text (page 3: '100 nevi and more'): "
                  "derived as total_nevus_count >= 100 when a count is known, else asserted.",
        points=["The >=100 operationalisation comes from section 3 (risk factors), not the box itself; "
                "boundary tested at 99/100."],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R7", status=RuleStatus.EXECUTABLE, scope="lesion_next_step", rec_no=7,
        title="Reflectance confocal microscopy for equivocal lesions, when available",
        page=9, section="7 Diagnostic approach",
        start="Recommendation 7 [83", end="Consensus rate: 100% (23/23)",
        instrument=DERM7PT_INSTRUMENT,
        population="Clinically/dermatoscopically equivocal skin lesions, at sites where RCM is available.",
        prerequisites={"all_of": [
            {"fact_is": ["image.dermoscopic_present", True]},
            {"fact_is": ["lesion.equivocal", True]},
            {"fact_is": ["site.rcm_available", True]},
        ]},
        exclusions=None,
        required=["image.dermoscopic_present", "lesion.equivocal", "site.rcm_available"],
        action={"code": "reflectance_confocal_microscopy",
                "text": "Reflectance confocal microscopy should be used for further evaluation of this "
                        "clinically/dermatoscopically equivocal lesion (available at this site).",
                "conditional_on": []},
        strength="B", evidence="1b", consensus="100% (23/23)", basis="evidence based statement",
        rationale="'when available' is a hard prerequisite on site capability (unknown unless asserted). "
                  "'Equivocal' is derived deterministically: undecidable checklist range, direct/checklist "
                  "disagreement, or abstained diagnosis.",
        points=["Whether the deterministic 'equivocal' derivation matches the guideline's clinical sense "
                "is an interpretation point for reviewer B."],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:P1", status=RuleStatus.PROSE_ONLY, scope="lesion_next_step", rec_no=None,
        title="Prose: excision should not be delayed when dermatoscopy unavailable and suspicion high",
        page=7, section="7.1 Clinical and dermatoscopic diagnosis",
        start="In the scenario where dermatoscopy is not available", end="excision should not be delayed.",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength=None, evidence=None, consensus=None, basis="narrative text",
        rationale="Not a numbered recommendation; 'clinical suspicion is high' needs clinician judgement "
                  "beyond the explicit passage. Recorded for adjudication; not executable in v1.",
        points=["Adjudicator may promote if a 'clinician.melanoma_suspected' assertion is accepted as 'high'."],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:P2", status=RuleStatus.PROSE_ONLY, scope="documentation", rec_no=None,
        title="Prose: photographic documentation before excision",
        page=7, section="7.1 Clinical and dermatoscopic diagnosis",
        start="Clinical and dermatoscopic photographic", end="before excision is recommended.",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength=None, evidence=None, consensus=None, basis="narrative text",
        rationale="Documentation practice statement without a recommendation box; not a diagnostic "
                  "next step.", points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:P3", status=RuleStatus.PROSE_ONLY, scope="histopathology", rec_no=None,
        title="Prose: histological examination mandatory after excision; complete excision if possible",
        page=10, section="7.2 Histopathologic diagnosis",
        start="Whenever a suspicious skin lesion is surgically excised", end="completely excised\nif possible.",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength=None, evidence=None, consensus=None, basis="narrative text",
        rationale="Post-excision histopathology practice; overlaps R3 and lies downstream of the "
                  "diagnostic next step.", points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:P4", status=RuleStatus.PROSE_ONLY, scope="lesion_next_step", rec_no=None,
        title="Prose: EFG rule (Elevated, Firm, Growing) for nodular melanoma prompting excision",
        page=7, section="7.1 Clinical and dermatoscopic diagnosis",
        start="Nodular\nmelanoma (NM) may lack", end="aggressive mela-\nnoma [53] .",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength=None, evidence=None, consensus=None, basis="narrative text",
        rationale="'Firm' and 'growing' are not observable from the locked record (only elevation is); "
                  "encoding would require unavailable facts.", points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R1", status=RuleStatus.OUT_OF_SCOPE, scope="staging", rec_no=1,
        title="Stage classification (AJCC 8th edition)", page=6,
        section="6. Staging examinations according to AJCC stages",
        start="Recommendation 1.", end="Consensus rate: 100% (19/19)",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength="GCP", evidence=None, consensus="100% (19/19)", basis="consensus based recommendation",
        rationale="Phase 0.1 excludes staging; retained so staging queries return no_catalogue_coverage.",
        points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R2", status=RuleStatus.OUT_OF_SCOPE, scope="staging_workup", rec_no=2,
        title="Lymph node ultrasound in primary melanoma pT1b and higher", page=6,
        section="6. Staging examinations according to AJCC stages",
        start="Recommendation 2 [41]", end="Consensus rate: 90 %; (19/21) 2 absten�ons",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength="B", evidence="2a", consensus="90% (19/21), 2 abstentions", basis="evidence based recommendation",
        rationale="Post-histopathology staging workup (requires pT stage); outside prediagnostic scope.",
        points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R8", status=RuleStatus.OUT_OF_SCOPE, scope="molecular", rec_no=8,
        title="BRAF status in stage III/IV", page=12, section="8. Molecular analysis and other biomarkers",
        start="Recommendation 8.", end="Consensus rate: 100% (22/22)",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength="B", evidence=None, consensus="100% (22/22)", basis="evidence based statement",
        rationale="Molecular testing by stage; outside prediagnostic scope.", points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R9", status=RuleStatus.OUT_OF_SCOPE, scope="follow_up", rec_no=9,
        title="Follow-up duration", page=14, section="11. Follow-up",
        start="Recommendation 9.", end="Consensus rate: 95% (20/21) 1 abstains",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength="GCP", evidence=None, consensus="95% (20/21), 1 abstains", basis="consensus based recommendation",
        rationale="Follow-up after diagnosis; outside prediagnostic scope.", points=[],
    ),
    _spec(
        rid=f"{CATALOGUE_ID}:R10", status=RuleStatus.OUT_OF_SCOPE, scope="follow_up", rec_no=10,
        title="Follow-up schedule", page=15, section="11.2 Recommendations for structured follow-up",
        start="Recommendation 10.", end="Consensus rate: 100% (21/21)",
        instrument=None, population="", prerequisites=None, exclusions=None, required=[], action=None,
        strength="GCP", evidence=None, consensus="100% (21/21)", basis="consensus based recommendation",
        rationale="Follow-up after diagnosis; outside prediagnostic scope.", points=[],
    ),
]




def load_source_pages() -> Dict[str, Any]:
    payload = json.loads(SOURCE_TEXT.read_text(encoding="utf-8"))
    for page in payload["pages"]:
        if sha256_text(page["text"]) != page["text_sha256"]:
            raise RuntimeError(f"page {page['pdf_page']} text hash mismatch in extracted source")
    return payload


def _slice_span(page_text: str, start: str, end: str) -> tuple[int, int, str]:
    s = page_text.find(start)
    if s < 0:
        raise ValueError(f"start marker not found: {start!r}")
    e = page_text.find(end, s)
    if e < 0:
        raise ValueError(f"end marker not found after start: {end!r}")
    e += len(end)
    return s, e, page_text[s:e]


def build_records(source: Mapping[str, Any]) -> List[RuleRecord]:
    pages = {p["pdf_page"]: p for p in source["pages"]}
    pdf_sha = source["source_pdf_sha256"]
    records: List[RuleRecord] = []
    for spec in REVIEWER_A_ENCODING:
        page = pages[spec["page"]]
        s, e, span = _slice_span(page["text"], spec["start"], spec["end"])
        rec = RuleRecord(
            rule_id=spec["rid"],
            catalogue_id=CATALOGUE_ID,
            catalogue_version=CATALOGUE_VERSION,
            status=spec["status"],
            scope=spec["scope"],
            recommendation_number=spec["rec_no"],
            title=spec["title"],
            source=SourceBinding(
                document_id=DOCUMENT_ID, doi=DOI, title=TITLE, issuer=ISSUER,
                pdf_sha256=pdf_sha, pdf_page=spec["page"], page_text_sha256=page["text_sha256"],
                section=spec["section"], span_text=span, span_sha256=sha256_text(span),
                span_char_start=s, span_char_end=e,
            ),
            authority=AUTHORITY,
            jurisdiction=JURISDICTION,
            guideline_version=GUIDELINE_VERSION,
            supersession=Supersession(status="current", valid_from=VALID_FROM, valid_until=VALID_UNTIL,
                                      superseded_by=None, checked_on=CHECKED_ON),
            instrument_id=spec["instrument"],
            target_population=spec["population"],
            prerequisites=Predicate(spec["prerequisites"]) if spec["prerequisites"] else None,
            exclusions=Predicate(spec["exclusions"]) if spec["exclusions"] else None,
            required_facts=tuple(spec["required"]),
            action=Action(spec["action"]["code"], spec["action"]["text"], tuple(spec["action"].get("conditional_on", []))) if spec["action"] else None,
            recommendation_strength=spec["strength"],
            evidence_level=spec["evidence"],
            consensus_rate=spec["consensus"],
            basis=spec["basis"],
            encoding_rationale=spec["rationale"],
            interpretation_points=tuple(spec["points"]),
        ).signed()
        records.append(rec)
    return records


def _review_field_names() -> List[str]:
    return [
        "status", "scope", "source.span_text", "source.pdf_page", "target_population",
        "prerequisites", "exclusions", "required_facts", "action", "recommendation_strength",
        "evidence_level", "consensus_rate", "supersession.valid_until", "instrument_id",
    ]


def build_catalogue(*, catalogue_path: Path = CATALOGUE_PATH, manifest_dir: Path = MANIFEST_DIR) -> Dict[str, Any]:
    source = load_source_pages()
    records = build_records(source)
    catalogue = RuleCatalogue(CATALOGUE_ID, CATALOGUE_VERSION, tuple(records))
    file_sha = catalogue.write_jsonl(catalogue_path)

    # Reload from disk and re-verify (save/reload parity is part of the gate).
    reloaded = RuleCatalogue.load_jsonl(catalogue_path, expected_sha256=catalogue.catalogue_sha256)
    assert reloaded.catalogue_sha256 == catalogue.catalogue_sha256

    fields = _review_field_names()
    reviewer_a = {
        "schema_version": "dermpact-phase4-reviewer-encoding-v1",
        "reviewer": "A",
        "identity": "Claude (Opus 4.6), session 2026-09-01",
        "catalogue_sha256": catalogue.catalogue_sha256,
        "records": [
            {**r.to_dict(), "span_text_display": normalise_display(r.source.span_text)}
            for r in records
        ],
    }
    reviewer_b_form = {
        "schema_version": "dermpact-phase4-reviewer-form-v1",
        "reviewer": "B",
        "identity": "(to be filled by the independent reviewer, e.g. Codex)",
        "instructions": (
            "For each rule and field, read ONLY the cited source span/page in the extracted text and "
            "record agree=true/false with a note. Do not consult reviewer A's rationale until the "
            "form is complete. Mark any field whose resolution needs clinical interpretation beyond "
            "the explicit passage as 'ambiguous'."
        ),
        "catalogue_sha256": catalogue.catalogue_sha256,
        "records": [
            {
                "rule_id": r.rule_id,
                "source_ref": {"pdf_page": r.source.pdf_page, "page_text_sha256": r.source.page_text_sha256,
                               "span_sha256": r.source.span_sha256},
                "fields": {f: {"agree": None, "ambiguous": False, "note": ""} for f in fields},
            }
            for r in records
        ],
    }
    adjudication_form = {
        "schema_version": "dermpact-phase4-adjudication-form-v1",
        "adjudicator": "(user)",
        "instructions": "Resolve each A/B disagreement ONLY from the source text; otherwise mark the "
                        "rule ambiguous_excluded.",
        "catalogue_sha256": catalogue.catalogue_sha256,
        "records": [{"rule_id": r.rule_id, "disagreements": [], "decision": None, "note": ""} for r in records],
    }

    manifest = {
        "schema_version": "dermpact-phase4-rule-catalog-manifest-v1",
        "status": "ENCODED_REVIEWER_A_ONLY",
        "review_status": {
            "reviewer_A": "complete", "reviewer_B": "pending", "adjudication": "pending",
            "hard_gate_to_phase5": "OPEN: dual independent source-fidelity review not yet complete",
        },
        "catalogue": {"id": CATALOGUE_ID, "version": CATALOGUE_VERSION,
                      "path": str(catalogue_path.relative_to(ROOT)).replace("\\", "/"),
                      "file_sha256": file_sha, "catalogue_sha256": catalogue.catalogue_sha256,
                      "n_records": len(records),
                      "n_executable": len(catalogue.executable),
                      "n_prose_only": sum(1 for r in records if r.status is RuleStatus.PROSE_ONLY),
                      "n_out_of_scope": sum(1 for r in records if r.status is RuleStatus.OUT_OF_SCOPE),
                      "n_ambiguous_excluded": sum(1 for r in records if r.status is RuleStatus.AMBIGUOUS_EXCLUDED)},
        "source": {"document_id": DOCUMENT_ID, "doi": DOI, "pdf_sha256": source["source_pdf_sha256"],
                   "extracted_text_path": str(SOURCE_TEXT.relative_to(ROOT)).replace("\\", "/"),
                   "extracted_text_sha256": _sha256_file(SOURCE_TEXT),
                   "valid_from": VALID_FROM, "valid_until": VALID_UNTIL, "supersession_checked_on": CHECKED_ON,
                   "license": "CC BY 4.0"},
        "glyph_normalisation": {k.encode("unicode_escape").decode("ascii"): v for k, v in GLYPH_NORMALISATION.items()},
        "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256_file(Path(__file__).resolve())},
        "hard_scope_checks": {"official_test_rows_loaded": 0, "images_opened": 0, "models_loaded": 0},
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sha_a = _atomic_json(manifest_dir / "reviewer_A_encoding.json", reviewer_a)
    sha_b = _atomic_json(manifest_dir / "reviewer_B_form.json", reviewer_b_form)
    sha_adj = _atomic_json(manifest_dir / "adjudication_form.json", adjudication_form)
    manifest["review_artifacts"] = {
        "reviewer_A_encoding_sha256": sha_a,
        "reviewer_B_form_sha256": sha_b,
        "adjudication_form_sha256": sha_adj,
    }
    manifest_sha = _atomic_json(manifest_dir / "catalog_manifest.json", manifest)
    Path(str(catalogue_path) + ".sha256").write_text(file_sha + "\n", encoding="ascii")
    return {**manifest, "manifest_sha256": manifest_sha}


def load_catalogue(*, catalogue_path: Path = CATALOGUE_PATH, manifest_dir: Path = MANIFEST_DIR) -> RuleCatalogue:
    manifest = json.loads((manifest_dir / "catalog_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["catalogue"]["catalogue_sha256"]
    if _sha256_file(catalogue_path) != manifest["catalogue"]["file_sha256"]:
        raise RuntimeError("catalogue file hash does not match its manifest")
    return RuleCatalogue.load_jsonl(catalogue_path, expected_sha256=expected)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.build == args.validate:
        parser.error("choose exactly one of --build or --validate")
    if args.build:
        result = build_catalogue()
        print(json.dumps({k: v for k, v in result.items() if k in ("status", "catalogue", "review_status", "manifest_sha256")},
                         indent=2, sort_keys=True))
    else:
        cat = load_catalogue()
        print(json.dumps({"status": "PASS", "catalogue_sha256": cat.catalogue_sha256,
                          "n_records": len(cat.records), "executable": [r.rule_id for r in cat.executable]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOGUE_ID", "CATALOGUE_PATH", "CATALOGUE_VERSION", "MANIFEST_DIR", "REVIEWER_A_ENCODING",
    "build_catalogue", "build_records", "load_catalogue", "load_source_pages", "normalise_display",
]
