"""N6 -- Guideline-update audit demo (appendix / governance story).

The catalog pins every recommendation to a content hash (sha256 of its verbatim text) and the host
fails CLOSED on any mismatch: GuidelineCatalog refuses to instantiate a record whose stored hash
does not match its text, and DecisionContract re-verifies the served record's hash before it may be
displayed. This script exercises that machinery end to end on a simulated guideline VERSION UPDATE,
entirely IN MEMORY -- the on-disk corpus is never modified.

The scenario: the top melanoma recommendation that bind("MEL") serves
(aad:pmid:30392755:table-x-excision-first-line, version 2019) is revised upstream. Its text
changes, so its content hash changes. We show:

  (i)   DETECTION -- a stale copy (new text still paired with the OLD hash) is caught in two
        independent places: the catalog integrity gate (fails to load, GuidelineIntegrityError) and
        the contract verification gate (safe_to_display flips to False with a named failure).
  (ii)  RE-BINDING -- once the catalog is updated to the revised record (new text + matching new
        hash + bumped version), it loads cleanly and bind("MEL") now serves the revised text/hash.
  (iii) BEFORE/AFTER CONTRACT DIFF -- for an affected melanoma case, the served guideline block is
        diffed old-vs-new (id, version, hash, text), which is exactly what an auditor would review
        to see what changed in the guidance a past decision relied on.

The update is clearly SIMULATED: the revised text is the original with an explicit
"[SIMULATED VERSION UPDATE -- governance demo, not real guidance]" marker appended, the version is a
"SIMULATED-..." string, and nothing is written to the corpus. This is a mechanism test, not a real
guideline edit and not authoritative clinical content.

Boundaries: zero paid calls, zero images, no disk writes to the corpus. Output:
research/evidence/derm7pt_v1/guideline_update_audit_demo.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.evidence.contract import build_contract  # noqa: E402
from medguard_v2.evidence.guideline_binding import (  # noqa: E402
    CATALOG_PATH,
    BindingResult,
    GuidelineCatalog,
    GuidelineIntegrityError,
    Recommendation,
)

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "guideline_update_audit_demo.json"

SIM_MARK = " [SIMULATED VERSION UPDATE -- governance demo, not real guidance]"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_raw_records() -> List[Dict]:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _guideline_block(binding: BindingResult) -> Dict:
    """The served-guideline view an auditor diffs (top recommendation only, for a clean diff)."""
    if not binding.covered or not binding.recommendations:
        return {"covered": binding.covered, "absence_reason": binding.absence_reason}
    top = binding.recommendations[0]
    return {
        "covered": True,
        "served_id": top.record_id,
        "source_version": top.source_version,
        "content_sha256": top.content_sha256,
        "citation_id": top.citation_id,
        "text": top.text,
    }


def build() -> Dict:
    raw_records = _read_raw_records()

    # ---- OLD catalog + the served melanoma recommendation ----------------------------------
    old_catalog = GuidelineCatalog(raw_records)
    old_binding = old_catalog.bind("MEL")
    assert old_binding.covered, "expected melanoma coverage in the catalog"
    served = old_binding.recommendations[0]
    target_id = served.record_id
    old_text = served.text
    old_hash = served.content_sha256
    old_version = served.source_version

    # locate the target's raw dict (by id) so we can build the updated variant
    target_idx = next(i for i, r in enumerate(raw_records) if str(r["id"]) == target_id)
    target_raw = raw_records[target_idx]

    # ---- construct the SIMULATED revised record --------------------------------------------
    new_text = old_text + SIM_MARK
    new_hash = _sha256(new_text)
    new_version = f"SIMULATED-{old_version}-rev2026"
    updated_raw = dict(target_raw)
    updated_raw["text"] = new_text
    updated_raw["content_sha256"] = new_hash
    updated_raw["source_version"] = new_version

    assert new_hash != old_hash, "revised text must produce a different hash"

    # ---- (i) DETECTION: catalog integrity gate on a STALE copy ------------------------------
    # Stale = the text was revised upstream but a cached record still carries the OLD hash.
    stale_raw = dict(updated_raw)
    stale_raw["content_sha256"] = old_hash  # new text, stale hash -> mismatch
    stale_records = list(raw_records)
    stale_records[target_idx] = stale_raw
    catalog_gate = {"fired": False}
    try:
        GuidelineCatalog(stale_records)
        catalog_gate["error"] = "integrity gate did NOT fire -- this is a bug"
    except GuidelineIntegrityError as exc:
        catalog_gate = {
            "fired": True,
            "exception": "GuidelineIntegrityError",
            "message": str(exc),
            "stale_record_id": target_id,
            "recomputed_hash_of_served_text": new_hash[:16] + "...",
            "stored_stale_hash": old_hash[:16] + "...",
        }

    # ---- (i) DETECTION: contract verification gate on a STALE served record -----------------
    # A previously-serialised contract could still carry the stale record even if the catalog were
    # bypassed. The contract re-verifies the hash itself, so it catches the drift independently.
    cases = load_cases()
    mel_case = next(c for c in cases if c.diagnosis_group == "MEL" and c.split == "test")
    valid_contract = build_contract(
        case_id=f"derm7pt-{mel_case.case_num}",
        classifier_diagnosis="MEL",
        classifier_confidence=0.90,
        criteria=dict(mel_case.criteria),
        catalog=old_catalog,
    )
    stale_rec = Recommendation(
        record_id=served.record_id,
        condition=served.condition,
        question_types=served.question_types,
        text=new_text,               # revised upstream
        citation_id=served.citation_id,
        source_version=new_version,
        authority_class=served.authority_class,
        content_sha256=old_hash,     # but still the stale hash
        license=served.license,
        distribution_status=served.distribution_status,
    )
    stale_binding = BindingResult(
        condition_group="MEL",
        catalog_condition=old_binding.catalog_condition,
        covered=True,
        recommendations=(stale_rec,),
        absence_reason=None,
    )
    stale_contract = dataclasses.replace(valid_contract, guideline=stale_binding)
    contract_gate = {
        "case_id": valid_contract.case_id,
        "safe_to_display_before_update": valid_contract.safe_to_display,
        "safe_to_display_with_stale_record": stale_contract.safe_to_display,
        "verification_failures_with_stale_record": stale_contract.verification_failures(),
        "fired": (not stale_contract.safe_to_display)
        and any("content hash" in f for f in stale_contract.verification_failures()),
    }

    # ---- (ii) RE-BINDING to the updated record ---------------------------------------------
    new_records = list(raw_records)
    new_records[target_idx] = updated_raw
    new_catalog = GuidelineCatalog(new_records)  # loads cleanly => integrity gate passes
    new_binding = new_catalog.bind("MEL")
    new_served = new_binding.recommendations[0]
    rebinding = {
        "new_catalog_loads": True,
        "n_records": len(new_catalog),
        "bind_MEL_now_serves": {
            "served_id": new_served.record_id,
            "source_version": new_served.source_version,
            "content_sha256": new_served.content_sha256,
            "hash_matches_text": new_served.verify(),
            "text_tail": new_served.text[-90:],
        },
        "rebound_to_same_record_id": new_served.record_id == target_id,
    }

    # ---- (iii) BEFORE/AFTER contract diff for the affected case -----------------------------
    contract_before = build_contract(
        case_id=f"derm7pt-{mel_case.case_num}",
        classifier_diagnosis="MEL",
        classifier_confidence=0.90,
        criteria=dict(mel_case.criteria),
        catalog=old_catalog,
    )
    contract_after = build_contract(
        case_id=f"derm7pt-{mel_case.case_num}",
        classifier_diagnosis="MEL",
        classifier_confidence=0.90,
        criteria=dict(mel_case.criteria),
        catalog=new_catalog,
    )
    before_block = _guideline_block(contract_before.guideline)
    after_block = _guideline_block(contract_after.guideline)
    diff = {
        "served_id_before": before_block["served_id"],
        "served_id_after": after_block["served_id"],
        "version_before": before_block["source_version"],
        "version_after": after_block["source_version"],
        "version_changed": before_block["source_version"] != after_block["source_version"],
        "hash_before": before_block["content_sha256"],
        "hash_after": after_block["content_sha256"],
        "hash_changed": before_block["content_sha256"] != after_block["content_sha256"],
        "text_changed": before_block["text"] != after_block["text"],
        "text_before": before_block["text"],
        "text_after": after_block["text"],
        "both_contracts_safe_to_display": bool(
            contract_before.safe_to_display and contract_after.safe_to_display
        ),
    }

    all_fired = catalog_gate["fired"] and contract_gate["fired"]

    return {
        "protocol": "derm7pt-guideline-update-audit-demo-n6-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "simulated": True,
        "scenario": (
            "In-memory simulation of a melanoma guideline VERSION UPDATE. The served recommendation "
            "aad:pmid:30392755:table-x-excision-first-line (version 2019) is revised upstream; its "
            "text and therefore its content hash change. The demo exercises the hash-mismatch "
            "detection, the re-binding, and the before/after contract diff. NOTHING is written to "
            "the corpus; the revised text is explicitly marked SIMULATED and is not real guidance."
        ),
        "boundaries": {
            "paid_calls": 0,
            "images_encoded": 0,
            "corpus_modified_on_disk": False,
            "catalog_path": str(CATALOG_PATH.relative_to(REPO)).replace("\\", "/"),
            "n_records": len(old_catalog),
        },
        "served_recommendation_before": {
            "served_id": target_id,
            "source_version": old_version,
            "content_sha256": old_hash,
            "text": old_text,
        },
        "simulated_update": {
            "new_source_version": new_version,
            "new_content_sha256": new_hash,
            "what_changed": "verbatim text revised (simulated marker appended) -> new sha256",
            "new_text": new_text,
        },
        "detection": {
            "catalog_integrity_gate": catalog_gate,
            "contract_verification_gate": contract_gate,
            "both_gates_fired": bool(all_fired),
            "note": (
                "Defense in depth: the catalog fails closed at LOAD (a stale record never "
                "instantiates), and the contract independently re-verifies the served hash before "
                "display (a stale record that reached a cached contract still cannot be shown)."
            ),
        },
        "rebinding": rebinding,
        "before_after_contract_diff": diff,
        "verdict": (
            "Governance loop demonstrated end to end: the hash-mismatch gate FIRED in both the "
            f"catalog ({catalog_gate['fired']}) and the contract ({contract_gate['fired']}) on the "
            "stale-hash version; the catalog re-bound cleanly to the updated record "
            f"(same id={rebinding['rebound_to_same_record_id']}, hash verifies="
            f"{rebinding['bind_MEL_now_serves']['hash_matches_text']}); and the before/after "
            f"contract diff surfaces the version change "
            f"({diff['version_before']} -> {diff['version_after']}), the hash change, and the exact "
            "text delta for the affected melanoma case. All done in memory; the corpus on disk is "
            "untouched."
        ),
    }


def main() -> None:
    t0 = time.perf_counter()
    r = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    d = r["detection"]
    print("\n=== N6 guideline-update audit demo ===")
    print(f"  served MEL record: {r['served_recommendation_before']['served_id']} "
          f"(v{r['served_recommendation_before']['source_version']})")
    print(f"  catalog integrity gate fired : {d['catalog_integrity_gate']['fired']}")
    print(f"  contract verify gate fired   : {d['contract_verification_gate']['fired']} "
          f"(safe_to_display {r['detection']['contract_verification_gate']['safe_to_display_before_update']} "
          f"-> {r['detection']['contract_verification_gate']['safe_to_display_with_stale_record']})")
    print(f"  re-bound to updated record   : {r['rebinding']['rebound_to_same_record_id']}, "
          f"hash verifies {r['rebinding']['bind_MEL_now_serves']['hash_matches_text']}")
    df = r["before_after_contract_diff"]
    print(f"  contract diff: version {df['version_before']} -> {df['version_after']}, "
          f"hash_changed {df['hash_changed']}, text_changed {df['text_changed']}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
