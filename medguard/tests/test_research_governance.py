import hashlib
import json
from pathlib import Path

from medguard.research.ledger import EvidenceLevel, RivalLedger


ROOT = Path(__file__).resolve().parents[2]


def test_rival_ledger_is_concise_full_text_verified_and_valid():
    """Every record here must be full-text audited with locally stored evidence.

    That is the property that makes any number in this ledger citable. The hard count
    is a tripwire: a record may only appear after its PDF is obtained, its text stored
    under papers/raw_texts/ with a sha256, and read. Search hits live in
    RIVAL_SCAN_QUEUE_*.json until then, and bumping this number without doing that work
    is the failure mode it exists to catch.
    """
    ledger = RivalLedger.load(ROOT / "research" / "RIVAL_LEDGER.json")
    assert not ledger.validate()
    assert len(ledger.records) == 10
    assert all(record.evidence_level is EvidenceLevel.VERIFIED for record in ledger.records)
    assert all(record.comparison_type == "reference-only" for record in ledger.records)
    assert {"mmed_rag_1", "rule_74", "dynarag_medical_135", "medgraphrag_133",
            "agentic_graph_rag_104",
            # 2026-08-08: the two closest works to MedGuard's provenance claim.
            "heg_tkg_139", "provenai_140"}.issubset(
        {record.rival_id for record in ledger.records})
    for record in ledger.records:
        for evidence in record.evidence:
            path = ROOT / evidence.locator
            assert path.exists()
            assert evidence.sha256, f"{record.rival_id} evidence must carry a sha256"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == evidence.sha256, (
                f"{record.rival_id}: {evidence.locator} changed since it was audited"
            )


def test_the_scan_queue_is_kept_separate_and_marked_uncitable():
    """Search hits must not leak into the audited ledger."""
    queue = json.loads((ROOT / "research" / "RIVAL_SCAN_QUEUE_2026-08-08.json").read_text(
        encoding="utf-8"))
    assert "NOT CITABLE" in queue["status"]
    audited = {r["rival_id"] for r in json.loads(
        (ROOT / "research" / "RIVAL_LEDGER.json").read_text(encoding="utf-8"))["records"]}
    queued = {c["rival_id"] for c in queue["candidates"]}
    graduated = {"provenai_2606_26449", "provenance_gap_clinical_2604_17114"}
    assert not (queued - graduated) & audited, (
        "a candidate cannot be in the queue and the audited ledger at the same time"
    )
    # The two withdrawn positioning claims must stay on record.
    withdrawn = {c["claim"] for c in queue["withdrawn_positioning_claims"]}
    assert any("ships code" in c for c in withdrawn)
    assert any("ECE" in c for c in withdrawn)


def test_preregistration_has_current_scope_metrics_and_novelty_boundary():
    protocol = json.loads((ROOT / "research" / "PREREGISTRATION.json").read_text(
        encoding="utf-8"))
    assert protocol["status"] == "design_frozen_artifacts_pending"
    assert protocol["scope"]["modality"] == "dermoscopy"
    assert protocol["scope"]["class_order"] == [
        "akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc"]
    assert protocol["primary_arm_roster"]["maximum_primary_arms"] == 6
    assert [arm["id"] for arm in protocol["ablation_study"]["arms"]] == [
        "A0", "A1", "A2", "A3", "A4"]
    assert protocol["scope"]["bcn_label_policy"]["actionable_binary"].startswith(
        "Retain SCC")
    assert protocol["primary_outcomes"]["diagnosis"] == "macro_f1"
    # v4 endpoint re-registration + v5 competed diagnosis + v6 endpoint classification,
    # each retained by the next rather than replaced.
    assert protocol["schema_version"] == 6
    assert "faithfulness" not in protocol["primary_outcomes"]
    assert protocol["primary_outcomes"]["provenance"] == "citation_integrity"
    assert protocol["primary_outcomes"]["actionable_fixed_specificity_level"] == 0.9
    assert protocol["statistics"]["alpha"] == 0.05
    assert protocol["statistics"]["bootstrap_replicates"] == 2000
    assert "ragas" in protocol["demoted_from_primary"]["faithfulness"].lower()
    # v5: diagnosis is a competed axis (superiority of the M-family over the B0 VLM).
    assert "superiority" in protocol["primary_outcomes"]["diagnosis_hypothesis"].lower()
    boundary = " ".join(protocol["novelty_boundary"]).lower()
    assert "not novel" in boundary and "provenance" in boundary
    assert "no paid generation" in protocol["authorization"].lower()


def test_experiment_spec_pins_design_and_keeps_unfrozen_artifacts_explicit():
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V3.json").read_text(
        encoding="utf-8"))
    assert spec["status"] == "design_frozen_runtime_lock_pending"
    assert spec["development_h2h_manifest_rule"]["expected_lesion_counts"]["total"] == 160
    assert [arm["id"] for arm in spec["ablation_matrix"]] == [
        "A0", "A1", "A2", "A3", "A4", "B1"]
    assert spec["scope"]["actionable_labels_bcn"] == [
        "akiec", "bcc", "melanoma", "scc"]
    freeze = {row["artifact"]: row["status"]
              for row in spec["artifact_freeze_ledger"]}
    assert freeze["dataset manifests"] == "frozen"
    assert freeze["development H2H manifest"] == "frozen"
    # The lock is not closeable while a scorer for a live endpoint is absent.
    assert freeze["RAGAS judge and evidence scorer versions"] == "pending"
    assert freeze["sealed-test access log"] == "pending_and_must_remain_empty"
    assert spec["current_generation_authorization"] == "none"
    assert spec["current_sealed_test_authorization"] == "none"


def test_v3_spec_records_the_adaptive_slot_the_defects_and_the_corpus_expansion():
    """v3 must carry v2's records forward and add the guideline-subset expansion.

    A spec that misdescribes a decision already taken launders a post-hoc choice
    as a pre-registered one; v2's D2 error ("no management claim backable") is
    corrected here rather than left standing.
    """
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V3.json").read_text(
        encoding="utf-8"))
    slot = spec["primary_h2h_roster"]["adaptive_slot"]
    assert slot["status"] == "selected"
    assert slot["selected"] == "B4"
    assert "not a full DynaRAG reproduction" in slot["selection_boundary"]
    declared = {row["id"] for row in spec["known_blocking_defects"]}
    assert {"D1", "D2", "D3", "D4", "D5"}.issubset(declared)
    # The expansion is recorded with both hashes so it is auditable.
    expansion = spec["supersession"]["recommendation_corpus"]
    assert expansion["records"] == 28 and expansion["was"] == 23
    assert spec["supersedes"] == "research/EXPERIMENT_SPEC_V2.json"


def test_historical_spec_receipts_remain_byte_identical():
    """v1, v2 and v3 are immutable historical anchors once superseded.

    v1's receipt is pinned by five frozen artifacts; editing it would invalidate
    the evidence bundle, the perception artifact and the development manifest at
    once. v3's receipt records the pre-v4 design (v3 spec + the pre-v4
    PREREGISTRATION.json); the CURRENT prereg is pinned by the v4 receipt, not this
    one, so v3's receipt must stay byte-identical even though the prereg has moved on.
    """
    anchors = {
        "research/EXPERIMENT_SPEC_V1.receipt.json":
            "5a990ece1c41b21d491b4919b5f31817c1522306846fbcd0058c84d1a05bbfd8",
        "research/EXPERIMENT_SPEC_V3.receipt.json":
            "8024d3708e2b98c7ba174d68ee67779679bb863745546aab519f515122b5d7e8",
        "research/EXPERIMENT_SPEC_V4.receipt.json":
            "1299f41046aeafd1c4e4f3a83cc161e997d4a93f04b39836082977fd534a0044",
    }
    for path, digest in anchors.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, (
            f"{path} changed. It is a historical anchor and must stay byte-identical; "
            f"record forward decisions in a new version instead."
        )
    # The supersession chain stays declared and self-consistent.
    v2 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V2.json").read_text(encoding="utf-8"))
    assert v2["supersedes"] == "research/EXPERIMENT_SPEC_V1.json"
    v3 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V3.json").read_text(encoding="utf-8"))
    assert v3["supersedes"] == "research/EXPERIMENT_SPEC_V2.json"
    for locator in v3["supersession"]["builders_still_read_v1"]:
        assert (ROOT / locator).is_file()
    v4 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V4.json").read_text(encoding="utf-8"))
    assert v4["supersedes"] == "research/EXPERIMENT_SPEC_V3.json"
    v5 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V5.json").read_text(encoding="utf-8"))
    assert v5["supersedes"] == "research/EXPERIMENT_SPEC_V4.json"
    v6 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    assert v6["supersedes"] == "research/EXPERIMENT_SPEC_V5.json"


def test_v6_receipt_design_binding_pins_match_the_files_on_disk():
    """v6 is current: the design tier must match disk; the observed tier may drift.

    v1 pinned the append-only results ledger and the working notes as
    design-binding, so ordinary appends invalidated it. Splitting the tiers keeps
    the freeze meaningful; the design tier must hold, the observed tier may drift.
    """
    receipt = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.receipt.json").read_text(
        encoding="utf-8"))
    binding = {row["path"] for row in receipt["design_binding_files"]}
    assert "research/EXPERIMENT_SPEC_V6.json" in binding
    assert "research/PREREGISTRATION.json" in binding
    for row in receipt["design_binding_files"]:
        raw = (ROOT / row["path"]).read_bytes()
        assert len(raw) == row["bytes"], f"{row['path']} size drifted from the v6 receipt"
        assert hashlib.sha256(raw).hexdigest() == row["sha256"], (
            f"{row['path']} is design-binding and changed since the v6 receipt was "
            f"created. Issue v7 and state the reason rather than editing v6 in place."
        )
    observed = {row["path"] for row in receipt["observed_files"]}
    assert "research/RESULTS_LEDGER.json" in observed, (
        "The results ledger is append-only and cannot be design-binding."
    )
    assert not (observed & binding), "a file cannot be both design-binding and observed"


def test_v5_receipt_is_now_historical_and_says_why():
    """v5 pinned the preregistration, which schema 6 changes.

    Per v5's own design_binding_rule that makes its receipt historical rather than
    broken. The v5 SPEC file itself must still be byte-identical to what v5 pinned --
    a superseded freeze is still a freeze, and editing it in place would erase the
    record of what was actually registered when the development H2H ran.
    """
    v5 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V5.receipt.json").read_text(
        encoding="utf-8"))
    v6 = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.receipt.json").read_text(
        encoding="utf-8"))
    assert v6["supersedes"] == "research/EXPERIMENT_SPEC_V5.receipt.json"
    assert "could not be satisfied as written" in v6["supersession_reason"]
    superseded = {row["path"]: row for row in v6["historical_versions"]}
    assert "research/EXPERIMENT_SPEC_V5.receipt.json" in superseded

    spec_pin = next(r for r in v5["design_binding_files"]
                    if r["path"] == "research/EXPERIMENT_SPEC_V5.json")
    raw = (ROOT / "research" / "EXPERIMENT_SPEC_V5.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec_pin["sha256"], (
        "EXPERIMENT_SPEC_V5.json changed after being superseded. A superseded spec is "
        "the record of what was registered when the development H2H ran; issue a new "
        "version instead of editing it."
    )


def test_v6_repairs_the_unsatisfiable_acceptance_condition():
    """v6's whole reason to exist: the v5 condition could not be met as written.

    Two of its four endpoints are undefined for a no-contract comparator and may not be
    imputed as zero, so only two were testable and the condition silently required both.
    """
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    assert spec["schema_version"] == "medguard-experiment-spec-v6"
    assert spec["supersedes"] == "research/EXPERIMENT_SPEC_V5.json"
    assert spec["current_generation_authorization"] == "none"
    assert spec["current_sealed_test_authorization"] == "none"

    prereg = json.loads((ROOT / "research" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert prereg["schema_version"] == 6
    assert "superiority" in prereg["primary_outcomes"]["diagnosis_hypothesis"].lower()

    # The families must agree between spec and preregistration.
    families = spec["registered_families"]
    assert (prereg["registered_inferential_families"]["evidence_use"]
            == families["evidence_use"]["members"])
    assert (prereg["registered_inferential_families"]["alignment"]
            == families["alignment"]["members"])


def test_v6_families_contain_endpoints_medguard_loses():
    """The anti-cherry-picking guard, enforced rather than promised.

    A family assembled after seeing the numbers is only credible if the rule that built
    it also admitted the losses. If someone later prunes the losing endpoints, this
    fails.
    """
    from medguard.eval.paired_endpoints import (
        V6_ALIGNMENT_FAMILY, V6_EVIDENCE_USE_FAMILY,
    )
    losing = {"ragas_answer_relevancy", "completeness", "copy_reference"}
    assert losing <= set(V6_ALIGNMENT_FAMILY), (
        "the alignment family must retain the endpoints MedGuard loses"
    )
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    assert losing <= set(spec["registered_families"]["alignment"]["members"])
    assert list(V6_EVIDENCE_USE_FAMILY) == spec["registered_families"]["evidence_use"]["members"]
    assert spec["honesty_declaration"]["written_after_seeing_development_numbers"] is True


def test_v6_bars_architectural_endpoints_from_every_family():
    """MedGuard's best numbers are its least inferential; they may not be tested."""
    from medguard.eval.paired_endpoints import (
        ARCHITECTURAL_ENDPOINTS, V6_INFERENTIAL_FAMILY,
    )
    assert not (set(ARCHITECTURAL_ENDPOINTS) & set(V6_INFERENTIAL_FAMILY))
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    declared = set(spec["endpoint_classification"]["architectural_reported_as_counts"])
    assert {"citation_integrity", "contract_completion"} <= declared
    assert not (declared & set(V6_INFERENTIAL_FAMILY))


def test_v6_closes_the_over_reliance_by_abstaining_hole():
    """A1 abstains on everything and therefore 'beats' MedGuard on over-reliance.

    An endpoint that rewards refusing to work needs a coverage guard, and the guard has
    to bind MedGuard too -- its own over-reliance advantage is partly bought by
    abstaining on 38 lesions.
    """
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    guard = spec["registered_families"]["alignment"]["over_reliance_coverage_guard"]
    assert "never answers scores a perfect 0.0" in guard["the_hole"]
    assert "AURC" in guard["registered_rule"]
    assert "constrains MedGuard" in guard["why_this_is_not_a_patch_to_protect_medguard"]
    assert any("over-reliance improvement claimed without joint coverage" in s
               for s in spec["stop_conditions"])


def test_v6_records_the_ablation_negatives_rather_than_only_the_win():
    """An ablation table where every component helps is a table nobody really ran."""
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    findings = spec["development_ablation_findings"]
    assert findings["A2_no_graph_relations"]["reporting"].startswith("honest null")
    # The verification-gate win must carry its cost in the same record.
    assert "coverage 0.7625 vs 0.9000" in findings["A4_no_verifier_gating"]["cost"]


def test_the_withdrawn_a3_negative_stays_withdrawn():
    """A3 removed nothing: remediation fired on 0/160, so A3 IS M.

    The retracted claim has to stay retracted in writing. If someone later re-reads the
    -0.0489 context-recall difference as a remediation effect, this fails.
    """
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    a3 = spec["development_ablation_findings"]["A3_no_bounded_remediation"]
    assert "WITHDRAWN" in a3["correction_2026_08_08"]
    assert len(a3["why_it_was_wrong"]) == 3
    assert "0 of 160" in a3["correct_finding"] or "0 lesions" in a3["correct_finding"]
    assert a3["where_remediation_IS_evaluated"] == "failure_typed_suite (see below)"


def test_the_a2_null_is_recorded_with_its_mechanism():
    """A bare null invites "they built a graph and it did nothing".

    The accurate finding is narrower: the graph acts only on the free-text channel, and
    the load-bearing channels are hash-bound and not graph-mediated. If the mechanism is
    ever dropped and only the null survives, this fails.
    """
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    mech = spec["graph_lineage_null_mechanism"]
    changed = mech["what_A2_actually_changed"]
    # The load-bearing channels must be recorded as untouched.
    assert changed["guideline_citations_identical"] == "160/160"
    assert changed["image_exemplars_identical"] == "160/160"
    assert changed["decisions_identical"] == "160/160"
    assert changed["pubmed_text_items_differing"] == 36
    assert "cannot satisfy an obligation" in mech["why_that_could_not_change_an_outcome"]
    # The null and the mechanism must be reported together, not separately.
    assert "Both, together." in mech["reporting_rule"]
    # The rival graph arm must not be collateral damage.
    assert "B5" in mech["does_not_affect_B5"]


def test_contribution_two_drops_the_graph_claim_and_keeps_the_hash_binding():
    """The restatement must narrow the claim, and the original must stay visible."""
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    restated = spec["contribution_claims_vs_evidence"]["contribution_2_restated"]
    assert "KG" in restated["was"], "the original wording must be preserved verbatim"
    assert "HASH-BOUND" in restated["now"]
    assert "KG" not in restated["now"] and "graph" not in restated["now"]
    assert "DROPPED, not defended" in restated["what_changed_and_why"]
    # It must now cite evidence that actually supports it.
    assert any("citation resolvability" in e for e in restated["evidence_now_cited_for_it"])


def test_remediation_is_evidenced_by_the_failure_suite_not_the_h2h():
    """The contribution survives, but only on the evidence that can actually bear it."""
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    base = spec["remediation_evidence_base"]
    results = base["results_real_retrieval"]
    assert results["F1_sparse_missing_evidence_EXPAND"]["recovery"] == 0.8875
    assert results["F2_query_evidence_misalignment_REFORMULATE"]["recovery"] == 0.8125
    assert results["F3_source_contradiction_INDEPENDENT_SOURCE"]["recovery"] == 0.4688
    # F4 must abstain; "recovering" from an image-evidence conflict would be the failure.
    assert results["F4_image_evidence_conflict_ABSTAIN"]["recovery"] == 0.0
    # The weakest action and the constructed-failure caveat must both be stated.
    assert any("0.4688" in c for c in base["mandatory_caveats"])
    assert any("CONSTRUCTED" in c for c in base["mandatory_caveats"])
    assert "0 of 160" in base["registered_contribution_wording"]
    for path in base["artifacts"]:
        assert (ROOT / path).is_file(), f"{path} is cited as evidence but is missing"


def test_v6_registers_the_judge_noise_floor_as_binding():
    """A significant difference between two copies of one system needs a floor."""
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    floor = spec["judge_noise_floor"]
    assert floor["measured_floor"]["ragas_context_recall"] == 0.0489
    assert "INDISTINGUISHABLE FROM JUDGE NOISE" in floor["registered_rule"]
    # Host-computed endpoints must stay exempt, and the floor must not be carried over.
    assert "deterministic and exempt" in floor["registered_rule"]
    assert "re-measured on the sealed run" in floor["registered_rule"]
    assert "not a general claim about RAGAS" in floor["scope_limit"]
    assert any("test-retest floor" in s for s in spec["stop_conditions"])


def test_v6_keeps_the_development_verdict_as_measured():
    """v6 must not retroactively rescue the development result."""
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V6.json").read_text(encoding="utf-8"))
    binding = spec["development_result_binding_statement"]
    assert "NOT met against B1" in binding["verdict_under_v5"]
    assert binding["must_appear_in_the_manuscript"] is True
    assert binding["may_not_be_restated_under_v6"] is True


def test_v5_spec_remains_a_valid_historical_registration():
    """v5 is now a historical anchor and validates its own frozen content."""
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V5.json").read_text(encoding="utf-8"))
    assert spec["schema_version"] == "medguard-experiment-spec-v5"
    assert spec["current_generation_authorization"] == "none"
    assert spec["current_sealed_test_authorization"] == "none"
    assert spec["supersession"]["recommendation_corpus"]["records"] == 66
    assert spec["matched_conditions"]["same_frozen_perception_output"] is False
    assert spec["matched_conditions"]["frozen_perception_output_shared_by_m_family"] is True
    assert "superiority" in spec["endpoints"]["primary"]["diagnosis_hypothesis"].lower()
    assert set(spec["generation_runtime_v5"]["defect_resolutions"]) == {"A2", "D4", "D3"}
    # The defect v6 exists to repair must stay recorded in v5.
    assert "registration_defect_2026_08_08" in spec


def test_v4_spec_reregisters_endpoints_and_records_the_corpus_expansion():
    """v4 completes the endpoint registration and the actinic-keratosis expansion.

    A spec that left RAGAS primary while the preregistration demoted it, or that
    still said 28 records, would launder a post-hoc choice as pre-registered. The
    spec and the prereg must agree.
    """
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V4.json").read_text(encoding="utf-8"))
    assert spec["schema_version"] == "medguard-experiment-spec-v4"
    assert spec["current_generation_authorization"] == "none"
    assert spec["current_sealed_test_authorization"] == "none"
    exp = spec["supersession"]["recommendation_corpus"]
    assert exp["records"] == 33 and exp["was"] == 28
    primary = spec["endpoints"]["primary"]
    assert "faithfulness" not in primary
    assert primary["provenance"] == "citation_integrity"
    assert primary["actionable_fixed_specificity_level"] == 0.9
    assert spec["endpoints"]["supporting"][0] == "ragas_faithfulness_demoted_from_primary"
    assert spec["statistics"]["alpha"] == 0.05 and spec["statistics"]["bootstrap_replicates"] == 2000
    defects = {d["id"]: d for d in spec["known_blocking_defects"]}
    assert "130 of 160" in defects["D1"]["consequence"]
    assert "14 of 29" in defects["D5"]["defect"]
    # v4 is now a historical anchor: it validates its own frozen content. The CURRENT
    # preregistration is pinned by the v5 receipt and cross-checked in the v5 test.


def test_implementation_cards_move_into_the_freeze_at_the_lock():
    """The cards are living now and binding later; the receipt must say when."""
    receipt = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V5.receipt.json").read_text(
        encoding="utf-8"))
    observed = {row["path"] for row in receipt["observed_files"]}
    assert "research/implementation_cards/H2H_IMPLEMENTATION_CARDS_V1.json" in observed
    assert "pre-H2H lock" in receipt["observed_rule"]
    cards = json.loads((
        ROOT / "research" / "implementation_cards" / "H2H_IMPLEMENTATION_CARDS_V1.json"
    ).read_text(encoding="utf-8"))
    assert cards["document_status"] == "living_prelock_specification"


def test_development_h2h_selection_rule_reproduces_expected_lesion_counts():
    spec = json.loads((ROOT / "research" / "EXPERIMENT_SPEC_V1.json").read_text(
        encoding="utf-8"))
    parent = json.loads((
        ROOT / spec["development_h2h_manifest_rule"]["parent"]
    ).read_text(encoding="utf-8"))
    labels = spec["scope"]["class_order"]
    expected = spec["development_h2h_manifest_rule"]["expected_lesion_counts"]
    by_label = {label: set() for label in labels}
    for record in parent["records"]:
        assert record["split"] != "test"
        if record["split"] == "calibration":
            by_label[record["label"]].add(record["cluster_id"])
    selected = {}
    for label in labels:
        ranked = sorted(
            by_label[label],
            key=lambda cluster_id: hashlib.sha256(
                f"20260716:{cluster_id}".encode("utf-8")).hexdigest(),
        )
        selected[label] = ranked[:min(30, len(ranked))]
        assert len(selected[label]) == expected[label]
    assert sum(map(len, selected.values())) == expected["total"]
