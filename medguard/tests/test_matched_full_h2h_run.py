"""The generation-executing H2H runner, driven entirely by a fake LLM client.

Every test here injects a fake ``LLMClient`` that returns canned outputs and makes
ZERO paid calls.  The runner is exercised end to end: per-arm rows, the B0-VLM
diagnosis path, M/A0 alias reuse, byte-identical cache replay, every scorer
producing a value, citation integrity scoring only guideline citations, and the
D3 independent-source action firing and changing the evidence set.
"""
import hashlib
import json

import pytest

from medguard.evidence.guideline_claims import build_guideline_claim_assets
from medguard.evidence.source_contradiction import (
    ContradictionFlag,
    SourceContradictionRegistry,
)
from medguard.pipeline.generator import VLMGenerator
from medguard.pipeline.guideline_retriever import FrozenGuidelineRetriever
from medguard.eval.h2h_protocol import ArmId
from medguard.eval.matched_full_h2h_run import (
    DEFAULT_ARMS,
    H2HRunCase,
    H2HRunDependencies,
    _RunContractBuilder,
    _RunRemediationTools,
    run_h2h,
)
from medguard.tests.c1_fixtures import perception


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


VALID_CONTRACT = {
    "claims": [{
        "claim_id": "c1", "claim_type": "diagnosis",
        "text": "The lesion is most consistent with melanoma.",
        "image_support": "supports", "target_concept": "melanoma",
        "target_state": "present", "image_finding_ids": ["f_commitment"],
        "evidence_ids": [], "entailment": "not_assessed", "contradiction": False,
    }],
    "unresolved_uncertainties": [],
}


class FakeLLMClient:
    """A scripted OpenAI-compatible client. Dispatches on prompt content."""

    def __init__(self, *, model="fake/vlm-32b", b0_label="melanoma",
                 malform_contract_calls=0):
        self.model = model
        self.call_log = []
        self.vision_prompts = []
        self.b0_label = b0_label
        self.malform_contract_calls = malform_contract_calls
        self._contract_calls = 0

    def vision(self, prompt, image_path=None, *, system="", max_tokens=512,
               temperature=0.2, seed=None):
        self.vision_prompts.append(prompt)
        self.call_log.append({
            "succeeded": True, "usage_available": True, "prompt_tokens": 5,
            "completion_tokens": 7, "total_tokens": 12, "latency_seconds": 0.01,
            "seed": seed,
        })
        if "single most likely diagnosis" in prompt:
            return json.dumps({"diagnosis": self.b0_label, "rationale": "asymmetric network"})
        if "EvidenceContract" in prompt:
            self._contract_calls += 1
            if self._contract_calls <= self.malform_contract_calls:
                return "definitely not json {{{"
            return json.dumps(VALID_CONTRACT)
        return "Grounded answer consistent with the committed diagnosis."


# --- fake evidence records ---------------------------------------------------------
def _text_record(doc, text, eligible=()):
    return {
        "id": doc, "document_id": doc, "citation_id": f"PMID:{doc}",
        "source_id": f"pmid:{doc}", "source": f"pmid:{doc}", "source_version": "2020",
        "authority": "peer_reviewed_secondary", "authority_class": "peer_reviewed_secondary",
        "text": text, "retrieval_channel": "dense", "content_sha256": _sha(text),
        "record_sha256": "d" * 64, "eligible_claim_types": list(eligible),
        "provenance_uri": f"https://pubmed/{doc}",
    }


def _image_record(doc, label, text):
    return {
        "id": doc, "document_id": doc, "citation_id": f"ISIC:{doc}",
        "source_id": f"isic:{doc}", "source": f"isic:{doc}",
        "source_version": "ham10000-train", "authority": "labeled_training_exemplar",
        "authority_class": "labeled_training_exemplar", "text": text,
        "retrieval_channel": "image_exemplar", "label": label, "split": "train",
        "content_sha256": _sha(text), "record_sha256": "e" * 64,
        "image_sha256": "f" * 64, "provenance_uri": f"https://isic/{doc}",
        "eligible_claim_types": ["image_exemplar_support"],
    }


def _guideline_record(doc, condition, text, eligible=("management",), source="aad"):
    return {
        "authority_class": "clinical_practice_guideline", "citation_id": f"{source.upper()}:{doc}",
        "condition": condition, "content_form": "verbatim_recommendation_text_projection",
        "content_sha256": _sha(text), "document_id": doc, "eligible_claim_types": list(eligible),
        "record_sha256": _sha("record:" + doc), "source_id": source,
        "source_pdf_sha256": "2" * 64, "source_version": "2019", "text": text,
        "section": "3.1", "article_page": 12,
    }


_GUIDELINE_RECORDS = [
    _guideline_record("aad:mel:1", "melanoma", "Excise suspected melanoma completely."),
    _guideline_record("aad:mel:2", "melanoma", "Refer melanoma to a specialist MDT.",
                      eligible=("referral",)),
    _guideline_record("nice:mel:3", "melanoma", "Assess pigmented lesions with dermoscopy.",
                      eligible=("assessment",), source="nice"),
    _guideline_record("aad:bcc:1", "basal_cell_carcinoma", "Excise BCC with a clear margin."),
    _guideline_record("nice:bcc:2", "basal_cell_carcinoma", "Offer surgical excision for BCC.",
                      source="nice"),
]


def _make_deps(tmp_path, *, client=None, contradiction_registry=None,
               text_records=None, image_for=None):
    client = client or FakeLLMClient()
    generator = VLMGenerator(client, max_schema_repairs=1, seed=0)
    retriever = FrozenGuidelineRetriever(_GUIDELINE_RECORDS)
    catalog, registry = build_guideline_claim_assets(retriever.records)

    default_text = text_records or [
        _text_record("pm:1", "Dermoscopy features of the lesion are described."),
        _text_record("pm:2", "Management overview for pigmented lesions."),
    ]

    def text_retrieve(query, k):
        return [dict(r) for r in default_text][:k]

    def image_retrieve(paths, k):
        label = image_for(paths[0]) if image_for else "melanoma"
        return [_image_record(f"ex:{label}", label, f"{label} exemplar text")][:k]

    def baseline_retrieve(arm_id, case):
        if arm_id == "B6":
            return [], True  # B6 abstains under its bounded loop
        return [_text_record("pm:base", "Baseline retrieved context.")], False

    return H2HRunDependencies(
        generator=generator, cache_root=str(tmp_path / "gen_cache"),
        guideline_retriever=retriever, guideline_catalog=catalog, guideline_registry=registry,
        text_retrieve=text_retrieve, image_retrieve=image_retrieve,
        query_for=lambda dx: f"dermoscopy {dx} diagnosis differential evidence",
        reformulated_query_for=lambda dx: f"{dx} dermoscopic diagnostic criteria evidence",
        baseline_retrieve=baseline_retrieve,
        contradiction_registry=contradiction_registry or SourceContradictionRegistry(),
    )


def _case(cluster, diagnosis):
    return H2HRunCase(
        case_id=f"lesion:{cluster}", cluster_id=cluster,
        perception=perception(diagnosis=diagnosis),
        ordered_view_case_ids=(f"{cluster}_v0",),
        ordered_image_paths=(f"img_{cluster}.jpg",),
    )


def _standard_cases():
    # case1 melanoma with a matching exemplar (M passes); case2 bcc with a mismatched
    # exemplar (M abstains -- absence of corroboration is not a free PASS).
    return [_case("c1", "melanoma"), _case("c2", "bcc")]


def _image_for(path):
    return "melanoma" if "c1" in path else "nevus"


TRUTH = {"c1": "melanoma", "c2": "bcc"}


# --- the whole runner behind a fake client ----------------------------------------
def test_full_run_produces_scored_rows_for_every_arm(tmp_path):
    client = FakeLLMClient()
    deps = _make_deps(tmp_path, client=client, image_for=_image_for)
    report = run_h2h(_standard_cases(), deps, TRUTH, n_boot=200)

    # Every arm produced a scored block.
    per_arm = report["scored"]["per_arm"]
    assert set(per_arm) == {arm.value for arm in DEFAULT_ARMS}
    assert report["aggregate"]["arm_rows"] == len(DEFAULT_ARMS) * 2

    m = per_arm["M"]
    b0 = per_arm["B0"]
    # Diagnosis: M commits the probe label (100% here); B0 uses the VLM (50%).
    assert m["diagnosis"]["accuracy"] == 1.0
    assert b0["diagnosis"]["accuracy"] == 0.5
    # Every endpoint is wired and produces a value for M.
    assert m["actionable"]["confusion_matrix"]
    assert m["selective"]["aurc"] is not None
    assert m["selective"]["coverage"] == 0.5  # passed c1, abstained c2
    assert m["provenance_citation_integrity"]["integrity_rate"] == 1.0
    assert m["process_contract_completion"]["applicable"] is True
    assert m["diagnostic_accuracy_ci"]["estimate"] == 1.0
    assert m["ragas_supporting"]["status"] == "mapped_not_executed"
    # B1 process is not_applicable, never zero.
    assert per_arm["B1"]["process_contract_completion"]["status"] == "not_applicable"

    # Zero paid calls: the only client is the injected fake.
    assert deps.generator.client is client
    assert report["boundaries"]["sealed_test_access"] is False
    assert all(event["succeeded"] for event in client.call_log)


def test_b0_diagnoses_via_the_vlm_and_is_scored(tmp_path):
    client = FakeLLMClient(b0_label="melanoma")
    deps = _make_deps(tmp_path, client=client, image_for=_image_for)
    report = run_h2h([_case("c1", "melanoma"), _case("c2", "bcc")], deps, TRUTH,
                     arms=[ArmId.B0], n_boot=100)
    b0 = report["scored"]["per_arm"]["B0"]
    # The VLM label (melanoma) is right on c1, wrong on c2 (truth bcc).
    assert b0["diagnosis"]["accuracy"] == 0.5
    # B0 has no probe confidence or contract.
    assert b0["selective"]["aurc"] is None
    assert b0["process_contract_completion"]["status"] == "not_applicable"


def test_m_and_a0_alias_reuse_one_generation(tmp_path):
    deps = _make_deps(tmp_path, image_for=_image_for)
    report = run_h2h([_case("c1", "melanoma")], deps, TRUTH,
                     arms=[ArmId.M, ArmId.A0], n_boot=50)
    m_row = report["cases"][0]["arms"]["M"]
    a0_row = report["cases"][0]["arms"]["A0"]
    # A0 is a declared alias: identical contract, zero marginal generation.
    assert a0_row["alias_of"] == "M"
    assert a0_row["contract_sha256"] == m_row["contract_sha256"]
    assert m_row["generation"]["remote_calls"] == 1
    assert a0_row["generation"]["remote_calls"] == 0
    assert a0_row["generation"]["cache_hit"] is True
    # And the scored A0 arm carries the alias annotation.
    assert report["scored"]["per_arm"]["A0"]["alias_of"] == "M"


def test_cache_replay_is_byte_identical(tmp_path):
    # First run populates the cache; a second run over the same root replays it.
    client_one = FakeLLMClient()
    deps_one = _make_deps(tmp_path, client=client_one, image_for=_image_for)
    first = run_h2h([_case("c1", "melanoma")], deps_one, TRUTH, arms=[ArmId.M], n_boot=50)

    client_two = FakeLLMClient()
    deps_two = _make_deps(tmp_path, client=client_two, image_for=_image_for)
    second = run_h2h([_case("c1", "melanoma")], deps_two, TRUTH, arms=[ArmId.M], n_boot=50)

    m_first = first["cases"][0]["arms"]["M"]
    m_second = second["cases"][0]["arms"]["M"]
    assert m_first["generation"]["cache_hit"] is False
    assert m_second["generation"]["cache_hit"] is True
    assert m_second["generation"]["remote_calls"] == 0
    # The replayed contract is byte-identical, and the second run made no VLM call.
    assert m_second["contract_sha256"] == m_first["contract_sha256"]
    assert client_two.vision_prompts == []


def test_schema_repair_path_is_exercised(tmp_path):
    client = FakeLLMClient(malform_contract_calls=1)
    deps = _make_deps(tmp_path, client=client, image_for=_image_for)
    report = run_h2h([_case("c1", "melanoma")], deps, TRUTH, arms=[ArmId.M], n_boot=50)
    m_row = report["cases"][0]["arms"]["M"]
    # One malformed response then a valid one: the generator repaired once.
    assert m_row["generation"]["schema_repairs"] == 1
    assert m_row["generation"]["remote_calls"] == 2
    assert m_row["generation_error"] is None
    assert m_row["final_action"] == "PASS"


def test_citation_integrity_scores_only_guideline_citations(tmp_path):
    deps = _make_deps(tmp_path, image_for=_image_for)
    report = run_h2h([_case("c1", "melanoma")], deps, TRUTH,
                     arms=[ArmId.M, ArmId.B1], n_boot=50)
    per_arm = report["scored"]["per_arm"]
    # M cites guideline records (all intact); the PubMed/exemplar evidence it also
    # carries is NOT scored against the guideline catalog.
    m_cite = per_arm["M"]["provenance_citation_integrity"]
    assert m_cite["total_citations"] >= 1
    assert m_cite["integrity_rate"] == 1.0
    m_row = report["cases"][0]["arms"]["M"]
    assert all(cid.startswith(("aad:", "nice:")) for cid in m_row["guideline_citations"])
    # B1 has no contract and no guideline citations -> None, not a free perfect score.
    assert per_arm["B1"]["provenance_citation_integrity"]["integrity_rate"] is None


def test_arm_that_cannot_corroborate_abstains_rather_than_free_passes(tmp_path):
    # A1 removes image-exemplar retrieval, so it cannot corroborate the commitment
    # and must abstain -- deleting evidence is not an improvement.
    deps = _make_deps(tmp_path, image_for=_image_for)
    report = run_h2h([_case("c1", "melanoma")], deps, TRUTH,
                     arms=[ArmId.M, ArmId.A1], n_boot=50)
    m_row = report["cases"][0]["arms"]["M"]
    a1_row = report["cases"][0]["arms"]["A1"]
    assert m_row["final_action"] == "PASS"
    assert a1_row["final_action"] == "ABSTAIN"
    assert a1_row["generation"]["remote_calls"] == 0


def test_no_paid_call_ever_reaches_a_real_gateway(tmp_path):
    # The runner only ever touches the injected fake; construct the real client's
    # namespace nowhere. This is a guard, not a network test.
    client = FakeLLMClient()
    deps = _make_deps(tmp_path, client=client, image_for=_image_for)
    run_h2h(_standard_cases(), deps, TRUTH, n_boot=100)
    # Every logged call is a fake success; the fake never authenticates or connects.
    assert client.call_log
    assert not hasattr(client, "api_key")


# --- D3: INDEPENDENT_SOURCE fires and changes the evidence set ---------------------
def test_d3_independent_source_fires_and_changes_the_evidence_set(tmp_path):
    from medguard.remediation.actions import Action
    from medguard.remediation.policy import DeterministicRemediationPolicy
    from medguard.remediation.runner import run_remediation

    flagged = _text_record("pm:bad", "This abstract disputes excision for melanoma.",
                           eligible=("assessment",))
    registry = SourceContradictionRegistry([
        ContradictionFlag(condition="melanoma", document_id="pm:bad"),
    ])
    deps = _make_deps(
        tmp_path, contradiction_registry=registry,
        text_records=[flagged, _text_record("pm:ok", "Neutral background context.")],
        image_for=lambda path: "melanoma",
    )
    deps.guideline_k = 2  # initial guideline tier smaller than the melanoma catalog
    case = _case("c1", "melanoma")

    builder = _RunContractBuilder(
        case, deps, image_exemplar_retrieval=True, graph_lineage=True,
    )
    contract, _chunks = builder.build_initial()
    initial_docs = {item.document_id for item in contract.evidence}
    # The flagged PubMed record is present and the contract detects a contradiction.
    assert "pm:bad" in initial_docs

    tools = _RunRemediationTools(builder)
    result = run_remediation(
        contract, DeterministicRemediationPolicy(), tools, max_steps=4,
    )
    final_docs = {item.document_id for item in result.contract.evidence}

    assert "INDEPENDENT_SOURCE" in tools.applied
    assert result.final_action is Action.PASS
    # The contradicted source is dropped and an independent guideline source is added.
    assert "pm:bad" not in final_docs
    independent = final_docs - initial_docs
    assert independent, "independent source did not change the evidence set"
    assert all(doc.startswith(("aad:", "nice:")) for doc in independent)


def test_calibrated_abstention_gate_abstains_low_confidence(tmp_path):
    from medguard.eval.selective_gate import CalibratedAbstentionGate

    deps = _make_deps(tmp_path, image_for=_image_for)
    # A tau above the melanoma fixture's 0.90 probe confidence forces c1 to abstain
    # even though its contract is PASS-eligible: the diagnostic gate, not the evidence
    # gate. Generation is skipped for the abstained lesion (no paid call).
    deps.abstention_gate = CalibratedAbstentionGate(
        diagnostic_confidence_tau=0.95, fitted=True, n_calibration=1)
    report = run_h2h([_case("c1", "melanoma")], deps, TRUTH, arms=[ArmId.M], n_boot=50)
    m_row = report["cases"][0]["arms"]["M"]
    assert m_row["final_action"] == "ABSTAIN"
    assert m_row["diagnostic_abstention"] is True
    assert m_row["generation"]["remote_calls"] == 0


def test_no_abstention_gate_answers_as_before(tmp_path):
    deps = _make_deps(tmp_path, image_for=_image_for)  # abstention_gate defaults to None
    report = run_h2h([_case("c1", "melanoma")], deps, TRUTH, arms=[ArmId.M], n_boot=50)
    m_row = report["cases"][0]["arms"]["M"]
    assert m_row["final_action"] == "PASS"
    assert m_row["diagnostic_abstention"] is False


def test_clean_source_text_strips_internal_biblio_refs():
    from medguard.eval.matched_full_h2h_run import _clean_source_text

    raw = "Dermatoscopy shall be used [72 - 80] .  Level of evidence: 1b [75, 76]"
    cleaned = _clean_source_text(raw)
    # The source document's own bracketed bibliography refs are removed; the
    # recommendation text and single-spaced formatting survive.
    assert "[72" not in cleaned and "[75" not in cleaned
    assert "Dermatoscopy shall be used" in cleaned
    assert "  " not in cleaned


def test_clean_source_text_strips_guideline_recommendation_scaffolding():
    from medguard.eval.matched_full_h2h_run import _clean_source_text

    raw = ("Recommendation 4 . Evidence based recommendation Level of recommendation A "
           "Dermatoscopy shall be used for the assessment of pigmented lesions. "
           "Level of evidence: 1b Consensus rate: 100% (19/19).")
    cleaned = _clean_source_text(raw)
    # The cited clinical statement survives VERBATIM -- only provenance scaffolding goes,
    # so the claim still restates the source exactly (citation integrity is unaffected).
    assert "Dermatoscopy shall be used for the assessment of pigmented lesions" in cleaned
    for scaffold in ("Recommendation 4", "Level of recommendation", "Level of evidence",
                     "Consensus rate", "Evidence based recommendation"):
        assert scaffold not in cleaned


def test_safe_answer_is_section_partitioned_with_cited_assessment(tmp_path):
    import re

    from medguard.eval.matched_full_h2h_run import (
        ASSESSMENT_SECTION_LABEL,
        DIAGNOSIS_SECTION_LABEL,
        VISUAL_FINDINGS_SECTION_LABEL,
        _safe_answer,
    )

    deps = _make_deps(tmp_path, image_for=_image_for)
    builder = _RunContractBuilder(
        _case("c1", "melanoma"), deps, image_exemplar_retrieval=True, graph_lineage=True,
    )
    contract, _chunks = builder.build_initial()
    answer = _safe_answer(contract)
    lines = answer.splitlines()

    visual_line = next(l for l in lines if l.startswith(VISUAL_FINDINGS_SECTION_LABEL))
    assessment_line = next(l for l in lines if l.startswith(ASSESSMENT_SECTION_LABEL))
    diagnosis_line = next(l for l in lines if l.startswith(DIAGNOSIS_SECTION_LABEL))

    # The Diagnosis line is exactly the frozen probe's committed label.
    assert diagnosis_line == f"{DIAGNOSIS_SECTION_LABEL} melanoma"
    # The Assessment section carries INDIVIDUAL single-bracket citations ([1] [2]),
    # never a grouped [1,2]; the Visual-findings section carries no citation.
    assert re.search(r"\[\d+\]", assessment_line)
    assert not re.search(r"\[\d+\s*,", assessment_line)
    assert not re.search(r"\[\d+\]", visual_line)


def _canonical_report(report):
    # The same canonical form the runner writes to disk; strips dict-ordering noise.
    return json.dumps(report, sort_keys=True, separators=(",", ":"), default=str)


def _six_cases():
    # Enough distinct lesions to actually exercise several worker threads.
    return [
        _case(f"c{i}", "melanoma" if i % 2 == 0 else "bcc") for i in range(6)
    ]


def _truth_for(cases):
    return {c.cluster_id: ("melanoma" if int(c.cluster_id[1:]) % 2 == 0 else "bcc")
            for c in cases}


def test_run_h2h_parallel_matches_sequential(tmp_path):
    # Parallelism must be RESULT-INVARIANT: a threaded run and a sequential run over the
    # same lesions produce a byte-identical scored report. Separate cache roots so both
    # actually generate (no replay masking); each parallel worker gets its own fake
    # generator/client via the factory.
    cases = _six_cases()
    truth = _truth_for(cases)

    deps_seq = _make_deps(tmp_path / "seq", image_for=lambda path: "melanoma")
    sequential = run_h2h(cases, deps_seq, truth, n_boot=200)

    deps_par = _make_deps(tmp_path / "par", image_for=lambda path: "melanoma")
    parallel = run_h2h(
        cases, deps_par, truth, n_boot=200, max_workers=4,
        generator_factory=lambda: VLMGenerator(FakeLLMClient(), max_schema_repairs=1, seed=0),
    )

    assert _canonical_report(sequential) == _canonical_report(parallel)
    # And it really produced every arm on every case.
    assert parallel["aggregate"]["arm_rows"] == len(DEFAULT_ARMS) * len(cases)


def test_b0_and_baseline_calls_are_cached_across_runs(tmp_path):
    # B0/baselines bypass the contract cache; the raw-call cache must make a warm re-run
    # (same cache_root) replay them for free -- the fix for the "re-runs re-pay" gotcha.
    client_one = FakeLLMClient()
    deps_one = _make_deps(tmp_path, client=client_one, image_for=_image_for)
    run_h2h([_case("c1", "melanoma")], deps_one, TRUTH,
            arms=[ArmId.B0, ArmId.B1], n_boot=20)
    assert client_one.vision_prompts  # cold run made real (fake) calls

    client_two = FakeLLMClient()
    deps_two = _make_deps(tmp_path, client=client_two, image_for=_image_for)  # same root
    run_h2h([_case("c1", "melanoma")], deps_two, TRUTH,
            arms=[ArmId.B0, ArmId.B1], n_boot=20)
    assert client_two.vision_prompts == []  # warm run: every B0/B1 call replayed from cache


def test_alignment_and_ece_blocks_are_scored(tmp_path):
    deps = _make_deps(tmp_path, image_for=_image_for)
    report = run_h2h(_standard_cases(), deps, TRUTH, n_boot=100)
    per_arm = report["scored"]["per_arm"]
    m = per_arm["M"]
    # ECE landed in the selective block (probe calibration).
    assert "ece" in m["selective"] and m["selective"]["ece"] is not None
    # Alignment block present with the canonical metrics + the architectural note.
    align = m["alignment"]
    assert align["applicable"] is True
    assert align["diagnosis_over_reliance"] == 0.0
    for key in ("over_reliance_confident_wrong", "copy_reference_ratio", "completeness"):
        assert key in align
    # B0 has no confidence -> no ECE; still gets an over-reliance rate over its answers.
    assert per_arm["B0"]["selective"]["ece"] is None


def test_reference_answer_is_label_plus_guideline_text(tmp_path):
    from medguard.eval.h2h_scoring import _reference_answer
    deps = _make_deps(tmp_path)
    ref = _reference_answer("melanoma", deps.guideline_retriever)
    assert ref.startswith("Diagnosis: melanoma.")
    assert "excise" in ref.lower() or "melanoma" in ref.lower()  # guideline text appended
    # A label with no licensed guideline condition -> just the diagnosis line.
    assert _reference_answer("nevus", deps.guideline_retriever).startswith("Diagnosis: nevus")
    assert _reference_answer("", deps.guideline_retriever) == ""


def test_parallel_run_requires_a_generator_factory(tmp_path):
    deps = _make_deps(tmp_path, image_for=lambda path: "melanoma")
    with pytest.raises(ValueError, match="generator_factory"):
        run_h2h(_six_cases(), deps, _truth_for(_six_cases()), n_boot=50, max_workers=4)


def test_parallel_run_rejects_a_model_mismatched_factory(tmp_path):
    # A factory whose client model differs from the base generator would split an arm's
    # cache across namespaces; the runner must refuse before generating.
    deps = _make_deps(tmp_path, image_for=lambda path: "melanoma")
    with pytest.raises(ValueError, match="model"):
        run_h2h(
            _six_cases(), deps, _truth_for(_six_cases()), n_boot=50, max_workers=4,
            generator_factory=lambda: VLMGenerator(
                FakeLLMClient(model="other/model"), seed=0),
        )


def test_independent_source_exhausted_when_no_alternative_guideline_remains_abstains(tmp_path):
    from medguard.remediation.actions import Action
    from medguard.remediation.policy import DeterministicRemediationPolicy
    from medguard.remediation.runner import run_remediation

    flagged = _text_record("pm:bad", "This abstract disputes excision for melanoma.",
                           eligible=("assessment",))
    registry = SourceContradictionRegistry([
        ContradictionFlag(condition="melanoma", document_id="pm:bad"),
    ])
    deps = _make_deps(
        tmp_path, contradiction_registry=registry, text_records=[flagged],
        image_for=lambda path: "melanoma",
    )
    # Consume the whole melanoma guideline tier so no independent source remains.
    deps.guideline_k = 3
    builder = _RunContractBuilder(
        _case("c1", "melanoma"), deps, image_exemplar_retrieval=True, graph_lineage=True,
    )
    contract, _chunks = builder.build_initial()
    tools = _RunRemediationTools(builder)
    result = run_remediation(contract, DeterministicRemediationPolicy(), tools, max_steps=4)
    assert "INDEPENDENT_SOURCE" in tools.applied
    assert result.final_action is Action.ABSTAIN
