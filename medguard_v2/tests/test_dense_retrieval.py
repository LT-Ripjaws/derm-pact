"""The dense arm is a MEASURED CONTROL and a WALLED-OFF exploration channel -- never a citation.

Two properties matter here, and both would be invisible in a retrieval-quality score:

1. **The verified core is untouched.** The keyed path's citation-resolvability stays exactly 1.0,
   and a dense chunk id can never be laundered into a resolvable citation: the existing in-loop
   gate rejects it and the resolvability metric excludes it.
2. **The dense channel is honestly labelled.** literature_lookup returns unverified, non-citable
   passages, recorded under its own tool name so it can never enter the guideline-context or
   fabrication accounting that key on search_guidelines.

No network, no model download: a deterministic HashingEmbedder drives the index, and a scripted
fake model drives the agent loop.
"""
from __future__ import annotations

import json
import unittest

from medguard_v2.agent.clinician_agent import (
    AgentAnswer,
    AgentTools,
    _build_system_prompt,
    run_agent,
)
from medguard_v2.agent.verification import _CITATION_TOKEN, citation_array_checks
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.eval.agent_eval import score_citations, score_fabrication
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.dense_retrieval import (
    CHUNK_ID_PREFIX,
    UNVERIFIED_DISCLAIMER,
    ChunkConfig,
    DenseGuidelineIndex,
    HashingEmbedder,
    build_chunks,
    format_unverified_chunks,
    load_source_records,
)
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog

CLEAN = {name: "absent" for name in CRITERION_NAMES}


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class ChunkingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_source_records()
        cls.catalog = GuidelineCatalog.load()
        cls.catalog_hashes = {
            r.content_sha256 for recs in cls.catalog._by_condition.values() for r in recs
        }

    def test_source_records_match_catalog_size(self):
        # The dense arm reads the SAME underlying text the keyed corpus is built from.
        self.assertEqual(len(self.records), len(self.catalog))

    def test_record_aligned_chunks_reverify_their_hash(self):
        # Dense's best case: one chunk per record, chunk text == record text, so every chunk
        # re-verifies a catalog content hash. This is the resolvability CEILING.
        chunks = build_chunks(self.records, ChunkConfig(mode="record_aligned"))
        self.assertEqual(len(chunks), len(self.records))
        reverify = sum(1 for c in chunks if c.content_sha256 in self.catalog_hashes)
        self.assertEqual(reverify, len(chunks))

    def test_mechanical_chunking_destroys_hash_provenance(self):
        # Realistic arm: fixed-size windows that ignore record boundaries. The curated boundaries
        # are gone, so almost nothing re-verifies -- the resolvability cost this control measures.
        chunks = build_chunks(self.records, ChunkConfig(mode="mechanical"))
        spanning = sum(1 for c in chunks if len(c.overlaps) > 1)
        self.assertGreater(spanning, 0, "mechanical chunks should merge/fragment records")
        reverify = sum(1 for c in chunks if c.content_sha256 in self.catalog_hashes)
        # Verbatim, hash-verifiable provenance is the exception, not the rule, once you re-chunk.
        self.assertLess(reverify, len(chunks) // 2)

    def test_chunk_ids_are_not_citation_shaped(self):
        # A chunk id must not resolve as a record and must not even look like an inline citation
        # token -- the first, structural half of the wall.
        for chunk in build_chunks(self.records, ChunkConfig(mode="mechanical")):
            self.assertTrue(chunk.chunk_id.startswith(CHUNK_ID_PREFIX))
            self.assertFalse(self.catalog.has_record(chunk.chunk_id))
            self.assertIsNone(_CITATION_TOKEN.fullmatch(chunk.chunk_id))


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class DenseIndexTest(unittest.TestCase):
    def test_build_and_retrieve_are_deterministic(self):
        idx_a = DenseGuidelineIndex.build(HashingEmbedder(), config=ChunkConfig(mode="mechanical"))
        idx_b = DenseGuidelineIndex.build(HashingEmbedder(), config=ChunkConfig(mode="mechanical"))
        # Same embedder + same chunks + same vectors -> same version hash.
        self.assertEqual(idx_a.version_hash, idx_b.version_hash)
        q = "melanoma excision margin management for a suspicious lesion"
        r1 = [x["chunk_id"] for x in idx_a.retrieve(q, k=5)]
        r2 = [x["chunk_id"] for x in idx_b.retrieve(q, k=5)]
        r3 = [x["chunk_id"] for x in idx_a.retrieve(q, k=5)]
        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(len(r1), 5)

    def test_retrieve_scores_descend_and_ids_carry_no_colon(self):
        idx = DenseGuidelineIndex.build(HashingEmbedder(), config=ChunkConfig(mode="mechanical"))
        hits = idx.retrieve("basal cell carcinoma surgical excision safety margin", k=4)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for h in hits:
            self.assertNotIn(":", h["chunk_id"])


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class LiteratureLookupWallTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()
        cls.index = DenseGuidelineIndex.build(
            HashingEmbedder(), config=ChunkConfig(mode="mechanical")
        )

    def _contract(self, diagnosis="MEL"):
        criteria = dict(CLEAN)
        criteria["pigment_network"] = "atypical"
        criteria["blue_whitish_veil"] = "present"
        return build_contract(
            case_id="dense-t1",
            classifier_diagnosis=diagnosis,
            classifier_confidence=0.9,
            criteria=criteria,
            catalog=self.catalog,
        )

    def test_lookup_returns_unverified_labelled_chunks(self):
        tools = AgentTools(self._contract(), self.catalog, dense_index=self.index)
        result = tools.literature_lookup("management of melanoma", k=3)
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["channel"], "unverified_exploration")
        self.assertFalse(result.payload["citable"])
        self.assertEqual(result.payload["disclaimer"], UNVERIFIED_DISCLAIMER)
        self.assertTrue(result.payload["chunks"])
        for chunk in result.payload["chunks"]:
            self.assertTrue(chunk["unverified"])
            self.assertFalse(chunk["citable"])
            self.assertTrue(chunk["chunk_id"].startswith(CHUNK_ID_PREFIX))

    def test_lookup_is_recorded_under_its_own_tool_name(self):
        # The wall for the RAGAS/fabrication accounting: literature_lookup must NEVER be recorded
        # as a search_guidelines call, or dense chunks would leak into guideline-context
        # reconstruction (both key on the 'search_guidelines' tool name).
        tools = AgentTools(self._contract(), self.catalog, dense_index=self.index)
        tools.literature_lookup("melanoma", k=2)
        self.assertEqual(tools.calls[-1]["tool"], "literature_lookup")
        self.assertFalse(any(c.get("tool") == "search_guidelines" for c in tools.calls))

    def test_lookup_unavailable_without_index(self):
        tools = AgentTools(self._contract(), self.catalog)  # no dense_index
        result = tools.literature_lookup("melanoma", k=3)
        self.assertFalse(result.ok)

    def test_chunk_id_in_citations_is_rejected_by_existing_gate(self):
        # The existing in-loop citation gate (unmodified) rejects any chunk id placed in the
        # citations array, and the resolvability metric counts it unresolvable.
        tools = AgentTools(self._contract(), self.catalog, dense_index=self.index)
        chunk_id = tools.literature_lookup("melanoma", k=1).payload["chunks"][0]["chunk_id"]

        issues = citation_array_checks([chunk_id], self.catalog)
        self.assertTrue(issues)
        self.assertIn(chunk_id, issues[0])

        answer = AgentAnswer(kind="answer", text="Assessment: see above.", citations=(chunk_id,))
        cite = score_citations(answer, self.catalog)
        self.assertEqual(cite["resolvability"], 0.0)
        self.assertEqual(cite["unresolvable_ids"], [chunk_id])

    def test_chunk_id_does_not_launder_a_fabricated_specific(self):
        # Routing a guideline specific through a dense chunk id must NOT make it look traceable.
        tools = AgentTools(self._contract(), self.catalog, dense_index=self.index)
        chunk_id = tools.literature_lookup("melanoma", k=1).payload["chunks"][0]["chunk_id"]
        answer = AgentAnswer(
            kind="answer",
            text="Excise with a 5-mm margin and use imiquimod.",
            citations=(chunk_id,),
        )
        fab = score_fabrication(answer, self.catalog)
        self.assertFalse(fab["cited_anything"])
        self.assertTrue(fab["fabricated_management"])

    def test_end_to_end_loop_bounces_a_chunk_citation(self):
        # A scripted model that looks up literature, then tries to cite a chunk id, is caught by
        # the host loop (unmodified citation_array_checks) on its first draft.
        idx = self.index
        tools_probe = AgentTools(self._contract(), self.catalog, dense_index=idx)
        chunk_id = tools_probe.literature_lookup("melanoma", k=1).payload["chunks"][0]["chunk_id"]

        replies = [
            json.dumps({"action": "tool", "tool": "literature_lookup",
                        "args": {"query": "melanoma management", "k": 2}}),
            json.dumps({"action": "answer", "text": "Assessment: background reading.",
                        "citations": [chunk_id]}),
            json.dumps({"action": "answer", "text": "Diagnosis: computed by the system.",
                        "citations": []}),
        ]
        seen = {"i": 0}

        def chat(_messages):
            reply = replies[min(seen["i"], len(replies) - 1)]
            seen["i"] += 1
            return reply

        answer = run_agent(
            contract=self._contract(), catalog=self.catalog, question="what should I do?",
            chat=chat, dense_index=idx,
        )
        first = answer.verification.get("issues_first_draft") or []
        self.assertTrue(any("does not resolve" in i for i in first))
        # And a literature_lookup call was actually made, under its own name.
        self.assertTrue(any(c.get("tool") == "literature_lookup" for c in answer.tool_calls))
        self.assertFalse(any(c.get("tool") == "search_guidelines" for c in answer.tool_calls))


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class VerifiedCoreUnchangedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def test_real_record_id_still_resolves_to_one(self):
        # The verified keyed path is untouched: a real record id resolves and re-verifies.
        served = self.catalog.bind("MEL").recommendations
        self.assertTrue(served)
        ids = tuple(r.record_id for r in served)
        answer = AgentAnswer(kind="answer", text="Assessment: quoted.", citations=ids)
        cite = score_citations(answer, self.catalog)
        self.assertEqual(cite["resolvability"], 1.0)
        self.assertEqual(cite["resolved"], cite["intact"])

    def test_default_prompt_has_no_literature_tool(self):
        # The default prompt bytes are unchanged: the literature tool appears ONLY when enabled.
        default = _build_system_prompt(None)
        self.assertNotIn("literature_lookup", default)
        self.assertIn("five tools", default)

        with_lit = _build_system_prompt(None, literature_tool=True)
        self.assertIn("literature_lookup", with_lit)
        self.assertIn("exploration-only", with_lit)
        self.assertIn("six tools", with_lit)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
