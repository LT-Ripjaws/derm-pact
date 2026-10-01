"""Unmodified Ragas 0.4.1 metrics with explicit, recorded judge transports."""
import asyncio
import copy
import json
import math
import os
from importlib.metadata import version

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics.collections import (AgentGoalAccuracyWithReference, AnswerRelevancy,
    FactualCorrectness, Faithfulness, ToolCallAccuracy, ToolCallF1)

from dermpact.core.agent.redesign.evidence import digest
from dermpact.core.agent.redesign.provider import JsonGatewayClient, ModelSettings
from .read_evaluation_context import read_evaluation_contexts

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


class GatewayJudge(InstructorBaseRagasLLM):
    def __init__(self):
        self.client = JsonGatewayClient(ModelSettings(model="openai/gpt-oss-120b",
            temperature=0, top_p=1, max_tokens=6000), allow_live=True)
        self.cache, self.calls = {}, []
        self.reuse_claims = False

    def generate(self, prompt, response_model):
        schema = response_model.model_json_schema()
        key = digest({"prompt": prompt, "schema": schema})
        if not self.reuse_claims or key not in self.cache:
            reply = self.client.complete("Return only JSON matching this schema: " + json.dumps(schema),
                                         [{"role": "user", "content": prompt}])
            self.calls.append({"prompt_sha256": key, "schema": schema,
                               "output": reply.value, "provider": reply.metadata})
            self.cache[key] = response_model.model_validate(reply.value)
        return self.cache[key]

    async def agenerate(self, prompt, response_model):
        return await asyncio.to_thread(self.generate, prompt, response_model)


class LocalEmbeddings(BaseRagasEmbedding):
    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                                        device="cpu", local_files_only=True)

    def embed_text(self, text, **kwargs):
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text, **kwargs)


class StandardReadScorer:
    def __init__(self, factual_judge=None, judge=None, atomicity="high"):
        if version("ragas") != "0.4.1":
            raise RuntimeError("This scoring protocol requires ragas==0.4.1")
        self.judge = judge if judge is not None else GatewayJudge()
        self.factual_judge = factual_judge if factual_judge is not None else self.judge
        self.profile = {"other_llm_metrics": getattr(self.judge, "configuration", {"model": "openai/gpt-oss-120b"}),
            "factual_judge": getattr(self.factual_judge, "configuration", {"model": "openai/gpt-oss-120b"}),
            "atomicity": atomicity, "coverage": "high"}
        self.results = {}
        self.metrics = {
            "faithfulness": Faithfulness(llm=self.judge),
            **{"factual_correctness_" + mode: FactualCorrectness(llm=self.factual_judge, mode=mode,
                atomicity=atomicity, coverage="high") for mode in ("f1", "precision", "recall")},
            "response_relevancy": AnswerRelevancy(llm=self.judge, embeddings=LocalEmbeddings(), strictness=3),
            "agent_goal_accuracy": AgentGoalAccuracyWithReference(llm=self.judge)}

    async def score(self, result, reference, *, completed_scores=None, expected_cache_key=None, metric_sink=None):
        answer, question = result["final_text"], reference["question"]
        # Use exactly what this rollout saw, even if receipt rendering changes later.
        contexts = read_evaluation_contexts(result)
        proposals = [e["command"] for e in result["trace"]
                     if e.get("node") in {"generate", "repair"} and e.get("command", {}).get("kind") == "read"]
        proposals = [{**c, "args": {k: v for k, v in c["args"].items()
                      if not (k == "detail" and v == "summary")}} for c in proposals]
        key = digest({"scoring_profile": self.profile, "answer": answer, "reference": reference, "contexts": contexts,
                      "proposals": proposals, "acquisition": result.get("settings", {}).get("acquisition")})
        if expected_cache_key is not None and expected_cache_key != key:
            raise ValueError("Partial grading inputs/configuration changed")
        if key in self.results:
            return {**copy.deepcopy(self.results[key]), "agent_outcome": result.get("outcome"),
                    "score_reused": True}
        dialogue = [HumanMessage(content=question), AIMessage(content=answer)]
        scores = copy.deepcopy(completed_scores or {})
        for name, metric in self.metrics.items():
            if name in scores:
                continue
            judge = self.factual_judge if name.startswith("factual") else self.judge
            judge.reuse_claims = name.startswith("factual")
            start = len(judge.calls)
            kwargs = ({"response": answer, "reference": reference["reference"]} if name.startswith("factual") else
                      {"user_input": dialogue, "reference": reference["reference_goal"]} if name == "agent_goal_accuracy" else
                      {"user_input": question, "response": answer})
            if name == "faithfulness":
                kwargs["retrieved_contexts"] = contexts
            try:
                value = await metric.ascore(**kwargs)
                finite = math.isfinite(float(value.value))
                scores[name] = {"value": float(value.value) if finite else None,
                    "status": "scored" if finite else "undefined", "reason": value.reason}
            except Exception as exc:
                # Keep the case and expose scoring failure, never drop it or assign 1.
                scores[name] = {"value": None, "status": "unscored", "error_type": type(exc).__name__}
            scores[name]["judge_calls"] = judge.calls[start:]
            if metric_sink is not None:
                metric_sink(key, name, scores[name])
        if result.get("settings", {}).get("acquisition") == "adaptive" and result.get("tool_metrics_applicable", True):
            tool_dialogue = [HumanMessage(content=question), AIMessage(content="", tool_calls=[
                ToolCall(name=c["tool"], args=c["args"]) for c in proposals])]
            for name, metric in (("tool_call_accuracy", ToolCallAccuracy(strict_order=False)), ("tool_call_f1", ToolCallF1())):
                values = [float((await metric.ascore(user_input=tool_dialogue,
                    reference_tool_calls=[ToolCall(**t) for t in alternative])).value)
                    for alternative in reference["tool_alternatives"]]
                scores[name] = {"value": max(values), "status": "scored", "alternative_scores": values}
        else:
            for name in ("tool_call_accuracy", "tool_call_f1"):
                scores[name] = {"value": None, "status": "not_applicable", "reason": "Host-selected acquisition"}
        grade = {"ragas_version": version("ragas"), "scores": scores,
                "scoring_profile": self.profile,
                "response_sha256": digest(answer), "reference_sha256": digest(reference),
                "contexts_sha256": digest(contexts), "empty_response": not answer.strip(),
                "agent_outcome": result.get("outcome"), "embedding_model": EMBEDDING_MODEL,
                "embedding_revision": EMBEDDING_REVISION,
                "goal_transcript_scope": "original question and delivered answer; internal drafts excluded",
                "score_cache_key": key, "score_reused": False}
        self.results[key] = copy.deepcopy(grade)
        return grade
