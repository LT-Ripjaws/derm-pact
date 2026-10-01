"""Kilo Luna transport for the unchanged Ragas prompts and study schema definitions."""
import asyncio
import json
from pathlib import Path
from ragas.llms.base import InstructorBaseRagasLLM
from dermpact.core.agent.redesign.evidence import digest
from dermpact.core.agent.redesign.provider import JsonGatewayClient, ModelSettings
from .codex_judge import INSTRUCTIONS, strict_schema


class KiloJudge(InstructorBaseRagasLLM):
    def __init__(self, artifacts, effort="low"):
        self.artifacts = Path(artifacts)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.settings = ModelSettings(model="openai/gpt-5.6-luna", backend="kilo", temperature=None,
            top_p=None, max_tokens=6000, timeout_s=180, retries=1, reasoning_enabled=True, reasoning_effort=effort)
        self.cache, self.calls = {}, []
        self.reuse_claims = True
        self.request_count = 1 + max((int(p.name.split("-", 1)[0]) for p in self.artifacts.iterdir()
            if p.is_dir() and p.name.split("-", 1)[0].isdigit()), default=-1)
        self.configuration = {"model": self.settings.model, "transport": "kilo_gateway",
            "reasoning_effort": effort, "instructions_sha256": digest(INSTRUCTIONS),
            "temperature_top_p": "not sent", "max_output_tokens": 6000, "timeout_s": 180,
            "transport_retries": 1, "tools_supplied": False, "conversation_scope": "single stateless evaluation prompt"}

    def generate(self, prompt, response_model):
        schema = response_model.model_json_schema()
        key = digest({"prompt": prompt, "schema": schema})
        if self.reuse_claims and key in self.cache:
            return response_model.model_validate(self.cache[key])
        call_dir = self.artifacts / f"{self.request_count:04d}-{key[:12]}"
        self.request_count += 1
        call_dir.mkdir(exist_ok=False)
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (call_dir / "instructions.md").write_text(INSTRUCTIONS, encoding="utf-8")
        (call_dir / "schema.json").write_text(json.dumps(strict_schema(schema)), encoding="utf-8")
        client = JsonGatewayClient(self.settings, allow_live=True, response_schema=strict_schema(schema),
                                   gateway_role="evaluation_judge")
        try:
            reply = client.complete(INSTRUCTIONS, [{"role": "user", "content": prompt}])
            (call_dir / "output.json").write_text(json.dumps(reply.value, ensure_ascii=False), encoding="utf-8")
            (call_dir / "metadata.json").write_text(json.dumps(reply.metadata, indent=2), encoding="utf-8")
            value = response_model.model_validate(reply.value)
        except Exception as exc:
            (call_dir / "error.json").write_text(json.dumps({"error_type": type(exc).__name__}), encoding="utf-8")
            raise
        self.cache[key] = value.model_dump()
        self.calls.append({"prompt_sha256": key, "schema": schema,
                           "output": self.cache[key], "provider": reply.metadata})
        return value

    async def agenerate(self, prompt, response_model):
        return await asyncio.to_thread(self.generate, prompt, response_model)
