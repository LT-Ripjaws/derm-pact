"""Explicit, bounded JSON transport with no model substitution or secret logging."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any
from types import SimpleNamespace

import httpx

from dermpact import config


@dataclass(frozen=True)
class ModelSettings:
    model: str = "qwen/qwen3-32b"
    temperature: float | None = 0.2
    top_p: float | None = 0.8
    max_tokens: int | None = 4096
    timeout_s: float = 90.0
    retries: int = 1
    reasoning_enabled: bool = False
    reasoning_effort: str | None = None
    backend: str = "kilo"

    def __post_init__(self):
        if not self.model or self.backend not in {"kilo", "codex_cli"}:
            raise ValueError("invalid model/decoding settings")
        if self.backend == "kilo" and ((self.temperature is not None and not 0 <= self.temperature <= 2)
                or (self.top_p is not None and not 0 < self.top_p <= 1) or self.max_tokens is None or self.max_tokens < 1):
            raise ValueError("invalid gateway decoding settings")
        if self.backend == "codex_cli" and any(v is not None for v in (self.temperature, self.top_p, self.max_tokens)):
            raise ValueError("Codex CLI does not expose these sampling/output-limit controls")
        if self.timeout_s <= 0 or not 0 <= self.retries <= 2:
            raise ValueError("invalid request budget")
        if self.reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError("invalid reasoning effort")


class GenerationError(RuntimeError):
    """Only a sanitized machine-readable reason is exposed to callers."""


@dataclass(frozen=True)
class JsonReply:
    value: dict[str, Any]
    metadata: dict[str, Any]


class JsonGatewayClient:
    def __init__(self, settings: ModelSettings, *, allow_live: bool = False,
                 transport: httpx.BaseTransport | None = None, response_schema: dict | None = None,
                 text_mode: bool = False, gateway_role: str = "generation"):
        if text_mode and response_schema is not None:
            raise ValueError("text mode cannot also request a JSON schema")
        self.settings = settings
        self.allow_live = allow_live
        self.transport = transport
        self.response_schema = response_schema
        self.text_mode = text_mode
        if gateway_role not in {"generation", "evaluation_judge"}:
            raise ValueError("unknown gateway role")
        self.gateway_role = gateway_role

    def complete(self, system: str, messages: list[dict[str, str]]) -> JsonReply:
        if not self.allow_live or os.environ.get("DERMPACT_ALLOW_LIVE_LLM") != "1":
            raise GenerationError("live_calls_disabled")
        if self.settings.backend == "codex_cli":
            from .codex_provider import complete
            return complete(self.settings, system, messages, self.response_schema, self.text_mode)
        if os.environ.get("DERMPACT_BLOCK_KILO") == "1" and not (
                self.gateway_role == "evaluation_judge" and os.environ.get("DERMPACT_ALLOW_KILO_JUDGE") == "1"):
            raise GenerationError("kilo_disabled_for_this_run")
        config.load_env()
        if not config.GATEWAY_API_KEY:
            raise GenerationError("missing_gateway_key")
        from openai import OpenAI
        from dermpact.pipeline.llm import LLMClient
        from dermpact.errors import LLMCallError
        start = time.perf_counter()
        captured = []
        http_client = httpx.Client(timeout=self.settings.timeout_s, transport=self.transport, follow_redirects=False)
        sdk = OpenAI(api_key=config.GATEWAY_API_KEY, base_url=config.GATEWAY_BASE_URL,
                     timeout=self.settings.timeout_s, max_retries=0, http_client=http_client)
        def create(**kwargs):
            if kwargs.get("temperature") is None:
                kwargs.pop("temperature", None)
            response = sdk.chat.completions.create(**kwargs)
            captured.append(response.model_dump())
            return response
        client = LLMClient(model=self.settings.model, max_retries=self.settings.retries + 1,
                           delay_sec=0, _client=SimpleNamespace(chat=SimpleNamespace(
                               completions=SimpleNamespace(create=create))))
        try:
            client.chat([{"role": "system", "content": system}, *messages],
                        max_tokens=self.settings.max_tokens, temperature=self.settings.temperature,
                        extra_body={**({"top_p": self.settings.top_p} if self.settings.top_p is not None else {}),
                                    "reasoning": {"enabled": self.settings.reasoning_enabled,
                                        **({"effort": self.settings.reasoning_effort} if self.settings.reasoning_effort else {})},
                                    **({} if self.text_mode else {"response_format": ({"type": "json_schema", "json_schema": {
                                        "name": "evaluation_result" if self.gateway_role == "evaluation_judge" else "read_command",
                                        "strict": True, "schema": self.response_schema}}
                                        if self.response_schema else {"type": "json_object"})})})
        except LLMCallError:
            kind = client.last_call_metadata.get("error_type", "unknown")
            code = {"PermissionDeniedError": "gateway_http_403", "AuthenticationError": "gateway_http_401",
                    "RateLimitError": "gateway_http_429"}.get(kind, "gateway_" + kind)
            raise GenerationError(code) from None
        finally:
            sdk.close()
        try:
            payload = captured[-1]
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            observed_model = payload.get("model")
            finish = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError):
            raise GenerationError("invalid_gateway_envelope") from None
        allowed_identity = {self.settings.model.lower(), self.settings.model.split("/")[-1].lower()}
        if self.settings.model == "qwen/qwen3-32b":
            allowed_identity.add("alibaba/qwen-3-32b")
        if not isinstance(observed_model, str) or observed_model.lower() not in allowed_identity:
            raise GenerationError("unexpected_or_missing_model_identity")
        if finish not in {"stop", "end_turn"}:
            raise GenerationError("incomplete_generation")
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("missing_final_content")
        # Reasoning is not a command. Do not recover a JSON substring from arbitrary text.
        clean = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL | re.IGNORECASE)
        if fenced:
            clean = fenced.group(1)
        try:
            value = {"text": content} if self.text_mode else json.loads(clean)
        except ValueError:
            raise GenerationError("invalid_command_json") from None
        if not isinstance(value, dict):
            raise GenerationError("command_not_object")
        usage = payload.get("usage") or {}
        return JsonReply(value, {
            "requested_model": self.settings.model, "observed_model": observed_model,
            "finish_reason": finish, "settings": asdict(self.settings),
            "reasoning_returned": bool(message.get("reasoning") or message.get("reasoning_content")),
            "usage": {k: v for k, v in usage.items() if k in {
                "prompt_tokens", "completion_tokens", "total_tokens", "cost", "completion_tokens_details"
            }},
            "transport": "project_LLMClient_OpenAI_SDK",
            "gateway_role": self.gateway_role,
            "output_format": "text" if self.text_mode else "json",
            "provider": payload.get("provider"),
            "attempts": len(client.call_log), "elapsed_s": time.perf_counter() - start,
        })
