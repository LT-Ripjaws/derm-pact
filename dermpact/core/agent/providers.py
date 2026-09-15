"""WP-AG: provider protocol, offline scripted double, and the httpx-gated live gateway adapter."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

from dermpact import config as cfg


@dataclass(frozen=True)
class ProviderReply:
    text: Optional[str] = None
    tool_calls: Tuple[Dict[str, Any], ...] = ()
    claims: Tuple[str, ...] = ()
    usage: Dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    provider_name: str = ""
    error: Optional[str] = None
    raw: Optional[str] = None


class LiveCallsDisabled(RuntimeError):
    """Raised by :meth:`LiveGatewayProvider.complete` before any transport call when live LLM calls are not both env-flagged and constructor-opted-in."""


class ScriptedProvider:
    """Offline double for controller tests: replies scripted in call order."""

    def __init__(self, replies: Sequence[Mapping[str, Any]]):
        self._replies = list(replies)
        self._index = 0
        self.provider_name = "scripted"
        self.asked: List[Dict[str, Any]] = []

    def complete(self, system: str, messages: Sequence[Mapping[str, Any]], tools_schema: Sequence[Mapping[str, Any]]) -> ProviderReply:
        self.asked.append({"system": system, "messages": list(messages), "tools_schema": list(tools_schema)})
        if self._index >= len(self._replies):
            return ProviderReply(provider_name=self.provider_name, error="scripted_replies_exhausted")
        reply = self._replies[self._index]
        self._index += 1
        if "tool" in reply:
            return ProviderReply(tool_calls=({"tool": reply["tool"], "args": dict(reply.get("args", {}))},),
                                 provider_name=self.provider_name)
        if "error" in reply:
            return ProviderReply(provider_name=self.provider_name, error=str(reply["error"]))
        return ProviderReply(text=str(reply.get("text", "")), claims=tuple(reply.get("claims", ())),
                             provider_name=self.provider_name)


def estimate_cost(prompt_tokens: int, completion_tokens: int, rate_in: float, rate_out: float) -> float:
    """USD cost estimate: ``rate_in``/``rate_out`` are per-1K-token rates."""
    return (prompt_tokens / 1000.0) * rate_in + (completion_tokens / 1000.0) * rate_out


def live_command(*, case_id: str = "<case_id>", model: Optional[str] = None) -> str:
    """The exact, costed live-run command WP-E writes into evaluation results as "deferred by user" -- never executed by this package."""
    model_name = model or cfg.GENERATOR_MODEL
    return (
        "$env:DERMPACT_ALLOW_LIVE_LLM='1'; .\\venv_p2\\Scripts\\python.exe -m dermpact.core.prototype run "
        f"--case {case_id} --scenario evidence_report --mode live --inference cached --model {model_name} "
        "--output research/experiments/dermpact_phase6/live_runs_v1"
    )


class LiveGatewayProvider:
    """httpx-only OpenAI-compatible gateway adapter."""

    def __init__(
        self, *, allow_live: bool = False, model: Optional[str] = None, base_url: Optional[str] = None,
        temperature: float = 0.0, timeout_s: float = 60.0, retries: int = 2,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        cfg.load_env()  # project-root/.env + documented aliases; never prints values
        self._allow_live = bool(allow_live)
        self._model = model or cfg.GENERATOR_MODEL
        self._base_url = base_url or cfg.GATEWAY_BASE_URL
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._retries = retries
        self._transport = transport
        self.provider_name = f"live_gateway:{self._model}"

    def _gated(self) -> None:
        if os.environ.get("DERMPACT_ALLOW_LIVE_LLM") != "1" or not self._allow_live:
            raise LiveCallsDisabled(
                "live LLM calls are deferred by user policy; set DERMPACT_ALLOW_LIVE_LLM=1 in the "
                "environment AND construct LiveGatewayProvider(allow_live=True) to enable them"
            )

    def complete(self, system: str, messages: Sequence[Mapping[str, Any]], tools_schema: Sequence[Mapping[str, Any]]) -> ProviderReply:
        self._gated()
        cfg.load_env()  # refresh from the current environment (project gateway pattern; never prints)
        api_key = cfg.GATEWAY_API_KEY  # read at call time; local only; never logged or returned
        if not api_key:
            raise RuntimeError("no gateway API key in environment")
        schema_text = json.dumps(list(tools_schema), sort_keys=True, separators=(",", ":"))
        wire_system = system + "\n\nAVAILABLE_TYPED_TOOLS_JSON:\n" + schema_text
        wire_messages: List[Dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            if "content" in message and role in {"user", "assistant"}:
                wire_messages.append({"role": role, "content": str(message.get("content", ""))})
            elif role == "assistant" and "tool_call" in message:
                wire_messages.append({
                    "role": "assistant",
                    "content": json.dumps(message["tool_call"], sort_keys=True, default=str),
                })
            elif role == "tool":
                wire_messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "host_event": "tool_result", "tool": message.get("tool"),
                        "result": message.get("result"),
                    }, sort_keys=True, default=str),
                })
            else:
                wire_messages.append({
                    "role": "user",
                    "content": json.dumps({"host_event": role, **{k: v for k, v in message.items() if k != "role"}},
                                          sort_keys=True, default=str),
                })
        body = {
            "model": self._model, "temperature": self._temperature,
            "messages": [{"role": "system", "content": wire_system}, *wire_messages],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        client_kwargs: Dict[str, Any] = {"timeout": self._timeout_s}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        start = time.perf_counter()
        last_exc: Optional[Exception] = None
        with httpx.Client(**client_kwargs) as client:
            for attempt in range(self._retries + 1):
                try:
                    resp = client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
                except httpx.TimeoutException as exc:
                    last_exc = exc
                    continue
                if resp.status_code >= 500:
                    last_exc = RuntimeError(f"gateway returned {resp.status_code}")
                    continue
                resp.raise_for_status()
                payload = resp.json()
                text = payload["choices"][0]["message"]["content"]
                raw_usage = payload.get("usage") or {}
                usage = {
                    str(key): value for key, value in raw_usage.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                latency_s = time.perf_counter() - start
                try:
                    clean = str(text).strip()
                    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.IGNORECASE | re.DOTALL)
                    parsed = json.loads(fenced.group(1) if fenced else clean)
                except (json.JSONDecodeError, TypeError):
                    return ProviderReply(error="malformed", raw=text, provider_name=self.provider_name,
                                         usage=usage, latency_s=latency_s)
                if isinstance(parsed, Mapping) and "tool" in parsed:
                    return ProviderReply(tool_calls=({"tool": parsed["tool"], "args": dict(parsed.get("args", {}))},),
                                         provider_name=self.provider_name, usage=usage, latency_s=latency_s, raw=text)
                if isinstance(parsed, Mapping) and "final" in parsed:
                    final = parsed["final"] or {}
                    return ProviderReply(text=str(final.get("text", "")), claims=tuple(final.get("claims", ())),
                                         provider_name=self.provider_name, usage=usage, latency_s=latency_s, raw=text)
                return ProviderReply(error="malformed", raw=text, provider_name=self.provider_name,
                                     usage=usage, latency_s=latency_s)

        latency_s = time.perf_counter() - start
        return ProviderReply(error="provider_failed", provider_name=self.provider_name, latency_s=latency_s,
                             raw=(str(last_exc) if last_exc is not None else "gateway call failed"))


__all__ = [
    "ProviderReply", "LiveCallsDisabled", "ScriptedProvider", "LiveGatewayProvider",
    "estimate_cost", "live_command",
]
