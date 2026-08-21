"""Fresh OpenAI-compatible LLM/VLM client for the MedGuard pipeline.

Independent of legacy baseline's src/. Talks to any OpenAI-compatible gateway (base_url +
key from medguard.config, which reads the env -- never a .env file). Vision messages
inline the image as a base64 data URI. The `openai` SDK is imported lazily so this
module (and the whole pipeline package) imports with no heavy deps and unit tests can
inject a fake client.
"""
from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import config
from ..errors import LLMCallError


class LLMClient:
    """Minimal chat/vision client over an OpenAI-compatible gateway."""

    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, max_retries: Optional[int] = None,
                 delay_sec: Optional[float] = None, _client: Any = None):
        self.api_key = api_key if api_key is not None else config.GATEWAY_API_KEY
        self.base_url = base_url or config.GATEWAY_BASE_URL
        self.model = model or config.GENERATOR_MODEL
        self.max_retries = config.API_MAX_RETRIES if max_retries is None else max_retries
        self.delay_sec = config.API_CALL_DELAY_SEC if delay_sec is None else delay_sec
        self._client = _client  # inject a fake in tests; real one built lazily
        self.call_log: List[Dict[str, Any]] = []
        self.last_call_metadata: Dict[str, Any] = {}

    def _ensure_client(self):
        if self._client is None:
            import os
            from openai import OpenAI  # lazy: only when a real call is made
            # re-read env at call time in case config.load_env() ran after construction
            key = self.api_key or os.environ.get("MEDGUARD_API_KEY") or os.environ.get("KILOCODE_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "no LLM gateway API key. Provide it via medguard/.env (copy .env.example) "
                    "or export KILOCODE_API_KEY, then run with --env or config.load_env().")
            self.api_key = key
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _image_data_uri(image_path: str) -> str:
        p = Path(image_path)
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _validated_seed(seed: Optional[int]) -> Optional[int]:
        if seed is None:
            return None
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        return seed

    def chat(self, messages: List[Dict[str, Any]], *, model: Optional[str] = None,
             max_tokens: int = 512, temperature: float = 0.2, extra_body: Any = None,
             seed: Optional[int] = None) -> str:
        """One chat completion -> assistant text. Retries with backoff on transient errors."""
        client = self._ensure_client()
        seed = self._validated_seed(seed)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            selected_model = model or self.model
            started = time.perf_counter()
            try:
                if self.delay_sec:
                    time.sleep(self.delay_sec)
                request = {
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **({"extra_body": extra_body} if extra_body else {}),
                }
                if seed is not None:
                    request["seed"] = seed
                resp = client.chat.completions.create(**request)
                usage = getattr(resp, "usage", None)
                metadata = {
                    "model": selected_model,
                    "attempt": attempt + 1,
                    "succeeded": True,
                    "latency_seconds": time.perf_counter() - started,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "usage_available": usage is not None,
                    "seed": seed,
                }
                self.call_log.append(metadata)
                self.last_call_metadata = dict(metadata)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 -- retry transient gateway errors
                last_err = e
                metadata = {
                    "model": selected_model,
                    "attempt": attempt + 1,
                    "succeeded": False,
                    "latency_seconds": time.perf_counter() - started,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "usage_available": False,
                    "error_type": type(e).__name__,
                    "seed": seed,
                }
                self.call_log.append(metadata)
                self.last_call_metadata = dict(metadata)
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        recent = self.call_log[-self.max_retries:]
        raise LLMCallError(
            f"LLM chat failed after {self.max_retries} attempts: {type(last_err).__name__}",
            usage_metadata={
                "remote_calls": len(recent),
                "failed_calls": sum(not event["succeeded"] for event in recent),
                "latency_seconds": sum(float(event["latency_seconds"]) for event in recent),
                "usage_available": False,
            },
        )

    def vision(self, prompt: str, image_path: Optional[str | Sequence[str]] = None, *, system: str = "",
               model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.2,
               seed: Optional[int] = None) -> str:
        """Prompt with an optional image (image_first, then text)."""
        content: List[Dict[str, Any]] = []
        image_paths = ([str(path) for path in image_path]
                       if isinstance(image_path, (list, tuple)) else
                       ([str(image_path)] if image_path else []))
        if len(set(image_paths)) != len(image_paths):
            raise ValueError("vision request contains duplicate image paths")
        for path in image_paths:
            content.append({"type": "image_url",
                            "image_url": {"url": self._image_data_uri(path)}})
        content.append({"type": "text", "text": prompt})
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.chat(
            messages, model=model, max_tokens=max_tokens, temperature=temperature,
            seed=seed,
        )
