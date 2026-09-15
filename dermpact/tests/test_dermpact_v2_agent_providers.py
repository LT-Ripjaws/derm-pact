"""WP-AG: provider protocol, ScriptedProvider double, and the httpx-gated LiveGatewayProvider (dermpact.core.agent.providers)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx

from dermpact.core.agent.providers import (
    LiveCallsDisabled,
    LiveGatewayProvider,
    ProviderReply,
    ScriptedProvider,
    estimate_cost,
    live_command,
)


class ScriptedProviderTest(unittest.TestCase):
    def test_tool_call_reply(self):
        provider = ScriptedProvider([{"tool": "read_reliability", "args": {"case_id": "c1"}}])
        reply = provider.complete("sys", [{"role": "user", "content": "hi"}], [])
        self.assertEqual(reply.tool_calls[0]["tool"], "read_reliability")
        self.assertEqual(reply.provider_name, "scripted")

    def test_final_reply(self):
        provider = ScriptedProvider([{"text": "done", "claims": ["p1"]}])
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.text, "done")
        self.assertEqual(reply.claims, ("p1",))

    def test_records_what_it_was_asked(self):
        provider = ScriptedProvider([{"text": "done"}])
        provider.complete("system-prompt", [{"role": "user", "content": "hello"}], [{"name": "t"}])
        self.assertEqual(len(provider.asked), 1)
        self.assertEqual(provider.asked[0]["system"], "system-prompt")

    def test_exhausted_replies_report_error(self):
        provider = ScriptedProvider([])
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.error, "scripted_replies_exhausted")

    def test_error_reply_scripted_directly(self):
        provider = ScriptedProvider([{"error": "boom"}])
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.error, "boom")


class EstimateCostTest(unittest.TestCase):
    def test_zero_tokens_zero_cost(self):
        self.assertEqual(estimate_cost(0, 0, 1.0, 2.0), 0.0)

    def test_scales_with_tokens_and_rate(self):
        cost = estimate_cost(1000, 500, 0.5, 1.5)
        self.assertAlmostEqual(cost, 0.5 * 1.0 + 1.5 * 0.5)


class LiveCommandTest(unittest.TestCase):
    def test_live_command_contains_env_flag_and_case_id(self):
        cmd = live_command(case_id="case-42")
        self.assertIn("case-42", cmd)
        self.assertIn("$env:DERMPACT_ALLOW_LIVE_LLM='1'", cmd)

    def test_production_schema_enumerates_amendable_fact_names(self):
        from dermpact.core.agent.prompts import TOOLS_SCHEMA
        fact_tool = next(tool for tool in TOOLS_SCHEMA if tool["name"] == "propose_fact_amendment")
        self.assertIn("clinician.melanoma_suspected", fact_tool["args"]["fact"])
        self.assertIn("site.rcm_available", fact_tool["args"]["fact"])


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _ok_response(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)
    return handler


class LiveGatewayGatingTest(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.get("DERMPACT_ALLOW_LIVE_LLM")
        os.environ.pop("DERMPACT_ALLOW_LIVE_LLM", None)

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop("DERMPACT_ALLOW_LIVE_LLM", None)
        else:
            os.environ["DERMPACT_ALLOW_LIVE_LLM"] = self._saved_env

    def test_refuses_without_env_flag_or_allow_live(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={})

        provider = LiveGatewayProvider(transport=_mock_transport(handler))
        with self.assertRaises(LiveCallsDisabled):
            provider.complete("sys", [], [])
        self.assertEqual(calls, [], "no transport call must happen when live calls are disabled")

    def test_refuses_with_env_flag_alone(self):
        os.environ["DERMPACT_ALLOW_LIVE_LLM"] = "1"
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={})

        provider = LiveGatewayProvider(allow_live=False, transport=_mock_transport(handler))
        with self.assertRaises(LiveCallsDisabled):
            provider.complete("sys", [], [])
        self.assertEqual(calls, [])

    def test_refuses_with_allow_live_alone(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={})

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler))
        with self.assertRaises(LiveCallsDisabled):
            provider.complete("sys", [], [])
        self.assertEqual(calls, [])


class LiveGatewayEnabledTest(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.get("DERMPACT_ALLOW_LIVE_LLM")
        os.environ["DERMPACT_ALLOW_LIVE_LLM"] = "1"
        self._saved_key = os.environ.get("DERMPACT_API_KEY")
        os.environ["DERMPACT_API_KEY"] = "test-key-not-real"

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop("DERMPACT_ALLOW_LIVE_LLM", None)
        else:
            os.environ["DERMPACT_ALLOW_LIVE_LLM"] = self._saved_env
        if self._saved_key is None:
            os.environ.pop("DERMPACT_API_KEY", None)
        else:
            os.environ["DERMPACT_API_KEY"] = self._saved_key

    def test_body_shape_and_header_present_without_printing_key(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps({"final": {"text": "ok", "claims": []}})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler), model="test-model")
        reply = provider.complete("system prompt", [{"role": "user", "content": "hi"}], [{"name": "read_reliability", "args": {"case_id": "str"}}])
        self.assertEqual(reply.text, "ok")
        self.assertTrue(seen["auth"].startswith("Bearer "))
        self.assertNotIn("test-key-not-real", repr(reply))  # key never echoed back in the reply
        self.assertEqual(seen["body"]["model"], "test-model")
        self.assertIn("system prompt", seen["body"]["messages"][0]["content"])
        self.assertIn("read_reliability", seen["body"]["messages"][0]["content"])
        self.assertNotIn("tools", seen["body"], "JSON-command protocol must not pretend to use native tools")
        self.assertEqual(reply.usage, {"prompt_tokens": 10, "completion_tokens": 5})

    def test_nested_gateway_usage_details_are_ignored(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps({"final": {"text": "ok", "claims": []}})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "cost_microdollars": 7, "prompt_tokens_details": {"cached_tokens": 3}},
            })
        reply = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler)).complete("sys", [], [])
        self.assertEqual(reply.usage, {"prompt_tokens": 10, "completion_tokens": 5, "cost_microdollars": 7})

    def test_tool_call_parsed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            content = json.dumps({"tool": "read_reliability", "args": {"case_id": "c1"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler))
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.tool_calls[0]["tool"], "read_reliability")

    def test_internal_tool_trace_is_encoded_as_valid_chat_content(self):
        seen = {}
        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            content = json.dumps({"final": {"text": "ok", "claims": []}})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler))
        provider.complete("sys", [
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_call": {"tool": "read_reliability", "args": {}}},
            {"role": "tool", "tool": "read_reliability", "result": {"ok": True}},
            {"role": "system", "content": "repair_request", "rejections": [{"code": "x"}]},
        ], [{"name": "read_reliability", "args": {}}])
        for message in seen["body"]["messages"]:
            self.assertIn(message["role"], {"system", "user", "assistant"})
            self.assertIsInstance(message["content"], str)
            self.assertNotIn("tool_call", message)
            self.assertNotIn("result", message)

    def test_fenced_json_is_accepted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "```json\n{\"final\":{\"text\":\"ok\",\"claims\":[]}}\n```"}}]})
        reply = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler)).complete("sys", [], [])
        self.assertEqual(reply.text, "ok")

    def test_malformed_json_reported(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler))
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.error, "malformed")

    def test_neither_tool_nor_final_reported_malformed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            content = json.dumps({"something_else": True})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler))
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.error, "malformed")

    def test_retries_on_5xx_then_succeeds(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(500)
            content = json.dumps({"final": {"text": "ok after retry", "claims": []}})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler), retries=2)
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.text, "ok after retry")
        self.assertGreaterEqual(calls["n"], 2)

    def test_persistent_5xx_reports_provider_failed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler), retries=1)
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.error, "provider_failed")

    def test_timeout_reports_provider_failed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler), retries=1)
        reply = provider.complete("sys", [], [])
        self.assertEqual(reply.error, "provider_failed")

    def test_missing_api_key_raises(self):
        os.environ.pop("DERMPACT_API_KEY", None)
        os.environ.pop("KILOCODE_API_KEY", None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        saved_env_file = os.environ.get("DERMPACT_ENV")
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / "isolated.env"
            isolated.write_text("", encoding="utf-8")
            os.environ["DERMPACT_ENV"] = str(isolated)
            try:
                provider = LiveGatewayProvider(allow_live=True, transport=_mock_transport(handler))
                with self.assertRaises(RuntimeError):
                    provider.complete("sys", [], [])
            finally:
                if saved_env_file is None:
                    os.environ.pop("DERMPACT_ENV", None)
                else:
                    os.environ["DERMPACT_ENV"] = saved_env_file


if __name__ == "__main__":
    unittest.main()
