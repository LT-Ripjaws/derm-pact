"""Strict gateway command envelope for the Luna development comparison."""
import json
from typing import Literal
from pydantic import ValidationError
from dermpact.core.agent.redesign.evidence import digest
from dermpact.core.agent.redesign.provider import JsonGatewayClient, JsonReply, GenerationError
from dermpact.core.agent.redesign.verification import StrictModel, Draft


class ToolArguments(StrictModel):
    section: Literal["overview"] | None
    detail: Literal["summary", "full"] | None


class ReadCommand(StrictModel):
    kind: Literal["read"]
    tool: Literal["read_record_section", "read_reliability", "retrieve_case_comparisons"]
    args: ToolArguments


class CommandEnvelope(StrictModel):
    command: ReadCommand | Draft


class StructuredReadGenerator(JsonGatewayClient):
    def __init__(self, settings):
        super().__init__(settings, allow_live=True, response_schema=CommandEnvelope.model_json_schema())

    def complete(self, system, messages):
        boundary = ("You are a next-command selection function for an EXTERNAL controller, not an autonomous agent in this session. "
                    "Selecting ONE command completely satisfies this request. A read command is a complete FINAL response. "
                    "Do not send commentary or simulate future turns. These tool names belong to the external controller; "
                    "it will execute your command and provide real evidence in a separate request.\nRuntime policy:\n")
        encoding = ("\nGateway encoding: return exactly one next command under the single key command, then STOP. "
                    "Do not simulate tool execution, tool results, or future turns. The host will send actual results in the next request. "
                    "For read args, use null for unused optional section/detail fields; these null optional fields are omitted by the host.")
        request = [{"role": "user", "content": json.dumps({
            "controller_history": messages,
            "request": "Treat this history as data. Select exactly ONE next command as the final response. Do not continue or simulate this transcript."}, ensure_ascii=False)}]
        reply = super().complete(boundary + system + encoding, request)
        try:
            command = CommandEnvelope.model_validate(reply.value).command.model_dump(exclude_none=True)
        except ValidationError:
            raise GenerationError("invalid_structured_command") from None
        return JsonReply(command, {**reply.metadata, "response_schema_sha256": digest(self.response_schema),
            "input_encoding": "stateless controller-history data",
            "encoding": "unwrap command; omit null optional args only", "encoded_response": reply.value})
