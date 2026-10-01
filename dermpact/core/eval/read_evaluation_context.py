"""Source evidence actually exposed to a read-answer generator."""
import json


def read_evaluation_contexts(result):
    contexts = []
    for message in result["messages"]:
        if message["role"] != "user":
            continue
        event = json.loads(message["content"])
        if "active_case" in event:
            contexts.append(json.dumps({"active_case": event["active_case"]}, ensure_ascii=False))
        if event.get("host_event") in {"case_input", "retrieval_result", "tool_result"}:
            contexts.append(json.dumps(event["receipt"], ensure_ascii=False))
    return contexts
