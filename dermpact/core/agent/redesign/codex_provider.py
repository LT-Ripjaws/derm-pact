"""Isolated Codex CLI generation using the same external-controller contracts."""
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .evidence import digest
from .provider import GenerationError, JsonReply


def complete(settings, system, messages, schema, text_mode):
    # Reuse the proven CLI/schema utilities only, never the judge's instructions.
    from dermpact.core.eval.codex_judge import executable, strict_schema
    if settings.model != "gpt-5.6-luna":
        raise GenerationError("unexpected_codex_generator_model")
    artifacts = Path(os.environ["DERMPACT_CODEX_GENERATION_DIR"]).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    call_dir = Path(tempfile.mkdtemp(prefix="call-", dir=artifacts))
    instructions = call_dir / "instructions.md"
    instructions.write_text(system, encoding="utf-8")
    prompt = "Return the next assistant response for this supplied conversation.\n" + json.dumps(messages, ensure_ascii=False)
    (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    schema_path = call_dir / "schema.json"
    if schema is not None:
        schema_path.write_text(json.dumps(strict_schema(schema)), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"CODEX_THREAD_ID", "CODEX_SESSION_ID"}}
    attempts = []
    started = time.monotonic()
    for attempt in range(settings.retries + 1):
        output = call_dir / f"output-{attempt}.txt"
        with tempfile.TemporaryDirectory(prefix="dermpact-generator-") as workspace:
            command = [executable(), "exec", "--model", settings.model, "--ephemeral", "--ignore-user-config",
                "--skip-git-repo-check", "--sandbox", "read-only", "--cd", workspace,
                "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "skill_search",
                "--disable", "apps", "--disable", "plugins", "--disable", "multi_agent",
                "--enable", "skip_host_skill_discovery", "-c", "approval_policy=never",
                "-c", "suppress_unstable_features_warning=true", "-c", "web_search=disabled",
                "-c", "project_doc_max_bytes=0", "-c", "memories.use_memories=false",
                "-c", "memories.generate_memories=false", "-c", f"model_reasoning_effort={settings.reasoning_effort}",
                "-c", f'model_instructions_file="{instructions.as_posix()}"',
                "--output-last-message", str(output), "--json"]
            if schema is not None:
                command += ["--output-schema", str(schema_path)]
            command += ["-"]
            call_started = time.monotonic()
            try:
                result = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="strict",
                    capture_output=True, timeout=settings.timeout_s, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except subprocess.TimeoutExpired:
                attempts.append({"error": "codex_timeout", "elapsed_s": time.monotonic() - call_started})
                (call_dir / "attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
                if attempt == settings.retries:
                    raise GenerationError("codex_timeout") from None
                continue
        events = []
        for line in result.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        item_types = [e["item"].get("type") for e in events if e.get("item")]
        observed = re.search(r"(?m)^model:\s*(\S+)\s*$", result.stderr)
        metadata = {"exit_code": result.returncode, "elapsed_s": time.monotonic() - call_started,
            "item_types": item_types, "event_types": [e.get("type") for e in events],
            "usage": [e["usage"] for e in events if e.get("usage")],
            "observed_model": observed.group(1) if observed else None,
            "stderr_sha256": digest(result.stderr),
            "diagnostics": [e["item"].get("message", "")[:600] for e in events if e.get("item", {}).get("type") == "error"]}
        attempts.append(metadata)
        (call_dir / "attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
        if set(item_types) - {"agent_message", "reasoning", "error"}:
            raise GenerationError("unexpected_codex_tool_execution")
        if observed and observed.group(1) != settings.model:
            raise GenerationError("unexpected_codex_model_identity")
        if result.returncode or not output.exists():
            if attempt < settings.retries:
                continue
            raise GenerationError("codex_execution_failed")
        content = output.read_text(encoding="utf-8").strip()
        if not content:
            raise GenerationError("missing_final_content")
        try:
            value = {"text": content} if text_mode else json.loads(content)
        except ValueError:
            raise GenerationError("invalid_command_json") from None
        if not isinstance(value, dict):
            raise GenerationError("command_not_object")
        usages = [u for a in attempts for u in a.get("usage", [])]
        prompt_tokens = sum(u.get("input_tokens", 0) for u in usages)
        completion_tokens = sum(u.get("output_tokens", 0) for u in usages)
        return JsonReply(value, {"requested_model": settings.model, "observed_model": metadata["observed_model"],
            "model_identity_basis": "Explicit CLI model selection; observed identity recorded only when emitted.",
            "settings": asdict(settings), "reasoning_effort": settings.reasoning_effort,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for u in usages)},
            "transport": "codex_exec", "provider": "Codex", "output_format": "text" if text_mode else "json",
            "attempts": len(attempts), "elapsed_s": time.monotonic() - started,
            "instructions_sha256": digest(system), "prompt_sha256": digest(prompt),
            "schema_sha256": digest(schema), "call_artifacts": str(call_dir),
            "unsupported_controls": ["temperature", "top_p", "max_tokens"], "tools_and_memory_disabled": True})
    raise GenerationError("codex_execution_failed")
