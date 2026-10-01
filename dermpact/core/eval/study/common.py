"""Small shared artifact/locking contract; importing this module makes no model call."""
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dermpact.core.agent.redesign.provider import ModelSettings

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / "research/experiments/dermpact_agent_redesign/study_v10"
BUNDLE = ROOT / "research/experiments/dermpact_agent_redesign/evaluation_v2"
GENERATOR = ModelSettings(model="gpt-5.6-luna", backend="codex_cli", temperature=None, top_p=None, max_tokens=None,
                          reasoning_enabled=True, reasoning_effort="medium")
VERIFIER = ModelSettings(model="gpt-5.6-luna", backend="codex_cli", temperature=None, top_p=None, max_tokens=None,
                         reasoning_enabled=True, reasoning_effort="low")
CONFIG = {"version": "dermpact-study-v10", "generator": asdict(GENERATOR), "verifier": asdict(VERIFIER),
    "judge": {"model": "openai/gpt-5.6-luna", "transport": "Kilo gateway", "effort": "low", "same_family": True},
    "ragas_version": "0.4.1", "factual_atomicity": "low", "factual_coverage": "high",
    "bootstrap_repetitions": 2000, "bootstrap_seed": 20260923,
    "read_arms": ["case_only", "standard_rag", "full", "no_verification", "check_only", "no_history", "raw_history"],
    "guideline_arms": ["constant_unknown", "naive_rag", "llm_rules", "forms", "executable"],
    "workflow_arms": ["full"], "persistence_arms": ["structured", "transcript"],
    "safety_read_arms": ["case_only", "standard_rag", "full"],
    "maximum_answer_repair": 1, "workflow_case_clusters": 30,
    "final_evaluation_requires_explicit_flag": True}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Immutable, atomic JSON result plus a hash sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError("Refusing to replace an existing artifact: " + str(path))
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
    Path(str(path) + ".sha256").write_text(sha(path) + "\n", encoding="ascii")


def checked(path):
    path = Path(path)
    if Path(str(path) + ".sha256").exists():
        if sha(path) != Path(str(path) + ".sha256").read_text(encoding="ascii").strip():
            raise ValueError("Artifact hash mismatch: " + str(path))
    return load(path)


def slug(task_id):
    return hashlib.sha256(task_id.encode()).hexdigest()[:20]


def result_path(root, suite, split, arm, task_id, phase="answers"):
    return Path(root) / phase / suite / split / arm / (slug(task_id) + ".json")


def execution_guard(root, split, *, allow_final=False):
    root = Path(root)
    lock = checked(root / "lock.json")
    for relative, expected in lock["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Source changed since freeze: " + relative)
    for relative, expected in lock["input_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError("Study input changed: " + relative)
    for relative, expected in lock.get("upstream_bindings", {}).items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Upstream evaluation binding changed: " + relative)
    from importlib.metadata import version
    for package, expected in lock.get("environment", {}).items():
        if package != "python" and version(package) != expected:
            raise ValueError("Dependency version changed: " + package)
    if split == "evaluation":
        if not allow_final:
            raise PermissionError("Final evaluation is excluded: pass --allow-final-evaluation only after explicit user authorization.")
        ready = checked(root / "READY.json")
        if ready.get("status") != "ready_for_evaluation" or ready["lock_sha256"] != sha(root / "lock.json"):
            raise PermissionError("The final readiness gate has not been satisfied")
    elif split != "development":
        raise ValueError("Unknown split")
    return lock


def call_cost(result):
    calls = result.get("calls", [])
    return {"model_calls": len(calls), "tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
            "provider_attempts": sum(c.get("attempts", 1) for c in calls),
            "reported_cost": sum(c["usage"]["cost"] for c in calls if isinstance(c.get("usage", {}).get("cost"), (int, float))),
            "cost_interpretation": "Provider-reported only; zero/missing cost is not proof of free inference."}


def stamp():
    return datetime.now(timezone.utc).isoformat()
