from __future__ import annotations

import os

from dermpact import config


def test_stdlib_dotenv_fallback_loads_aliases_without_overwriting_process_env(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "KILOCODE_API_KEY=fake-test-key-not-secret\n"
        "KILOCODE_BASE_URL=https://example.invalid/gateway\n"
        "KILO_GENERATOR_MODEL=example/generator\n"
        "JUDGE_MODEL=example/judge\n"
        "API_CALL_DELAY_SEC=0.25\n",
        encoding="utf-8",
    )
    for name in ("DERMPACT_API_KEY", "KILOCODE_API_KEY", "DERMPACT_GATEWAY_BASE_URL", "KILOCODE_BASE_URL",
                 "DERMPACT_GENERATOR_MODEL", "KILO_GENERATOR_MODEL", "DERMPACT_JUDGE_MODEL", "JUDGE_MODEL",
                 "DERMPACT_API_DELAY", "API_CALL_DELAY_SEC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DERMPACT_GENERATOR_MODEL", "process/wins")
    config._load_env_file_without_dependency(env_file)
    assert os.environ["KILOCODE_API_KEY"] == "fake-test-key-not-secret"
    assert os.environ["DERMPACT_GENERATOR_MODEL"] == "process/wins"
    assert os.environ["KILO_GENERATOR_MODEL"] == "example/generator"
