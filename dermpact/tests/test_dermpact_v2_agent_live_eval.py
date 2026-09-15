from __future__ import annotations

from dermpact.core.eval import agent_live_eval as live


def test_live_preflight_requires_both_explicit_gate_and_key(monkeypatch, tmp_path):
    isolated_env = tmp_path / "isolated.env"
    isolated_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("DERMPACT_ENV", str(isolated_env))
    monkeypatch.delenv("DERMPACT_ALLOW_LIVE_LLM", raising=False)
    monkeypatch.delenv("DERMPACT_API_KEY", raising=False)
    monkeypatch.delenv("KILOCODE_API_KEY", raising=False)
    assert live.live_preflight()[0] is False
    monkeypatch.setenv("DERMPACT_ALLOW_LIVE_LLM", "1")
    assert live.live_preflight()[0] is False
    monkeypatch.setenv("DERMPACT_API_KEY", "test-presence-only")
    assert live.live_preflight() == (True, "ready")


def test_main_refuses_before_building_tasks_or_sending(monkeypatch, tmp_path):
    isolated_env = tmp_path / "isolated.env"
    isolated_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("DERMPACT_ENV", str(isolated_env))
    monkeypatch.delenv("DERMPACT_ALLOW_LIVE_LLM", raising=False)
    monkeypatch.setattr(live, "load_frozen_tasks",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")))
    assert live.main(["--out-dir", str(tmp_path)]) == 2


def test_frozen_task_inventory_is_hash_verified():
    tasks = live.load_frozen_tasks()
    assert len(tasks) == 54
    assert len({task["task_id"] for task in tasks}) == 54
    assert live.PRIMARY_LIVE_MODEL == "qwen/qwen3-vl-32b-instruct"


def test_smoke_profile_is_five_prespecified_cross_family_tasks():
    selected = live.select_tasks(live.load_frozen_tasks(), profile="smoke5", limit=None)
    assert [task["task_id"] for task in selected] == list(live.SMOKE5_TASK_IDS)
    assert len({task["family"] for task in selected}) == 5


def test_exact_task_selection_is_ordered_and_deduplicated():
    tasks = live.load_frozen_tasks()
    selected = live.select_tasks(
        tasks, profile="full", limit=None,
        task_ids=["case_comparison-p1", "evidence_report-p1", "case_comparison-p1"],
    )
    assert [task["task_id"] for task in selected] == ["case_comparison-p1", "evidence_report-p1"]


def test_live_task_preserves_underlying_runner_error(monkeypatch):
    monkeypatch.setattr(live.offline, "_run_task_through_graph", lambda *_args, **_kwargs: {"error": "root-cause"})
    result = live.run_live_task({"message": "x", "kind": "paraphrase"}, live.TASK_INVENTORY.parent, model=live.PRIMARY_LIVE_MODEL)
    assert result["error"] == "root-cause"


def test_confirmatory_inventory_has_two_grounded_tasks_per_real_case(tmp_path):
    from collections import Counter
    from dermpact.core.eval import agent_confirmatory_inventory as confirmatory

    protocol = confirmatory.build(tmp_path)
    tasks = live.load_frozen_tasks(tmp_path / "task_inventory.json")
    counts = Counter(task["case_id"] for task in tasks)
    assert len(tasks) == 100
    assert len(counts) == 50
    assert set(counts.values()) == {2}
    assert {task["family"] for task in tasks} == {"evidence_report", "case_comparison"}
    assert protocol["hard_scope"]["official_or_final_rows_loaded"] == 0


def test_confirmatory_case_selection_is_100_label_free_fold_balanced():
    from collections import Counter
    from dermpact.core import inference_client

    original = set(inference_client._load_reserved_case_ids())
    selected = inference_client._resolve_confirmatory_cases()
    assert len(selected) == 100
    assert original.issubset(selected)
    assert Counter(case.fold for case in selected.values()) == {fold: 20 for fold in range(5)}


def test_confirmatory300_freezes_three_tasks_and_one_ragas_task_per_case(tmp_path):
    from collections import Counter
    import json
    from dermpact.core.eval import agent_confirmatory300_inventory as confirmatory

    case_inventory = tmp_path / "cases"
    case_inventory.mkdir()
    rows = [{"case_id": f"case-{index:03d}", "fold": index % 5} for index in range(100)]
    (case_inventory / "inventory_manifest.json").write_text(json.dumps({
        "n_cases": 100, "rows": rows, "demo_subset": [], "generated_at": "fixed",
    }), encoding="utf-8")
    out_dir = tmp_path / "tasks"
    protocol = confirmatory.build(case_inventory, out_dir)
    full = live.load_frozen_tasks(out_dir / "task_inventory_300.json")
    ragas = live.load_frozen_tasks(out_dir / "task_inventory_ragas100.json")
    assert len(full) == 300 and len(ragas) == 100
    assert set(Counter(task["case_id"] for task in full).values()) == {3}
    assert len({task["case_id"] for task in ragas}) == 100
    assert all(task["action"]["args"].get("section") == "overview"
               for task in full if task["family"] == "evidence_report")
    assert protocol["hard_scope"]["official_or_final_rows_loaded"] == 0


def test_confirmatory_fact_metrics_score_valid_required_claims():
    import json
    from dermpact.core.eval import agent_confirmatory_metrics as metrics

    context = {"propositions": [
        {"id": "p1", "path_or_ref": "record.overview.heads.diagnosis_head.provisional_class"},
        {"id": "p2", "path_or_ref": "record.overview.heads.melanoma_head.probability"},
        {"id": "p3", "path_or_ref": "unrelated.detail"},
    ]}
    scored = metrics._fact_metrics({
        "family": "evidence_report", "claims": ["p1", "p2"],
        "grounding_contexts": [json.dumps(context)],
    })
    assert scored["fact_precision"] == 1.0
    assert scored["required_fact_traceability"] == 1.0
    assert scored["fact_f1"] == 1.0
