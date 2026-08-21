"""Dependency-light runner for the maintained MedGuard suite."""
from __future__ import annotations

import importlib
import inspect
import io
import traceback
import unittest
from unittest import SkipTest


TEST_MODULES = [
    "medguard.tests.test_evidence_contract",
    "medguard.tests.test_c1_contract_scorer",
    "medguard.tests.test_c1_perception_binding",
    "medguard.tests.test_evidence_relation_assessor",
    "medguard.tests.test_atomic_claim_catalog",
    "medguard.tests.test_medguard_development_preflight",
    "medguard.tests.test_h2h_implementation_cards",
    "medguard.tests.test_h2h_protocol",
    "medguard.tests.test_h2h_generation",
    "medguard.tests.test_h2h_b4_adapter",
    "medguard.tests.test_h2h_b6_adapter",
    "medguard.tests.test_h2h_evidence_graph",
    "medguard.tests.test_matched_b0_b1_validate",
    "medguard.tests.test_matched_full_h2h_validate",
    "medguard.tests.test_deterministic_remediation",
    "medguard.tests.test_evidence_generator",
    "medguard.tests.test_evidence_pipeline_ops",
    "medguard.tests.test_generation_cache",
    "medguard.tests.test_process_trust",
    "medguard.tests.test_harness",
    "medguard.tests.test_metrics",
    "medguard.tests.test_statistics",
    "medguard.tests.test_manifest",
    "medguard.tests.test_control_graph",
    "medguard.tests.test_graph_audit",
    "medguard.tests.test_hybrid_retriever",
    "medguard.tests.test_image_retriever",
    "medguard.tests.test_gmm_adaptive_selector",
    "medguard.tests.test_dynarag_gmm_component_audit",
    "medguard.tests.test_audit_derm_embedding_duplicates",
    "medguard.tests.test_audit_derm_near_duplicates",
    "medguard.tests.test_freeze_bcn_external",
    "medguard.tests.test_freeze_derm_retrieval_benchmark",
    "medguard.tests.test_freeze_medguard_h2h_development",
    "medguard.tests.test_freeze_medguard_h2h_perception",
    "medguard.tests.test_derm_evidence_freeze",
    "medguard.tests.test_guideline_scope_v1",
    "medguard.tests.test_derm_guideline_recommendations_v1",
    "medguard.tests.test_guideline_catalog_feasibility",
    "medguard.tests.test_guideline_retriever",
    "medguard.tests.test_win_feasibility",
    "medguard.tests.test_derm_guideline_retrieval_benchmark_v1",
    "medguard.tests.test_freeze_ham_nested_calibration",
    "medguard.tests.test_derm_encoder_compare",
    "medguard.tests.test_derm_encoder_fusion",
    "medguard.tests.test_derm_external_bcn",
    "medguard.tests.test_derm_retrieval_repair_v2",
    "medguard.tests.test_derm_retrieval_tournament",
    "medguard.tests.test_research_governance",
    "medguard.tests.test_results_ledger",
]


def _required_parameters(function) -> list:
    """Names of parameters the caller must supply, i.e. pytest fixtures.

    Fixture-taking tests are skipped by this runner and must be run under
    pytest; see the note printed at the end of main().
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return []
    return [
        parameter.name
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]


def main() -> int:
    failures = []
    skipped = []
    total = 0
    for module_name in TEST_MODULES:
        print(f"== {module_name} ==")
        try:
            module = importlib.import_module(module_name)
        except Exception:
            print("  FAIL import")
            print("    " + traceback.format_exc().replace("\n", "\n    "))
            failures.append(f"{module_name}.__import__")
            continue
        for name in sorted(value for value in dir(module) if value.startswith("test_")):
            function = getattr(module, name)
            if not callable(function):
                continue
            total += 1
            if _required_parameters(function):
                # This runner is stdlib-only and cannot supply pytest fixtures.
                # Report rather than fail, so the gap stays visible instead of
                # being counted as a pass.
                needed = ", ".join(_required_parameters(function))
                print(f"  SKIP {name}: needs pytest fixture(s): {needed}")
                skipped.append(f"{module_name}.{name}")
                continue
            try:
                function()
                print(f"  PASS {name}")
            except SkipTest as exc:
                print(f"  SKIP {name}: {exc}")
                skipped.append(f"{module_name}.{name}")
            except Exception:
                print(f"  FAIL {name}")
                print("    " + traceback.format_exc().replace("\n", "\n    "))
                failures.append(f"{module_name}.{name}")
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        if suite.countTestCases():
            stream = io.StringIO()
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            total += result.testsRun
            skipped.extend(f"{module_name}.{test}" for test, _reason in result.skipped)
            failures.extend(f"{module_name}.{test}" for test, _trace in
                            [*result.failures, *result.errors])
            print("  " + stream.getvalue().strip().replace("\n", "\n  "))
    passed = total - len(failures) - len(skipped)
    print(f"\n{passed}/{total} passed, {len(skipped)} skipped, {len(failures)} failed")
    if skipped:
        print("SKIPPED: " + ", ".join(skipped))
        print("  Skips are not passes. Fixture-taking tests run under:")
        print("  python -m pytest medguard/tests -q")
    if failures:
        print("FAILURES: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
