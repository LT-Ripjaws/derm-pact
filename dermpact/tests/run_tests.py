"""Dependency-light test runner; run fixture-taking tests with `python -m pytest dermpact/tests -q`."""
from __future__ import annotations

import importlib
import inspect
import io
import traceback
import unittest
from pathlib import Path
from unittest import SkipTest

TESTS_DIR = Path(__file__).resolve().parent


def _required_parameters(function) -> list[str]:
    params = inspect.signature(function).parameters.values()
    return [p.name for p in params if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]


def _discover() -> list[str]:
    return [f"dermpact.tests.{p.stem}" for p in sorted(TESTS_DIR.glob("test_*.py"))]


def main() -> int:
    total = 0
    failures: list[str] = []
    skipped: list[str] = []
    for module_name in _discover():
        try:
            module = importlib.import_module(module_name)
        except SkipTest as exc:
            print(f"SKIP module {module_name}: {exc}")
            continue
        except Exception:
            print(f"FAIL import {module_name}")
            print("  " + traceback.format_exc().replace("\n", "\n  "))
            failures.append(module_name)
            continue
        print(module_name)
        for name in sorted(v for v in dir(module) if v.startswith("test_")):
            function = getattr(module, name)
            if not callable(function):
                continue
            total += 1
            needed = _required_parameters(function)
            if needed:
                print(f"  SKIP {name}: needs pytest fixture(s): {', '.join(needed)}")
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
            skipped.extend(f"{module_name}.{t}" for t, _ in result.skipped)
            failures.extend(f"{module_name}.{t}" for t, _ in [*result.failures, *result.errors])
            print("  " + stream.getvalue().strip().replace("\n", "\n  "))
    passed = total - len(failures) - len(skipped)
    print(f"\n{passed}/{total} passed, {len(skipped)} skipped, {len(failures)} failed")
    if failures:
        print("FAILURES: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
