"""Tiny fallback test runner (use it if pytest is not installed):  python tests/run_tests.py"""
import importlib.util, inspect, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
failed = passed = 0
for f in sorted(ROOT.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            try:
                fn(); passed += 1; print(f"PASS {f.stem}::{name}")
            except Exception:
                failed += 1; print(f"FAIL {f.stem}::{name}"); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
