"""Fallback runner when pytest is not installed:  python tests/run_tests.py"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_requirements as t
fails = 0
for n in sorted(x for x in dir(t) if x.startswith("test_")):
    try:
        getattr(t, n)(); print(f"PASS {n}")
    except Exception:
        fails += 1; print(f"FAIL {n}"); traceback.print_exc()
print(f"\n{'ALL TESTS PASSED' if not fails else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
