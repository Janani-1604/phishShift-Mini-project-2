"""One command to reproduce everything: python run_all.py"""
import subprocess, sys
steps = ["generate_data.py", "prepare_data.py", "train_eval.py", "robustness.py"]
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, s], check=True)
print("\n=== tests ===", flush=True)
try:
    import pytest  # noqa
    cmd = [sys.executable, "-m", "pytest", "-v", "tests"]
except ImportError:
    cmd = [sys.executable, "tests/run_tests.py"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2500:] + r.stderr[-1500:])
open("reports/pytest_log.txt", "w").write(r.stdout + r.stderr)
print("\nDone. Start the app with:  streamlit run app.py")
