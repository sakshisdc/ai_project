"""Run this first:  python check_setup.py   - tells you exactly what is wrong with your setup."""
import importlib, platform, sys
from pathlib import Path

root = Path(__file__).resolve().parent
ok = True
def line(flag, msg):
    global ok
    ok = ok and flag
    print(("[ OK ] " if flag else "[FAIL] ") + msg)

print(f"Python {platform.python_version()}  ({sys.executable})")
line(sys.version_info >= (3, 9), "Python >= 3.9 (3.10+ recommended)")
for mod, pip in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                 ("joblib", "joblib"), ("matplotlib", "matplotlib")):
    try:
        m = importlib.import_module(mod)
        line(True, f"{pip} {getattr(m, '__version__', '')}")
    except ImportError:
        line(False, f"{pip} missing  ->  \"{sys.executable}\" -m pip install -r requirements.txt")
for rel in ("adaptive_minimax/game.py", "scripts/01_baselines.py", "web/index.html", "data/positions.csv", "models/classifier.joblib"):
    here = (root / rel).exists()
    line(here, f"file {rel}" + ("" if here else "   <-- wrong folder? open the folder that contains serve.py"))
if ok:
    try:
        sys.path.insert(0, str(root))
        from adaptive_minimax.experiments import load_artifacts, make_agents
        make_agents(load_artifacts())
        line(True, "models load and all 6 agents build")
    except Exception as e:  # noqa: BLE001
        line(False, f"loading models failed: {type(e).__name__}: {e}")
print("\nEverything looks good - run:  python serve.py" if ok else "\nFix the [FAIL] lines above, then run this again.")
