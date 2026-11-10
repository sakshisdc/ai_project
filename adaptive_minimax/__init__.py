"""Adaptive Minimax Search with Intelligent Cutoff Depth Control.

Package layout (see README.md for the full workflow):

    game.py          Connect Four (bitboards)                   -> Phase 1
    evaluation.py    static heuristic evaluation                -> Phase 1
    features.py      per-position features used by all models   -> Phase 3/4
    search.py        minimax / alpha-beta / iterative deepening
                     / selective extension, hard node budgets   -> Phase 1, 2, 5
    policies.py      rule-based (C) and ML-based (D) cutoffs    -> Phase 5
    fastmodels.py    fast pure-python inference for sklearn     -> Phase 5
    error_models.py  p(e|x): binned (E) and learned (F)         -> Phase 4
    voc.py           Monte-Carlo value-of-computation search    -> Phase 5
    agents.py        Agents A-F                                 -> Phase 2, 5
    dataset.py       training-data generation + regret labels   -> Phase 3
    training.py      model training / validation                -> Phase 4
    experiments.py   benchmarks, tournaments, ablations         -> Phase 6
    plotting.py      figures                                    -> Phase 6
"""
import os
import sys
import tempfile
import warnings

__version__ = "1.0.2"

# Friendly message when the wrong Python interpreter is used (very common in VS Code / Colab).
_missing = []
for _mod, _pip in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                   ("sklearn", "scikit-learn"), ("joblib", "joblib")):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pip)
if _missing:
    raise SystemExit(
        "\nMissing Python packages: " + ", ".join(_missing)
        + "\nThis Python is: " + sys.executable
        + "\nFix:  \"" + sys.executable + "\" -m pip install -r requirements.txt"
        + "\n(In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' and pick the one you installed packages into.)\n"
    )

# Configure MPLCONFIGDIR if not already set to prevent non-writable home dir warnings on Windows
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

# Scikit-learn backwards compatibility shim for unpickling pre-trained GradientBoosting models:
# Models pickled under scikit-learn <= 1.8 refer to `_loss.CyHalfBinomialLoss`, whereas modern
# scikit-learn places this under `sklearn._loss._loss`.
try:
    import sklearn._loss._loss as _cy_loss  # noqa: F401
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _cy_loss
except (ImportError, AttributeError):
    pass


# Saved models may have been pickled with a different scikit-learn version; they load fine, so keep the console clean.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
