"""Central place for paths and the main hyper-parameters.

Every script imports from here, so changing a value in one place changes the
whole pipeline. Command-line flags of the scripts override the most important
ones (see `python scripts/<name>.py --help`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"

# Ensure essential output directories exist
for _d in (DATA_DIR, MODEL_DIR, RESULT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

POSITIONS_CSV = DATA_DIR / "positions.csv"
CALIBRATION_CSV = RESULT_DIR / "baseline_calibration.csv"

# ---- Phase 3: dataset / labelling -------------------------------------------
REF_DEPTH = 7          # deep reference search (full window on every root move)
SHALLOW_DEPTH = 1      # the "cheap" decision whose regret we try to reduce
EXTRA_DEPTH = 3        # the "additional computation" that is simulated
LABEL_RHO = 0.25       # label = 1  if  regret reduction per 1000 extra nodes >= RHO
                       #   (i.e. "this extra search paid for itself")

# ---- Phase 6: experiment budgets (nodes per move) ---------------------------
BUDGETS = [100, 300, 1000, 3000]
TOURNAMENT_BUDGETS = [300, 1000]

# ---- Splits (by complete game, never by position) ---------------------------
TEST_FRACTION = 0.25
VAL_FRACTION = 0.15
SEED = 0
