# Adaptive Minimax Search: Codebase Audit & Update Report

**Project**: Adaptive Minimax Search with Intelligent Cutoff Depth Control (Connect Four 6x7, Agents A–F, MC-VOC Scheduler)  
**Location**: `adaptive-minimax/`  
**Date**: October 31, 2026  
**Status**: All audits completed, fixes applied, test suite expanded (19 -> 27 passing tests), and verified end-to-end.

---

## 1. Executive Summary

The codebase implements a complete 6-phase research workflow comparing six Connect Four search strategies under identical node budgets:
- **A**: Fixed-depth alpha-beta (calibrated against node budgets).
- **B**: Iterative deepening with alpha-beta (partial-iteration baseline).
- **C**: Rule-based selective extension at the horizon (tactical threats + eval swings).
- **D**: Machine-learning selective extension (gradient boosting classifier).
- **E**: Monte-Carlo Value-of-Computation (VOC) best-first search with empirical binned error table.
- **F**: Proposed Monte-Carlo VOC with learned heteroscedastic error distribution ($\mu(x), \sigma(x)$).

While the mathematical core and experimental methodology are strong, several critical runtime bugs, compatibility issues, packaging omissions, and platform limitations prevented the pipeline from running out of the box on modern Python environments and Windows systems.

All issues have been resolved, audited, and verified.

---

## 2. Issues Discovered & Resolved

### 2.1 Critical Runtime & Dependency Issues

| Issue | Root Cause | Impact | Resolution |
|---|---|---|---|
| **Scikit-Learn Pickling Mismatch** | Models pickled under scikit-learn $\le 1.8$ reference `_loss.CyHalfBinomialLoss`. In modern scikit-learn, this moved to `sklearn._loss._loss`. | `ModuleNotFoundError: No module named '_loss'` when loading `classifier.joblib` and `error_learned.joblib`. Crashed scripts 04 and 05. | Added backward-compatibility alias in `adaptive_minimax/__init__.py` mapping `sys.modules['_loss'] = sklearn._loss._loss`. |
| **Missing `pyproject.toml` / Pytest Discovery** | No package configuration or pythonpath definition existed. | Running `pytest` directly in the project root failed with `ModuleNotFoundError: No module named 'adaptive_minimax'`. | Created standard `pyproject.toml` with `[tool.pytest.ini_options] pythonpath = ["."]`. |
| **Incomplete `requirements.txt`** | `scipy` was omitted from `requirements.txt` despite being required by `scikit-learn`. | Clean installs in fresh virtualenvs could fail or rely on pip transitive resolution. | Added `scipy>=1.10` to `requirements.txt` and `pyproject.toml`. |
| **Windows Matplotlib Cache Warning** | Default `~/.matplotlib` is often non-writable in sandboxed or Windows corporate profiles. | Matplotlib printed verbose warning on every plot invocation. | Configured automatic `MPLCONFIGDIR` to a writable temp directory in `__init__.py`. |

### 2.2 Algorithmic, Data & Logic Fixes

| Component | Issue | Resolution |
|---|---|---|
| **`game.State` Value Equality** | `State` lacked `__eq__` and `__hash__` methods, relying solely on Python object identity (`id`). | Added `__hash__` returning `self.key` and `__eq__` comparing bitboards, move count, and win state. Enables sets and dict keys. |
| **`experiments.play_game` Self-Play Collision** | Stats were keyed by `id(a)` and `id(b)`. If an agent played against itself (`a is b`), stats dictionary keys collided, corrupting node counts and reporting a 100% win rate for Player 1. | Keyed statistics by player slot (`0` and `1`) and determined winners by player index instead of object identity. |
| **Tournament Opening Deduplication** | `experiments.tournament` selected random 2-ply openings with replacement. | Added deduplication loop ensuring distinct 2-ply opening configurations up to `n_openings`. |
| **`fastmodels._compile_tree` Single-Class Trees** | If a decision tree or random forest saw only 1 class during training, `tree.value` shape is `(n_nodes, 1, 1)` and `tree` lacks `classes_`. Accessing `v[:, 1]` raised `IndexError`. | Added class-cardinality check and passed `classes_` from the estimator into `_compile_tree` to handle single-class classifiers cleanly. |
| **Hardcoded `GroupKFold(n_splits=5)`** | `training.train_classifier` had hardcoded `n_splits=5`. If train split had $<5$ games (e.g. quick smoke tests), it crashed with `ValueError`. | Added dynamic split bounding: `n_splits = max(2, min(5, len(np.unique(gtr))))` and fallback for single-group datasets. |
| **`plotting.fig_allocation` Crash on Partial Runs** | `fig_allocation` hardcoded `budget=1000`. Running `04_run_experiments.py --quick` only computes budgets 100 and 300, causing an empty slice and `ValueError` in `pd.qcut`. | Dynamically inspect available budgets in `benchmark_raw.csv` and safely fallback to highest available budget. |
| **Multiprocessing Resource Cleanup** | `dataset.generate_dataset` did not wrap worker pool processing in a `try...finally` block to call `pool.close()` and `pool.join()`. | Added `try...finally` resource cleanup to avoid orphan worker processes on Windows. |
| **`search.SearchResult` Typing** | `res.n_done` was assigned dynamically, requiring `# type: ignore[attr-defined]`. | Added `n_done: int = 0` as a first-class field in the dataclass. |
| **Evaluation Cache Management** | `evaluation.py` lacked a public cache reset function. | Added `clear_cache()` for deterministic benchmarks and test isolation. |

### 2.3 Workflow & Usability Improvements

1. **Cross-Platform Execution**:
   - Added `run_all.ps1` (PowerShell for Windows, supports `-Quick`).
   - Added `run_all.bat` (Command Prompt for Windows, supports `--quick`).
   - Maintained `run_all.sh` (Bash for Linux/macOS).
2. **CLI Enhancements Across All Scripts**:
   - `scripts/01_baselines.py`: Added `--games`, `--max-depth`, `--quick`.
   - `scripts/02_generate_dataset.py`: Added `--out` to prevent quick test runs from overwriting the canonical 3,032-position dataset.
   - `scripts/03_train_models.py`: Added `--only` (`all`, `classifier`, `error`) and `--data`.
   - `scripts/05_make_plots.py`: Added `argparse` support.
3. **Expanded Test Suite**:
   - Created `tests/test_features_and_policies.py` testing feature extraction, rule-based policies, ML policies, state equality, calibration lookup, single-class model compilation, and self-play tournament games.
   - Total test count increased from 19 to 27 tests (100% pass rate).

---

## 3. File Modification Summary

```
adaptive-minimax/
├── pyproject.toml                         [NEW] Package metadata & pytest config
├── run_all.ps1                            [NEW] Windows PowerShell end-to-end runner
├── run_all.bat                            [NEW] Windows Command Prompt runner
├── AUDIT_AND_UPDATES.md                   [NEW] In-tree documentation of updates
├── requirements.txt                       [MODIFIED] Added scipy>=1.10
├── README.md                              [MODIFIED] Documented new CLI flags, runners, test count
├── adaptive_minimax/
│   ├── __init__.py                        [MODIFIED] Scikit-learn unpickle shim & MPLCONFIGDIR
│   ├── config.py                          [MODIFIED] Auto-creation of data/models/results/figures dirs
│   ├── game.py                            [MODIFIED] Implemented State.__eq__ and State.__hash__
│   ├── evaluation.py                      [MODIFIED] Added clear_cache()
│   ├── search.py                          [MODIFIED] Fixed SearchResult.n_done dataclass field
│   ├── policies.py                        [MODIFIED] Guarded None parent_eval in _features
│   ├── fastmodels.py                      [MODIFIED] Single-class tree compilation & zero-scale guard
│   ├── error_models.py                    [MODIFIED] Dynamic group count & finalize safety guard
│   ├── dataset.py                         [MODIFIED] try/finally pool cleanup with pool.join()
│   ├── training.py                        [MODIFIED] Dynamic CV splits & robust F1 thresholding
│   ├── experiments.py                     [MODIFIED] Safe calibration loader, self-play stats fix, opening deduplication
│   └── plotting.py                        [MODIFIED] Safe budget fallback in fig_allocation & bounds in fig_calibration
├── scripts/
│   ├── 01_baselines.py                    [MODIFIED] Added CLI args (--games, --max-depth, --quick)
│   ├── 02_generate_dataset.py             [MODIFIED] Added --out argument
│   ├── 03_train_models.py                 [MODIFIED] Added CLI args (--only, --data)
│   └── 05_make_plots.py                   [MODIFIED] Added CLI parser
└── tests/
    └── test_features_and_policies.py      [NEW] 8 new comprehensive unit tests
```

---

## 4. Verification Results

### 4.1 Unit Tests
```
cd adaptive-minimax
pytest
============================= 27 passed in 3.10s ==============================
```
Also verified with standalone runner:
```
python tests/run_tests.py
27 passed, 0 failed
```

### 4.2 End-to-End Pipeline
Executed `powershell -File .\run_all.ps1 -Quick`:
- **Phase 1**: 27 unit tests passed.
- **Phase 2**: Fixed-depth & iterative deepening baseline calibration completed.
- **Phase 3**: Dataset generation completed.
- **Phase 4**: Classifier comparison (logistic regression, decision tree, random forest, gradient boosting) and error models trained.
- **Phase 5+6**: Tuning, position benchmark, head-to-head tournament, and ablation study completed.
- **Phase 6**: All 7 PNG figures generated in `results/figures/`:
  - `01_classifier_comparison.png`
  - `02_error_models.png`
  - `03_budget_curves.png`
  - `04_tournament.png`
  - `05_allocation.png`
  - `06_ablation.png`
  - `07_sigma_calibration.png`
