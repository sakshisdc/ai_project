# Adaptive Minimax Search with Intelligent Cutoff Depth Control

**Sakshi Chaudhari – B.Tech DSAI, IIT Roorkee**

Can an ML-assisted, probabilistic *Value-of-Computation* (VOC) scheduler spend a fixed search budget
more effectively than fixed-depth minimax, iterative deepening, rule-based adaptive search and a
non-learned probabilistic VOC controller?

This repository implements the complete workflow of the project proposal (Phases 1-6) on **Connect Four
(6x7)**, with six agents compared under identical node budgets and the same evaluation function.

---------------------------------------------------------------------------------------------------

## 1. Quick start

> **Web demo game modes:** `python serve.py`, then open http://localhost:8000 (or http://127.0.0.1:8000).
> * **Human vs AI** and **AI vs Human**: the AI plays automatically - it opens the game when it is Player 1 and replies
>   instantly after every move you make (no "Step AI Move" click). *Undo* takes back a full turn (your move and the AI's reply).
> * **Human vs Human** and **AI vs AI**: unchanged (manual clicks / "Step AI Move" / "Auto-Play Match").
> * Pick the players with the two drop-downs; the banner under them shows the current mode. Default: Human (Red) vs Agent F (Yellow).

> **Not working? Three ways to run it**
> 1. **Check your setup first:** `python check_setup.py` (it names the exact problem: wrong Python, missing package, wrong folder).
> 2. **VS Code:** *File -> Open Folder* and choose the folder that contains `serve.py` (not its parent). Pick your interpreter
>    (Ctrl+Shift+P -> *Python: Select Interpreter*), run `pip install -r requirements.txt` in its terminal, then use the
>    **Run and Debug** panel -> *Web demo (serve.py)*, or type `python serve.py` and open the printed URL.
> 3. **Google Colab:** upload this zip to https://colab.research.google.com via `Adaptive_Minimax_Colab.ipynb`
>    (File -> Upload notebook), then Run all. The web demo opens inside the notebook.
> Windows shortcut: double-click `start_web.bat`. Never open `web/index.html` directly - it needs `serve.py`.

> **Windows users:** extract the zip first (Right-click -> Extract All), open PowerShell **in this folder**
> (the one containing `scripts\`, `tests\` and `run_all.ps1`), then run `.\run_all.ps1 -Quick`
> (or `run_all.bat --quick` in CMD). Do not use `run_all.sh` on Windows. If PowerShell blocks scripts:
> `powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick`.
> `python serve.py` starts the interactive web demo at http://127.0.0.1:8000.
> Note: the *quick* runner retrains the models from a small dataset and overwrites `models/` and `results/`;
> make a copy of the folder first if you want to keep the committed full-run numbers.

```bash
pip install -r requirements.txt

python tests/run_tests.py                     # Phase 1  correctness tests (or: pytest -q)
python scripts/01_baselines.py                # Phase 2  fixed-depth / ID baselines + calibration
python scripts/02_generate_dataset.py         # Phase 3  labelled dataset   -> data/positions.csv
python scripts/03_train_models.py             # Phase 4  train + validate   -> models/*.joblib
python scripts/04_run_experiments.py          # Phase 5+6 tuning, benchmark, tournament, ablation
python scripts/05_make_plots.py               # figures  -> results/figures/
# or run everything:
#   Linux/macOS: ./run_all.sh
#   Windows (PowerShell): .\run_all.ps1 [-Quick]
#   Windows (CMD): run_all.bat [--quick]
```

Smoke test in ~1 minute: `python scripts/01_baselines.py --quick`, then `python scripts/04_run_experiments.py --quick`, then `python scripts/05_make_plots.py`.

### How to train (what each step does)
| Step | Command | What is learned / produced |
|---|---|---|
| Dataset | `02_generate_dataset.py --games 600` | Samples positions from random + noisy self-play games, runs a shallow search, a deep reference search (depth 7, exact Q for every move), simulates extra computation, and labels it by *regret reduction per 1000 nodes* (use `--out` to customize output path) |
| Classifier (agent D) | `03_train_models.py --only classifier` | Compares logistic regression, decision tree, random forest, gradient boosting with **group-k-fold CV by game**; picks the best, tunes the threshold on validation games, scores once on test games |
| Error model (agent F) | `03_train_models.py --only error` | Gradient-boosting regressors for mu(x) and sigma(x) of the evaluation error `e = v_ref - v_static`; compared with a non-learned binned table (agent E) and a constant Gaussian |
| Tuning | `04_run_experiments.py --only tune` | C, D and VOC hyper-parameters chosen on **validation** games only |

Splits are by complete game (train 60 % / val 15 % / test 25 %); **all reported numbers use test games only.**
Useful flags:
* `01_baselines.py`: `--quick`, `--games`, `--max-depth`
* `02_generate_dataset.py`: `--games`, `--ref-depth`, `--processes`, `--out`
* `03_train_models.py`: `--only` (`all` / `classifier` / `error`), `--data`
* `04_run_experiments.py`: `--quick`, `--only` (`tune` / `benchmark` / `tournament` / `ablation`), `--budgets`, `--n-positions`, `--openings`, `--pairs` (results of partial runs are merged into the CSVs, so a long run can be split).
* `05_make_plots.py`: generates all publication figures to `results/figures/`

---------------------------------------------------------------------------------------------------

## 2. Repository layout

```
adaptive_minimax/
  game.py          Connect Four bitboards (State, play, win detection)
  evaluation.py    shared static evaluation (69 windows + centre bonus, squashed to +-0.8)
  features.py      9 position features: phase, branching, eval, |eval|, immediate wins (own/opp),
                   open threes (own/opp), swing (= evaluation instability vs parent)
  search.py        minimax, alpha-beta, iterative deepening, hard node budget, selective extension
  policies.py      rule-based (C) and ML-based (D) extension policies
  fastmodels.py    compiles sklearn trees/forests/GBMs/linear models to microsecond pure-python closures
  error_models.py  p(e|x) Gaussian: constant / binned (E) / learned GBM (F)
  voc.py           best-first Monte-Carlo VOC search (E, F)
  agents.py        agents A-F
  dataset.py       Phase 3 sampling + regret labelling
  training.py      Phase 4 model comparison / validation
  experiments.py   Phase 2 + 6: baselines, tuning, benchmark, tournament, ablation
  plotting.py      figures
scripts/           01_baselines ... 05_make_plots (thin CLI wrappers)
tests/             27 tests (game rules, alpha-beta == minimax, budgets, compiled models, VOC invariants, features & policies)
data/ models/ results/   generated artefacts (committed: the numbers below can be reproduced from them)
```

---------------------------------------------------------------------------------------------------

## 3. The agents

| | Strategy | Idea |
|---|---|---|
| **A** | fixed-depth alpha-beta | depth chosen so its *average* cost fits the budget (calibration, Phase 2) |
| **B** | iterative deepening + alpha-beta | strong practical baseline; uses the partial last iteration |
| **C** | B + **rule-based** selective extension | at the horizon, extend 1 ply if someone has an immediate win or the eval swing is large |
| **D** | B + **ML** selective extension | same hook, but a classifier "is extra search worthwhile?" decides |
| **E** | **probabilistic VOC**, empirical *binned* error table, Monte Carlo | non-learned controller |
| **F** | **probabilistic VOC**, *learned* (GBM) error distribution, Monte Carlo | **proposed** |

**Node accounting is identical everywhere:** 1 node = 1 position generated; every extension-policy query is
charged 1 extra node. B-F enforce the budget *hard* (tests check it). A cannot be interrupted, so it is
calibrated and may overshoot on individual moves (this is what "budget compliance" measures).

### The VOC scheduler (agents E/F) – `voc.py`
Best-first search over an explicit tree. At each step:
1. candidates = PV leaf of the top-k root moves + "challenger" (2nd-best child) leaf at every node on those PVs;
2. leaf value belief `N(static + mu(x), sigma(x)^2)` from the error model;
3. alpha-beta-style relevance window prunes leaves that cannot change the root decision;
4. M antithetic samples are pushed through the ancestors (negamax) to the root;
5. `VOC(l) = mean_m max_a Q_a^(m) - max(max_{a'!=a_l} Q_a', mean_m Q_{a_l}^(m)) - lambda*cost(l)`;
6. expand the best leaf, back values up, repeat until the budget is gone (optionally: until net VOC <= 0).

**Two deliberate deviations from the proposal's formula – please read:**
* The proposal subtracts `max_a Q_a` evaluated at the leaf's mean. For a leaf under a MIN node this can be
  *negative* (E[min] < min E). I use the best action's expected utility *under the same samples* as baseline;
  this is >= 0 (Jensen) and exactly 0 for irrelevant leaves (unit-tested).
* Expanding a leaf is modelled as revealing a sample of its "true" value (value of perfect information about
  that leaf, a standard myopic approximation); lambda converts nodes to utility.

### The "novel addition" as implemented
* **Regret-based labels** (`dataset.py`): shallow decision -> deep reference -> regret -> simulate extra search
  -> label by regret reduction relative to node cost. The same dataset gives the error targets `e = v_ref - v_static`.
* **Evaluation-error distributions** (`error_models.py`): Gaussian `N(mu(x), sigma(x)^2)`; mu = systematic bias,
  sigma = uncertainty. sigma uses out-of-fold residuals and shrinkage toward the global variance (the first
  version was over-confident on a few positions: test NLL 1.99 -> -0.05 after the fix).

---------------------------------------------------------------------------------------------------

## 4. Results (Connect Four, held-out test games)

Dataset: 3032 positions / 600 games (746 test positions). Reference = depth-7 full-window search.
Benchmark = 400 random held-out test positions. Tournament = 60 games per matchup (30 openings x both colours).
All numbers are from `results/*.csv`; figures in `results/figures/`.

### 4.1 Learned components (Phase 4)
| "extra search worthwhile?" classifier | group-CV AUC | test AUC |
|---|---|---|
| logistic regression | 0.629 | 0.607 |
| decision tree | 0.669 | 0.671 |
| random forest | 0.706 | 0.676 |
| **gradient boosting (selected)** | **0.726** | **0.694** |

| error model p(e\|x) (test) | NLL (lower better) | RMSE of mu | 1-sigma / 2-sigma coverage |
|---|---|---|---|
| constant Gaussian | 0.525 | 0.407 | 0.75 / 0.93 |
| binned empirical (E) | 0.272 | 0.399 | 0.69 / 0.93 |
| **learned GBM (F)** | **-0.051** | **0.286** | 0.80 / 0.94 |

### 4.2 Decision quality vs reference (mean regret, lower = better; optimal-move rate in brackets)
| agent | 100 nodes | 300 | 1000 | 3000 |
|---|---|---|---|---|
| A fixed depth | 0.052 (.78) | 0.027 (.81) | 0.020 (.87) | 0.019 (.82) |
| B iterative deepening | 0.049 (.75) | 0.043 (.79) | 0.021 (.82) | 0.018 (.86) |
| C rule adaptive | 0.116 (.66) | 0.039 (.77) | 0.033 (.79) | 0.025 (.82) |
| D ML adaptive | 0.041 (.76) | 0.050 (.76) | 0.030 (.81) | 0.018 (.86) |
| E VOC, binned | 0.041 (.76) | 0.025 (.79) | 0.022 (.81) | 0.022 (.80) |
| **F VOC, learned** | **0.021 (.82)** | **0.017 (.82)** | **0.015 (.83)** | **0.013 (.85)** |

### 4.3 Head-to-head score of the first agent (win = 1, draw = 0.5; 95 % CI about +-0.12)
| matchup | 300 nodes | 1000 nodes |
|---|---|---|
| F vs A | 0.575 | 0.508 |
| F vs B | 0.675 | 0.667 |
| F vs C | 0.733 | 0.658 |
| F vs D | 0.808 | 0.675 |
| F vs E | 0.500 | 0.542 |
| C / D / E vs B | 0.51 / 0.52 / 0.69 | 0.45 / 0.45 / 0.48 |

### 4.4 Ablation of F at 1000 nodes (300 test positions)
| variant | regret | nodes used |
|---|---|---|
| **F full** | **0.0148** | 718 |
| without bias mu(x) (mu=0) | 0.0318 | 753 |
| constant sigma instead of sigma(x) | 0.0211 | 624 |
| E: binned table instead of learned | 0.0246 | 656 |
| M = 8 / 128 samples | 0.0147 / 0.0151 | 725 / 718 |
| k_root = 6 candidate lines (more spread out) | 0.0285 | 735 |
| + stop rule (net VOC <= 0), lambda = 1e-4 / 1e-3 / 1e-2 | 0.018 / 0.023 / 0.038 | 23 / 19 / 8 |
| B iterative deepening | 0.0234 | 804 |

---------------------------------------------------------------------------------------------------

## 5. Honest reading of the results

**What the data support**
* F has the lowest decision regret at every budget and the highest optimal-move rate at 100-300 nodes. F beats
  B, C and D head-to-head (about 0.66-0.81 score; the 95 % interval excludes 0.5 for most, not all, of these).
* The *learned bias term* is the most important ingredient (regret 0.015 -> 0.032 without it), then the
  *learned heteroscedastic sigma* (-> 0.021). Learned > binned > constant on NLL and on regret.
* The stop rule works as a compute saver: ~20 nodes still gives regret 0.018-0.023 (comparable to B with 800).

**What they do not support / caveats**
* **F is not clearly better than A or E in games**: F vs A = 0.51 at 1000 nodes, F vs E = 0.50-0.54 (all within noise
  with 60 games). E also beat B at 300 but not at 1000. Single run, no repeated seeds.
* **Equal *node* budgets are not equal *time* budgets.** Python F costs ~75 ms per move at 1000 nodes vs 2.4 ms for B
  (3000 nodes: 268 ms vs 9 ms). Feature extraction, model inference and Monte Carlo are not counted as nodes
  (only policy queries of C/D are). A time-matched comparison would be far less favourable to F; a C++/vectorised
  implementation would narrow the gap but this was not tested.
* **ML cutoff (D) did not help**: the classifier is weak (test AUC 0.69), tuning pushed the threshold to 0.7 (it
  rarely extends), so D ~ B. The proposal's hypothesis "learning beats manual rules" is *not* supported for the
  cutoff classifier; it is supported only for the error-distribution model inside VOC.
* The reference is itself a depth-7 heuristic search, not the game-theoretic truth; at 3000 nodes agents reach depth
  5-6, so differences at large budgets are small and partly noise. The depth-7 row of
  `baseline_calibration.csv` is trivially 0 regret (it *is* the reference).
* Dataset is small (3032 positions, one game, one evaluation function); the optional "recursive vs best-first"
  comparison of Phase 5 is realised as C/D (recursive extension) vs E/F (best-first).
* Myopic VOC is zero whenever no *single* expansion can flip the decision, so the stop rule is conservative in
  logic but, empirically, did not hurt much; with it off the budget is always used.
* Budget compliance: A overshoots on ~32-39 % of moves (61-68 % compliant); B-F are 100 % by construction.

**Search allocation (`results/figures/05_allocation.png`).** At 1000 nodes the VOC agents search very *selectively*:
F reaches a deepest ply of ~20 on quiet positions (B/C/D: ~5), with k_root = 1 (chosen on validation) it follows a
narrow principal line rather than a broad tree. All agents use fewer nodes in tactical positions (an immediate win
or forced reply ends the search early), and E/F use fewer nodes when the branching factor is small. Depth grows
with evaluation instability for F (12 -> 20 plies from low to high swing). This is *consistent with* adaptive
allocation, but it was not tested whether the extra depth is what causes the lower regret.

## 6. Extending
Other games: implement `State` (`legal_moves`, `play`, `winning_moves`) + an `analyze()` evaluation and keep the
feature names. Better VOC: multi-leaf (non-myopic) VOC, expansion that reveals a *partial* correction, time-based
costs. Tests: `pytest -q` (27 tests).
