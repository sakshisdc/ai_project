"""Phase 2 (baselines) and Phase 6 (controlled experiments).

Everything here is deterministic given the seeds, uses the SAME evaluation
function for every agent and the SAME node accounting, and evaluates only on
held-out games (`split == "test"`; tuning uses `split == "val"`).

Main entry points
    load_artifacts / make_agents      build agents A-F from the trained models
    run_baselines                     Phase 2: calibration of A + baseline quality table
    tune_policies                     pick C / D hyper-parameters on VALIDATION positions
    position_benchmark                decision quality / efficiency / compliance / allocation
    tournament                        head-to-head games (colours swapped, shared openings)
    ablation                          which component actually helps?
"""
from __future__ import annotations

import json
import random
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .agents import (FixedDepthAgent, IterativeDeepeningAgent, MLAdaptiveAgent,
                     RuleAdaptiveAgent, VOCAgent)
from .error_models import ConstantErrorModel
from .fastmodels import compile_estimator
from .game import State
from .search import search_fixed

TUNING_JSON = C.RESULT_DIR / "tuning.json"

DEFAULT_TUNING = {
    "C": {"swing_threshold": 0.5, "max_ext": 2},
    "D": {"threshold": None, "max_ext": 2},      # None -> use the F1-optimal threshold from training
    "VOC": {"lam": 2e-4, "n_samples": 32, "k_root": 3, "stop": False},
}


# ===================================================================== artifacts
def load_calibration() -> dict:
    if C.CALIBRATION_CSV.exists():
        try:
            df = pd.read_csv(C.CALIBRATION_CSV)
            if not df.empty and "depth" in df.columns and "mean_nodes" in df.columns:
                return {int(d): float(n) for d, n in zip(df.depth, df.mean_nodes)}
        except Exception:
            pass
    from .agents import DEFAULT_CALIBRATION
    return dict(DEFAULT_CALIBRATION)


def load_artifacts() -> dict:
    """Load trained models. If the saved .joblib files cannot be read (e.g. a different
    scikit-learn version than the one they were trained with), retrain them automatically
    from data/positions.csv (about a minute) and load again."""
    try:
        return _load_artifacts()
    except Exception as exc:  # noqa: BLE001 - any unpickling problem triggers a retrain
        print(f"[adaptive_minimax] saved models could not be loaded ({type(exc).__name__}: {exc}).\n"
              f"[adaptive_minimax] Retraining them from {C.POSITIONS_CSV} for this scikit-learn version ...", flush=True)
        from .training import load_dataset, train_classifier, train_error_models
        df = load_dataset(C.POSITIONS_CSV)
        train_classifier(df)
        train_error_models(df)
        return _load_artifacts()


def _load_artifacts() -> dict:
    clf = joblib.load(C.MODEL_DIR / "classifier.joblib")
    art = {
        "calibration": load_calibration(),
        "clf_fn": compile_estimator(clf["model"]),
        "clf_threshold": float(clf["threshold"]),
        "clf_name": clf["name"],
        "err": {k: joblib.load(C.MODEL_DIR / f"error_{k}.joblib") for k in ("constant", "binned", "learned")},
        "tuning": json.loads(TUNING_JSON.read_text()) if TUNING_JSON.exists() else DEFAULT_TUNING,
    }
    return art


def make_agents(art: dict, tuning: dict | None = None) -> dict:
    t = tuning or art["tuning"]
    d_thr = t["D"]["threshold"] if t["D"]["threshold"] is not None else art["clf_threshold"]
    v = t["VOC"]
    voc_kw = dict(lam=v["lam"], n_samples=v["n_samples"], k_root=v["k_root"], stop_on_nonpositive=v["stop"])
    return {
        "A": FixedDepthAgent(calibration=art["calibration"]),
        "B": IterativeDeepeningAgent(),
        "C": RuleAdaptiveAgent(t["C"]["swing_threshold"], t["C"]["max_ext"]),
        "D": MLAdaptiveAgent(art["clf_fn"], d_thr, t["D"]["max_ext"]),
        "E": VOCAgent(art["err"]["binned"], "E", "E: VOC + binned errors", **voc_kw),
        "F": VOCAgent(art["err"]["learned"], "F", "F: VOC + learned errors", **voc_kw),
    }


LABELS = {"A": "A fixed-depth", "B": "B iter. deepening", "C": "C rule adaptive",
          "D": "D ML adaptive", "E": "E VOC (binned)", "F": "F VOC (learned)"}


# ============================================================ position evaluation
def row_state(row) -> State:
    return State(int(row["cur"]), int(row["mask"]), int(row["ply"]), False)


def _regret(row, move: int) -> float:
    q = np.array([row[f"q{c}"] for c in range(7)], dtype=float)
    if move < 0 or move >= 7 or np.isnan(q[move]):
        return 1.0  # illegal move penalty
    best_q = np.nanmax(q)
    return float(best_q - q[move])


def evaluate_on_positions(agent, df: pd.DataFrame, budget: int, name: str | None = None) -> pd.DataFrame:
    """Let `agent` choose a move in every position; score against the deep reference."""
    out = []
    for _, row in df.iterrows():
        move, info = agent.choose(row_state(row), budget)
        reg = _regret(row, move)
        out.append(dict(agent=name or agent.name, budget=budget, regret=reg, optimal=reg < 1e-9,
                        tactical=int(row["my_win_now"] + row["opp_win_now"] > 0),
                        branching=row["branching"], phase=row["phase"], swing=row["swing"], **info))
    return pd.DataFrame(out)


def _subsample(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)


def position_benchmark(agents: dict, df: pd.DataFrame, budgets=C.BUDGETS, n_positions: int | None = None,
                       verbose: bool = True) -> pd.DataFrame:
    df = _subsample(df, n_positions)
    parts = []
    for b in budgets:
        for name, ag in agents.items():
            t0 = time.time()
            parts.append(evaluate_on_positions(ag, df, b, name))
            if verbose:
                r = parts[-1]
                print(f"  budget {b:5d}  agent {name}  regret {r.regret.mean():.4f}  "
                      f"optimal {r.optimal.mean():.3f}  nodes {r.nodes.mean():7.0f}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    return pd.concat(parts, ignore_index=True)


def summarize_benchmark(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["agent", "budget"])
    return g.agg(mean_regret=("regret", "mean"), optimal_rate=("optimal", "mean"),
                 mean_nodes=("nodes", "mean"), mean_ms=("time", lambda x: 1000 * x.mean()),
                 compliance=("within_budget", "mean"), mean_depth=("max_ply", "mean"),
                 n=("regret", "size")).reset_index()


# ================================================================ Phase 2 baselines
def run_baselines(n_games: int = 40, max_depth: int = 7, verbose: bool = True):
    """Phase 2: node/time/quality of fixed depths and of iterative deepening.

    Writes results/baseline_calibration.csv (used by agent A to map a node budget
    to a depth) and results/baseline_iterative_deepening.csv.
    """
    from .dataset import label_position, sample_game
    rows = []
    for g in range(n_games):
        for s, p in sample_game(g, 12345, 6):
            rows.append((s, label_position(s, p)))
    if verbose:
        print(f"  {len(rows)} calibration positions")

    cal = []
    for d in range(1, max_depth + 1):
        nodes, times, reg, opt = [], [], [], []
        for s, lab in rows:
            t0 = time.perf_counter()
            r = search_fixed(s, d)
            times.append(time.perf_counter() - t0)
            nodes.append(r.nodes)
            reg.append(_regret(lab, r.move))
            opt.append(reg[-1] < 1e-9)
        cal.append(dict(depth=d, mean_nodes=np.mean(nodes), median_nodes=np.median(nodes),
                        p90_nodes=np.percentile(nodes, 90), mean_ms=1000 * np.mean(times),
                        mean_regret=np.mean(reg), optimal_rate=np.mean(opt)))
        if verbose:
            print("  fixed depth", {k: round(float(v), 3) for k, v in cal[-1].items()})
    pd.DataFrame(cal).to_csv(C.CALIBRATION_CSV, index=False)

    idb = []
    agent = IterativeDeepeningAgent()
    for b in C.BUDGETS:
        reg, opt, nodes, ms, dep = [], [], [], [], []
        for s, lab in rows:
            mv, info = agent.choose(s, b)
            reg.append(_regret(lab, mv))
            opt.append(reg[-1] < 1e-9)
            nodes.append(info["nodes"]); ms.append(1000 * info["time"]); dep.append(info["depth"])
        idb.append(dict(budget=b, mean_nodes=np.mean(nodes), mean_ms=np.mean(ms), mean_depth=np.mean(dep),
                        mean_regret=np.mean(reg), optimal_rate=np.mean(opt)))
        if verbose:
            print("  iterative deepening", {k: round(float(v), 3) for k, v in idb[-1].items()})
    pd.DataFrame(idb).to_csv(C.RESULT_DIR / "baseline_iterative_deepening.csv", index=False)
    return pd.DataFrame(cal), pd.DataFrame(idb)


# ================================================================== tuning (val)
def tune_policies(art: dict, val_df: pd.DataFrame, n_positions: int = 250,
                  budgets=(300, 1000), verbose: bool = True) -> dict:
    """Choose C / D hyper-parameters on VALIDATION positions (never on test)."""
    val = _subsample(val_df, n_positions)
    tuning = json.loads(json.dumps(DEFAULT_TUNING))
    tuning["D"]["threshold"] = art["clf_threshold"]

    def score(agent) -> float:
        return float(np.mean([evaluate_on_positions(agent, val, b).regret.mean() for b in budgets]))

    best = (1e9, None)
    for sw in (0.35, 0.5, 0.7):
        for me in (1, 2):
            s = score(RuleAdaptiveAgent(sw, me))
            if verbose:
                print(f"  C swing={sw} max_ext={me}: val regret {s:.4f}")
            best = min(best, (s, (sw, me)), key=lambda x: x[0])
    tuning["C"] = {"swing_threshold": best[1][0], "max_ext": best[1][1]}

    best = (1e9, None)
    for thr in sorted({art["clf_threshold"], 0.3, 0.5, 0.7}):
        for me in (1, 2):
            s = score(MLAdaptiveAgent(art["clf_fn"], thr, me))
            if verbose:
                print(f"  D threshold={thr:.2f} max_ext={me}: val regret {s:.4f}")
            best = min(best, (s, (thr, me)), key=lambda x: x[0])
    tuning["D"] = {"threshold": best[1][0], "max_ext": best[1][1]}

    # VOC settings (shared by E and F; tuned with the learned model F, same protocol as C / D)
    best = (1e9, None)
    for k in (1, 3):
        for m in (16, 32):
            s = score(VOCAgent(art["err"]["learned"], "F", "F", lam=2e-4, n_samples=m, k_root=k))
            if verbose:
                print(f"  VOC k_root={k} n_samples={m}: val regret {s:.4f}")
            best = min(best, (s, (k, m)), key=lambda x: x[0])
    tuning["VOC"].update({"k_root": best[1][0], "n_samples": best[1][1]})

    TUNING_JSON.write_text(json.dumps(tuning, indent=2))
    if verbose:
        print("tuning ->", TUNING_JSON, tuning)
    return tuning


# ================================================================== tournaments
def play_game(a, b, budget: int, opening: list[int], a_first: bool) -> dict:
    """Play one game; returns the score of `a` (1 win / .5 draw / 0 loss) + move statistics."""
    s = State.from_moves(opening)
    players = (a, b) if a_first else (b, a)
    stats = {0: [], 1: []}
    a_idx = 0 if a_first else 1
    turn = 0
    while not s.is_terminal:
        turn_idx = turn % 2
        p = players[turn_idx]
        move, info = p.choose(s, budget)
        stats[turn_idx].append(info)
        s = s.play(move)
        turn += 1
    if s.won:                      # the player who just moved won
        winner_idx = (turn - 1) % 2
        score = 1.0 if winner_idx == a_idx else 0.0
    else:
        score = 0.5
    sa = stats[a_idx]
    mean_nodes = float(np.mean([i["nodes"] for i in sa])) if sa else 0.0
    mean_ms = float(1000 * np.mean([i["time"] for i in sa])) if sa else 0.0
    mean_comp = float(np.mean([i["within_budget"] for i in sa])) if sa else 1.0
    return dict(score=score, plies=turn, nodes_a=mean_nodes, ms_a=mean_ms, compliance_a=mean_comp)


def tournament(agents: dict, pairs, budgets=C.TOURNAMENT_BUDGETS, n_openings: int = 12,
               seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Every pair plays `n_openings` openings x 2 colours at every budget."""
    rng = random.Random(seed)
    openings = []
    attempts = 0
    while len(openings) < n_openings and attempts < 1000:
        attempts += 1
        s = State()
        op = []
        for _ in range(2):
            m = rng.choice(s.legal_moves())
            op.append(m)
            s = s.play(m)
        if op not in openings:
            openings.append(op)
    rows = []
    for b in budgets:
        for x, y in pairs:
            t0 = time.time()
            for i, op in enumerate(openings):
                for first in (True, False):
                    r = play_game(agents[x], agents[y], b, op, first)
                    rows.append(dict(budget=b, agent=x, opponent=y, opening=i, first=first, **r))
            if verbose:
                sub = pd.DataFrame(rows)
                sub = sub[(sub.budget == b) & (sub.agent == x) & (sub.opponent == y)]
                print(f"  budget {b:5d}  {x} vs {y}: score {sub.score.mean():.3f} "
                      f"over {len(sub)} games [{time.time() - t0:.0f}s]", flush=True)
    return pd.DataFrame(rows)


def summarize_tournament(games: pd.DataFrame) -> pd.DataFrame:
    g = games.groupby(["budget", "agent", "opponent"])
    out = g.agg(games=("score", "size"), score=("score", "mean"),
                wins=("score", lambda x: (x == 1).sum()), draws=("score", lambda x: (x == .5).sum()),
                losses=("score", lambda x: (x == 0).sum()),
                mean_nodes=("nodes_a", "mean"), mean_ms=("ms_a", "mean"),
                compliance=("compliance_a", "mean")).reset_index()
    out["ci95"] = 1.96 * np.sqrt(np.maximum(out.score * (1 - out.score), 0.05) / out.games)
    return out


# ===================================================================== ablations
def ablation(art: dict, df: pd.DataFrame, budget: int = 1000, n_positions: int = 300,
             verbose: bool = True) -> pd.DataFrame:
    """Which component of the proposed agent F actually matters?"""
    t = art["tuning"]["VOC"]
    base = dict(lam=t["lam"], n_samples=t["n_samples"], k_root=t["k_root"], stop_on_nonpositive=t["stop"])
    err = art["err"]
    from .error_models import LearnedErrorModel
    nobias = LearnedErrorModel(err["learned"].mu_model, err["learned"].var_model, False, err["learned"].global_var)

    def voc(model, name, **over):
        kw = {**base, **over}
        return VOCAgent(model, name, name, **kw)

    variants = {
        "F  full (learned mu, sigma)": voc(err["learned"], "F"),
        "F  - bias term (mu = 0)": voc(nobias, "F-nobias"),
        "F  - heteroscedastic sigma (constant Gaussian)": voc(err["constant"], "F-const"),
        "E  binned empirical table (non-learned)": voc(err["binned"], "E"),
        "F  M = 8 samples": voc(err["learned"], "F-M8", n_samples=8),
        "F  M = 128 samples": voc(err["learned"], "F-M128", n_samples=128),
        "F  k_root = 1 candidate line": voc(err["learned"], "F-k1", k_root=1),
        "F  k_root = 6 candidate lines": voc(err["learned"], "F-k6", k_root=6),
        "F  + stop rule, lambda = 1e-4": voc(err["learned"], "F-stop1e-4", lam=1e-4, stop_on_nonpositive=True),
        "F  + stop rule, lambda = 1e-3": voc(err["learned"], "F-stop1e-3", lam=1e-3, stop_on_nonpositive=True),
        "F  + stop rule, lambda = 1e-2": voc(err["learned"], "F-stop1e-2", lam=1e-2, stop_on_nonpositive=True),
        "B  iterative deepening (reference)": IterativeDeepeningAgent(),
    }
    sub = _subsample(df, n_positions)
    rows = []
    for name, ag in variants.items():
        r = evaluate_on_positions(ag, sub, budget, name)
        rows.append(dict(variant=name, mean_regret=r.regret.mean(), optimal_rate=r.optimal.mean(),
                         mean_nodes=r.nodes.mean(), mean_ms=1000 * r.time.mean(), n=len(r)))
        if verbose:
            print(f"  {name:48s} regret {rows[-1]['mean_regret']:.4f}  optimal {rows[-1]['optimal_rate']:.3f}  "
                  f"nodes {rows[-1]['mean_nodes']:.0f}", flush=True)
    return pd.DataFrame(rows)
