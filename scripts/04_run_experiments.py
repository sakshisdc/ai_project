#!/usr/bin/env python
"""Phase 5/6: tune C and D on validation games, then run the controlled experiments
on HELD-OUT test games:  position benchmark, tournament, ablation.

    python scripts/04_run_experiments.py                 # full run
    python scripts/04_run_experiments.py --quick         # small smoke run
    python scripts/04_run_experiments.py --only ablation # one stage: tune|benchmark|tournament|ablation

Outputs (results/): tuning.json, benchmark_raw.csv, benchmark_summary.csv,
tournament_games.csv, tournament_summary.csv, ablation.csv
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from adaptive_minimax import config as C
from adaptive_minimax import experiments as X

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["tune", "benchmark", "tournament", "ablation"])
    ap.add_argument("--n-positions", type=int, default=None, help="test positions for the benchmark")
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="only run these budgets (results are MERGED into the existing CSVs, so a long run "
                         "can be split into several calls)")
    ap.add_argument("--pairs", nargs="+", default=None, help='tournament matchups to run, e.g. FA FB (default: all)')
    ap.add_argument("--openings", type=int, default=None, help="openings per matchup (x2 colours)")
    a = ap.parse_args()
    q = a.quick
    n_pos = a.n_positions or (60 if q else 400)
    n_open = a.openings or (2 if q else 12)

    df = pd.read_csv(C.POSITIONS_CSV)
    val, test = df[df.split == "val"], df[df.split == "test"]
    R = C.RESULT_DIR
    run = lambda s: a.only in (None, s)

    def merge(new, path):
        """Replace the rows of the budgets just computed, keep the rest."""
        if path.exists():
            old = pd.read_csv(path)
            keys = [c for c in ("budget", "agent", "opponent") if c in new.columns]
            drop = old.merge(new[keys].drop_duplicates(), on=keys, how="left", indicator=True)["_merge"] == "both"
            new = pd.concat([old[~drop.values], new], ignore_index=True)
        new.to_csv(path, index=False)
        return new

    art = X.load_artifacts()
    if run("tune"):
        print("== tuning C and D on validation positions ==")
        X.tune_policies(art, val, n_positions=60 if q else 250)
        art = X.load_artifacts()
    agents = X.make_agents(art)

    if run("benchmark"):
        print("== position benchmark (held-out test positions) ==")
        raw = X.position_benchmark(agents, test, a.budgets or (C.BUDGETS[:2] if q else C.BUDGETS), n_pos)
        raw = merge(raw, R / "benchmark_raw.csv")
        summ = X.summarize_benchmark(raw)
        summ.to_csv(R / "benchmark_summary.csv", index=False)
        print(summ.round(4).to_string(index=False))

    if run("tournament"):
        print("== tournament ==")
        pairs = [(x, y) for x in "CDEF" for y in "AB"] + [("F", "C"), ("F", "D"), ("F", "E")]
        if a.pairs:
            pairs = [(p[0], p[1]) for p in a.pairs]
        games = X.tournament(agents, pairs[:3] if q else pairs, a.budgets or (C.TOURNAMENT_BUDGETS[:1] if q else C.TOURNAMENT_BUDGETS), n_open)
        games = merge(games, R / "tournament_games.csv")
        ts = X.summarize_tournament(pd.read_csv(R / "tournament_games.csv"))
        ts.to_csv(R / "tournament_summary.csv", index=False)
        print(ts.round(3).to_string(index=False))

    if run("ablation"):
        print("== ablation ==")
        ab = X.ablation(art, test, 1000, 60 if q else 300)
        ab.to_csv(R / "ablation.csv", index=False)
