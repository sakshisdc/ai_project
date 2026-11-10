#!/usr/bin/env python
"""Phase 2: fixed-depth / iterative-deepening baselines.

    python scripts/01_baselines.py

Writes results/baseline_calibration.csv (depth -> mean nodes, used by agent A to
map a node budget to a depth) and results/baseline_iterative_deepening.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_minimax.experiments import run_baselines

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=40, help="number of games for baseline calibration (default: 40)")
    ap.add_argument("--max-depth", type=int, default=7, help="maximum fixed depth to calibrate (default: 7)")
    ap.add_argument("--quick", action="store_true", help="quick smoke calibration (5 games, depth 4)")
    args = ap.parse_args()
    games = 5 if args.quick else args.games
    depth = 4 if args.quick else args.max_depth
    run_baselines(n_games=games, max_depth=depth)
