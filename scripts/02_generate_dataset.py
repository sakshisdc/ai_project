#!/usr/bin/env python
"""Phase 3: generate the labelled dataset  ->  data/positions.csv

    python scripts/02_generate_dataset.py                 # default: 1000 games
    python scripts/02_generate_dataset.py --games 100     # quick smoke run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_minimax import config as C
from adaptive_minimax.dataset import generate_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=1000, help="number of games to sample from")
    ap.add_argument("--per-game", type=int, default=6, help="positions sampled per game")
    ap.add_argument("--ref-depth", type=int, default=C.REF_DEPTH, help="deep reference search depth")
    ap.add_argument("--processes", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", type=str, default=str(C.POSITIONS_CSV), help=f"output CSV path (default: {C.POSITIONS_CSV})")
    a = ap.parse_args()
    generate_dataset(a.games, a.per_game, a.seed, a.processes, a.ref_depth, out_path=Path(a.out))
