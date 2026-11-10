#!/usr/bin/env python
"""Phase 4: train + validate the learned components.

    python scripts/03_train_models.py

Reads  data/positions.csv   (from 02_generate_dataset.py)
Writes models/*.joblib, results/classifier_comparison.csv, results/error_model_metrics.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_minimax import config as C
from adaptive_minimax.training import load_dataset, train_classifier, train_error_models

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=str(C.POSITIONS_CSV), help=f"path to positions.csv (default: {C.POSITIONS_CSV})")
    ap.add_argument("--only", choices=["all", "classifier", "error"], default="all", help="which component to train")
    args = ap.parse_args()

    df = load_dataset(Path(args.data))
    print(f"dataset: {len(df)} positions, {df.game_id.nunique()} games; split sizes {df.split.value_counts().to_dict()}")

    if args.only in ("all", "classifier"):
        print("== (1) 'is extra search worthwhile?' classifier ==")
        train_classifier(df)
    if args.only in ("all", "error"):
        print("== (2) evaluation-error distributions p(e|x) ==")
        train_error_models(df)
