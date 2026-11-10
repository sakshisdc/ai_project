#!/usr/bin/env python
"""Draw all figures from the CSV files in results/  ->  results/figures/*.png"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_minimax.plotting import make_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    make_all()
