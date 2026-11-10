#!/usr/bin/env bash
# Run the complete pipeline (Phase 1 -> 6).  Add --quick to 04 for a fast smoke test.
set -e
python tests/run_tests.py                       # Phase 1: correctness tests
python scripts/01_baselines.py                  # Phase 2
python scripts/02_generate_dataset.py --games 600  # Phase 3
python scripts/03_train_models.py               # Phase 4
python scripts/04_run_experiments.py            # Phase 5 + 6
python scripts/05_make_plots.py                 # figures
