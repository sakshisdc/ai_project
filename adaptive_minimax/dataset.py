"""Phase 3 - generate the labelled training dataset.

For every sampled position `s` (the "root" of a local decision problem):

    1. run a SHALLOW search (depth SHALLOW_DEPTH) and record the chosen move;
    2. run a deep REFERENCE search (depth REF_DEPTH, full window on every root
       move) -> exact reference utilities Q_ref(s, a) for all moves a;
    3. regret of a move a:   regret(a) = max_b Q_ref(b) - Q_ref(a);
    4. simulate ADDITIONAL computation (search to depth EXTRA_DEPTH), record the
       new move, its regret and the extra nodes it cost;
    5. label the computation by how much it reduced regret relative to its cost:

           benefit_per_knode = (regret_shallow - regret_extra) / (extra_nodes/1000)
           label             = 1  if benefit_per_knode >= LABEL_RHO  else 0

Additionally the evaluation ERROR of the static evaluation is stored,

           e = v_ref(s) - v_static(s),          v_ref = max_a Q_ref(s, a)

which is the training target of the error-distribution models p(e|x) (agents E/F).

Positions come from random games and noisy self-play at all game stages. Each row
carries a `game_id`; train/validation/test splits are made BY GAME (never by
position) because positions of one game are strongly correlated.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import time

import numpy as np
import pandas as pd

from . import config as C
from .evaluation import static_eval
from .features import FEATURE_NAMES, extract_features
from .game import State, WIDTH
from .search import AlphaBeta


# ------------------------------------------------------------------ sampling
def _noisy_selfplay_move(state: State, rng: random.Random, p_random: float) -> int:
    """With prob. p_random a random move, otherwise a random near-best depth-2 move."""
    legal = state.legal_moves()
    if rng.random() < p_random:
        return rng.choice(legal)
    q = AlphaBeta().search_depth(state, 2, full_window=True).root_values
    best = max(q.values())
    return rng.choice([m for m, v in q.items() if v >= best - 0.03])


def sample_game(game_id: int, seed: int, n_positions: int) -> list[tuple[State, State | None]]:
    """Play one game and return `n_positions` (state, parent_state) samples.

    Samples are stratified over the three game stages (early / middle / late) so
    that all phases are represented. Only non-terminal states are kept.
    """
    rng = random.Random(seed * 1_000_003 + game_id)
    p_random = (1.0, 0.5, 0.25, 0.1)[game_id % 4]          # mixture of random and self-play
    hist: list[tuple[State, State | None]] = [(State(), None)]
    s = State()
    while not s.is_terminal:
        prev = s
        s = s.play(_noisy_selfplay_move(s, rng, p_random))
        if not s.is_terminal:
            hist.append((s, prev))
    cand = [h for h in hist if h[0].n >= 2 and len(h[0].legal_moves()) >= 2]
    if not cand:
        return []
    thirds = [[h for h in cand if lo <= h[0].n < hi] for lo, hi in ((0, 10), (10, 22), (22, 43))]
    picked = []
    per = max(1, n_positions // 3)
    for grp in thirds:
        rng.shuffle(grp)
        picked += grp[:per]
    return picked


# ------------------------------------------------------------------ labelling
def label_position(state: State, prev: State | None, ref_depth: int = C.REF_DEPTH,
                   shallow: int = C.SHALLOW_DEPTH, extra: int = C.EXTRA_DEPTH,
                   rho: float = C.LABEL_RHO) -> dict:
    """Compute features, reference Q-values, regrets and labels for one position."""
    feats = extract_features(state, static_eval(prev) if prev is not None else None)
    ref = AlphaBeta().search_depth(state, ref_depth, full_window=True)
    q_ref = ref.root_values
    best_q = max(q_ref.values())

    e_s = AlphaBeta()
    move_s = e_s.search_depth(state, shallow).move
    e_x = AlphaBeta()
    move_x = e_x.search_depth(state, extra).move

    regret_s = best_q - q_ref[move_s]
    regret_x = best_q - q_ref[move_x]
    extra_nodes = max(e_x.nodes - e_s.nodes, 1)
    reduction = regret_s - regret_x
    bpk = reduction / (extra_nodes / 1000.0)

    row = dict(zip(FEATURE_NAMES, feats))
    row.update(
        cur=state.cur, mask=state.mask, ply=state.n,
        v_ref=best_q, error=best_q - feats[FEATURE_NAMES.index("static_eval")],
        move_shallow=move_s, move_extra=move_x,
        regret_shallow=regret_s, regret_extra=regret_x, reduction=reduction,
        extra_nodes=extra_nodes, benefit_per_knode=bpk, label=int(bpk >= rho),
        ref_nodes=ref.nodes,
    )
    for c in range(WIDTH):
        row[f"q{c}"] = q_ref.get(c, np.nan)
    return row


def _worker(args):
    game_id, seed, n_pos, kw = args
    return [dict(game_id=game_id, **label_position(s, p, **kw))
            for s, p in sample_game(game_id, seed, n_pos)]


def assign_splits(df: pd.DataFrame, seed: int = C.SEED,
                  test_frac: float = C.TEST_FRACTION, val_frac: float = C.VAL_FRACTION) -> pd.DataFrame:
    """Add a `split` column (train / val / test) assigned per COMPLETE GAME."""
    games = np.array(sorted(df["game_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_test, n_val = int(len(games) * test_frac), int(len(games) * val_frac)
    split_of = {}
    for i, g in enumerate(games):
        split_of[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    df = df.copy()
    df["split"] = df["game_id"].map(split_of)
    return df


def generate_dataset(n_games: int = 400, positions_per_game: int = 6, seed: int = C.SEED,
                     processes: int | None = None, ref_depth: int = C.REF_DEPTH,
                     out_path=C.POSITIONS_CSV, verbose: bool = True) -> pd.DataFrame:
    """Generate, label, split and save the dataset (Phase 3 deliverable)."""
    kw = dict(ref_depth=ref_depth)
    jobs = [(g, seed, positions_per_game, kw) for g in range(n_games)]
    processes = processes or os.cpu_count() or 1
    rows, t0 = [], time.time()
    pool = None
    try:
        if processes <= 1:
            it = map(_worker, jobs)
        else:
            pool = mp.Pool(processes)
            it = pool.imap_unordered(_worker, jobs, chunksize=2)
        for i, r in enumerate(it, 1):
            rows += r
            if verbose and i % 25 == 0:
                print(f"  games {i}/{n_games}  positions {len(rows)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    df = assign_splits(pd.DataFrame(rows), seed)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"saved {len(df)} positions from {df.game_id.nunique()} games -> {out_path}")
        print(df.groupby("split").agg(n=("label", "size"), pos_rate=("label", "mean"),
                                      mean_regret=("regret_shallow", "mean")))
    return df
