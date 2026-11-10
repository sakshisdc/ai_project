"""Per-position features (the "Methodology" table of the proposal).

Every learned component (ML cutoff classifier D, error models E/F) uses exactly
the same feature vector, so training data and search-time queries match.

    phase        game stage   n_stones / 42
    branching    number of legal moves
    static_eval  static score (side-to-move view)
    abs_eval     |static_eval|
    my_win_now   playable columns where the mover wins immediately
    opp_win_now  playable columns where the opponent WOULD win -> forced reply
    my_threes    windows with 3 own stones + 1 empty   (tactical potential)
    opp_threes   same for the opponent
    swing        |eval(s) + eval(parent)|  -> evaluation instability: after a
                 quiet move the evaluation, seen from the other side, should be
                 about the negative of the parent's; a big jump means the
                 evaluation is reacting to something tactical.

The proposal's remaining features (search depth, remaining budget, root
relevance) depend on the search context rather than on the position, so they are
handled inside the search engines (see search.py / voc.py).
"""
from __future__ import annotations
from .evaluation import analyze

FEATURE_NAMES = [
    "phase", "branching", "static_eval", "abs_eval",
    "my_win_now", "opp_win_now", "my_threes", "opp_threes", "swing",
]
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def extract_features(state, parent_eval: float | None = None) -> list[float]:
    """Feature vector of `state`.

    parent_eval : static evaluation of the PARENT position (from the parent's
                  side-to-move view). None -> swing = 0 (e.g. the game root).
    """
    ev, mt, ot = analyze(state)
    my_win = len(state.winning_moves(state.cur))
    opp_win = len(state.winning_moves(state.opp))
    swing = 0.0 if parent_eval is None else abs(ev + parent_eval)
    return [state.n / 42.0, float(len(state.legal_moves())), ev, abs(ev),
            float(my_win), float(opp_win), float(mt), float(ot), swing]


def tactical_volatility(feats) -> int:
    """Integer tactical-volatility score (immediate wins for either side)."""
    return int(feats[IDX["my_win_now"]] + feats[IDX["opp_win_now"]])
