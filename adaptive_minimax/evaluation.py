"""Static evaluation function shared by EVERY agent (fair comparison).

Heuristic: for each of the 69 four-cell windows that contains stones of only
one player, add a weight that grows with the number of stones (1, 4, 16);
add a small centre-column bonus. The score is squashed to (-0.8, 0.8) with tanh
so that it is always below the value of a proven win (>= 0.91).

`analyze(state)` returns (score, my_threes, opp_threes):
    my_threes  = number of windows with 3 of my stones + 1 empty cell
    opp_threes = same for the opponent
These counts are free by-products used as "tactical" features.
"""
from __future__ import annotations
import math
from .game import HEIGHT, WIDTH, H1, COLMASK

_SCORE_SCALE = 25.0
_MAX_ABS = 0.8
_WEIGHTS = (0, 1, 4, 16, 0)


def _build_windows() -> list[int]:
    def bit(c, r):
        return 1 << (c * H1 + r)
    wins = []
    for r in range(HEIGHT):                      # horizontal
        for c in range(WIDTH - 3):
            wins.append(sum(bit(c + i, r) for i in range(4)))
    for c in range(WIDTH):                       # vertical
        for r in range(HEIGHT - 3):
            wins.append(sum(bit(c, r + i) for i in range(4)))
    for c in range(WIDTH - 3):                   # diagonals
        for r in range(HEIGHT - 3):
            wins.append(sum(bit(c + i, r + i) for i in range(4)))
        for r in range(3, HEIGHT):
            wins.append(sum(bit(c + i, r - i) for i in range(4)))
    return wins


WINDOWS = _build_windows()
assert len(WINDOWS) == 69

_CACHE: dict[int, tuple[float, int, int]] = {}
_CACHE_LIMIT = 400_000


def analyze(state) -> tuple[float, int, int]:
    """Return (static_eval, my_threes, opp_threes) from the side-to-move view."""
    key = state.key
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    cur, opp = state.cur, state.opp
    score = 0
    mt = ot = 0
    for w in WINDOWS:
        a = cur & w
        b = opp & w
        if a:
            if b:
                continue
            c = a.bit_count()
            score += _WEIGHTS[c]
            if c == 3:
                mt += 1
        elif b:
            c = b.bit_count()
            score -= _WEIGHTS[c]
            if c == 3:
                ot += 1
    centre = COLMASK[3]
    score += 3 * ((cur & centre).bit_count() - (opp & centre).bit_count())
    val = _MAX_ABS * math.tanh(score / _SCORE_SCALE)
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.clear()
    res = (val, mt, ot)
    _CACHE[key] = res
    return res


def clear_cache() -> None:
    """Clear the static evaluation cache."""
    _CACHE.clear()


def static_eval(state) -> float:
    """Heuristic value of a NON-terminal state for the side to move."""
    return analyze(state)[0]
