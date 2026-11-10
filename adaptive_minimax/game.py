"""Connect Four (6 rows x 7 columns) implemented with bitboards.

Bit layout (Pascal Pons' scheme): every column uses HEIGHT+1 = 7 bits, the top
bit of each column is a permanent empty "sentinel" so that shifts never wrap
from one column into the next.

    .  .  .  .  .  .  .       row 5  (top)   bit index = col*7 + row
    5 12 19 26 33 40 47
    4 11 18 25 32 39 46
    ...
    0  7 14 21 28 35 42       row 0  (bottom)

A `State` stores
    cur   : stones of the player whose turn it is
    mask  : stones of both players
    n     : number of stones played so far
    won   : True if the player who JUST moved completed four in a row
States are immutable; `play(col)` returns a new State.
"""
from __future__ import annotations

HEIGHT, WIDTH = 6, 7
H1 = HEIGHT + 1
N_CELLS = HEIGHT * WIDTH

# Search the centre first: it dramatically improves alpha-beta pruning.
COLUMN_ORDER = (3, 2, 4, 1, 5, 0, 6)

BOTTOM = [1 << (c * H1) for c in range(WIDTH)]
TOP = [1 << (HEIGHT - 1 + c * H1) for c in range(WIDTH)]
COLMASK = [((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH)]


def win_value(n_stones: int) -> float:
    """Utility (for the winner) of a win after `n_stones` stones were played.

    Faster wins are slightly better, so agents finish games instead of
    dithering. Always >= 0.91, i.e. strictly above any heuristic score
    (the heuristic is clipped to +-0.8, see evaluation.py).
    """
    return 1.0 - 0.002 * n_stones


def has_four(b: int) -> bool:
    """True if bitboard `b` contains four stones in a row (any direction)."""
    m = b & (b >> H1)                      # horizontal
    if m & (m >> (2 * H1)):
        return True
    m = b & (b >> HEIGHT)                  # diagonal  "\"
    if m & (m >> (2 * HEIGHT)):
        return True
    m = b & (b >> (HEIGHT + 2))            # diagonal  "/"
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    m = b & (b >> 1)                       # vertical
    return bool(m & (m >> 2))


class State:
    __slots__ = ("cur", "mask", "n", "won")

    def __init__(self, cur: int = 0, mask: int = 0, n: int = 0, won: bool = False):
        self.cur, self.mask, self.n, self.won = cur, mask, n, won

    # ------------------------------------------------------------------ basics
    @property
    def opp(self) -> int:
        """Stones of the player who is NOT to move."""
        return self.cur ^ self.mask

    @property
    def key(self) -> int:
        """Hashable unique id of the position (used for caches)."""
        return (self.mask << 49) | self.cur

    def __hash__(self) -> int:
        return self.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.cur == other.cur and self.mask == other.mask and self.n == other.n and self.won == other.won

    @property
    def is_terminal(self) -> bool:
        return self.won or self.n >= N_CELLS

    def terminal_value(self) -> float:
        """Utility for the SIDE TO MOVE at a terminal state (negamax convention)."""
        if self.won:
            return -win_value(self.n)       # the previous mover won -> we lost
        return 0.0                           # draw

    def legal_moves(self) -> list[int]:
        m = self.mask
        return [c for c in COLUMN_ORDER if not m & TOP[c]]

    # ------------------------------------------------------------------- moves
    def play(self, col: int) -> "State":
        mask = self.mask
        new_mask = mask | (mask + BOTTOM[col])
        stone = new_mask ^ mask
        won = has_four(self.cur | stone)    # did the mover just win?
        return State(self.cur ^ mask, new_mask, self.n + 1, won)

    def winning_moves(self, stones: int | None = None) -> list[int]:
        """Playable columns where `stones` (default: side to move) would win at once."""
        stones = self.cur if stones is None else stones
        mask = self.mask
        out = []
        for c in COLUMN_ORDER:
            if mask & TOP[c]:
                continue
            stone = (mask + BOTTOM[c]) & COLMASK[c]
            if has_four(stones | stone):
                out.append(c)
        return out

    # --------------------------------------------------------------- utilities
    @staticmethod
    def from_moves(moves) -> "State":
        s = State()
        for c in moves:
            s = s.play(int(c))
        return s

    @staticmethod
    def from_string(rows: str) -> "State":
        """Build a state from an ASCII picture (top row first; 'X' moves first).

        Characters: 'X' / 'O' / '.'. Gravity is NOT checked, so stones must be
        supported by the caller. Used by the unit tests.
        """
        lines = [l.strip() for l in rows.strip().splitlines()]
        assert len(lines) == HEIGHT and all(len(l) == WIDTH for l in lines)
        x = o = 0
        for r, line in enumerate(lines):
            row = HEIGHT - 1 - r
            for c, ch in enumerate(line):
                bit = 1 << (c * H1 + row)
                if ch == "X":
                    x |= bit
                elif ch == "O":
                    o |= bit
        n = (x | o).bit_count()
        x_to_move = (n % 2 == 0)
        cur = x if x_to_move else o
        return State(cur, x | o, n, False)

    def __str__(self) -> str:
        first_is_cur = (self.n % 2 == 0)
        x = self.cur if first_is_cur else self.opp
        o = self.opp if first_is_cur else self.cur
        out = []
        for row in range(HEIGHT - 1, -1, -1):
            line = ""
            for c in range(WIDTH):
                bit = 1 << (c * H1 + row)
                line += "X" if x & bit else ("O" if o & bit else ".")
            out.append(line)
        out.append("0123456")
        return "\n".join(out)

    __repr__ = __str__
