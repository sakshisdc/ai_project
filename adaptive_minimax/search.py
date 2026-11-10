"""Recursive search engines (Phase 1, 2 and 5).

Everything uses the NEGAMAX convention: a value is always from the point of view
of the side to move at that node, and  value(parent) = max(-value(child)).

Node accounting (identical in every engine, including voc.py)
-------------------------------------------------------------
    1 node = 1 position generated with `State.play`.
Features/policy queries made by the adaptive cutoff policies are charged
`POLICY_QUERY_COST` extra nodes, so no agent gets "free" computation.

Classes
-------
AlphaBeta        minimax with alpha-beta pruning, optional hard node budget,
                 optional *selective extension policy* (agents C and D).
Helper functions
    minimax_value        plain minimax (no pruning) - used by tests as ground truth
    search_fixed         fixed-depth alpha-beta (agent A)
    search_iterative     iterative deepening (agent B) and, with a policy, C / D
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .evaluation import static_eval
from .game import N_CELLS, State, win_value

INF = 10.0
WIN_THRESHOLD = 0.9          # |value| above this means a proven win / loss
POLICY_QUERY_COST = 1        # nodes charged per extension-policy query


class BudgetExceeded(Exception):
    """Raised inside the recursion when the hard node budget is hit."""


@dataclass
class SearchResult:
    move: int | None
    value: float
    nodes: int
    depth: int                       # deepest *base* depth fully / partially searched
    max_ply: int = 0                 # deepest ply reached incl. extensions
    n_ext: int = 0                   # number of selective extensions granted
    completed: bool = True           # False if the budget aborted the last iteration
    root_values: dict = field(default_factory=dict)   # exact Q only when full_window=True
    n_done: int = 0                  # root moves completed in the last iteration


class AlphaBeta:
    """Alpha-beta search with optional budget and selective extensions.

    Parameters
    ----------
    budget      hard cap on nodes (None = unlimited)
    policy      object with `should_extend(state, parent, alpha, beta) -> bool`.
                Consulted ONLY at the horizon (remaining depth 0).
    max_ext     maximum number of 1-ply extensions along any single path
    gate        "root relevance" gate: no extension is considered when the
                alpha-beta window is narrower than this (the node can hardly
                influence the root decision).
    """

    def __init__(self, budget: int | None = None, policy=None,
                 max_ext: int = 0, gate: float = 0.0):
        self.budget = budget
        self.policy = policy
        self.max_ext = max_ext if policy is not None else 0
        self.gate = gate
        self.nodes = 0
        self.max_ply = 0
        self.n_ext = 0

    # ------------------------------------------------------------ bookkeeping
    def _tick(self, k: int = 1) -> None:
        if self.budget is not None and self.nodes + k > self.budget:
            raise BudgetExceeded            # never exceed the hard budget
        self.nodes += k

    # -------------------------------------------------------------- recursion
    def _ab(self, s: State, parent: State, depth: int, alpha: float, beta: float,
            ext: int, ply: int) -> float:
        """Value of non-terminal `s` (side-to-move view), `depth` plies left."""
        if ply > self.max_ply:
            self.max_ply = ply
        if depth <= 0:
            # ---- horizon: stop, or buy one more ply? -------------------------
            if (self.policy is not None and ext < self.max_ext
                    and (min(beta, 1.0) - max(alpha, -1.0)) >= self.gate):
                self._tick(POLICY_QUERY_COST)
                if self.policy.should_extend(s, parent, alpha, beta):
                    depth, ext = 1, ext + 1
                    self.n_ext += 1
                else:
                    return static_eval(s)
            else:
                return static_eval(s)

        best = -INF
        for col in s.legal_moves():
            self._tick()
            child = s.play(col)
            if child.won:
                v = win_value(child.n)
            elif child.n >= N_CELLS:
                v = 0.0
            else:
                v = -self._ab(child, s, depth - 1, -beta, -alpha, ext, ply + 1)
            if v > best:
                best = v
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
        return best

    # ------------------------------------------------------------------- root
    def search_depth(self, state: State, depth: int, order: list[int] | None = None,
                     full_window: bool = False) -> SearchResult:
        """Search every root move to `depth` plies.

        full_window=True searches each root move with the full (-inf, inf)
        window: slower, but returns EXACT root values for all moves (used to
        build the deep reference Q-values of the dataset).
        """
        moves = list(order) if order else state.legal_moves()
        alpha = -INF
        best_move, best_val = None, -INF
        values: dict[int, float] = {}
        n_done = 0
        completed = True
        try:
            for col in moves:
                self._tick()
                child = state.play(col)
                if child.won:
                    v = win_value(child.n)
                elif child.n >= N_CELLS:
                    v = 0.0
                elif full_window:
                    v = -self._ab(child, state, depth - 1, -INF, INF, 0, 1)
                else:
                    v = -self._ab(child, state, depth - 1, -INF, -alpha, 0, 1)
                values[col] = v
                n_done += 1
                if v > best_val:
                    best_val, best_move = v, col
                    alpha = max(alpha, v)
        except BudgetExceeded:
            completed = False
        res = SearchResult(best_move, best_val, self.nodes, depth, self.max_ply,
                           self.n_ext, completed, values if full_window else {}, n_done=n_done)
        return res

    def iterative_deepening(self, state: State, max_depth: int | None = None) -> SearchResult:
        """Depth 1, 2, 3, ... until the budget is spent (or a win is proven).

        If the budget aborts iteration d>1 we still use its partial result
        whenever the previous best move was fully searched first (the classic
        "use the partial iteration" rule) - this keeps the baseline strong.
        """
        legal = state.legal_moves()
        max_depth = max_depth or (N_CELLS - state.n)
        order = None
        final = SearchResult(legal[0], 0.0, 0, 0)
        for d in range(1, max_depth + 1):
            res = self.search_depth(state, d, order)
            if res.completed:
                final = res
                order = [res.move] + [m for m in legal if m != res.move]
                if res.value >= WIN_THRESHOLD:
                    break
            else:
                # Budget hit mid-iteration. The partial result is trustworthy
                # when at least one root move was finished (with the previous
                # best move searched first, its value is exact and every
                # improvement found afterwards is exact too).
                if res.move is not None and res.n_done >= 1:
                    final = SearchResult(res.move, res.value, self.nodes, d,
                                         self.max_ply, self.n_ext, False, n_done=res.n_done)
                break
        final.nodes = self.nodes
        final.max_ply = self.max_ply
        final.n_ext = self.n_ext
        return final


# --------------------------------------------------------------- convenience
def search_fixed(state: State, depth: int) -> SearchResult:
    """Agent A: fixed-depth alpha-beta, no budget."""
    eng = AlphaBeta()
    res = eng.search_depth(state, depth)
    res.nodes = eng.nodes
    return res


def search_iterative(state: State, budget: int, policy=None, max_ext: int = 0,
                     gate: float = 0.0) -> SearchResult:
    """Agent B (policy=None) or C / D (policy given): iterative deepening, hard budget."""
    eng = AlphaBeta(budget=budget, policy=policy, max_ext=max_ext, gate=gate)
    return eng.iterative_deepening(state)


def minimax_value(state: State, depth: int) -> float:
    """Plain minimax (no pruning) value of `state` - ground truth for the tests."""
    if state.won:
        return state.terminal_value()
    if state.n >= N_CELLS:
        return 0.0
    if depth == 0:
        return static_eval(state)
    return max(-minimax_value(state.play(c), depth - 1) for c in state.legal_moves())
