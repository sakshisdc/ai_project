"""Phase 1 correctness tests: game rules, alpha-beta == minimax, budgets, tactics.

Run with   pytest -q        (or:  python tests/run_tests.py   if pytest is missing)
"""
import random

from adaptive_minimax.agents import (IterativeDeepeningAgent, RuleAdaptiveAgent, VOCAgent)
from adaptive_minimax.error_models import ConstantErrorModel
from adaptive_minimax.game import N_CELLS, State, has_four, win_value
from adaptive_minimax.search import AlphaBeta, minimax_value, search_fixed, search_iterative


def random_position(rng, lo=3, hi=16):
    while True:
        s = State()
        for _ in range(rng.randint(lo, hi)):
            s = s.play(rng.choice(s.legal_moves()))
            if s.is_terminal:
                break
        if not s.is_terminal:
            return s


# ------------------------------------------------------------------- game rules
def test_gravity_and_column_full():
    s = State()
    for _ in range(6):
        s = s.play(0)
    assert 0 not in s.legal_moves() and len(s.legal_moves()) == 6


def test_vertical_win():
    s = State.from_moves([0, 1, 0, 1, 0, 1, 0])
    assert s.won and s.terminal_value() == -win_value(7)


def test_horizontal_diagonal_wins():
    assert State.from_moves([0, 0, 1, 1, 2, 2, 3]).won                      # horizontal
    assert State.from_moves([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]).won          # diagonal /


def test_no_wrap_around_between_columns():
    # top three cells of column 0 + bottom cell of column 1 are adjacent as bit
    # indices (3,4,5 and 7) but separated by the empty sentinel bit 6: no line
    assert not has_four((0b111 << 3) | (1 << 7))


def test_draw_possible_and_full_board_count():
    rng = random.Random(0)
    for _ in range(300):                       # random games always end consistently
        s = State()
        while not s.is_terminal:
            s = s.play(rng.choice(s.legal_moves()))
        assert s.won or s.n == N_CELLS


def test_from_string_roundtrip():
    s = State.from_moves([3, 3, 2, 4])
    t = State.from_string(str(s).split("\n0123456")[0])
    assert (t.cur, t.mask, t.n) == (s.cur, s.mask, s.n)


# --------------------------------------------------------------------- search
def test_alpha_beta_equals_minimax():
    rng = random.Random(1)
    for _ in range(25):
        s = random_position(rng)
        for d in (1, 2, 3):
            assert abs(AlphaBeta().search_depth(s, d).value - minimax_value(s, d)) < 1e-9


def test_full_window_root_values_are_exact():
    rng = random.Random(2)
    for _ in range(10):
        s = random_position(rng)
        res = AlphaBeta().search_depth(s, 3, full_window=True)
        for col, v in res.root_values.items():
            assert abs(v - -minimax_value(s.play(col), 2)) < 1e-9 or s.play(col).won


def test_takes_immediate_win_and_blocks_threat():
    win = State.from_moves([0, 0, 1, 1, 2, 2])            # X to move, col 3 wins
    block = State.from_moves([6, 0, 6, 1, 5, 2])          # X must stop O's 0-1-2 -> col 3
    for agent in (IterativeDeepeningAgent(), RuleAdaptiveAgent(),
                  VOCAgent(ConstantErrorModel(0.0, 0.2), "E", "E")):
        assert agent.choose(win, 200)[0] == 3
        assert agent.choose(block, 200)[0] == 3


def test_hard_budget_is_never_exceeded():
    rng = random.Random(3)
    agents = [IterativeDeepeningAgent(), RuleAdaptiveAgent(),
              VOCAgent(ConstantErrorModel(0.0, 0.2), "E", "E")]
    for _ in range(15):
        s = random_position(rng, 4, 14)
        for budget in (20, 100, 400):
            for ag in agents:
                move, info = ag.choose(s, budget)
                assert info["nodes"] <= budget, (ag.name, info, budget)
                assert move in s.legal_moves()


def test_iterative_deepening_reaches_same_value_as_fixed_depth():
    rng = random.Random(4)
    for _ in range(8):
        s = random_position(rng)
        r_id = AlphaBeta().iterative_deepening(s, max_depth=3)       # unlimited budget, stop at depth 3
        r_fx = search_fixed(s, 3)
        assert abs(r_id.value - r_fx.value) < 1e-9
