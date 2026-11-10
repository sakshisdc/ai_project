"""The six agents compared in the project (proposal table, Phase 5/6).

    A  FixedDepthAgent         fixed-depth minimax + alpha-beta
    B  IterativeDeepeningAgent iterative deepening + alpha-beta       (strong baseline)
    C  RuleAdaptiveAgent       B + rule-based selective extension
    D  MLAdaptiveAgent         B + ML-classifier selective extension
    E  VOCAgent(binned)        probabilistic MC-VOC + empirical (binned) error table
    F  VOCAgent(learned)       probabilistic MC-VOC + learned (GBM) error model  <- proposed

All agents share the same evaluation function and the same node accounting.
Every agent exposes

    choose(state, budget) -> (move, info)

with `info = {"nodes", "time", "depth", "max_ply", "n_ext", "within_budget"}`.

Budget handling
---------------
B, C, D, E, F enforce the budget HARD (they can never exceed it).
A cannot be interrupted mid-search, so - as in a normal fixed-depth engine - it
is *calibrated*: for a node budget N we use the deepest depth whose average cost
(measured in Phase 2, `results/baseline_calibration.csv`) is <= N. Individual A
moves may therefore overshoot, which is exactly what the "budget compliance"
metric measures.
"""
from __future__ import annotations

import time

from .policies import MLPolicy, RuleBasedPolicy
from .search import AlphaBeta, search_fixed, search_iterative
from .voc import VOCSearch

DEFAULT_CALIBRATION = {1: 8, 2: 30, 3: 110, 4: 300, 5: 900, 6: 2500, 7: 7000}  # fallback only


class Agent:
    name = "?"
    label = "?"

    def choose(self, state, budget: int):
        t0 = time.perf_counter()
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0], dict(nodes=0, time=0.0, depth=0, max_ply=0, n_ext=0, within_budget=True)
        res = self._search(state, budget)
        dt = time.perf_counter() - t0
        move = res.move if res.move is not None else legal[0]
        info = dict(nodes=res.nodes, time=dt, depth=res.depth, max_ply=res.max_ply,
                    n_ext=res.n_ext, within_budget=res.nodes <= budget)
        return move, info

    def _search(self, state, budget):  # pragma: no cover
        raise NotImplementedError


class FixedDepthAgent(Agent):
    name = "A"
    label = "A: fixed-depth"

    def __init__(self, depth: int | None = None, calibration: dict | None = None):
        self.depth = depth
        self.calibration = calibration or DEFAULT_CALIBRATION

    def depth_for(self, budget: int) -> int:
        if self.depth is not None:
            return self.depth
        ok = [d for d, n in sorted(self.calibration.items()) if n <= budget]
        return max(ok) if ok else 1

    def _search(self, state, budget):
        return search_fixed(state, self.depth_for(budget))


class IterativeDeepeningAgent(Agent):
    name = "B"
    label = "B: iterative deepening"

    def _search(self, state, budget):
        return search_iterative(state, budget)


class RuleAdaptiveAgent(Agent):
    name = "C"
    label = "C: rule-based adaptive"

    def __init__(self, swing_threshold: float = 0.50, max_ext: int = 2, gate: float = 0.05):
        self.policy = RuleBasedPolicy(swing_threshold)
        self.max_ext, self.gate = max_ext, gate

    def _search(self, state, budget):
        return search_iterative(state, budget, self.policy, self.max_ext, self.gate)


class MLAdaptiveAgent(Agent):
    name = "D"
    label = "D: ML adaptive cutoff"

    def __init__(self, predict_proba, threshold: float, max_ext: int = 2, gate: float = 0.05):
        self.policy = MLPolicy(predict_proba, threshold)
        self.max_ext, self.gate = max_ext, gate

    def _search(self, state, budget):
        return search_iterative(state, budget, self.policy, self.max_ext, self.gate)


class VOCAgent(Agent):
    """E (binned error model) or F (learned error model)."""

    def __init__(self, error_model, name: str, label: str, lam: float = 2e-4,
                 n_samples: int = 32, k_root: int = 3, stop_on_nonpositive: bool = False):
        self.name, self.label = name, label
        self.engine = VOCSearch(error_model, lam, n_samples, k_root, stop_on_nonpositive)

    def _search(self, state, budget):
        return self.engine.search(state, budget)
