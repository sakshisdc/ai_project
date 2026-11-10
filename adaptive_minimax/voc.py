"""Best-first search driven by a Monte-Carlo estimate of the Value of Computation
(agents E and F).

Main loop (one iteration = one "computation" = expanding one leaf)
------------------------------------------------------------------
  1. build the CANDIDATE set: PV leaf of the top-k root moves + the "challenger"
     (second-best child) leaf at every node on those PVs.
  2. for each candidate leaf  l  with belief  v_l ~ N(static + mu(x), sigma(x)^2):
        a. alpha-beta style relevance window: skip l if no value inside
           [v_l - 3 sigma, v_l + 3 sigma] could change the root decision.
        b. draw M samples of l's value (antithetic Gaussian), push each through
           its ancestors (negamax) up to the root, and obtain the root utility
           max_a Q_a^(m)(l).
        c. VOC(l) = mean_m max_a Q_a^(m)  -  max( max_{a != a_l} Q_a, mean_m Q_{a_l}^(m) )
                    - lambda * cost(l)
           (cost = number of children that expanding l generates).
           NOTE: the baseline is "the best action's expected utility under the
           same samples" rather than max_a Q_a evaluated at the leaf's mean.
           A minimax over an uncertain leaf has a lower expectation than the
           minimax of its mean, so the plain formula can go negative for leaves
           under a MIN node. This version is >= 0 (Jensen) and is exactly 0 when
           the leaf cannot change the decision.
  3. expand the candidate with the highest positive net VOC; back the new
     values up the tree (this is the minimax / alpha-beta bound update).
  4. stop when  (a) no candidate has positive net VOC (only if
     `stop_on_nonpositive=True`), (b) the hard node budget is exhausted, or
     (c) a forced win has been proven.
     Myopic VOC is 0 whenever no SINGLE expansion can flip the decision, even if
     several together could, so rule (a) can fire too early; with
     `stop_on_nonpositive=False` the budget is always spent and VOC is used as
     the scheduler (see the stop-rule ablation in the results).

The error model (binned = agent E, learned GBM = agent F) is the ONLY difference
between E and F.

Approximation note: expanding a leaf is modelled as revealing a sample of its
"true" value (value of perfect information about that leaf, a standard myopic
meta-reasoning approximation); lambda converts nodes into utility units.
"""
from __future__ import annotations

import time

import numpy as np

from .evaluation import analyze
from .features import IDX, extract_features
from .game import N_CELLS, State
from .search import INF, WIN_THRESHOLD, SearchResult

BELIEF_CLIP = 0.95


class Node:
    __slots__ = ("state", "parent", "move", "value", "static", "sigma", "children",
                 "best", "depth", "branching", "terminal")

    def __init__(self, state, parent, move, depth):
        self.state, self.parent, self.move, self.depth = state, parent, move, depth
        self.children = None
        self.best = None
        self.value = 0.0
        self.static = 0.0
        self.sigma = 0.0
        self.branching = 0
        self.terminal = False


class VOCSearch:
    def __init__(self, error_model, lam: float = 2e-4, n_samples: int = 32,
                 k_root: int = 3, stop_on_nonpositive: bool = False, seed: int = 0):
        self.model = error_model
        self.lam = lam
        self.k_root = k_root
        self.stop = stop_on_nonpositive
        half = np.random.default_rng(seed).standard_normal(max(1, n_samples // 2))
        self.Z = np.concatenate([half, -half])      # antithetic samples (fixed -> low noise)
        self.nodes = 0
        self.max_depth = 0

    # ------------------------------------------------------------ tree build
    def _new(self, state: State, parent: Node | None, move: int | None, depth: int) -> Node:
        self.nodes += 1
        node = Node(state, parent, move, depth)
        if depth > self.max_depth:
            self.max_depth = depth
        if state.won:
            node.terminal, node.value = True, state.terminal_value()
        elif state.n >= N_CELLS:
            node.terminal, node.value = True, 0.0
        else:
            feats = extract_features(state, parent.static if parent is not None else None)
            mu, sigma = self.model.predict(feats)
            node.static = feats[IDX["static_eval"]]
            node.value = max(-BELIEF_CLIP, min(BELIEF_CLIP, node.static + mu))
            node.sigma = sigma
            node.branching = int(feats[IDX["branching"]])
        return node

    @staticmethod
    def _refresh(node: Node) -> None:
        """Recompute value / best child of an expanded node (negamax)."""
        best = None
        for c in node.children:
            if best is None or c.value < best.value:
                best = c
        node.best = best
        node.value = -best.value

    def _expand(self, node: Node) -> None:
        kids = []
        for col in node.state.legal_moves():
            kids.append(self._new(node.state.play(col), node, col, node.depth + 1))
        node.children = kids
        self._refresh(node)
        # back the new value up towards the root (stop when nothing changes)
        n = node.parent
        while n is not None:
            old = n.value
            self._refresh(n)
            if n.value == old:
                break
            n = n.parent

    # ------------------------------------------------------------ candidates
    @staticmethod
    def _pv_leaf(node: Node):
        while node.children:
            node = node.best
        return None if node.terminal else node

    def _candidates(self, root: Node) -> list[Node]:
        out: dict[int, Node] = {}
        for top in sorted(root.children, key=lambda c: c.value)[: self.k_root]:
            node = top
            while True:
                if node.terminal:
                    break
                if node.children is None:
                    out[id(node)] = node
                    break
                kids = sorted(node.children, key=lambda c: c.value)
                if len(kids) > 1:                       # challenger line
                    leaf = self._pv_leaf(kids[1])
                    if leaf is not None:
                        out[id(leaf)] = leaf
                node = kids[0]
        return list(out.values())

    # ------------------------------------------------------------------- VOC
    def _gross_voc(self, leaf: Node, root: Node, q_best: float) -> float:
        """Monte-Carlo estimate of E[max_a Q_a | sampled leaf value] - max_a Q_a."""
        # path leaf -> ... -> c0 (the root child that contains the leaf)
        path = [leaf]
        while path[-1].parent is not root:
            path.append(path[-1].parent)
        c0 = path[-1]
        q_other = -INF
        for c in root.children:
            if c is not c0 and -c.value > q_other:
                q_other = -c.value
        # best value of the siblings at each level (excluded: the path child)
        others = []
        for i in range(len(path) - 1):
            child, par = path[i], path[i + 1]
            o = -INF
            for x in par.children:
                if x is not child and -x.value > o:
                    o = -x.value
            others.append(o)

        m, s = leaf.value, leaf.sigma
        # ---- alpha-beta style relevance window of the leaf value --------------
        lo, hi = -INF, -q_other
        for o in reversed(others):
            lo, hi = -hi, min(-lo, -o)
        if lo >= hi or hi <= m - 3 * s or lo >= m + 3 * s:
            return 0.0

        # ---- Monte-Carlo propagation through the ancestors ---------------------
        w = np.clip(m + s * self.Z, -1.0, 1.0)      # sampled leaf values
        for o in others:
            w = np.maximum(o, -w)                    # value of the next ancestor
        q_m = -w                                      # utility of the leaf's root move
        best_m = np.maximum(q_other, q_m)             # root utility per sample (act AFTER knowing)
        act_now = max(q_other, float(q_m.mean()))     # best utility acting under current beliefs
        return float(best_m.mean() - act_now)         # >= 0 by Jensen's inequality

    # ----------------------------------------------------------------- search
    def search(self, state: State, budget: int) -> SearchResult:
        self.nodes = 0
        self.max_depth = 0
        legal = state.legal_moves()
        if len(legal) == 1:
            return SearchResult(legal[0], 0.0, 0, 0)

        root = self._new(state, None, None, 0)
        self.nodes = 0                                   # the root itself is free
        root.static = analyze(state)[0]
        self._expand(root)
        expansions = 0
        stopped_early = False

        while True:
            q_best = -root.best.value
            if q_best >= WIN_THRESHOLD:                  # forced win proven
                break
            cands = self._candidates(root)
            best_c, best_key = None, None
            for leaf in cands:
                if self.nodes + leaf.branching > budget:  # hard budget
                    continue
                gross = self._gross_voc(leaf, root, q_best)
                net = gross - self.lam * leaf.branching
                # ranking: net VOC; ties (e.g. all-zero "gridlock" where no single
                # expansion can flip the decision) -> most uncertain, then shallowest
                key = (net, leaf.sigma, -leaf.depth)
                if best_key is None or key > best_key:
                    best_c, best_key = leaf, key
            if best_c is None:
                break
            if best_key[0] <= 0.0 and self.stop:          # no positive net VOC -> stop
                stopped_early = True
                break
            self._expand(best_c)
            expansions += 1

        move = root.best.move
        return SearchResult(move, -root.best.value, self.nodes, self.max_depth,
                            self.max_depth, expansions, not stopped_early)
