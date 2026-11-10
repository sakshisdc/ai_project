"""Selective-extension ("cutoff") policies used by agents C and D.

At the search horizon the engine asks the policy:  "extend this node by one more
ply, or stop and trust the static evaluation?"  Both policies see the same
features and the same relevance gate (see search.AlphaBeta), so the ONLY
difference between C and D is how the decision is made.
"""
from __future__ import annotations

from .evaluation import static_eval
from .features import IDX, extract_features


class ExtensionPolicy:
    def should_extend(self, state, parent, alpha: float, beta: float) -> bool:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _features(state, parent):
        p_eval = static_eval(parent) if parent is not None else None
        return extract_features(state, p_eval)


class RuleBasedPolicy(ExtensionPolicy):
    """Agent C: hand-written volatility rules (no learning).

    Extend when the position is *tactically volatile*:
      * somebody has an immediate win (a forced reply is coming), or
      * the evaluation swung by >= `swing_threshold` relative to the parent
        (the evaluation is reacting to something it cannot fully resolve).
    """

    def __init__(self, swing_threshold: float = 0.50):
        self.swing_threshold = swing_threshold

    def should_extend(self, state, parent, alpha, beta) -> bool:
        f = self._features(state, parent)
        if f[IDX["my_win_now"]] + f[IDX["opp_win_now"]] > 0:
            return True
        return f[IDX["swing"]] >= self.swing_threshold


class MLPolicy(ExtensionPolicy):
    """Agent D: a learned classifier predicts "extra search is worthwhile".

    `predict_proba` is a compiled fast function (see fastmodels.py) returning
    P(worthwhile | features). Extend when it is >= `threshold`.
    """

    def __init__(self, predict_proba, threshold: float):
        self.predict_proba = predict_proba
        self.threshold = threshold

    def should_extend(self, state, parent, alpha, beta) -> bool:
        return self.predict_proba(self._features(state, parent)) >= self.threshold
