"""Models of the evaluation-error distribution  p(e | x)  (Phase 4).

    e = v_ref(s) - v_static(s)

where v_ref is the value of s from a deeper reference search and v_static the
heuristic value, both from the side-to-move view. We assume a Gaussian:

    e | x  ~  N( mu(x), sigma(x)^2 )

    mu(x)     systematic evaluation bias (the static eval is too optimistic /
              pessimistic in this kind of position)
    sigma(x)  how uncertain the static evaluation is in this kind of position

Three interchangeable implementations (all expose `predict(feats) -> (mu, sigma)`):

    ConstantErrorModel   one global Gaussian             (ablation)
    BinnedErrorModel     NON-learned empirical table     (agent E)
    LearnedErrorModel    gradient-boosting regressors    (agent F)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from .fastmodels import compile_estimator
from .features import IDX

SIGMA_FLOOR = 0.05
VAR_SHRINK = 0.25      # weight of the global residual variance in sigma^2(x) (robustness)


class ErrorModel:
    name = "base"

    def predict(self, feats) -> tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    def finalize(self) -> "ErrorModel":
        """Build fast runtime closures (called after fit / after joblib.load)."""
        return self


class ConstantErrorModel(ErrorModel):
    name = "constant"

    def __init__(self, mu: float = 0.0, sigma: float = 0.2):
        self.mu, self.sigma = mu, sigma

    @classmethod
    def fit(cls, X: np.ndarray, e: np.ndarray, use_bias: bool = True):
        return cls(float(e.mean()) if use_bias else 0.0, max(float(e.std()), SIGMA_FLOOR))

    def predict(self, feats):
        return self.mu, self.sigma


class BinnedErrorModel(ErrorModel):
    """Agent E: empirical mean / std of the error inside hand-defined feature bins.

    Bins: game phase (3) x tactical flag (2) x branching (2) = 12 cells.
    Cells with fewer than `min_count` samples fall back to the global Gaussian.
    This is *empirical* but not *learned*: no model is fitted, only counted.
    """
    name = "binned"

    def __init__(self, table: dict, global_mu: float, global_sigma: float):
        self.table, self.global_mu, self.global_sigma = table, global_mu, global_sigma

    @staticmethod
    def bin_of(feats) -> tuple[int, int, int]:
        phase = feats[IDX["phase"]]
        p = 0 if phase < 0.25 else (1 if phase < 0.6 else 2)
        tact = int(feats[IDX["my_win_now"]] + feats[IDX["opp_win_now"]] > 0)
        br = int(feats[IDX["branching"]] >= 7)
        return p, tact, br

    @classmethod
    def fit(cls, X: np.ndarray, e: np.ndarray, min_count: int = 20, use_bias: bool = True):
        gmu = float(e.mean()) if use_bias else 0.0
        gsd = max(float(e.std()), SIGMA_FLOOR)
        cells: dict = {}
        for x, ei in zip(X, e):
            cells.setdefault(cls.bin_of(x), []).append(ei)
        table = {}
        for k, v in cells.items():
            if len(v) >= min_count:
                v = np.asarray(v)
                table[k] = (float(v.mean()) if use_bias else 0.0, max(float(v.std()), SIGMA_FLOOR))
        return cls(table, gmu, gsd)

    def predict(self, feats):
        return self.table.get(self.bin_of(feats), (self.global_mu, self.global_sigma))


class LearnedErrorModel(ErrorModel):
    """Agent F: two gradient-boosting regressors, one for mu(x), one for sigma(x).

    Step 1  fit  mu(x)  on  e.
    Step 2  get OUT-OF-FOLD residuals r = e - mu_oof(x)  (grouped by game, so
            the variance model never sees over-fitted residuals) and fit a
            second regressor on r^2;  sigma^2(x) = (1-w)*pred + w*global residual
            variance (shrinkage guards against over-confident tiny sigmas).
    """
    name = "learned"

    def __init__(self, mu_model, var_model, use_bias: bool = True, global_var: float = 0.05):
        self.mu_model, self.var_model, self.use_bias = mu_model, var_model, use_bias
        self.global_var = global_var
        self._mu = self._var = None
        self.finalize()

    def finalize(self):
        self._mu = compile_estimator(self.mu_model)
        self._var = compile_estimator(self.var_model)
        return self

    @staticmethod
    def _gbr(seed=0):
        return GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08,
                                         subsample=0.8, min_samples_leaf=15, random_state=seed)

    @classmethod
    def fit(cls, X: np.ndarray, e: np.ndarray, groups: np.ndarray, use_bias: bool = True,
            seed: int = 0):
        mu_model = cls._gbr(seed).fit(X, e)
        n_unique_groups = len(np.unique(groups))
        if n_unique_groups >= 2:
            n_splits = max(2, min(5, n_unique_groups))
            mu_oof = cross_val_predict(cls._gbr(seed), X, e, groups=groups,
                                       cv=GroupKFold(n_splits=n_splits))
        else:
            mu_oof = mu_model.predict(X)
        var_model = cls._gbr(seed + 1).fit(X, (e - mu_oof) ** 2)
        return cls(mu_model, var_model, use_bias, float(np.mean((e - mu_oof) ** 2)))

    def predict(self, feats):
        if self._mu is None or self._var is None:
            self.finalize()
        mu = self._mu(feats) if self.use_bias else 0.0
        var = (1 - VAR_SHRINK) * max(self._var(feats), 0.0) + VAR_SHRINK * self.global_var
        return mu, max(var, SIGMA_FLOOR ** 2) ** 0.5

    # joblib: closures are not picklable -> drop them and rebuild on load
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mu"] = d["_var"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.finalize()
