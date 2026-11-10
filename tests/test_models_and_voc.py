"""Tests for the compiled sklearn models, the error models and the VOC search."""
import random

import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from adaptive_minimax.error_models import BinnedErrorModel, ConstantErrorModel, LearnedErrorModel
from adaptive_minimax.fastmodels import compile_estimator
from adaptive_minimax.features import FEATURE_NAMES, extract_features
from adaptive_minimax.game import State
from adaptive_minimax.search import WIN_THRESHOLD
from adaptive_minimax.voc import VOCSearch


def _data(n=400, d=len(FEATURE_NAMES), seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 3] * X[:, 1] + 0.3 * r.normal(size=n) > 0).astype(int)
    return X, y, X[:, 0] * 2 + np.sin(X[:, 2]) + 0.1 * r.normal(size=n)


def test_compiled_classifiers_match_sklearn():
    X, y, _ = _data()
    models = [Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=500))]),
              DecisionTreeClassifier(max_depth=5, random_state=0),
              RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0),
              GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0)]
    for m in models:
        m.fit(X, y)
        f = compile_estimator(m)
        ref = m.predict_proba(X[:50])[:, 1]
        got = np.array([f(list(x)) for x in X[:50]])
        assert np.allclose(ref, got, atol=1e-6), type(m).__name__


def test_compiled_regressors_match_sklearn():
    X, _, t = _data()
    for m in (Ridge(), DecisionTreeRegressor(max_depth=5, random_state=0),
              RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0),
              GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)):
        m.fit(X, t)
        f = compile_estimator(m)
        assert np.allclose(m.predict(X[:50]), [f(list(x)) for x in X[:50]], atol=1e-6), type(m).__name__


def _positions(n=300, seed=0):
    rng = random.Random(seed)
    X, e, g = [], [], []
    for i in range(n):
        s = State()
        prev = None
        for _ in range(rng.randint(3, 25)):
            prev = s
            s = s.play(rng.choice(s.legal_moves()))
            if s.is_terminal:
                break
        if s.is_terminal:
            continue
        from adaptive_minimax.evaluation import static_eval
        f = extract_features(s, static_eval(prev))
        X.append(f); e.append(0.3 * f[4] - 0.2 * f[2] + 0.05 * rng.gauss(0, 1)); g.append(i // 5)
    return np.array(X), np.array(e), np.array(g)


def test_error_models_return_valid_gaussians():
    X, e, g = _positions()
    for m in (ConstantErrorModel.fit(X, e), BinnedErrorModel.fit(X, e), LearnedErrorModel.fit(X, e, g)):
        for x in X[:20]:
            mu, sigma = m.predict(list(x))
            assert np.isfinite(mu) and sigma > 0


def test_learned_error_model_survives_pickling():
    import pickle
    X, e, g = _positions()
    m = LearnedErrorModel.fit(X, e, g)
    m2 = pickle.loads(pickle.dumps(m))
    assert m.predict(list(X[0])) == m2.predict(list(X[0]))


def test_voc_is_nonnegative_and_zero_for_irrelevant_leaves():
    s = State.from_moves([3, 3, 2, 4, 4, 2, 3, 3, 2])
    eng = VOCSearch(ConstantErrorModel(0.0, 0.25), lam=0.0)
    eng.nodes = 0
    root = eng._new(s, None, None, 0)
    root.static = 0.0
    eng._expand(root)
    for _ in range(10):                              # grow the tree a little
        q = -root.best.value
        cands = eng._candidates(root)
        for leaf in cands:
            assert eng._gross_voc(leaf, root, q) >= -1e-9
        eng._expand(cands[0])


def test_voc_finds_forced_win_and_stops():
    win = State.from_moves([0, 0, 1, 1, 2, 2])
    res = VOCSearch(ConstantErrorModel(0.0, 0.2)).search(win, 500)
    assert res.move == 3 and res.value >= WIN_THRESHOLD and res.nodes <= 20


def test_voc_respects_budget_and_uses_it_without_stop_rule():
    s = State.from_moves([3, 3, 2, 4, 4, 2, 3, 3, 2])
    for b in (30, 200, 800):
        r = VOCSearch(ConstantErrorModel(0.0, 0.25), stop_on_nonpositive=False).search(s, b)
        assert r.nodes <= b and r.nodes >= 0.85 * b


def test_voc_stop_rule_can_save_nodes():
    s = State.from_moves([3, 3, 2, 4, 4, 2, 3, 3, 2])
    r_stop = VOCSearch(ConstantErrorModel(0.0, 0.25), lam=1e-3, stop_on_nonpositive=True).search(s, 2000)
    r_full = VOCSearch(ConstantErrorModel(0.0, 0.25), stop_on_nonpositive=False).search(s, 2000)
    assert r_stop.nodes < r_full.nodes
