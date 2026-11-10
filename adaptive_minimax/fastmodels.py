"""Fast single-sample inference for the scikit-learn models used in search.

Calling `model.predict_proba` on one row costs 0.1 - 5 ms, which is far too slow
when the search asks for a prediction at thousands of nodes per move (and it
would silently distort every "equal time budget" comparison). This module
"compiles" fitted sklearn estimators into tiny pure-python closures
`f(x: list[float]) -> float` that evaluate in microseconds.

Supported estimators (also inside `Pipeline([StandardScaler, est])`):
    LogisticRegression, Ridge/LinearRegression,
    DecisionTree{Classifier,Regressor}, RandomForest{Classifier,Regressor},
    GradientBoosting{Classifier,Regressor}

Classifiers return P(class 1); regressors return the predicted value.
`tests/test_fastmodels.py` checks every compiled model against sklearn.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _compile_tree(tree, classifier: bool, classes=None):
    """Return f(x) -> leaf value (P(class1) for classifiers) for one sklearn Tree."""
    feat = tree.feature.tolist()
    thr = tree.threshold.tolist()
    left = tree.children_left.tolist()
    right = tree.children_right.tolist()
    if classifier:
        v = tree.value[:, 0, :]
        if v.shape[1] > 1:
            val = (v[:, 1] / np.maximum(v.sum(axis=1), 1e-12)).tolist()
        else:
            cls_val = 1.0 if classes is not None and len(classes) == 1 and classes[0] == 1 else 0.0
            val = [cls_val] * len(v)
    else:
        val = tree.value[:, 0, 0].tolist()

    def f(x):
        node = 0
        while left[node] != -1:
            node = left[node] if x[feat[node]] <= thr[node] else right[node]
        return val[node]
    return f


def compile_estimator(est):
    """Compile a fitted sklearn estimator/pipeline into f(x_list) -> float."""
    if isinstance(est, Pipeline):
        *pre, final = [s for _, s in est.steps]
        if len(pre) == 1 and isinstance(pre[0], StandardScaler):
            mean = pre[0].mean_.tolist()
            scale = [s if s != 0 else 1.0 for s in pre[0].scale_.tolist()]
            inner = compile_estimator(final)
            return lambda x: inner([(xi - m) / s for xi, m, s in zip(x, mean, scale)])
        if not pre:
            return compile_estimator(final)
        raise TypeError("unsupported pipeline")

    if isinstance(est, LogisticRegression):
        w = est.coef_[0].tolist()
        b = float(est.intercept_[0])
        return lambda x: _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)

    if isinstance(est, (Ridge, LinearRegression)):
        w = np.ravel(est.coef_).tolist()
        b = float(np.ravel(est.intercept_)[0])
        return lambda x: sum(wi * xi for wi, xi in zip(w, x)) + b

    if isinstance(est, DecisionTreeClassifier):
        return _compile_tree(est.tree_, True, classes=getattr(est, "classes_", None))
    if isinstance(est, DecisionTreeRegressor):
        return _compile_tree(est.tree_, False)

    if isinstance(est, (RandomForestClassifier, RandomForestRegressor)):
        clf = isinstance(est, RandomForestClassifier)
        classes = getattr(est, "classes_", None) if clf else None
        trees = [_compile_tree(t.tree_, clf, classes=classes) for t in est.estimators_]
        n = len(trees)
        return lambda x: sum(t(x) for t in trees) / n

    if isinstance(est, (GradientBoostingClassifier, GradientBoostingRegressor)):
        clf = isinstance(est, GradientBoostingClassifier)
        trees = [_compile_tree(t[0].tree_, False) for t in est.estimators_]
        lr = float(est.learning_rate)
        # recover the constant initial raw score robustly (version independent)
        x0 = np.zeros((1, est.n_features_in_))
        raw0 = float(est.decision_function(x0)[0]) if clf else float(est.predict(x0)[0])
        init = raw0 - lr * sum(t([0.0] * est.n_features_in_) for t in trees)
        if clf:
            return lambda x: _sigmoid(init + lr * sum(t(x) for t in trees))
        return lambda x: init + lr * sum(t(x) for t in trees)

    raise TypeError(f"cannot compile {type(est).__name__}")
