"""Phase 4 - train and validate the learned components.

(1) "Is extra search worthwhile?" classifier  (used by agent D)
    Candidates: logistic regression, decision tree, random forest, gradient
    boosting. Compared with GROUP k-fold cross-validation on the training games
    (groups = complete games, so correlated positions of one game never sit on
    both sides of a split). The best model by CV ROC-AUC is refit on the training
    split, its decision threshold is chosen on the validation split (max F1) and
    it is finally scored ONCE on the held-out test games.

(2) Evaluation-error distribution p(e|x)  (agents E and F)
    Gaussian N(mu(x), sigma(x)^2); three variants (constant / binned / learned)
    scored by Gaussian negative log-likelihood and interval coverage on the
    validation and test games.

Outputs
    models/classifier.joblib         {"model", "name", "threshold", "features"}
    models/error_{constant,binned,learned}.joblib
    results/classifier_comparison.csv
    results/error_model_metrics.csv
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .error_models import BinnedErrorModel, ConstantErrorModel, LearnedErrorModel
from .features import FEATURE_NAMES


# ------------------------------------------------------------------- helpers
def load_dataset(path=C.POSITIONS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def xy(df: pd.DataFrame):
    return df[FEATURE_NAMES].to_numpy(float)


def classifier_zoo(seed: int = 0) -> dict:
    return {
        "logistic_regression": Pipeline([("sc", StandardScaler()),
                                         ("m", LogisticRegression(class_weight="balanced", max_iter=2000))]),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30,
                                                class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=60, max_depth=7, min_samples_leaf=15,
                                                class_weight="balanced_subsample", random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08,
                                                        subsample=0.8, random_state=seed),
    }


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    max_f1 = max(f1s) if f1s else 0.0
    if max_f1 == 0:
        return 0.5
    return float(grid[int(np.argmax(f1s))])


# -------------------------------------------------------------- 1. classifier
def train_classifier(df: pd.DataFrame, out_dir=C.MODEL_DIR, res_dir=C.RESULT_DIR, verbose=True) -> dict:
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, ytr, gtr = xy(tr), tr.label.to_numpy(), tr.game_id.to_numpy()
    rows = []
    n_unique_groups = len(np.unique(gtr))
    n_splits = max(2, min(5, n_unique_groups))
    for name, model in classifier_zoo().items():
        t0 = time.time()
        if n_unique_groups >= 2:
            cv = GroupKFold(n_splits=n_splits)
            p_cv = cross_val_predict(model, Xtr, ytr, groups=gtr, cv=cv, method="predict_proba")[:, 1]
        else:
            model.fit(Xtr, ytr)
            p_cv = model.predict_proba(Xtr)[:, 1]
        rows.append(dict(model=name, cv_auc=roc_auc_score(ytr, p_cv),
                         cv_avg_precision=average_precision_score(ytr, p_cv), fit_seconds=time.time() - t0))
        if verbose:
            print(f"  CV {name:22s} AUC={rows[-1]['cv_auc']:.3f}  AP={rows[-1]['cv_avg_precision']:.3f}")
    comp = pd.DataFrame(rows)

    best_name = comp.sort_values("cv_auc", ascending=False).iloc[0]["model"]
    model = classifier_zoo()[best_name].fit(Xtr, ytr)
    thr = best_f1_threshold(va.label.to_numpy(), model.predict_proba(xy(va))[:, 1])

    # final held-out scores for ALL models (refit on train) - for the report
    test_rows = []
    for name, m in classifier_zoo().items():
        m = m.fit(Xtr, ytr)
        t_thr = best_f1_threshold(va.label.to_numpy(), m.predict_proba(xy(va))[:, 1])
        p = m.predict_proba(xy(te))[:, 1]
        yt = te.label.to_numpy()
        test_rows.append(dict(model=name, test_auc=roc_auc_score(yt, p),
                              test_avg_precision=average_precision_score(yt, p),
                              threshold=t_thr, test_f1=f1_score(yt, p >= t_thr, zero_division=0),
                              test_precision=precision_score(yt, p >= t_thr, zero_division=0),
                              test_recall=recall_score(yt, p >= t_thr, zero_division=0)))
    comp = comp.merge(pd.DataFrame(test_rows), on="model")
    comp["selected"] = comp.model == best_name
    comp.to_csv(res_dir / "classifier_comparison.csv", index=False)
    joblib.dump({"model": model, "name": best_name, "threshold": thr, "features": FEATURE_NAMES},
                out_dir / "classifier.joblib")
    if verbose:
        print(comp.round(3).to_string(index=False))
        print(f"selected: {best_name}  (threshold {thr:.2f})  positive rate train={ytr.mean():.2f}")
    return {"name": best_name, "threshold": thr, "table": comp}


# ------------------------------------------------------- 2. error distribution
def gaussian_scores(model, X: np.ndarray, e: np.ndarray) -> dict:
    mu_sig = np.array([model.predict(list(x)) for x in X])
    mu, sig = mu_sig[:, 0], mu_sig[:, 1]
    z = (e - mu) / sig
    nll = 0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2
    return dict(nll=float(nll.mean()), rmse_mu=float(np.sqrt(np.mean((e - mu) ** 2))),
                cover_1sigma=float(np.mean(np.abs(z) <= 1)), cover_2sigma=float(np.mean(np.abs(z) <= 2)),
                mean_sigma=float(sig.mean()))


def train_error_models(df: pd.DataFrame, out_dir=C.MODEL_DIR, res_dir=C.RESULT_DIR, verbose=True) -> pd.DataFrame:
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, etr, gtr = xy(tr), tr.error.to_numpy(), tr.game_id.to_numpy()
    models = {
        "constant": ConstantErrorModel.fit(Xtr, etr),
        "binned": BinnedErrorModel.fit(Xtr, etr),
        "learned": LearnedErrorModel.fit(Xtr, etr, gtr),
    }
    rows = []
    for name, m in models.items():
        joblib.dump(m, out_dir / f"error_{name}.joblib")
        for split_name, part in (("val", va), ("test", te)):
            rows.append(dict(model=name, split=split_name,
                             **gaussian_scores(m, xy(part), part.error.to_numpy())))
    out = pd.DataFrame(rows)
    out.to_csv(res_dir / "error_model_metrics.csv", index=False)
    if verbose:
        print(out.round(3).to_string(index=False))
        print("(nll: lower is better; coverage should be ~0.68 / ~0.95 for a calibrated Gaussian)")
    return out


def train_all(verbose=True):
    df = load_dataset()
    if verbose:
        print(f"dataset: {len(df)} positions, {df.game_id.nunique()} games; "
              f"split sizes {df.split.value_counts().to_dict()}")
        print("== (1) 'is extra search worthwhile?' classifier ==")
    clf = train_classifier(df, verbose=verbose)
    if verbose:
        print("== (2) evaluation-error distributions p(e|x) ==")
    err = train_error_models(df, verbose=verbose)
    return clf, err
