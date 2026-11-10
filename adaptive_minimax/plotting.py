"""Figures for the report (Phase 6). Every function reads the CSV files written by
the experiment scripts and saves a PNG in results/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .experiments import LABELS

COLORS = {"A": "#8c8c8c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#2ca02c", "E": "#9467bd", "F": "#d62728"}
MARK = {"A": "s", "B": "o", "C": "^", "D": "v", "E": "D", "F": "*"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path)


# --------------------------------------------------------------------- models
def fig_classifier_comparison():
    df = pd.read_csv(C.RESULT_DIR / "classifier_comparison.csv")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.cv_auc, 0.4, label="group-CV (train games)")
    ax.bar(x + 0.2, df.test_auc, 0.4, label="held-out test games")
    ax.set_xticks(x, [m.replace("_", "\n") for m in df.model])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("'Is extra search worthwhile?' classifier (selected: "
                 f"{df[df.selected].model.iloc[0].replace('_', ' ')})")
    ax.legend()
    _save(fig, "01_classifier_comparison.png")


def fig_error_models():
    df = pd.read_csv(C.RESULT_DIR / "error_model_metrics.csv")
    df = df[df.split == "test"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    axs[0].bar(df.model, df.nll, color=["#8c8c8c", "#9467bd", "#d62728"])
    axs[0].set_ylabel("Gaussian NLL on test games (lower = better)")
    axs[0].set_title("Error-distribution models p(e|x)")
    w = 0.35
    x = np.arange(len(df))
    axs[1].bar(x - w / 2, df.cover_1sigma, w, label="within 1 sigma")
    axs[1].bar(x + w / 2, df.cover_2sigma, w, label="within 2 sigma")
    axs[1].axhline(0.683, ls="--", c="k", lw=0.8)
    axs[1].axhline(0.954, ls="--", c="k", lw=0.8)
    axs[1].set_xticks(x, df.model)
    axs[1].set_title("Calibration (dashed = ideal Gaussian)")
    axs[1].legend(loc="lower right")
    _save(fig, "02_error_models.png")


# ------------------------------------------------------------------ benchmark
def fig_budget_curves(summary: pd.DataFrame):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for a in sorted(summary.agent.unique()):
        d = summary[summary.agent == a].sort_values("budget")
        kw = dict(color=COLORS[a], marker=MARK[a], label=LABELS[a], lw=2 if a == "F" else 1.2,
                  ms=10 if a == "F" else 5)
        axs[0].plot(d.budget, d.mean_regret, **kw)
        axs[1].plot(d.budget, d.optimal_rate, **kw)
        axs[2].plot(d.mean_nodes, d.mean_regret, **kw)
    axs[0].set(xscale="log", xlabel="node budget per move", ylabel="mean decision regret (lower = better)",
               title="Decision regret vs budget")
    axs[1].set(xscale="log", xlabel="node budget per move", ylabel="agreement with reference best move",
               title="Agreement with deep reference")
    axs[2].set(xscale="log", xlabel="nodes actually used per move", ylabel="mean decision regret",
               title="Efficiency: regret vs nodes used")
    axs[1].legend(fontsize=7)
    _save(fig, "03_budget_curves.png")


def fig_tournament(summary: pd.DataFrame):
    budgets = sorted(summary.budget.unique())
    fig, axs = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 4.2), squeeze=False)
    for ax, b in zip(axs[0], budgets):
        d = summary[summary.budget == b].copy()
        d["label"] = d.agent + " vs " + d.opponent
        d = d.sort_values(["opponent", "agent"])
        y = np.arange(len(d))
        ax.barh(y, d.score, xerr=d.ci95, color=[COLORS[a] for a in d.agent], alpha=0.85)
        ax.axvline(0.5, c="k", lw=0.8)
        ax.set_yticks(y, d.label)
        ax.set_xlim(0, 1)
        ax.set_xlabel("score of the first agent (win=1, draw=0.5), 95% CI")
        ax.set_title(f"Head-to-head, {b} nodes / move")
    _save(fig, "04_tournament.png")


def fig_allocation(bench: pd.DataFrame, budget: int = 1000):
    if bench.empty:
        return
    avail_budgets = sorted(bench.budget.unique())
    if budget not in avail_budgets:
        budget = int(1000 if 1000 in avail_budgets else avail_budgets[-1])
    d = bench[(bench.budget == budget) & bench.agent.isin(["B", "C", "D", "E", "F"])].copy()
    if d.empty or len(d) < 3:
        return
    d["branch_bin"] = pd.cut(d.branching, [0, 5, 6, 7], labels=["<=5", "6", "7"])
    d["swing_bin"] = pd.qcut(d.swing.rank(method="first"), min(3, len(d)), labels=["low", "mid", "high"][:min(3, len(d))])
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for row, (metric, lab) in enumerate((("max_ply", "deepest ply reached"), ("nodes", "nodes used"))):
        for ax, (col, title) in zip(axs[row], (("tactical", "tactical volatility\n(0 = quiet, 1 = immediate win/threat)"),
                                               ("branch_bin", "branching factor"),
                                               ("swing_bin", "evaluation instability (swing)"))):
            piv = d.groupby([col, "agent"], observed=True)[metric].mean().unstack("agent")
            piv.plot(kind="bar", ax=ax, color=[COLORS[a] for a in piv.columns], rot=0, width=0.8)
            ax.set_ylabel(lab)
            ax.set_xlabel(title)
            ax.legend(fontsize=6, ncol=5)
    fig.suptitle(f"Search allocation at {budget} nodes/move (held-out positions)")
    _save(fig, "05_allocation.png")


def fig_ablation(abl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = abl.iloc[::-1]
    cols = ["#d62728" if v.startswith("F  full") else "#1f77b4" if v.startswith("B") else "#9aa" for v in d.variant]
    ax.barh(d.variant, d.mean_regret, color=cols)
    for y, (r, n) in enumerate(zip(d.mean_regret, d.mean_nodes)):
        ax.text(r, y, f"  {r:.3f}  ({n:.0f} nodes)", va="center", fontsize=7)
    ax.set_xlabel("mean decision regret on held-out positions (lower = better)")
    ax.set_title("Ablation study")
    ax.set_xlim(0, d.mean_regret.max() * 1.35)
    _save(fig, "06_ablation.png")


def fig_calibration():
    """Predicted sigma vs observed error spread for the learned model on test positions."""
    import joblib
    from .training import xy
    if not C.POSITIONS_CSV.exists() or not (C.MODEL_DIR / "error_learned.joblib").exists():
        return
    df = pd.read_csv(C.POSITIONS_CSV)
    te = df[df.split == "test"]
    if te.empty:
        return
    m = joblib.load(C.MODEL_DIR / "error_learned.joblib")
    ms = np.array([m.predict(list(x)) for x in xy(te)])
    z = pd.DataFrame({"sigma": ms[:, 1], "resid": te.error.to_numpy() - ms[:, 0]})
    q_bins = min(8, max(2, len(z) // 10))
    z["bin"] = pd.qcut(z.sigma, q_bins, duplicates="drop")
    g = z.groupby("bin", observed=True).agg(pred=("sigma", "mean"), obs=("resid", "std"))
    if g.empty or g.pred.isna().all():
        return
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(g.pred, g.obs, "o-", c=COLORS["F"])
    lim = max(float(g.pred.max()), float(g.obs.max())) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(xlabel="predicted sigma(x)", ylabel="observed std of error", title="Learned sigma(x) calibration (test)")
    _save(fig, "07_sigma_calibration.png")


def make_all():
    """Draw every figure for which the input CSVs exist."""
    R = C.RESULT_DIR
    pairs = [("classifier_comparison.csv", fig_classifier_comparison), ("error_model_metrics.csv", fig_error_models)]
    for f, fn in pairs:
        if (R / f).exists():
            fn()
    if C.POSITIONS_CSV.exists() and (C.MODEL_DIR / "error_learned.joblib").exists():
        fig_calibration()
    if (R / "benchmark_summary.csv").exists():
        fig_budget_curves(pd.read_csv(R / "benchmark_summary.csv"))
    if (R / "benchmark_raw.csv").exists():
        fig_allocation(pd.read_csv(R / "benchmark_raw.csv"), budget=1000)
    if (R / "tournament_summary.csv").exists():
        fig_tournament(pd.read_csv(R / "tournament_summary.csv"))
    if (R / "ablation.csv").exists():
        fig_ablation(pd.read_csv(R / "ablation.csv"))
