"""
Data-driven feature-selection diagnostics for reviewer response.

Produces:
  - fig_feature_importance.png
  - fig_feature_correlation.png
  - fig_acf_pacf_by_pattern.png
  - table_feature_selection_summary.csv
  - feature_selection_rationale.md

Usage:
  python feature_selection_analysis.py
  python feature_selection_analysis.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_engineering import Config
from pharmacy_pipeline import FEATURE_COLS, PipelineArtifacts, build_pipeline, compute_demand_stats

FEATURE_GROUPS: dict[str, tuple[str, str]] = {
    "lag1": ("Lag", "1-week demand"),
    "lag2": ("Lag", "2-week demand"),
    "lag4": ("Lag", "4-week demand"),
    "lag8": ("Lag", "8-week demand"),
    "lag12": ("Lag", "12-week demand"),
    "lag52": ("Lag", "52-week seasonality"),
    "ma4": ("Rolling", "4-week mean of lag1"),
    "ma12": ("Rolling", "12-week mean of lag1"),
    "cv": ("Rolling", "26-week CV of lag1"),
    "ratio_lag1_ma4": ("Ratio", "lag1 / ma4"),
    "ratio_ma4_ma12": ("Ratio", "ma4 / ma12"),
    "cumu_qty": ("Intermittency", "cumulative volume"),
    "nonzero_ratio": ("Intermittency", "non-zero week ratio"),
    "weeks_since_last": ("Intermittency", "weeks since last non-zero"),
    "woy_sin": ("Seasonality", "ISO week sine"),
    "woy_cos": ("Seasonality", "ISO week cosine"),
    "hist_mean_nonzero": ("Static", "train-period mean non-zero qty"),
    "hist_total_vol": ("Static", "train-period total volume"),
    "is_new_drug": ("Cold-start", "no train static; ATC prior used"),
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _acf(y: np.ndarray, nlags: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.zeros(nlags + 1)
    y = y - y.mean()
    denom = float(np.dot(y, y))
    if denom <= 0:
        return np.zeros(nlags + 1)
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(float(np.dot(y[k:], y[:-k]) / denom))
    return np.asarray(out)


def _pacf_yw(y: np.ndarray, nlags: int) -> np.ndarray:
    """Yule-Walker PACF (no statsmodels dependency)."""
    acf_vals = _acf(y, nlags)
    pacf_vals = [1.0]
    phi = np.zeros((nlags, nlags))
    sigma = float(np.var(y)) if y.size else 0.0
    for k in range(1, nlags + 1):
        if k == 1:
            pacf_vals.append(float(acf_vals[1]))
            phi[0, 0] = acf_vals[1]
            sigma *= 1.0 - acf_vals[1] ** 2
            continue
        r = acf_vals[1 : k + 1]
        R = np.array([[acf_vals[abs(i - j)] for j in range(k)] for i in range(k)], dtype=float)
        try:
            a = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            pacf_vals.append(0.0)
            continue
        pacf_vals.append(float(a[-1]))
        phi[k - 1, :k] = a
        sigma *= 1.0 - a[-1] ** 2
    return np.asarray(pacf_vals)


def pick_exemplar_drugs(demand_stats: pd.DataFrame) -> dict[str, Any]:
    """One median-volume exemplar per Syntetos-Boylan category."""
    exemplars: dict[str, Any] = {}
    for cat in ("Smooth", "Intermittent", "Erratic", "Lumpy"):
        sub = demand_stats[demand_stats["Category"] == cat].copy()
        if sub.empty:
            continue
        med = sub["mean_qty"].median()
        row = sub.iloc[(sub["mean_qty"] - med).abs().argsort()[:1]].iloc[0]
        exemplars[cat] = row["drugid"]
    return exemplars


def load_or_fit_importance(
    art: PipelineArtifacts,
    out_dir: Path,
    *,
    lgb_params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    imp_path = out_dir / "feature_importance.csv"
    if imp_path.is_file():
        imp = pd.read_csv(imp_path)
        if {"feature", "importance_gain"}.issubset(imp.columns):
            return imp.sort_values("importance_gain", ascending=False).reset_index(drop=True)

    from benchmark_eval import _load_best_lgbm_params, train_lgbm_tweedie

    workdir = os.path.dirname(os.path.abspath(__file__))
    cfg = Config(workdir=workdir)
    params = lgb_params or _load_best_lgbm_params(cfg, out_dir)
    _, _, _ = train_lgbm_tweedie(
        art,
        lgb_params=params,
        include_zero_weeks=True,
        save_importance_path=imp_path,
    )
    return pd.read_csv(imp_path).sort_values("importance_gain", ascending=False).reset_index(drop=True)


def build_selection_table(
    art: PipelineArtifacts,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    panel = art.train_fit[FEATURE_COLS + ["weekly_qty"]].copy()
    rows: list[dict[str, Any]] = []
    imp_map = importance.set_index("feature")
    total_gain = float(importance["importance_gain"].sum()) or 1.0

    for feat in FEATURE_COLS:
        grp, rationale = FEATURE_GROUPS.get(feat, ("Other", ""))
        x = panel[feat]
        y = panel["weekly_qty"].fillna(0.0)
        mask = x.notna() & y.notna()
        spearman = float(x[mask].corr(y[mask], method="spearman")) if mask.sum() > 10 else np.nan
        gain = float(imp_map.loc[feat, "importance_gain"]) if feat in imp_map.index else 0.0
        split = int(imp_map.loc[feat, "importance_split"]) if feat in imp_map.index else 0
        rows.append(
            {
                "feature": feat,
                "group": grp,
                "rationale": rationale,
                "spearman_with_target": round(spearman, 4) if np.isfinite(spearman) else np.nan,
                "importance_gain": round(gain, 2),
                "importance_gain_pct": round(100.0 * gain / total_gain, 2),
                "importance_split": split,
            }
        )
    out = pd.DataFrame(rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    out["importance_rank"] = np.arange(1, len(out) + 1)
    return out


def plot_feature_importance(importance: pd.DataFrame, out_dir: Path) -> None:
    imp = importance.sort_values("importance_gain", ascending=True).copy()
    total = float(imp["importance_gain"].sum()) or 1.0
    imp["gain_pct"] = 100.0 * imp["importance_gain"] / total

    colors = []
    for f in imp["feature"]:
        grp = FEATURE_GROUPS.get(f, ("Other", ""))[0]
        palette = {
            "Lag": "#457B9D",
            "Rolling": "#1D3557",
            "Ratio": "#A8DADC",
            "Intermittency": "#E63946",
            "Seasonality": "#F4A261",
            "Static": "#2A9D8F",
            "Cold-start": "#BDBDBD",
        }
        colors.append(palette.get(grp, "#6C757D"))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["feature"], imp["gain_pct"], color=colors)
    for i, (pct, g) in enumerate(zip(imp["gain_pct"], imp["importance_gain"])):
        ax.text(pct + 0.3, i, f"{pct:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Relative gain importance (%)")
    ax.set_title("LightGBM-Tweedie feature importance (2024 holdout model)")
    ax.set_xlim(0, max(imp["gain_pct"]) * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_feature_importance.png", dpi=150)
    plt.close(fig)


def plot_feature_correlation(art: PipelineArtifacts, out_dir: Path) -> None:
    panel = art.train_fit[FEATURE_COLS + ["weekly_qty"]].copy()
    corr = panel.corr(method="spearman")
    corr = corr.rename(index={"weekly_qty": "target (weekly_qty)"}, columns={"weekly_qty": "target"})

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.35 or (labels[j] == "target" and i < len(FEATURE_COLS)):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_title("Spearman correlation among features and weekly demand (training set)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_feature_correlation.png", dpi=150)
    plt.close(fig)
    corr.to_csv(out_dir / "table_feature_spearman_correlation.csv", encoding="utf-8-sig")


def plot_acf_pacf_by_pattern(
    art: PipelineArtifacts,
    demand_stats: pd.DataFrame,
    exemplars: dict[str, Any],
    out_dir: Path,
    *,
    nlags: int = 26,
) -> None:
    cats = [c for c in ("Smooth", "Intermittent", "Erratic", "Lumpy") if c in exemplars]
    if not cats:
        return

    fig, axes = plt.subplots(len(cats), 2, figsize=(10, 2.6 * len(cats)), squeeze=False)
    for i, cat in enumerate(cats):
        drug = exemplars[cat]
        g = art.df_weekly_padded[art.df_weekly_padded["drugid"] == drug].sort_values("week")
        y = g["weekly_qty"].fillna(0.0).to_numpy(dtype=float)
        cutoff = art.cutoff
        weeks = pd.to_datetime(g["week"])
        y_tr = y[weeks < cutoff]
        if y_tr.size < nlags + 5:
            y_tr = y

        acf_vals = _acf(y_tr, nlags)
        pacf_vals = _pacf_yw(y_tr, nlags)
        lags = np.arange(nlags + 1)

        ax_acf, ax_pacf = axes[i, 0], axes[i, 1]
        ax_acf.bar(lags, acf_vals, width=0.6, color="#457B9D")
        ax_acf.axhline(0, color="gray", lw=0.8)
        ax_acf.axhline(1.96 / np.sqrt(max(len(y_tr), 1)), color="red", ls="--", lw=0.8)
        ax_acf.axhline(-1.96 / np.sqrt(max(len(y_tr), 1)), color="red", ls="--", lw=0.8)
        ax_acf.set_title(f"{cat} exemplar ({drug}) — ACF")
        ax_acf.set_xlabel("Lag (weeks)")

        ax_pacf.bar(lags[1:], pacf_vals[1:], width=0.6, color="#E63946")
        ax_pacf.axhline(0, color="gray", lw=0.8)
        ax_pacf.axhline(1.96 / np.sqrt(max(len(y_tr), 1)), color="red", ls="--", lw=0.8)
        ax_pacf.axhline(-1.96 / np.sqrt(max(len(y_tr), 1)), color="red", ls="--", lw=0.8)
        ax_pacf.set_title(f"{cat} exemplar ({drug}) — PACF")
        ax_pacf.set_xlabel("Lag (weeks)")

    fig.suptitle("Weekly demand autocorrelation by Syntetos–Boylan pattern (training window)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_acf_pacf_by_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [{"Category": c, "exemplar_drugid": exemplars[c]} for c in cats]
    ).to_csv(out_dir / "table_acf_exemplar_drugs.csv", index=False, encoding="utf-8-sig")


def write_rationale_md(table: pd.DataFrame, out_dir: Path) -> None:
    top5 = table.head(5)
    lines = [
        "# Feature selection rationale (reviewer response draft)",
        "",
        "We used a **hybrid** approach: domain-informed candidate features from intermittent-demand forecasting literature, "
        "then **data-driven confirmation** on the full training panel (all SKUs, all drug-week rows before 2024).",
        "",
        "## 1. Candidate generation (literature + demand taxonomy)",
        "",
        "- **Short- and medium-term lags** (1, 2, 4, 8, 12 weeks) and **annual seasonality** (lag 52): standard for weekly pharmaceutical demand.",
        "- **Rolling means / CV** and **ratio features**: capture level shifts and volatility (Syntetos–Boylan erratic/lumpy patterns).",
        "- **Intermittency counters** (`nonzero_ratio`, `weeks_since_last`, `cumu_qty`): Croston/SBA-style intermittent demand.",
        "- **Calendar encodings** (`woy_sin`, `woy_cos`): smooth weekly seasonality without 52 dummy variables.",
        "- **Drug-level static history** (`hist_mean_nonzero`, `hist_total_vol`, `is_new_drug`): cold-start and scale heterogeneity across 1,480+ SKUs.",
        "",
        "Policy dummies (`in_policy`, `covid_burst`, `flu_burst`) and `lag26` were **candidates** but **not retained** in the final 19-feature set "
        "(near-zero gain importance or redundant with calendar/rolling terms).",
        "",
        "## 2. Data-driven validation",
        "",
        "| Diagnostic | File | Finding |",
        "|------------|------|---------|",
        "| LightGBM gain importance | `fig_feature_importance.png` | Rolling means (`ma4`, `ma12`) and `hist_mean_nonzero` dominate; lags and seasonality contribute; `is_new_drug` ≈ 0 (rare in training). |",
        "| Spearman correlation with target | `fig_feature_correlation.png` | Demand-linked features correlate with `weekly_qty`; engineered ratios add orthogonal signal beyond raw lags. |",
        "| ACF / PACF by demand pattern | `fig_acf_pacf_by_pattern.png` | Smooth SKUs show persistent ACF; intermittent/lumpy SKUs show weak short-lag ACF → justifies multi-horizon lags + intermittency features. |",
        "",
        "**Top 5 features by gain importance:**",
        "",
        "| Rank | Feature | Group | Gain (%) | Spearman with target |",
        "|------|---------|-------|----------|----------------------|",
    ]
    for _, r in top5.iterrows():
        lines.append(
            f"| {int(r['importance_rank'])} | {r['feature']} | {r['group']} | {r['importance_gain_pct']}% | {r['spearman_with_target']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Why this is not post-hoc overfitting",
            "",
            "- Feature set was **fixed before** 2024 holdout evaluation; importance and correlations are computed on **training data only**.",
            "- The same features are used for **all 1,480 SKUs** (no per-SKU feature cherry-picking).",
            "- Hyperparameter tuning used a temporal validation fold within training; feature list was not re-optimized on the test year.",
            "",
        ]
    )
    (out_dir / "feature_selection_rationale.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_selection_analysis(out_dir: Path) -> pd.DataFrame:
    workdir = os.path.dirname(os.path.abspath(__file__))
    cfg = Config(workdir=workdir)
    out_dir = _ensure_dir(out_dir)
    print("Feature selection analysis...", flush=True)
    art = build_pipeline(cfg)
    demand_stats = compute_demand_stats(art.df_weekly_padded)
    exemplars = pick_exemplar_drugs(demand_stats)

    importance = load_or_fit_importance(art, out_dir)
    table = build_selection_table(art, importance)
    table.to_csv(out_dir / "table_feature_selection_summary.csv", index=False, encoding="utf-8-sig")

    plot_feature_importance(importance, out_dir)
    plot_feature_correlation(art, out_dir)
    plot_acf_pacf_by_pattern(art, demand_stats, exemplars, out_dir)
    write_rationale_md(table, out_dir)

    print(f"Done. See {out_dir}/fig_feature_*.png and feature_selection_rationale.md", flush=True)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature-selection diagnostics for reviewer response")
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()
    workdir = os.path.dirname(os.path.abspath(__file__))
    run_feature_selection_analysis(Path(workdir) / args.output_dir)


if __name__ == "__main__":
    main()
