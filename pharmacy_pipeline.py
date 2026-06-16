"""
Weekly pharmacy panel: raw sales CSV to model_data and train-test splits.
Shared by feature_engineering.py and benchmark_eval.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from feature_engineering import (
    Config,
    _floor_to_monday,
    add_weekly_features_from_padded,
    compute_drug_static_from_train_slice,
    merge_static_with_priors_safe,
)

FEATURE_COLS = [
    "lag1", "lag2", "lag4", "lag8", "lag12", "lag52",
    "ma4", "ma12", "cv",
    "ratio_lag1_ma4", "ratio_ma4_ma12",
    "cumu_qty", "nonzero_ratio",
    "weeks_since_last", "woy_sin", "woy_cos",
    "hist_mean_nonzero", "hist_total_vol",
    "is_new_drug",
]

ENGINEERED_EXTRA_COLS = [
    "cv_short", "hist_max_vol", "in_policy", "covid_burst", "flu_burst",
    "event_target", "qty_target", "weekly_qty",
]


@dataclass
class PipelineArtifacts:
    cfg: Config
    df_weekly_raw: pd.DataFrame
    df_clean: pd.DataFrame
    df_weekly_padded: pd.DataFrame
    model_data: pd.DataFrame
    static: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_fit: pd.DataFrame
    test_fit: pd.DataFrame
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    cutoff: pd.Timestamp = pd.Timestamp("2024-01-01")
    test_end: pd.Timestamp = pd.Timestamp("2024-12-31")
    atc_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def pad_zeros_per_drug(df_weekly: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (drugid, atc), g in df_weekly.groupby(["drugid", "atc_lvl3"], sort=False):
        g = g.sort_values("week")
        full_weeks = pd.date_range(g["week"].min(), g["week"].max(), freq="W-MON")
        gg = g.set_index("week").reindex(full_weeks)
        gg.index.name = "week"
        gg = gg.reset_index()
        gg["drugid"] = drugid
        gg["atc_lvl3"] = atc
        for c in ["weekly_qty", "in_policy", "covid_burst", "flu_burst"]:
            if c in gg.columns:
                gg[c] = gg[c].fillna(0)
            else:
                gg[c] = 0
        parts.append(gg)
    if not parts:
        return df_weekly.copy()
    return pd.concat(parts, ignore_index=True).sort_values(["drugid", "week"]).reset_index(drop=True)


def audit_negative_values(df_raw: pd.DataFrame, *, exclude_dyg: bool = True) -> pd.DataFrame:
    """Row-level negative quantity audit."""
    df = df_raw.copy()
    if exclude_dyg and "stock" in df.columns:
        df = df[df["stock"] != "DYG"]
    qty = pd.to_numeric(df.get("weekly_qty", pd.Series(dtype=float)), errors="coerce")
    n_total = len(df)
    n_neg = int((qty < 0).sum())
    n_zero = int((qty == 0).sum())
    n_pos = int((qty > 0).sum())
    n_na = int(qty.isna().sum())
    rows = [
        {"level": "row", "metric": "total_rows", "value": n_total},
        {"level": "row", "metric": "negative_rows", "value": n_neg},
        {"level": "row", "metric": "negative_pct", "value": round(100.0 * n_neg / max(n_total, 1), 4)},
        {"level": "row", "metric": "zero_rows", "value": n_zero},
        {"level": "row", "metric": "positive_rows", "value": n_pos},
        {"level": "row", "metric": "na_rows", "value": n_na},
    ]
    if n_neg > 0:
        rows.append({"level": "row", "metric": "negative_qty_sum", "value": float(qty[qty < 0].sum())})
        rows.append({"level": "row", "metric": "negative_qty_min", "value": float(qty.min())})
    return pd.DataFrame(rows)


def audit_weekly_negative_share(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Share of drug-weeks with negative net quantity after aggregation."""
    q = pd.to_numeric(df_weekly["weekly_qty"], errors="coerce").fillna(0.0)
    n = len(q)
    n_neg = int((q < 0).sum())
    rows = [
        {"level": "drug_week", "metric": "total_rows", "value": n},
        {"level": "drug_week", "metric": "negative_rows", "value": n_neg},
        {"level": "drug_week", "metric": "negative_pct", "value": round(100.0 * n_neg / max(n, 1), 4)},
        {"level": "drug_week", "metric": "zero_rows", "value": int((q == 0).sum())},
        {"level": "drug_week", "metric": "positive_rows", "value": int((q > 0).sum())},
    ]
    if n_neg > 0:
        rows.append({"level": "drug_week", "metric": "negative_qty_sum", "value": float(q[q < 0].sum())})
        rows.append({"level": "drug_week", "metric": "negative_qty_min", "value": float(q.min())})
    return pd.DataFrame(rows)


def normalize_row_level_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to row-level or weekly table with drugid, week, weekly_qty."""
    out = df.copy()
    out["week"] = pd.to_datetime(out["week"])
    out["weekly_qty"] = pd.to_numeric(out["weekly_qty"], errors="coerce").fillna(0.0)
    out["drugid"] = out["drugid"].astype(str).str.strip()
    for col, default in [("stock", "DYA"), ("department", ""), ("visit_cat", "")]:
        if col not in out.columns:
            out[col] = default
    return out


def merge_main_and_supplement_sales(
    workdir: str,
    main_file: str = "data/df_weekly_sales.csv",
    supplement_file: str = "data/weekly_from_xlsx.csv",
    supplement_from: str = "2025-01-01",
) -> pd.DataFrame:
    """Concatenate main CSV (through 2024) with supplemental weekly CSV (2025+), no overlapping weeks."""
    w = Path(workdir)
    main = normalize_row_level_sales(pd.read_csv(w / main_file, encoding="gbk"))
    sup_path = w / supplement_file
    if not sup_path.is_file():
        return main
    sup = normalize_row_level_sales(pd.read_csv(sup_path))
    cut = pd.Timestamp(supplement_from)
    main = main[main["week"] < cut]
    sup = sup[sup["week"] >= cut]
    combined = pd.concat([main, sup], ignore_index=True)
    return combined.sort_values(["drugid", "week"]).reset_index(drop=True)


def build_pipeline(
    cfg: Optional[Config] = None,
    *,
    keep_negative_rows: bool = False,
    weekly_raw: Optional[pd.DataFrame] = None,
) -> PipelineArtifacts:
    cfg = cfg or Config()
    workdir = cfg.workdir or os.path.dirname(os.path.abspath(__file__))

    if weekly_raw is not None:
        df_weekly_raw = normalize_row_level_sales(weekly_raw)
    else:
        df_weekly_raw = pd.read_csv(os.path.join(workdir, cfg.sales_file), encoding="gbk")
    policy_df = pd.read_csv(os.path.join(workdir, cfg.policy_file), encoding="gbk")
    atc_df = pd.read_csv(os.path.join(workdir, cfg.atc_file), encoding="gbk")

    policy_df = policy_df.copy()
    policy_df["start_date"] = pd.to_datetime(policy_df["start_date"])
    policy_df["end_date"] = pd.to_datetime(policy_df["end_date"])

    rows = []
    for _, r in policy_df.iterrows():
        start = _floor_to_monday(pd.Series([r["start_date"]])).iloc[0]
        end = _floor_to_monday(pd.Series([r["end_date"]])).iloc[0]
        if pd.isna(start) or pd.isna(end):
            continue
        weeks = pd.date_range(start=start, end=end, freq="W-MON")
        if len(weeks) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "drugid": r["drugid"],
                    "week": weeks,
                    "in_policy_flag": 1,
                }
            )
        )
    policy_weeks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["drugid", "week", "in_policy_flag"]
    )
    policy_weeks = policy_weeks.drop_duplicates(subset=["drugid", "week"])

    df_clean = df_weekly_raw.copy()
    if "stock" in df_clean.columns:
        df_clean = df_clean[df_clean["stock"] != "DYG"]
    df_clean["week"] = pd.to_datetime(df_clean["week"])
    df_clean["weekly_qty"] = pd.to_numeric(df_clean["weekly_qty"], errors="coerce").fillna(0.0)
    if not keep_negative_rows:
        # Exclude negative rows before aggregation (report frequency via audit_negative_values)
        df_clean = df_clean[df_clean["weekly_qty"] >= 0]

    atc_map = atc_df[["drugid", "atc_lvl3"]].drop_duplicates()
    df_clean = df_clean.merge(atc_map, on="drugid", how="left")
    df_clean = df_clean.merge(
        policy_weeks[["drugid", "week", "in_policy_flag"]], on=["drugid", "week"], how="left"
    )
    df_clean["in_policy"] = df_clean["in_policy_flag"].fillna(0).astype(int)
    df_clean["covid_burst"] = (
        (df_clean["week"] >= pd.Timestamp("2022-12-01")) & (df_clean["week"] <= pd.Timestamp("2023-04-01"))
    ).astype(int)
    df_clean["flu_burst"] = (
        (df_clean["week"] >= pd.Timestamp("2023-09-01")) & (df_clean["week"] <= pd.Timestamp("2024-02-01"))
    ).astype(int)
    df_clean["week"] = _floor_to_monday(df_clean["week"])

    df_weekly = (
        df_clean.groupby(["drugid", "atc_lvl3", "week"], as_index=False)
        .agg(
            weekly_qty=("weekly_qty", "sum"),
            in_policy=("in_policy", "max"),
            covid_burst=("covid_burst", "max"),
            flu_burst=("flu_burst", "max"),
        )
        .sort_values(["drugid", "week"])
    )
    df_weekly = df_weekly[df_weekly["atc_lvl3"].notna()].copy()
    df_weekly_padded = pad_zeros_per_drug(df_weekly)

    df = add_weekly_features_from_padded(df_weekly_padded)
    cutoff = pd.Timestamp(cfg.cutoff_date)
    test_end = pd.Timestamp(cfg.test_end_date)
    train_start = pd.Timestamp(cfg.train_start_date) if getattr(cfg, "train_start_date", "") else None
    train_cutoff_ind = df[df["week"] < cutoff].copy()
    if train_start is not None:
        train_cutoff_ind = train_cutoff_ind[train_cutoff_ind["week"] >= train_start].copy()
    static = compute_drug_static_from_train_slice(train_cutoff_ind)
    model_data = merge_static_with_priors_safe(df, static, train_cutoff_ind)

    train_df = model_data[model_data["week"] < cutoff].copy()
    if train_start is not None:
        train_df = train_df[train_df["week"] >= train_start].copy()
    test_df = model_data[(model_data["week"] >= cutoff) & (model_data["week"] <= test_end)].copy()
    train_fit = train_df[train_df["qty_target"].notna()].copy()
    test_fit = test_df.copy()

    return PipelineArtifacts(
        cfg=cfg,
        df_weekly_raw=df_weekly_raw,
        df_clean=df_clean,
        df_weekly_padded=df_weekly_padded,
        model_data=model_data,
        static=static,
        train_df=train_df,
        test_df=test_df,
        train_fit=train_fit,
        test_fit=test_fit,
        cutoff=cutoff,
        test_end=test_end,
        atc_df=atc_df,
    )


def compute_demand_stats(
    df_weekly_padded: pd.DataFrame,
    *,
    window_start: str = "2023-01-01",
    window_end: str = "2024-01-01",
) -> pd.DataFrame:
    """Syntetos-Boylan ADI/CV demand categories (default 2023 window)."""
    w0 = pd.Timestamp(window_start)
    w1 = pd.Timestamp(window_end)
    sub = df_weekly_padded[(df_weekly_padded["week"] >= w0) & (df_weekly_padded["week"] < w1)]
    stats = (
        sub.groupby("drugid", as_index=False)
        .agg(
            n_weeks=("week", "count"),
            n_nonzero=("weekly_qty", lambda s: int((s > 0).sum())),
            mean_qty=("weekly_qty", "mean"),
            sd_qty=("weekly_qty", "std"),
            atc_lvl3=("atc_lvl3", "first"),
        )
    )
    stats["ADI"] = np.where(stats["n_nonzero"] > 0, stats["n_weeks"] / stats["n_nonzero"], 52.0)
    stats["CV"] = np.where(stats["mean_qty"] > 0, stats["sd_qty"] / stats["mean_qty"], 0.0)
    stats["CV"] = stats["CV"].fillna(0.0)

    def _cat(row: pd.Series) -> str:
        adi, cv = float(row["ADI"]), float(row["CV"])
        if adi < 1.32 and cv < 0.49:
            return "Smooth"
        if adi >= 1.32 and cv < 0.49:
            return "Intermittent"
        if adi < 1.32 and cv >= 0.49:
            return "Erratic"
        if adi >= 1.32 and cv >= 0.49:
            return "Lumpy"
        return "Unknown"

    stats["Category"] = stats.apply(_cat, axis=1)
    stats["Zero_Pct"] = (1 - stats["n_nonzero"] / stats["n_weeks"].clip(lower=1)) * 100.0
    return stats
