"""
Python version of: Supplemental Document S1.R

Pipeline:
- Load weekly sales + policy + ATC (GBK encoded CSVs)
- Expand policy to drug-week flags (weekly, Monday week start)
- Clean + aggregate to drug-week (no department leakage)
- Pad zeros for continuous weekly index per drug
- Feature engineering (lags, rolling means, CV, ratios, cumu, intermittency, seasonality)
- Leakage-safe static features computed on training period only; ATC/global priors for cold-start drugs + `is_new_drug`
- Train LightGBM Tweedie (one-step) and evaluate (global + per-drug); `WeeklyPredictor` for causal incremental inference
- Optional hybrid ARIMA fallback for top-volume drugs
- Sensitivity analysis for tweedie_variance_power

Outputs (written to current working directory):
- global_metrics_one_step.csv
- per_drug_metrics_one_step.csv
- worst_errors_one_step.csv
- global_metrics_final.csv
- eval_final_per_drug.csv
- sensitivity_tweedie_variance_power.csv
- feature_importance.csv
"""

from __future__ import annotations

import os
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# History buffer for lag52 and 26-week rolling statistics on lag1
_MAX_QTY_HISTORY = 128


def _floor_to_monday(d: pd.Series) -> pd.Series:
    """Equivalent to lubridate::floor_date(date, 'week', week_start=1) (Monday)."""
    d = pd.to_datetime(d)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def _wmape(pred: np.ndarray, act: np.ndarray) -> float:
    denom = np.sum(act) + 1e-9
    return float(np.sum(np.abs(pred - act)) / denom)


def _weeks_since_last_nonzero(y: np.ndarray) -> np.ndarray:
    """
    Mirrors the R loop:
      if previous week > 0 => reset to 1 else increment.
    """
    n = len(y)
    out = np.empty(n, dtype=np.int32)
    ctr = 1
    for i in range(n):
        if i == 0:
            out[i] = 1
        else:
            if y[i - 1] > 0:
                ctr = 1
            else:
                ctr += 1
            out[i] = ctr
    return out


def _iso_woy_sin_cos(week_ts: pd.Timestamp) -> tuple[float, float]:
    iso = pd.Timestamp(week_ts).isocalendar()
    woy = int(iso.week)
    wsin = float(np.sin(2 * np.pi * woy / 52.0))
    wcos = float(np.cos(2 * np.pi * woy / 52.0))
    return wsin, wcos


def _rolling_mean_std_ddof1(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    m = float(np.mean(x))
    if x.size < 2:
        return m, 0.0
    sd = float(np.std(x, ddof=1))
    return m, sd


@dataclass
class DrugWeeklyState:
    """Per-drug incremental quantity history for online features aligned with batch pandas."""

    drugid: Any
    atc_lvl3: Any
    qty_hist: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_QTY_HISTORY))
    cumu_qty: float = 0.0
    nonzero_count: int = 0
    row_number: int = 0
    prev_qty: float = 0.0
    wsl_ctr: int = 1
    hist_mean_nonzero: float = 0.0
    hist_total_vol: float = 0.0
    is_new_drug: float = 0.0

    def update(self, qty: float) -> None:
        q = float(qty)
        self.row_number += 1
        if self.row_number == 1:
            self.wsl_ctr = 1
        else:
            if self.prev_qty > 0:
                self.wsl_ctr = 1
            else:
                self.wsl_ctr += 1
        self.prev_qty = q
        self.cumu_qty += q
        if q > 0:
            self.nonzero_count += 1
        self.qty_hist.append(q)


def build_feature_row_from_state(state: DrugWeeklyState, target_week: pd.Timestamp) -> dict[str, float]:
    """
    After `feed_actual` has injected demand through the prior week, build one-step features
    for `target_week` (causal, no future leakage). lag1 is the most recent week;
    cumu_qty / nonzero_ratio / row_number use observed weeks only.
    Batch training rows from pandas groupby include the current week in cumu_qty (R convention);
    values may differ slightly—use this function for online inference.
    """
    q = np.asarray(list(state.qty_hist), dtype=float)
    wsin, wcos = _iso_woy_sin_cos(target_week)

    def lag(k: int) -> float:
        if k <= 0 or len(q) < k:
            return 0.0
        return float(q[-k])

    lag1, lag2, lag4, lag8, lag12, lag52 = lag(1), lag(2), lag(4), lag(8), lag(12), lag(52)

    tail4 = q[-4:] if len(q) >= 4 else q
    tail12 = q[-12:] if len(q) >= 12 else q
    ma4 = float(np.mean(tail4)) if tail4.size else 0.0
    ma12 = float(np.mean(tail12)) if tail12.size else 0.0

    tail26 = q[-26:] if len(q) >= 26 else q
    rm, sd = _rolling_mean_std_ddof1(tail26)
    cv = float(sd / rm) if rm > 0 else 0.0

    ratio_lag1_ma4 = float(lag1 / ma4) if ma4 > 0 else 0.0
    ratio_ma4_ma12 = float(ma4 / ma12) if ma12 > 0 else 0.0

    rn = max(int(state.row_number), 1)
    nonzero_ratio = float(state.nonzero_count) / float(rn)

    lag1_tail4 = q[-4:] if len(q) >= 4 else q
    m4, sd4 = _rolling_mean_std_ddof1(lag1_tail4)
    cv_short = float(sd4 / (m4 + 1e-6)) if lag1_tail4.size else 0.0

    return {
        "lag1": lag1,
        "lag2": lag2,
        "lag4": lag4,
        "lag8": lag8,
        "lag12": lag12,
        "lag52": lag52,
        "ma4": ma4,
        "ma12": ma12,
        "cv": cv,
        "ratio_lag1_ma4": ratio_lag1_ma4,
        "ratio_ma4_ma12": ratio_ma4_ma12,
        "cumu_qty": float(state.cumu_qty),
        "nonzero_ratio": nonzero_ratio,
        "weeks_since_last": float(state.wsl_ctr),
        "woy_sin": wsin,
        "woy_cos": wcos,
        "hist_mean_nonzero": float(state.hist_mean_nonzero),
        "hist_total_vol": float(state.hist_total_vol),
        "is_new_drug": float(state.is_new_drug),
    }


def merge_static_with_priors_safe(
    df: pd.DataFrame,
    static: pd.DataFrame,
    train_cutoff_ind: pd.DataFrame,
) -> pd.DataFrame:
    """Merge train-period static; ATC/global priors for cold-start drugs; adds is_new_drug."""
    drug_atc_train = train_cutoff_ind.groupby("drugid", as_index=False).agg(atc_lvl3=("atc_lvl3", "first"))
    static_drugids = set(static["drugid"].tolist())

    sta = static.merge(drug_atc_train, on="drugid", how="left")
    atc_prior = (
        sta.groupby("atc_lvl3", as_index=False)[["hist_mean_nonzero", "hist_total_vol"]]
        .mean()
        .rename(
            columns={
                "hist_mean_nonzero": "atc_hist_mean_nonzero",
                "hist_total_vol": "atc_hist_total_vol",
            }
        )
    )
    global_mean_nz = float(sta["hist_mean_nonzero"].mean()) if len(sta) else 0.0
    global_tot = float(sta["hist_total_vol"].mean()) if len(sta) else 0.0

    m = df.merge(static, on="drugid", how="left")
    m = m.merge(atc_prior, on="atc_lvl3", how="left")

    m["hist_mean_nonzero"] = m["hist_mean_nonzero"].fillna(m["atc_hist_mean_nonzero"]).fillna(global_mean_nz)
    m["hist_total_vol"] = m["hist_total_vol"].fillna(m["atc_hist_total_vol"]).fillna(global_tot)
    m["is_new_drug"] = (~m["drugid"].isin(static_drugids)).astype(np.float64)

    drop_cols = [c for c in m.columns if c.startswith("atc_hist_")]
    m = m.drop(columns=drop_cols, errors="ignore")
    return m


def add_weekly_features_from_padded(df_weekly_padded: pd.DataFrame) -> pd.DataFrame:
    """
    Batch feature engineering on padded drug-week panel (drugid, atc_lvl3, week, weekly_qty).
    Returns lag/rolling/qty_target columns without static merges.
    """
    df = df_weekly_padded.copy()
    df["weekly_qty"] = pd.to_numeric(df["weekly_qty"], errors="coerce").fillna(0.0)
    df["event_target"] = (df["weekly_qty"] > 0).astype(int)
    df["qty_target"] = df["weekly_qty"].where(df["weekly_qty"] > 0, np.nan).astype(float)
    df = df.sort_values(["drugid", "week"]).reset_index(drop=True)
    g = df.groupby("drugid", sort=False)
    for k in [1, 2, 4, 8, 12, 26, 52]:
        df[f"lag{k}"] = g["weekly_qty"].shift(k).fillna(0.0)
    df["ma4"] = g["lag1"].transform(lambda s: s.rolling(4, min_periods=1).mean()).fillna(0.0)
    df["ma12"] = g["lag1"].transform(lambda s: s.rolling(12, min_periods=1).mean()).fillna(0.0)
    roll_mean_26 = g["lag1"].transform(lambda s: s.rolling(26, min_periods=1).mean())
    roll_sd_26 = g["lag1"].transform(lambda s: s.rolling(26, min_periods=1).std(ddof=1))
    df["roll_mean"] = roll_mean_26.fillna(0.0)
    df["roll_sd"] = roll_sd_26.fillna(0.0)
    df["cv"] = np.where(df["roll_mean"] > 0, df["roll_sd"] / df["roll_mean"], 0.0)
    df["cumu_qty"] = g["weekly_qty"].cumsum()
    df["nonzero_count"] = g["weekly_qty"].transform(lambda s: (s > 0).cumsum())
    df["row_number"] = g.cumcount() + 1
    df["nonzero_ratio"] = df["nonzero_count"] / df["row_number"]
    df["weeks_since_last"] = g["weekly_qty"].transform(
        lambda s: pd.Series(_weeks_since_last_nonzero(s.to_numpy()), index=s.index)
    )
    iso = pd.to_datetime(df["week"]).dt.isocalendar()
    df["woy"] = iso.week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * df["woy"] / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * df["woy"] / 52.0)
    df["ratio_lag1_ma4"] = np.where(df["ma4"] > 0, df["lag1"] / df["ma4"], 0.0)
    df["ratio_ma4_ma12"] = np.where(df["ma12"] > 0, df["ma4"] / df["ma12"], 0.0)

    def _cv_short(s: pd.Series) -> pd.Series:
        m = s.rolling(4, min_periods=1).mean()
        sd = s.rolling(4, min_periods=1).std(ddof=1)
        return (sd / (m + 1e-6)).fillna(0.0)

    df["cv_short"] = g["lag1"].transform(_cv_short)
    return df


def compute_drug_static_from_train_slice(train_slice: pd.DataFrame) -> pd.DataFrame:
    """Drug-level static features aggregated from a training week slice only."""

    def _hist_mean_nonzero(sub: pd.DataFrame) -> float:
        nz = sub.loc[sub["weekly_qty"] > 0, "qty_target"]
        if len(nz) == 0:
            return 0.0
        v = float(np.nanmean(nz.to_numpy()))
        return v if np.isfinite(v) else 0.0

    out_cols = ["drugid", "hist_mean_nonzero", "hist_total_vol", "hist_max_vol"]
    if train_slice.empty:
        return pd.DataFrame(columns=out_cols)

    # Explicit per-drug aggregation (avoids pandas groupby.apply edge cases)
    rows: list[dict[str, Any]] = []
    for drugid, sub in train_slice.groupby("drugid", sort=False):
        w = sub["weekly_qty"].to_numpy()
        rows.append(
            {
                "drugid": drugid,
                "hist_mean_nonzero": _hist_mean_nonzero(sub),
                "hist_total_vol": float(np.nansum(w)),
                "hist_max_vol": float(np.nanmax(w)) if len(w) else 0.0,
            }
        )
    static = pd.DataFrame(rows)
    for c in ["hist_mean_nonzero", "hist_total_vol", "hist_max_vol"]:
        static[c] = static[c].fillna(0.0)
    return static


class WeeklyPredictor:
    """
    Online incremental one-step forecasting with DrugWeeklyState and a trained LightGBM model.
    Static features use individual history when available, else ATC/global priors (same as merge_static_with_priors_safe).
    """

    def __init__(
        self,
        feature_cols: list[str],
        model: Any,
        drug_static: dict[Any, tuple[float, float]],
        atc_prior: dict[Any, tuple[float, float]],
        global_prior: tuple[float, float],
        drug_atc: dict[Any, Any],
        best_iteration: Optional[int] = None,
    ):
        self.feature_cols = feature_cols
        self.model = model
        self.drug_static = drug_static
        self.atc_prior = atc_prior
        self.global_prior = global_prior
        self.drug_atc = drug_atc
        self.best_iteration = best_iteration
        self.states: dict[Any, DrugWeeklyState] = {}

    def _ensure_state(self, drugid: Any, atc_lvl3: Any = None) -> DrugWeeklyState:
        """Create empty state from priors for drugs not yet seen in feed_actual (e.g. first appear in test)."""
        st = self.states.get(drugid)
        if st is not None:
            return st
        atc = atc_lvl3 if atc_lvl3 is not None else self.drug_atc.get(drugid)
        hm, ht, ind = self._prior_for(drugid, atc)
        st = DrugWeeklyState(
            drugid=drugid,
            atc_lvl3=atc,
            hist_mean_nonzero=hm,
            hist_total_vol=ht,
            is_new_drug=ind,
        )
        self.states[drugid] = st
        return st

    def _prior_for(self, drugid: Any, atc: Any) -> tuple[float, float, float]:
        """Return hist_mean_nonzero, hist_total_vol, is_new_drug."""
        if drugid in self.drug_static:
            hm, ht = self.drug_static[drugid]
            return float(hm), float(ht), 0.0
        if atc is not None and not (isinstance(atc, float) and np.isnan(atc)) and atc in self.atc_prior:
            hm, ht = self.atc_prior[atc]
            return float(hm), float(ht), 1.0
        hm, ht = self.global_prior
        return float(hm), float(ht), 1.0

    def feed_actual(self, drugid: Any, week: pd.Timestamp, qty: float, atc_lvl3: Any = None) -> None:
        st = self._ensure_state(drugid, atc_lvl3)
        st.update(float(qty))

    def features_for_next_week(
        self, drugid: Any, next_week: pd.Timestamp, atc_lvl3: Any = None
    ) -> np.ndarray:
        st = self._ensure_state(drugid, atc_lvl3)
        row = build_feature_row_from_state(st, pd.Timestamp(next_week))
        return np.array([[row[c] for c in self.feature_cols]], dtype=float)

    def predict_next(self, drugid: Any, next_week: pd.Timestamp, atc_lvl3: Any = None) -> np.ndarray:
        X = self.features_for_next_week(drugid, next_week, atc_lvl3=atc_lvl3)
        kwargs = {}
        if self.best_iteration is not None:
            kwargs["num_iteration"] = self.best_iteration
        return self.model.predict(X, **kwargs)

    def retrain_model(
        self,
        df_weekly_padded: pd.DataFrame,
        feature_cols: list[str],
        *,
        train_through_week: Optional[pd.Timestamp] = None,
        lookback_weeks: int = 104,
        lgb_params: Optional[dict[str, Any]] = None,
        nrounds: int = 2000,
        early_stopping_rounds: int = 80,
        valid_week_frac: float = 0.1,
        min_data_in_leaf: int = 20,
        tweedie_variance_power: float = 1.3,
        seed: int = 2025,
        verbosity: int = -1,
    ) -> None:
        """
        Retrain LightGBM on the most recent lookback_weeks ending at train_through_week.
        Updates self.model, best_iteration, drug_static, and ATC/global priors.
        Requires full df_weekly_padded so lag52 and related features have enough history.
        """
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install lightgbm") from e

        df_feat = add_weekly_features_from_padded(df_weekly_padded)
        df_feat["week"] = pd.to_datetime(df_feat["week"])
        tt = pd.Timestamp(train_through_week if train_through_week is not None else df_feat["week"].max())
        all_weeks = np.sort(df_feat["week"].unique())
        all_weeks = all_weeks[all_weeks <= tt]
        if len(all_weeks) == 0:
            raise ValueError("retrain_model: no weeks <= train_through_week")
        if len(all_weeks) > lookback_weeks:
            window_arr = all_weeks[-lookback_weeks:]
        else:
            window_arr = all_weeks
        # Do not feed np.datetime64 via .tolist() into isin (nanosecond ints break Timestamp matching)
        week_win = pd.DatetimeIndex(window_arr)

        train_slice = df_feat[df_feat["week"].isin(week_win)].copy()
        static = compute_drug_static_from_train_slice(train_slice)
        model_data = merge_static_with_priors_safe(train_slice, static, train_slice)
        train_fit_all = model_data[model_data["qty_target"].notna()].copy()
        if train_fit_all.empty:
            raise ValueError("retrain_model: no positive-demand rows in window")

        w_sorted = week_win.sort_values()
        n_valid = max(2, int(np.ceil(len(w_sorted) * valid_week_frac)))
        valid_week_set = set(w_sorted[-n_valid:])
        train_week_set = set(w_sorted) - valid_week_set
        if len(train_week_set) == 0:
            train_week_set = set(w_sorted)
            valid_week_set = set()

        train_fit = train_fit_all[train_fit_all["week"].isin(train_week_set)]
        valid_fit = train_fit_all[train_fit_all["week"].isin(valid_week_set)] if valid_week_set else None

        X_tr = train_fit[feature_cols].to_numpy(dtype=float)
        y_tr = train_fit["qty_target"].to_numpy(dtype=float)
        params = dict(lgb_params) if lgb_params else {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_variance_power,
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_data_in_leaf": min_data_in_leaf,
            "force_col_wise": True,
            "seed": seed,
            "verbosity": verbosity,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        if valid_fit is not None and len(valid_fit) > 0:
            X_va = valid_fit[feature_cols].to_numpy(dtype=float)
            y_va = valid_fit["qty_target"].to_numpy(dtype=float)
            dvalid = lgb.Dataset(X_va, label=y_va)
            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=nrounds,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
        else:
            model = lgb.train(params=params, train_set=dtrain, num_boost_round=nrounds)

        bi = getattr(model, "best_iteration", None)
        self.best_iteration = int(bi) if bi is not None else int(nrounds)
        self.model = model

        pr_dict, atc_d, glb = build_priors_for_predictor(static, train_slice)
        self.drug_static = pr_dict
        self.atc_prior = atc_d
        self.global_prior = glb


@dataclass
class Config:
    # Default: directory containing this script (makes the script portable across machines)
    workdir: str = ""
    sales_file: str = "data/df_weekly_sales.csv"
    policy_file: str = "data/policy_df.csv"
    atc_file: str = "data/atc_df.csv"
    cutoff_date: str = "2024-01-01"
    test_end_date: str = "2024-12-31"
    train_start_date: str = ""
    """If set, static features and train_df restrict to weeks on/after this date."""
    tweedie_variance_power: float = 1.3
    nrounds: int = 3000
    early_stopping_rounds: int = 100
    seed: int = 2025
    run_hybrid_arima: bool = True
    topN_arima: int = 30
    variance_powers: tuple[float, ...] = (1.2, 1.3, 1.5, 1.8)


def build_priors_for_predictor(
    static: pd.DataFrame,
    train_cutoff_ind: pd.DataFrame,
) -> tuple[dict[Any, tuple[float, float]], dict[Any, tuple[float, float]], tuple[float, float]]:
    """Individual and ATC-level priors for WeeklyPredictor from training static table."""
    drug_atc_train = train_cutoff_ind.groupby("drugid", as_index=False).agg(atc_lvl3=("atc_lvl3", "first"))
    sta = static.merge(drug_atc_train, on="drugid", how="left")
    ap = sta.groupby("atc_lvl3")[["hist_mean_nonzero", "hist_total_vol"]].mean()
    drug_static: dict[Any, tuple[float, float]] = {
        r["drugid"]: (float(r["hist_mean_nonzero"]), float(r["hist_total_vol"])) for _, r in static.iterrows()
    }
    atc_prior: dict[Any, tuple[float, float]] = {
        idx: (float(row["hist_mean_nonzero"]), float(row["hist_total_vol"])) for idx, row in ap.iterrows()
    }
    g_mean = float(sta["hist_mean_nonzero"].mean()) if len(sta) else 0.0
    g_tot = float(sta["hist_total_vol"].mean()) if len(sta) else 0.0
    return drug_static, atc_prior, (g_mean, g_tot)
