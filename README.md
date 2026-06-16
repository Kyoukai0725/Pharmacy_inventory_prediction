# Pharmacy demand forecasting — supplementary code

Python reproduction of the LightGBM-Tweedie weekly demand forecasting pipeline, benchmark comparisons, sensitivity analyses, and financial validation described in the manuscript and reviewer responses.

## Contents

| File | Role |
|------|------|
| `pharmacy_pipeline.py` | Build weekly panel, ADI/CV labels, train/test split, negative-value audit |
| `feature_engineering.py` | Lag/rolling features, static merges, `WeeklyPredictor` for online inference |
| `benchmark_eval.py` | Main entry: grid search, benchmarks (RF, SBA, optional ARIMA/LSTM), tables & figures |
| `xlsx_weekly_io.py` | Stream-read hospital dispensing Excel exports |
| `xlsx_to_weekly_csv.py` | CLI to aggregate Excel → `data/weekly_from_xlsx.csv` |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Data files

Place the following under `data/` (not included in this repository because of hospital data-use restrictions):

| File | Description |
|------|-------------|
| `df_weekly_sales.csv` | Weekly drug-level sales through 2024 (columns: `drugid`, `week`, `weekly_qty`, …) |
| `policy_df.csv` | Policy / formulary flags (GBK-encoded) |
| `atc_df.csv` | ATC level-3 mapping per drug |
| `real_orders.csv` | Manual vs AI order lines for financial validation (optional) |
| `weekly_from_xlsx.csv` | Optional 2025+ weeks from Excel export (see below) |

To build `weekly_from_xlsx.csv` from a hospital movement query export:

```bash
python xlsx_to_weekly_csv.py . --glob "*2602005*.xlsx" -o data/weekly_from_xlsx.csv
```

## Main commands

Primary 2024 holdout benchmark (skips slow ARIMA/LSTM):

```bash
python benchmark_eval.py --skip-arima --skip-lstm
```

Full benchmark including optional models:

```bash
python benchmark_eval.py
```

Other analyses:

```bash
python benchmark_eval.py --validation-sensitivity      # rolling-origin + 2025 OOT
python benchmark_eval.py --financial-only              # AI vs manual orders
python benchmark_eval.py --negative-rows-sensitivity # negative-row handling
python benchmark_eval.py --zero-week-ablation        # zero-week training ablation
python benchmark_eval.py --no-grid                     # skip hyperparameter grid search
```

Results are written to `outputs/` by default (`--output-dir` to override).

## Key outputs

- `table_benchmark_global.csv` — global wMAPE / MAE / Bias for all models
- `table_lgbm_grid_search.csv`, `table_lgbm_best_params.csv` — hyperparameter search
- `table_overfit_train_valid_test.csv` — train / valid / test fit
- `predictions_test_weekly.csv` — per drug-week predictions on 2024 holdout
- `validation_holdout_rationale.md` — holdout design narrative (with `--validation-sensitivity`)
- `table_financial_*.csv` — financial validation (with `--financial-only` and `real_orders.csv`)

## Modeling defaults

- **Train:** weeks before `2024-01-01`; **test:** calendar year 2024
- **LightGBM:** Tweedie objective, training includes zero-demand weeks (`weekly_qty` label)
- **Hyperparameters:** 18-combination grid on the last ~10% of training weeks; select by validation wMAPE
- **Negative rows:** dropped before weekly aggregation (frequency reported in `table_negative_value_audit.csv`)

## Citation

If you use this code, please cite the associated manuscript (details in the paper).
