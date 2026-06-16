"""
Aggregate dispensing Excel to drug x week CSV (much smaller than raw export).

Usage:
  python xlsx_to_weekly_csv.py
  python xlsx_to_weekly_csv.py . --glob "*2602005*.xlsx" -o data/weekly_from_xlsx.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from xlsx_weekly_io import aggregate_excel_to_weekly


def main() -> None:
    ap = argparse.ArgumentParser(description="xlsx → drug×week CSV")
    ap.add_argument(
        "workdir",
        nargs="?",
        default="",
        help="Working directory (default: script directory)",
    )
    ap.add_argument(
        "--glob",
        dest="excel_glob",
        default="*2602005*.xlsx",
        help="Glob for Excel files (default *2602005*.xlsx)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="data/weekly_from_xlsx.csv",
        help="Output CSV path relative to workdir",
    )
    args = ap.parse_args()

    workdir = Path(args.workdir or os.path.dirname(os.path.abspath(__file__)))
    matches = list(workdir.glob(args.excel_glob))
    if not matches:
        raise SystemExit(f"No file matching {args.excel_glob} in {workdir}")
    xlsx_path = matches[0]

    df = aggregate_excel_to_weekly(xlsx_path)
    out_path = workdir / args.output
    df["week"] = pd.to_datetime(df["week"]).dt.strftime("%Y-%m-%d")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path} | rows={len(df)} | drugs={df['drugid'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
