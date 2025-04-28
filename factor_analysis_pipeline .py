
#!/usr/bin/env python3
"""factor_analysis_pipeline.py — Generic ML-Varimax factor-analysis CLI
=====================================================================
This script distils the workflow used in the CSCEC green-transition paper
into a *dataset-agnostic* command-line tool.

Main features
-------------
* Accepts **any tidy** CSV / Excel / Parquet file (all numeric columns or
  user-specified indicators).
* Performs **KMO** and **Bartlett** tests to check suitability for factor
  analysis.
* Decides the number of factors by the *Kaiser* rule (*eigenvalue > 1*) or
  a user-supplied integer.
* Extracts factors with **maximum-likelihood** estimation + **Varimax**
  rotation (via the *factor_analyzer* package).
* Generates -->  
  • *scree.png* — eigenvalue scree plot  
  • *factor_loadings.csv* — rotated loadings  
  • *factor_scores.csv* — row-level scores + composite & rank
* Composite score = variance-weighted sum of factor scores.
* Runs on **Python ≥ 3.8** (uses `from __future__ import annotations`).

Usage example
-------------
```bash
python factor_analysis_pipeline.py \
    --input green_data.xlsx \
    --id-col Year \
    --indicators X1,X2,X3,X4 \
    --reverse X4 \
    --n-factors auto \
    --output-prefix results/green
```
Install requirements:
```bash
pip install -r requirements.txt
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from factor_analyzer import (
        FactorAnalyzer,
        calculate_bartlett_sphericity,
        calculate_kmo,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "Missing optional dependency 'factor_analyzer'.\n"
        "Install it via   pip install factor_analyzer   and try again."
    )

###############################################################################
# Utility helpers                                                             #
###############################################################################

def load_table(path: Path) -> pd.DataFrame:
    """Load CSV / Excel / Parquet by file suffix."""
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    # Fallback—assume CSV (handles .csv, .txt, no suffix, etc.)
    return pd.read_csv(path)

def choose_indicators(df: pd.DataFrame, indicators_arg: Optional[str]) -> List[str]:
    """Return list of indicator columns (user-supplied or all numeric)."""
    if indicators_arg:
        cols = [c.strip() for c in indicators_arg.split(",") if c.strip()]
    else:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        sys.exit("No numeric columns detected; specify --indicators explicitly.")
    return cols

def reverse_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """In-place multiply chosen columns by -1 (bad → good)."""
    for c in cols:
        if c in df.columns:
            df[c] = -df[c]
        else:
            print(f"[WARN] --reverse column '{c}' not found; skipping.")

def kmo_bartlett(X: np.ndarray) -> Tuple[float, float, float]:
    """Return overall KMO, Bartlett χ² and p-value."""
    _kmo_per_item, kmo_model = calculate_kmo(X)
    chi2, p = calculate_bartlett_sphericity(X)
    return kmo_model, chi2, p

def decide_n_factors(X: np.ndarray, arg: str) -> int:
    """Kaiser rule (eigenvalue > 1) or force explicit integer."""
    if arg != "auto":
        n = int(arg)
        if n < 1:
            sys.exit("--n-factors must be ≥ 1")
        return n
    eigvals, _ = np.linalg.eig(np.corrcoef(X, rowvar=False))
    n = int((eigvals > 1).sum()) or 1  # guarantee ≥ 1
    return n

def make_scree_plot(eigvals: np.ndarray, prefix: Path) -> None:
    """Save scree plot PNG next to other artefacts."""
    plt.figure()
    plt.plot(range(1, len(eigvals) + 1), eigvals, marker="o")
    plt.title("Scree Plot")
    plt.xlabel("Factor Number")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(prefix.with_suffix("_scree.png"), dpi=300)
    plt.close()

###############################################################################
# Main pipeline                                                               #
###############################################################################
def run_factor_analysis(
    df: pd.DataFrame,
    indicators: List[str],
    id_col: Optional[str],
    reverse_cols: Iterable[str],
    n_factors_spec: str,
    out_prefix: Path,
) -> None:
    """Core logic split out for easier unit testing."""

    # Prepare directory (prefix may contain sub-dir)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Copy & preprocess indicator matrix
    X_df = df[indicators].copy()
    reverse_columns(X_df, reverse_cols)
    X_df = X_df.dropna()
    X = X_df.to_numpy()
    if X.size == 0:
        sys.exit("After dropping NA rows, no data left to analyse.")

    # Suitability tests
    kmo_model, chi2, p = kmo_bartlett(X)
    print(f"KMO (overall): {kmo_model:.3f}")
    print(f"Bartlett χ²={chi2:.1f}, p={p:.3e}")

    # Decide # factors
    n_factors = decide_n_factors(X, n_factors_spec)
    print(f"Extracting {n_factors} factor(s)…")

    # Fit ML-Varimax FA
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="ml")
    fa.fit(X)

    # Scree plot (sorted descending)
    ev, _ = np.linalg.eig(np.corrcoef(X, rowvar=False))
    make_scree_plot(np.sort(ev)[::-1], out_prefix)

    # Save loadings
    loadings = pd.DataFrame(
        fa.loadings_, index=indicators, columns=[f"F{i+1}" for i in range(n_factors)]
    )
    loadings.to_csv(out_prefix.with_suffix("_factor_loadings.csv"))
    print("Rotated loadings saved.")

    # Factor scores + composite
    scores = fa.transform(X)
    scores_df = pd.DataFrame(scores, columns=[f"F{i+1}" for i in range(n_factors)])
    if id_col:
        scores_df.insert(0, id_col, df.loc[X_df.index, id_col].values)

    var_exp = fa.get_factor_variance()[1]
    weights = var_exp / var_exp.sum()
    scores_df["Composite"] = scores @ weights
    scores_df["Rank"] = scores_df["Composite"].rank(ascending=False, method="min").astype(int)
    scores_df.to_csv(out_prefix.with_suffix("_factor_scores.csv"), index=False)
    print("Scores & composite saved.")

    # Console summary
    print("\n=== Variance Explained (%) ===")
    for i, v in enumerate(var_exp, 1):
        print(f"F{i}: {v*100:.2f}")
    print(f"Total: {var_exp.sum()*100:.2f}")
    print(f"\nOutputs written with prefix '{out_prefix}_*'\n")

###############################################################################
# CLI wrapper                                                                 #
###############################################################################
def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ML-Varimax factor analysis on a dataset.")
    ap.add_argument("--input", "-i", required=True, help="CSV/Excel/Parquet file to analyse")
    ap.add_argument("--id-col", default=None, help="Non-numeric identifier (e.g. Year)")
    ap.add_argument(
        "--indicators",
        default=None,
        help="Comma-separated indicator columns (default = all numeric)",
    )
    ap.add_argument("--reverse", default=None, help="Columns whose sign should be flipped")
    ap.add_argument(
        "--n-factors",
        default="auto",
        help="'auto' (eigen>1) or explicit integer",
    )
    ap.add_argument(
        "--output-prefix",
        default="fa_results",
        help="Prefix (can include directory) for output artefacts",
    )
    return ap.parse_args()

def main() -> None:  # pragma: no cover
    args = parse_cli()
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file '{in_path}' not found.")

    df_raw = load_table(in_path)
    indicators = choose_indicators(df_raw, args.indicators)
    reverse_cols = [c.strip() for c in (args.reverse or "").split(",") if c.strip()]

    run_factor_analysis(
        df=df_raw,
        indicators=indicators,
        id_col=args.id_col,
        reverse_cols=reverse_cols,
        n_factors_spec=args.n_factors,
        out_prefix=Path(args.output_prefix),
    )

if __name__ == "__main__":
    main()
