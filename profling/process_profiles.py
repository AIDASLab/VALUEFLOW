#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------
# Helpers
# -------------------------------
def find_groups(intensity_cols):
    """Group intensity columns by theory prefix."""
    groups = {"pvq": [], "mft": [], "duty": [], "rights": []}
    for c in intensity_cols:
        prefix = c.split("_", 1)[0]
        if prefix in groups:
            groups[prefix].append(c)
    for k in groups:
        groups[k].sort()
    return groups

def normalize_row_group(values: pd.Series, method: str, step: float, eps: float):
    """
    Normalize a 1D Series (one row's group columns).
    Returns a Series aligned to input index (NaNs preserved).
    Methods:
      - rank: dense rank starting at 0, multiplied by 'step' (e.g., 0, step, 2*step, ...)
      - minmax: (x - min) / (max - min + eps)
      - zscore: (x - mean) / (std + eps)
      - percentile: rank / (n-1 + eps) in [0,1]
    """
    out = pd.Series(index=values.index, dtype=float)
    mask = values.notna()
    if mask.sum() <= 1:
        out[mask] = 0.0
        return out

    v = values[mask].astype(float)

    if method == "rank":
        ranks = v.rank(method="dense", ascending=True) - 1.0
        out[mask] = ranks * step
    elif method == "minmax":
        vmin, vmax = v.min(), v.max()
        out[mask] = (v - vmin) / (max(vmax - vmin, eps))
    elif method == "zscore":
        mu, sig = v.mean(), v.std(ddof=0)
        out[mask] = (v - mu) / (sig + eps)
    elif method == "percentile":
        ranks = v.rank(method="dense", ascending=True) - 1.0
        denom = max(len(np.unique(v)) - 1, 1)
        out[mask] = ranks / denom
    else:
        raise ValueError(f"Unknown method: {method}")

    out[~mask] = np.nan
    return out

def hybrid_per_column(s: pd.Series, alpha: float = 0.5, eps: float = 1e-8):
    """
    Compute hybrid score for a single column across rows:
      mag = x / max(|x|)           -> [-1, 1] (preserves sign)
      pct = 2 * rank_pct - 1       -> [-1, 1] (relative standing)
      zrob = tanh(((x - median)/(MAD+eps))/2)  # optional diagnostic
      hybrid = alpha*mag + (1-alpha)*pct
    Returns (hybrid, mag, pct, zrob) as Series aligned to s.index.
    """
    x = pd.to_numeric(s, errors="coerce")
    mask = x.notna()
    mag = pd.Series(index=x.index, dtype=float)
    pct_signed = pd.Series(index=x.index, dtype=float)
    zrob = pd.Series(index=x.index, dtype=float)

    if mask.sum() == 0:
        return x*0.0, x*0.0, x*0.0, x*0.0

    xv = x[mask].astype(float)
    max_abs = np.abs(xv).max()
    mag[mask] = xv / (max_abs if max_abs > 0 else 1.0)

    pct = xv.rank(pct=True)  # [0,1]
    pct_signed[mask] = 2 * pct - 1  # [-1,1]

    med = xv.median()
    mad = (np.abs(xv - med)).median()
    z = (xv - med) / (mad if mad > 0 else 1.0)
    zrob[mask] = np.tanh(z / 2.0)

    hybrid = pd.Series(index=x.index, dtype=float)
    hybrid[mask] = alpha * mag[mask] + (1.0 - alpha) * pct_signed[mask]
    hybrid[~mask] = np.nan
    mag[~mask] = np.nan
    pct_signed[~mask] = np.nan
    zrob[~mask] = np.nan
    return hybrid, mag, pct_signed, zrob

# -------------------------------
# Main transform
# -------------------------------
def normalize_profiles(
    in_csv: Path,
    out_dir: Path,
    method: str = "rank",
    step: float = 1.0,
    axis: str = "row",
    eps: float = 1e-8,
    alpha: float = 0.5,
    add_extras: bool = False,
):
    """
    axis='row': normalize within each theory group *per row* (attribute profile).
    axis='col': normalize each column *across rows*.

    method supports: rank | minmax | zscore | percentile | hybrid
      - hybrid is only meaningful with axis='col' and replaces each *_intensity column
        by the hybrid score; if --add_extras is passed, also writes *_mag, *_pct, *_zrob, *_hybrid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_csv)

    intensity_cols = [c for c in df.columns if c.endswith("_intensity")]
    if not intensity_cols:
        raise ValueError("No *_intensity columns found.")

    groups = find_groups(intensity_cols)
    df_norm = df.copy()

    if method == "hybrid":
        if axis != "col":
            raise ValueError("For method='hybrid', use --axis col (hybrid is column-wise).")
        # Compute hybrid per column across rows
        for col in intensity_cols:
            h, m, p, z = hybrid_per_column(df[col], alpha=alpha, eps=eps)
            df_norm[col] = h  # replace intensity with hybrid score
            if add_extras:
                df_norm[col + "_mag"] = m
                df_norm[col + "_pct"] = p
                df_norm[col + "_zrob"] = z
                df_norm[col + "_hybrid"] = h
    else:
        if axis == "row":
            # Per attribute (row), normalize within each theory group
            for idx, row in df.iterrows():
                for gname, gcols in groups.items():
                    if not gcols:
                        continue
                    vals = row[gcols]
                    norm_vals = normalize_row_group(vals, method=method, step=step, eps=eps)
                    df_norm.loc[idx, gcols] = norm_vals.values
        elif axis == "col":
            # Across attributes (rows), normalize each theory group column independently
            for gname, gcols in groups.items():
                for col in gcols:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() <= 1:
                        df_norm[col] = 0.0
                        continue
                    if method == "rank":
                        ranks = s.rank(method="dense", ascending=True) - 1.0
                        df_norm[col] = ranks * step
                    elif method == "minmax":
                        vmin, vmax = s.min(), s.max()
                        df_norm[col] = (s - vmin) / (max(vmax - vmin, eps))
                    elif method == "zscore":
                        mu, sig = s.mean(), s.std(ddof=0)
                        df_norm[col] = (s - mu) / (sig + eps)
                    elif method == "percentile":
                        ranks = s.rank(method="dense", ascending=True) - 1.0
                        denom = max(len(np.unique(s.dropna())) - 1, 1)
                        df_norm[col] = ranks / denom
                    else:
                        raise ValueError(f"Unknown method: {method}")
        else:
            raise ValueError("axis must be 'row' or 'col'.")

    # Save
    base = in_csv.stem
    suffix = f"norm_{method}_{axis}"
    if method == "rank":
        suffix += f"_step{int(step) if float(step).is_integer() else step}"
    if method == "hybrid":
        suffix += f"_alpha{alpha}"
        if add_extras:
            suffix += "_extras"
    out_path = out_dir / f"{base}__{suffix}.csv"
    df_norm.to_csv(out_path, index=False)
    print(f"Saved → {out_path} (rows={len(df_norm)}, cols={len(df_norm.columns)})")

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize attribute value profiles per theory group.")
    parser.add_argument("--in_csv", type=Path, default=Path(""))
    parser.add_argument("--out_dir", type=Path, default=Path(""))
    parser.add_argument("--method", choices=["rank", "minmax", "zscore", "percentile", "hybrid"], default="rank",
                        help="Normalization method. Use 'hybrid' to combine absolute magnitude and relative standing.")
    parser.add_argument("--step", type=float, default=1.0,
                        help="Spacing for rank method: 0, step, 2*step, ... (only for method=rank).")
    parser.add_argument("--axis", choices=["row", "col"], default="row",
                        help="row: normalize within each theory group per row. col: normalize per column across rows. For method=hybrid, use 'col'.")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid weight: score = alpha*mag + (1-alpha)*pct. (only for method=hybrid)")
    parser.add_argument("--add_extras", action="store_true",
                        help="When using method=hybrid, also output *_mag, *_pct, *_zrob, *_hybrid helper columns.")
    args = parser.parse_args()

    normalize_profiles(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        method=args.method,
        step=args.step,
        axis=args.axis,
        eps=args.eps,
        alpha=args.alpha,
        add_extras=args.add_extras,
    )
