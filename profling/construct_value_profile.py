#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ============================== Utils ==============================

def distance_col_for(intensity_col: str) -> str:
    return intensity_col[:-10] + "_distance" if intensity_col.endswith("_intensity") else ""

def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def _normalize_abs(x, max_abs=10.0):
    x = _safe_numeric(x)
    return (x.abs() / max_abs).clip(0.0, 1.0)

def per_value_score_signed(df: pd.DataFrame, intensity_col: str, *, alpha=1.0, beta=0.0, max_abs=10.0):
    """
    Base per-row signed score for a value:
      base = x * (1 - d)^alpha * (|x|/M)^beta
    where x is intensity in [-M,M], d in [0,1] if distance col exists.
    """
    x = _safe_numeric(df.get(intensity_col, np.nan))
    dcol = distance_col_for(intensity_col)
    if dcol in df.columns:
        d = _safe_numeric(df[dcol]).clip(0.0, 1.0)
        w_rel = (1.0 - d).pow(alpha)
    else:
        w_rel = pd.Series(1.0, index=df.index)

    w_mag = _normalize_abs(x, max_abs=max_abs).pow(beta) if beta else 1.0
    base = x * w_rel * w_mag
    return base.where(~x.isna(), other=np.nan)

def weighted_mean_intensity(df_attr: pd.DataFrame, value_col: str, keys, key_col: str, weight_by_distance: bool):
    sub = df_attr[df_attr[key_col].isin(keys)]
    if sub.empty or value_col not in sub.columns:
        return np.nan
    x = _safe_numeric(sub[value_col])
    if not weight_by_distance:
        return float(x.mean(skipna=True))
    dcol = distance_col_for(value_col)
    if dcol in sub.columns:
        d = _safe_numeric(sub[dcol]).clip(0.0, 1.0)
        w = 1.0 - d
        mask = ~(x.isna() | w.isna())
        if not mask.any():
            return np.nan
        w_eff = w[mask]
        nz = w_eff > 0
        if not nz.any():
            return np.nan
        x_eff = x[mask][nz]
        w_eff = w_eff[nz]
        return float(np.dot(x_eff, w_eff) / w_eff.sum())
    return float(x.mean(skipna=True))

def global_mean_intensity(df_attr: pd.DataFrame, value_col: str, weight_by_distance: bool):
    if value_col not in df_attr.columns:
        return np.nan
    x = _safe_numeric(df_attr[value_col])
    if not weight_by_distance:
        return float(x.mean(skipna=True))
    dcol = distance_col_for(value_col)
    if dcol in df_attr.columns:
        d = _safe_numeric(df_attr[dcol]).clip(0.0, 1.0)
        w = 1.0 - d
        mask = ~(x.isna() | w.isna())
        if not mask.any():
            return float(x.mean(skipna=True))
        w_eff = w[mask]
        nz = w_eff > 0
        if not nz.any():
            return float(x.mean(skipna=True))
        x_eff = x[mask][nz]
        w_eff = w_eff[nz]
        return float(np.dot(x_eff, w_eff) / w_eff.sum())
    return float(x.mean(skipna=True))

# ============================== Core ==============================

def select_by_significance_and_profile(
    attribute_dir: Path,
    out_dir: Path,
    key_col: str,
    *,
    alpha: float = 1.0,         # exponent on (1-d) in base score
    beta: float = 0.0,          # exponent on |x|/M in base score
    max_abs: float = 10.0,
    method: str = "zscore",     # 'zscore' or 'margin'
    gamma: float = 1.0,         # exponent on divergence term
    require_positive: bool = False,  # if True, clamp divergence at >=0
    top_n: int = 5,
    profile_weight_by_distance: bool = True,
    seed: int = 123,
    save_suffix: str = ""
):
    """
    For each value and attribute:
      1) base score per (query) from x, distance, and magnitude (signed).
      2) divergence vs other attributes for the same query (z-score or margin).
      3) significance = (divergence_clamped_or_abs)**gamma.
      4) rank by significance, select top_n queries.
      5) profile = mean of ORIGINAL intensities for those queries (weighted optional).
         (if none selected due to NaNs, fallback to global mean)

    Saves:
      - selections CSV (per (attribute,value,query): base, divergence, significance, rank)
      - profile CSV (rows=attributes, cols=value intensities)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Load attributes
    attr_files = sorted([p for p in attribute_dir.glob("*.csv") if p.is_file()])
    if not attr_files:
        raise FileNotFoundError(f"No attribute CSVs found in {attribute_dir}")

    # Detect value columns
    all_value_cols = set()
    for f in attr_files:
        df = pd.read_csv(f)
        all_value_cols.update([c for c in df.columns if c.endswith("_intensity")])
    value_cols = sorted(all_value_cols)
    if not value_cols:
        raise RuntimeError("No *_intensity columns found in inputs.")

    # Build long table with base score per (attribute,value,query)
    long_rows = []
    for f in attr_files:
        df = pd.read_csv(f)
        if key_col not in df.columns:
            raise RuntimeError(f"key_col '{key_col}' not found in {f}")
        # attribute name (prefer embedded column if consistent)
        if "attribute" in df.columns and df["attribute"].nunique() == 1:
            attr_name = str(df["attribute"].iloc[0])
        else:
            attr_name = f.stem

        for vcol in value_cols:
            if vcol not in df.columns:
                continue
            base = per_value_score_signed(df, vcol, alpha=alpha, beta=beta, max_abs=max_abs)
            long_rows.append(pd.DataFrame({
                key_col: df[key_col],
                "attribute": attr_name,
                "value_col": vcol,
                "base": base
            }))

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    if long_df.empty:
        raise RuntimeError("No rows assembled. Check key_col and inputs.")

    # Aggregate duplicates (same key, attribute, value)
    long_df = long_df.groupby([key_col, "attribute", "value_col"], as_index=False)["base"].mean()

    # For each value, compute divergence per (key, attribute) by comparing to others
    sel_rows = []   # selection table rows
    selected_map = {}  # (attribute, value_col) -> selected list of keys

    for vcol in value_cols:
        sub = long_df[long_df["value_col"] == vcol].copy()
        if sub.empty:
            continue

        pivot_base = sub.pivot(index=key_col, columns="attribute", values="base")
        attrs = list(pivot_base.columns)
        if len(attrs) == 0:
            continue

        # Precompute row-wise stats used by divergence methods
        if method == "zscore":
            row_mean = pivot_base.mean(axis=1, skipna=True)
            row_std  = pivot_base.std(axis=1, skipna=True).replace(0, np.nan)
        elif method == "margin":
            # nothing global to precompute
            pass
        else:
            raise ValueError("method must be 'zscore' or 'margin'")

        for attr in attrs:
            s_attr = pivot_base[attr]

            if method == "zscore":
                # z = (x - mean_others) / std_others
                # compute mean_others via adjustment: (n*row_mean - s_attr) / (n-1)
                n = pivot_base.count(axis=1)  # per-row non-NaN attr count
                mean_others = (row_mean * n - s_attr) / (n - 1)
                # std_others: recompute excluding attr (safe fallback to row_std if small n)
                # For robustness: if n < 2 → no others → use NaN
                std_others = row_std.copy()
                std_others[n < 2] = np.nan
                z = (s_attr - mean_others) / (std_others + 1e-8)
                div = z
            else:  # margin
                others = pivot_base.drop(columns=[attr]) if len(attrs) > 1 else None
                if others is None or others.shape[1] == 0:
                    div = pd.Series(index=pivot_base.index, dtype=float)
                else:
                    other_max = others.max(axis=1, skipna=True)
                    div = s_attr - other_max  # can be negative

            # significance
            if require_positive:
                sig = div.clip(lower=0.0)
            else:
                sig = div.abs()
            if gamma != 1.0:
                sig = sig.pow(gamma)

            # rank (ties are resolved by base value, then random to stabilize)
            # Build a DataFrame to sort with secondary keys
            temp = pd.DataFrame({
                "sig": sig,
                "base": s_attr,
                "rand": rng.random(len(sig))
            }, index=sig.index)
            temp = temp.sort_values(by=["sig", "base", "rand"], ascending=[False, False, False])

            # select top_n with non-NaN significance; if none, try fallback to base
            top_keys = list(temp.index[temp["sig"].notna()][:top_n])
            if len(top_keys) == 0:
                # fallback: rank by |base|
                temp2 = pd.DataFrame({"base_abs": s_attr.abs(), "rand": rng.random(len(s_attr))}, index=s_attr.index)
                temp2 = temp2.sort_values(by=["base_abs", "rand"], ascending=[False, False])
                top_keys = list(temp2.index[:top_n])

            selected_map[(attr, vcol)] = top_keys

            # Store rows for selections CSV
            rec = sub[[key_col]].drop_duplicates().set_index(key_col)
            rec["attribute"] = attr
            rec["value"] = vcol
            rec["base"] = s_attr
            rec["divergence"] = div
            rec["significance"] = sig
            # attach rank (1..), NaN gets rank after valid ones
            ranks = pd.Series(index=temp.index, data=np.arange(1, len(temp) + 1))
            rec["rank"] = ranks.reindex(rec.index)
            # keep only selected top_keys for export brevity? user might want all. We'll keep selected only:
            rec_sel = rec.loc[rec.index.intersection(top_keys)].reset_index()
            sel_rows.append(rec_sel)

    # Build and save selections CSV
    selections_df = pd.concat(sel_rows, ignore_index=True) if sel_rows else pd.DataFrame(
        columns=[key_col, "attribute", "value", "base", "divergence", "significance", "rank"]
    )
    sel_name = f"selections_sig-{method}_a{alpha:g}_b{beta:g}_g{gamma:g}_top{top_n}{'_pos' if require_positive else ''}{save_suffix}.csv"
    sel_path = out_dir / sel_name
    selections_df.to_csv(sel_path, index=False)
    print(f"Saved selections → {sel_path} (rows={len(selections_df)})")

    # ================== Build profiles from selections ==================
    profile_rows = []
    for f in attr_files:
        df_attr = pd.read_csv(f)
        if key_col not in df_attr.columns:
            raise RuntimeError(f"key_col '{key_col}' not found in {f} while building profiles.")
        if "attribute" in df_attr.columns and df_attr["attribute"].nunique() == 1:
            attr_name = str(df_attr["attribute"].iloc[0])
        else:
            attr_name = f.stem

        row = {"attribute": attr_name}
        for vcol in value_cols:
            keys = selected_map.get((attr_name, vcol), [])
            if keys:
                val = weighted_mean_intensity(df_attr, vcol, set(keys), key_col, profile_weight_by_distance)
            else:
                # extreme edge-case: missing any candidates → global average so you don't get blanks
                val = global_mean_intensity(df_attr, vcol, profile_weight_by_distance)
            row[vcol] = val
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)[["attribute"] + value_cols]
    prof_name = f"profile_sig-{method}_a{alpha:g}_b{beta:g}_g{gamma:g}_top{top_n}{'_wdist' if profile_weight_by_distance else ''}{save_suffix}.csv"
    prof_path = out_dir / prof_name
    profile_df.to_csv(prof_path, index=False)
    print(f"Saved profile  → {prof_path} (rows={len(profile_df)}, cols={len(profile_df.columns)})")

# ============================== CLI ==============================

def main():
    ap = argparse.ArgumentParser(
        description="For each (attribute, value): compute significance per query, rank, select top-k, and build a profile."
    )
    ap.add_argument("--attribute_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--key_col", type=str, required=True,
                    help="Column that uniquely identifies the same query across attributes (e.g., 'question' or 'question_cleaned').")

    # base score
    ap.add_argument("--alpha", type=float, default=1.0, help="Exponent on (1 - distance) in base score.")
    ap.add_argument("--beta", type=float, default=0.0, help="Exponent on |intensity|/M in base score (0 disables).")
    ap.add_argument("--max_abs", type=float, default=10.0, help="Max abs intensity (M).")

    # significance
    ap.add_argument("--method", type=str, default="zscore", choices=["zscore", "margin"],
                    help="Divergence method across attributes for the same query.")
    ap.add_argument("--gamma", type=float, default=1.0, help="Exponent on divergence term for significance.")
    ap.add_argument("--require_positive", action="store_true",
                    help="If set, negative divergences are clamped to 0 (only positive direction counts).")
    ap.add_argument("--top_n", type=int, default=5, help="Top-K per (attribute,value).")

    # profiling
    ap.add_argument("--no_profile_weight_by_distance", action="store_true",
                    help="If set, do NOT weight by (1 - distance) when averaging intensities in profiles.")

    # misc
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save_suffix", type=str, default="")

    args = ap.parse_args()

    select_by_significance_and_profile(
        attribute_dir=args.attribute_dir,
        out_dir=args.out_dir,
        key_col=args.key_col,
        alpha=args.alpha,
        beta=args.beta,
        max_abs=args.max_abs,
        method=args.method,
        gamma=args.gamma,
        require_positive=args.require_positive,
        top_n=args.top_n,
        profile_weight_by_distance=not args.no_profile_weight_by_distance,
        seed=args.seed,
        save_suffix=args.save_suffix
    )

if __name__ == "__main__":
    main()
