#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------- LLM interface ---------------------------- #
try:
    from models import LLMModel
except Exception:
    LLMModel = None

# --------------------------- Prompts ---------------------------------- #
PROMPT_DEFAULT = """You are given a value from {theory_name} and {num_texts} short texts.

**{label_name}:** {value}
**Definition of {label_name_lower}:** {definition}

Below are the texts:
{enumerated_texts}

Your task is to **rank these texts** based on how strongly each one reflects (i.e., supports or opposes) the given value.

- Rank **1** to the text that **most strongly supports** the value.
- Rank **{num_texts}** to the text that **most strongly opposes** the value.
- Rank texts that are **neutral or unrelated** somewhere in the **middle**.

Return a list of integers indicating the **ranking order** of the texts from most supportive to most opposing.

Format your answer exactly as follows:
A list like `[3, 1, 4, 2, 5]` (no explanation).
""".strip()

PROMPT_ONESHOT = """You are given a value from Schwartz’s theory of basic human values and {num_texts} short texts.

---

### Example

**Value Label:** Benevolence
**Definition:** Preserving and enhancing the welfare of those with whom one is in frequent personal contact (e.g., friends, family).

[1] I trained every day to win the regional sports competition.
[2] I sincerely helped my friend move into his new apartment even though I was exhausted.
[3] I tricked my friend into lending me money and never paid him back.
[4] I listened patiently while my classmate vented about her problems.
[5] I finally bought a new TV for myself after saving for months.

**Answer:** [2, 4, 5, 1, 3]

---

Now your task:

**Value Label:** {value}
**Definition:** {definition}

{enumerated_texts}

Rank the texts from **most supportive** (1) to **most opposing** ({num_texts}) based on how strongly each reflects the given value.
Neutral or unrelated texts should be placed in the middle.

**Format your answer exactly as follows:**
A list like `[3, 1, 4, 2, 5]` (no extra text).
""".strip()

def build_prompt(theory: str, target_value: str, definition: str, texts: List[str], fmt: str) -> str:
    tl = theory.lower()
    if tl == "pvq":
        label_name = "Value Label"; theory_name = "Schwartz’s theory of basic human values"
    elif tl == "mft":
        label_name = "Foundation Label"; theory_name = "Moral Foundations Theory"
    elif tl == "duty":
        label_name = "Duty Label"; theory_name = "Ross's Prima Facie Duty Theory"
    elif tl in ("right", "rights"):
        label_name = "Right Label"; theory_name = "Vasak's Three Generations of Rights"
        theory = "right"
    else:
        label_name = "Value Label"; theory_name = theory

    enumerated_texts = "\n".join(f"[{i+1}] {t.strip()}" for i, t in enumerate(texts))
    tpl = PROMPT_DEFAULT if fmt == "default" else PROMPT_ONESHOT
    return tpl.format(
        theory_name=theory_name,
        num_texts=len(texts),
        label_name=label_name,
        label_name_lower=label_name.lower(),
        value=target_value,
        definition=definition,
        enumerated_texts=enumerated_texts,
    )

# --------------------------- Parsing model outputs -------------------- #
def parse_rank_list(resp: str, k_expected: int) -> List[int] | None:
    m = re.search(r"\[([\d,\s]+)\]", resp)
    if not m:
        return None
    try:
        ranks = [int(x.strip()) for x in m.group(1).split(",")]
    except Exception:
        return None
    if sorted(ranks) != list(range(1, k_expected + 1)):
        return None
    return ranks

# --------------------------- Definition loading ----------------------- #
def load_definition_json(path: str, target_value: str, theory: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    defs = data.get("definitions", {})

    if target_value in defs:
        return defs[target_value]

    # tolerant matches (lower, hyphen/underscore/space)
    tvs = {
        target_value,
        target_value.lower(),
        target_value.replace("-", "_"),
        target_value.replace("_", "-"),
        target_value.replace(" ", "_"),
        target_value.replace(" ", "-"),
        target_value.lower().replace("-", "_"),
        target_value.lower().replace("_", "-"),
        target_value.lower().replace(" ", "_"),
        target_value.lower().replace(" ", "-"),
    }
    for k, v in defs.items():
        if k in tvs or k.lower() in tvs:
            return v

    raise ValueError(f"Definition for '{target_value}' not found in {path}")

# --------------------------- PL Estimation ---------------------------- #
def estimate_scores(resp_df: pd.DataFrame,
                    per_row_orders: Dict[int, List[List[int]]],
                    per_row_local_utils: Dict[int, List[Dict[int, float]]],
                    I_MIN=-10.0, I_MAX=10.0,
                    iters=150, lr=0.1, lambda_prior=1e-3) -> pd.DataFrame:
    """
    Bounded, window-consistent PL with fixed anchors (utilities == known intensities).
    Estimates only the response intensity per row (mapped via isotonic calibration).
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    def exp_clip(x: float) -> float:
        return float(np.exp(np.clip(x, -20.0, 20.0)))

    # Gather anchor pairs for global calibration
    anchor_pairs = []
    for _, utils_list in per_row_local_utils.items():
        for lu in utils_list:
            for a_id, I in lu.items():
                if a_id == 0 or I is None or (isinstance(I, float) and np.isnan(I)):
                    continue
                I = float(np.clip(I, I_MIN, I_MAX))
                anchor_pairs.append((I, I))  # anchor utility == intensity

    if not anchor_pairs:
        raise ValueError("No anchor intensities found for calibration.")

    anchors_u, anchors_I = zip(*{(ua, Ia) for (ua, Ia) in anchor_pairs})
    anchors_u = np.array(anchors_u, dtype=float)
    anchors_I = np.array(anchors_I, dtype=float)

    iso = IsotonicRegression(y_min=I_MIN, y_max=I_MAX, out_of_bounds="clip")
    iso.fit(anchors_u, anchors_I)

    grad_tol = 1e-6
    bisection_max_iter = 200
    eps = 1e-6

    def grad_ll(u_resp: float, orders, utils_list) -> float:
        eu = exp_clip(u_resp)
        g = 0.0
        for order_local, local_utils in zip(orders, utils_list):
            exp_fixed = {}
            for i, I in local_utils.items():
                if i == 0 or I is None or (isinstance(I, float) and np.isnan(I)):
                    continue
                exp_fixed[i] = exp_clip(float(np.clip(I, I_MIN, I_MAX)))
            for j in range(len(order_local)):
                R = order_local[j:]
                denom = 0.0
                resp_in_R = False
                for idx in R:
                    if idx == 0:
                        denom += eu; resp_in_R = True
                    else:
                        denom += exp_fixed.get(idx, 0.0)
                if order_local[j] == 0:
                    g += 1.0
                if resp_in_R and denom > 0.0:
                    g -= eu / denom
        if lambda_prior > 0:
            g -= lambda_prior * u_resp
        return g

    def solve_bisection(orders, utils_list, lo=-20.0, hi=20.0):
        g_lo = grad_ll(lo, orders, utils_list)
        g_hi = grad_ll(hi, orders, utils_list)
        if g_lo == 0.0:
            return lo
        if g_hi == 0.0:
            return hi
        if g_lo * g_hi > 0:
            return None
        for _ in range(bisection_max_iter):
            mid = 0.5 * (lo + hi)
            g_mid = grad_ll(mid, orders, utils_list)
            if abs(g_mid) < grad_tol or (hi - lo) < 1e-6:
                return mid
            if g_lo * g_mid > 0:
                lo, g_lo = mid, g_mid
            else:
                hi, g_hi = mid, g_mid
        return 0.5 * (lo + hi)

    out_rows = []
    for ridx, _ in resp_df.iterrows():
        orders = per_row_orders.get(ridx, [])
        utils_list = per_row_local_utils.get(ridx, [])

        if not orders:
            s_hat = np.nan
        else:
            u_star = solve_bisection(orders, utils_list)
            if u_star is None:
                u = 0.0
                for _ in range(iters):
                    prev = u
                    g = grad_ll(u, orders, utils_list)
                    u = float(np.clip(u + lr * g, -20.0, 20.0))
                    if abs(u - prev) < 1e-6:
                        break
                u_star = float(u)
            u_hat = float(np.clip(u_star, -20.0, 20.0))
            s_hat = float(iso.predict([u_hat])[0])

            # Window-level guardrails
            row_anchor_Is = []
            wins_total, losses_total = 0, 0
            for order_local, local_utils in zip(orders, utils_list):
                for i, I in local_utils.items():
                    if i == 0 or I is None or (isinstance(I, float) and np.isnan(I)):
                        continue
                    row_anchor_Is.append(float(np.clip(I, I_MIN, I_MAX)))
                pos = {idx: p for p, idx in enumerate(order_local)}
                if 0 in pos:
                    p_new = pos[0]
                    for idx, p_idx in pos.items():
                        if idx == 0:
                            continue
                        if p_new < p_idx:
                            wins_total += 1
                        elif p_new > p_idx:
                            losses_total += 1

            if row_anchor_Is:
                w_min = float(np.min(row_anchor_Is))
                w_max = float(np.max(row_anchor_Is))
                if losses_total > 0 and wins_total == 0:
                    s_hat = float(np.nextafter(w_min, -np.inf))
                elif wins_total > 0 and losses_total == 0:
                    s_hat = float(np.nextafter(w_max, np.inf))
                else:
                    s_hat = max(w_min, min(w_max, s_hat))
                    if abs(s_hat - w_min) < 1e-12:
                        s_hat = min(w_max - eps, w_min + eps)
                    elif abs(s_hat - w_max) < 1e-12:
                        s_hat = max(w_min + eps, w_max - eps)

            s_hat = float(np.clip(s_hat, I_MIN, I_MAX))

        out_rows.append({"row_idx": ridx, "intensity": s_hat})

    return pd.DataFrame(out_rows)

# --------------------------- Helpers ---------------------------------- #
THEORY_COL_MAP = {
    "pvq":   [("pvq_value_1","pvq_distance_1"), ("pvq_value_2","pvq_distance_2"), ("pvq_value_3","pvq_distance_3")],
    "mft":   [("mft_value_1","mft_distance_1"), ("mft_value_2","mft_distance_2"), ("mft_value_3","mft_distance_3")],
    "duty":  [("duty_value_1","duty_distance_1"), ("duty_value_2","duty_distance_2"), ("duty_value_3","duty_distance_3")],
    "rights":[("rights_value_1","rights_distance_1"), ("rights_value_2","rights_distance_2"), ("rights_value_3","rights_distance_3")],
}

def _value_fname_candidates(value_name: str) -> list[str]:
    """Generate robust filename stems for a value label."""
    bases = set()
    v0 = value_name.strip(); bases.add(v0)
    v1 = v0.replace(" ", "_"); bases.add(v1)
    bases.add(v1.replace("-", "_")); bases.add(v1.replace("_", "-"))
    bases.add(v0.replace("-", "_")); bases.add(v0.replace("_", "-"))
    lowers = {b.lower() for b in bases}; bases |= lowers
    seen, ordered = set(), []
    for b in bases:
        if b not in seen:
            seen.add(b); ordered.append(b)
    return ordered

def load_value_db(theory: str, value_name: str) -> pd.DataFrame:
    """
    Load DB at data/final_ratings/{value}_{theory}_flag_aggregated.csv
    (must have columns: text, final_rating). Tries multiple filename variants.
    """
    theory_norm = "right" if theory in ("right", "rights") else theory
    tried = []
    for stem in _value_fname_candidates(value_name):
        fname = f"{stem}_{theory_norm}_flag_aggregated.csv"
        path = os.path.join("data", "final_ratings", fname)
        tried.append(path)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "text" not in df.columns:
                raise ValueError(f"DB missing 'text' column: {path}")
            if "final_rating" not in df.columns:
                raise ValueError(f"DB missing 'final_rating' column: {path}")
            return df.dropna(subset=["text", "final_rating"]).reset_index(drop=True)
    raise FileNotFoundError(f"DB not found for {theory}:{value_name}. Tried:\n- " + "\n- ".join(tried))

def prepare_buckets(db_df: pd.DataFrame, k: int) -> Dict[int, List[int]]:
    buckets = defaultdict(list)
    min_score, max_score = float(db_df["final_rating"].min()), float(db_df["final_rating"].max())
    bin_edges = np.linspace(min_score, max_score, k)
    b = pd.cut(db_df["final_rating"], bins=bin_edges, labels=False, include_lowest=True)
    b = b.fillna(k - 2).astype(int)
    for idx, lab in enumerate(b):
        buckets[int(lab)].append(idx)
    return buckets

def sample_opponents(n_samples: int, db_texts: List[str], buckets: Dict[int, List[int]]) -> List[int]:
    sampled_ids = []
    for i in range(n_samples):
        if i in buckets and buckets[i]:
            sampled_ids.append(np.random.choice(buckets[i]))
        else:
            sampled_ids.append(np.random.randint(0, len(db_texts)))
    return sampled_ids

# --------------------------- Main Runner ------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Evaluate intensities per theory for top3-per-theory CSV.")
    parser.add_argument("--input_csv",
                        default="",
                        help="CSV with query/response and *_value_i, *_distance_i columns.")
    parser.add_argument("--out_dir",
                        default="",
                        help="Directory to save the per-theory output CSV.")
    parser.add_argument("--theory", required=True, choices=["pvq", "mft", "duty", "rights"],
                        help="Which theory to process (one at a time).")
    # Defaults mirroring your .sh
    parser.add_argument("--eval_llm", default="google/gemma-3-27b-it")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--iter", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--prompt_format", choices=["default","oneshot"], default="default")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Prepare LLM
    if LLMModel is None:
        raise RuntimeError("models.LLMModel not found. Ensure models.py is available in PYTHONPATH.")
    model = LLMModel(model=args.eval_llm, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    # Load input
    in_df = pd.read_csv(args.input_csv)
    text_col = "response" if "response" in in_df.columns else "query"
    if text_col not in in_df.columns:
        raise ValueError("Input CSV must contain 'query' or 'response' column for evaluation.")

    # Work on a copy; only add columns for this theory
    out_df = in_df.copy()
    theory = args.theory
    print(f"\n[+] Processing theory: {theory}")

    # Definitions path for this theory
    def_path = os.path.join("data", f"{'right' if theory=='rights' else theory}.json")

    # Cache per (value) to avoid re-reading DB/defs
    db_cache: Dict[str, Dict] = {}

    # Which columns to use for this theory
    value_cols = THEORY_COL_MAP[theory]

    # Preload DBs/defs for all values that appear in the file (for this theory)
    for (val_col, dist_col) in value_cols:
        if val_col not in out_df.columns:
            continue
        present_values = out_df[val_col].dropna().astype(str).str.strip().unique()
        for value_name in present_values:
            if not value_name or value_name.lower() == "nan":
                continue
            if value_name not in db_cache:
                db_df = load_value_db("right" if theory == "rights" else theory, value_name)
                db_texts = db_df["text"].astype(str).tolist()
                db_scores = db_df["final_rating"].astype(float).tolist()
                if len(db_texts) == 0:
                    raise ValueError(f"Empty DB for {theory}:{value_name}")
                buckets = prepare_buckets(db_df, args.k)
                definition = load_definition_json(def_path, value_name, "right" if theory=="rights" else theory)
                db_cache[value_name] = {
                    "texts": db_texts,
                    "scores": db_scores,
                    "buckets": buckets,
                    "definition": definition,
                }

    # Evaluate per value-slot (1..3)
    for slot_idx, (val_col, dist_col) in enumerate(value_cols, start=1):
        print(f"    [-] Evaluating slot {slot_idx}: ({val_col}, {dist_col})")
        if val_col not in out_df.columns:
            print(f"        [!] Missing {val_col}; skipping.")
            continue

        mask = out_df[val_col].notna() & (out_df[val_col].astype(str).str.strip() != "")
        if not mask.any():
            print("        [!] No rows with this slot populated; skipping.")
            continue

        rows = out_df[mask].copy().reset_index().rename(columns={"index": "orig_idx"})
        resp_df = pd.DataFrame({
            "query": rows[text_col].astype(str),
            "dataset": "",   # not used downstream
            "response": rows[text_col].astype(str),
        })

        per_row_orders: Dict[int, List[List[int]]] = defaultdict(list)
        per_row_local_utils: Dict[int, List[Dict[int, float]]] = defaultdict(list)

        # Build prompts
        all_prompts, meta = [], []
        for ridx, r in rows.iterrows():
            text = str(r[text_col]).strip()
            target_value = str(r[val_col]).strip()
            cache = db_cache[target_value]
            db_texts = cache["texts"]
            db_scores = cache["scores"]
            buckets   = cache["buckets"]
            definition = cache["definition"]

            for _ in range(args.m):
                sampled_ids = sample_opponents(args.k - 1, db_texts, buckets)
                sampled_texts = [db_texts[i] for i in sampled_ids]
                texts = [text] + sampled_texts
                prompt = build_prompt(theory, target_value, definition, texts, args.prompt_format)
                all_prompts.append(prompt)
                meta.append({"local_ridx": ridx, "db_ids": sampled_ids, "target_value": target_value})

        # Query LLM
        print(f"        [*] Running LLM on {len(all_prompts)} prompts (k={args.k}, m={args.m})…")
        outputs = model(all_prompts)

        # Parse rankings
        for out, mrec in zip(outputs, meta):
            local_ridx = mrec["local_ridx"]
            db_ids = mrec["db_ids"]
            k = 1 + len(db_ids)
            ranks = parse_rank_list(out, k_expected=k)
            if ranks is None:
                continue
            order_local = [rank - 1 for rank in ranks]
            local_utils = {0: 0.0}
            cache = db_cache[mrec["target_value"]]
            for j, db_id in enumerate(db_ids):
                local_utils[j + 1] = float(cache["scores"][db_id])
            per_row_orders[local_ridx].append(order_local)
            per_row_local_utils[local_ridx].append(local_utils)

        # Estimate intensities for these rows
        est_df = estimate_scores(resp_df, per_row_orders, per_row_local_utils,
                                 I_MIN=-10.0, I_MAX=10.0,
                                 iters=max(args.iter, 50), lr=args.lr, lambda_prior=1e-3)

        # Write back to out_df under column: {theory}_{value}_intensity
        for _, r in rows.iterrows():
            local_ridx = int(r.name)
            orig_idx = int(r["orig_idx"])
            t_value = str(r[val_col]).strip()
            col_name = f"{theory}_{t_value}_intensity".replace(" ", "_").replace("-", "_")
            val = est_df.loc[est_df["row_idx"] == local_ridx, "intensity"]
            out_df.loc[orig_idx, col_name] = float(val.iloc[0]) if len(val) else np.nan

    # Save per-theory file
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"response_value_top3_with_intensities_{theory}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Saved intensities for {theory} → {out_path}")


if __name__ == "__main__":
    main()
