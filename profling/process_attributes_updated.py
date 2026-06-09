#!/usr/bin/env python3
"""
Build attribute CSVs with distribution-weighted intensities.

- Load original OpinionQA JSON (with duplicates).
- Load GPT JSONL to get candidate responses ("bullets") per unique question.
- Load all theory intensity CSVs and build response -> intensities lookup.
- For each original row:
    * align options, gold_distribution, bullets
    * for each intensity col, compute expected intensity:
        sum(p_i * x_i) / sum(p_i over available candidates)
      (renormalizes over candidates that actually have intensity rows)
- Save one CSV per attribute under BASE/attribute/<ATTRIBUTE>.csv

Paths are set for the user's project layout.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# ----------------------------- Config -----------------------------
BASE = Path("")
ORIG_JSON = BASE / "steerable_test_opinionqa.json"
GPT_JSONL = Path("")

THEORY_CSVS = [
    BASE / "response_value_top3_with_intensities_mft_with_intensities_mft.csv",
    BASE / "response_value_top3_with_intensities_pvq_with_intensities_pvq.csv",
    BASE / "response_value_top3_with_intensities_duty_with_intensities_duty.csv",
    BASE / "response_value_top3_with_intensities_rights_with_intensities_rights.csv",
]

OUT_DIR = BASE / "attribute"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Utils -----------------------------
RE_BACKTICK_BLOCK = re.compile(r"```(.*?)```", re.DOTALL)
RE_BULLET = re.compile(r"^\s*-\s+(.*)\s*$")

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")

def parse_gpt_content_to_bullets(content: str) -> List[str]:
    m = RE_BACKTICK_BLOCK.search(content or "")
    block = m.group(1) if m else (content or "")
    lines = block.strip().splitlines()

    bullets = []
    for line in lines:
        m2 = RE_BULLET.match(line)
        if m2:
            bullets.append(m2.group(1).strip())

    if not bullets:
        try:
            idx = next(i for i, ln in enumerate(lines) if "response candidates:" in ln.lower())
            for ln in lines[idx + 1:]:
                m3 = RE_BULLET.match(ln)
                if m3:
                    bullets.append(m3.group(1).strip())
        except StopIteration:
            pass
    return bullets

def load_gpt_jsonl(path: Path) -> Dict[int, List[str]]:
    idx_to_bullets: Dict[int, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            custom_id = rec.get("custom_id", "")
            m = re.search(r"request-(\d+)", custom_id or "")
            if not m:
                continue
            req_idx = int(m.group(1))

            body = (((rec.get("response") or {}).get("body")) or {})
            choices = (body.get("choices") or [])
            if not choices:
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            idx_to_bullets[req_idx] = parse_gpt_content_to_bullets(content)
    return idx_to_bullets

def dedup_original_in_order(orig: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for rec in orig:
        q = (rec.get("question") or "").strip()
        if q not in seen:
            seen.add(q)
            unique.append(rec)
    return unique

def extract_question_cleaned(q_field: str) -> str:
    s = q_field or ""
    q_idx = s.lower().find("question:")
    if q_idx != -1:
        s = s[q_idx + len("question:") :]
    ans_idx = s.lower().find("\nanswer")
    if ans_idx != -1:
        s = s[:ans_idx]
    return s.strip()

# ----------------------------- Load intensity lookup -----------------------------
def load_intensity_lookup(paths: List[Path]) -> pd.DataFrame:
    """
    Returns a dataframe indexed by 'response', with all *_intensity columns merged in.
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "query" in df.columns and "response" not in df.columns:
            df = df.rename(columns={"query": "response"})
        keep = ["response"] + [c for c in df.columns if c.endswith("_intensity")]
        df = df[keep].copy()
        dfs.append(df)
    # outer-merge across theory files on 'response'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="response", how="outer")
    merged = merged.drop_duplicates(subset=["response"])
    merged = merged.set_index("response")
    return merged

# ----------------------------- Main -----------------------------
def main():
    # Load GPT bullets
    idx_to_bullets = load_gpt_jsonl(GPT_JSONL)
    num_gpt = len(idx_to_bullets)

    # Load original JSON (may have duplicates)
    orig = json.loads(ORIG_JSON.read_text(encoding="utf-8"))
    total_rows = len(orig)

    # Unique-order list for aligning indices
    unique_order = dedup_original_in_order(orig)
    num_unique = len(unique_order)

    if num_gpt != num_unique:
        print(f"[WARN] GPT responses ({num_gpt}) != unique questions ({num_unique}). "
              f"Will map where indices are available.")

    # Build question -> bullets (aligned by unique index)
    question_to_bullets: Dict[str, List[str]] = {}
    for i, urec in enumerate(unique_order):
        q_text = (urec.get("question") or "").strip()
        question_to_bullets[q_text] = idx_to_bullets.get(i, [])

    # Load response -> intensities lookup
    df_lookup = load_intensity_lookup(THEORY_CSVS)
    all_intensity_cols = [c for c in df_lookup.columns if c.endswith("_intensity")]
    all_intensity_cols.sort()

    # Accumulate rows per attribute
    attr_to_rows: Dict[str, List[dict]] = {}

    for rec in orig:
        q_full = (rec.get("question") or "").strip()
        attribute = rec.get("attribute", "")
        options = (rec.get("options") or [])  # list of option texts
        probs = (rec.get("gold_distribution") or [])  # list of floats
        bullets = question_to_bullets.get(q_full, [])
        q_clean = extract_question_cleaned(q_full)

        # Align lengths conservatively
        n = min(len(options), len(bullets), len(probs)) if options else min(len(bullets), len(probs))
        if n == 0:
            # No candidates to compute expectation
            row = {
                "question": q_full,
                "question_cleaned": q_clean,
                "attribute": attribute,
            }
            for col in all_intensity_cols:
                row[col] = np.nan
            attr_to_rows.setdefault(attribute, []).append(row)
            continue

        # For each candidate, fetch its intensity vector if available
        candidate_responses = bullets[:n]
        candidate_probs = np.array(probs[:n], dtype=float)

        # Keep only candidates that exist in the lookup
        mask_available = [resp in df_lookup.index for resp in candidate_responses]
        if not any(mask_available):
            # No available intensities
            row = {
                "question": q_full,
                "question_cleaned": q_clean,
                "attribute": attribute,
            }
            for col in all_intensity_cols:
                row[col] = np.nan
            attr_to_rows.setdefault(attribute, []).append(row)
            continue

        # Renormalize probabilities over available candidates
        avail_indices = [i for i, ok in enumerate(mask_available) if ok]
        p_avail = candidate_probs[avail_indices]
        p_sum = float(p_avail.sum())
        if p_sum <= 0:
            # Degenerate distribution; treat as uniform over available
            p_avail = np.ones_like(p_avail) / len(p_avail)
        else:
            p_avail = p_avail / p_sum

        # Stack available intensity rows
        df_avail = df_lookup.loc[[candidate_responses[i] for i in avail_indices], all_intensity_cols]

        # Expected intensity per column
        expected = {}
        for col in all_intensity_cols:
            col_vals = df_avail[col].to_numpy(dtype=float)
            # Some intensities may be NaN for some candidates; mask them and renormalize p
            valid_mask = ~np.isnan(col_vals)
            if not valid_mask.any():
                expected[col] = np.nan
                continue
            p_eff = p_avail[valid_mask]
            v_eff = col_vals[valid_mask]
            s = p_eff.sum()
            if s <= 0:
                expected[col] = np.nan
            else:
                p_eff = p_eff / s
                expected[col] = float(np.dot(p_eff, v_eff))

        # Compose row
        row = {
            "question": q_full,
            "question_cleaned": q_clean,
            "attribute": attribute,
        }
        row.update(expected)
        attr_to_rows.setdefault(attribute, []).append(row)

    # Write per-attribute CSVs
    base_cols = ["question", "question_cleaned", "attribute"]
    out_cols = base_cols + all_intensity_cols

    for attr, rows in attr_to_rows.items():
        df_attr = pd.DataFrame(rows)
        # ensure consistent col order
        for c in out_cols:
            if c not in df_attr.columns:
                df_attr[c] = np.nan
        df_attr = df_attr[out_cols]
        fname = sanitize_filename(attr if attr else "UNKNOWN") + ".csv"
        out_path = OUT_DIR / fname
        df_attr.to_csv(out_path, index=False)
        print(f"Saved → {out_path} (rows={len(df_attr)})")

    print(f"[DONE] total_input={total_rows}, attributes={len(attr_to_rows)}, "
          f"intensity_cols={len(all_intensity_cols)}")

if __name__ == "__main__":
    main()
