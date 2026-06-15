import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import re
import json
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm

from models import LLMModel


OPINIONQA_ATTR_CATEGORY = {
    "POLPARTY": "political party",
    "POLIDEOLOGY": "political ideology",
    "RELIG": "religion",
    "RACE": "race",
    "EDUCATION": "education",
    "INCOME": "income",
    "CREGION": "region in the United States",
    "SEX": "sex",
}

GROUP_PREFIXES = {
    "Reg": "CREGION_",
    "Edu": "EDUCATION_",
    "Inc": "INCOME_",
    "Ideo": "POLIDEOLOGY_",
    "Par": "POLPARTY_",
    "Race": "RACE_",
    "Relig": "RELIG_",
    "Sex": "SEX_",
}


def prefix_with_attribute(attribute):
    parts = (attribute or "").split("_", 1)
    if len(parts) != 2:
        return ""
    cat, label = parts
    cat_name = OPINIONQA_ATTR_CATEGORY.get(cat, cat.title().lower())
    return f"In terms of {cat_name}, you are {label}. "


def build_prompt_default(question, attribute):
    return f"{prefix_with_attribute(attribute)}Please respond to the following question. Return only the single letter. {question.strip()}"


def build_prompt_value_profile(question, attribute, bullet_lines):
    prefix = prefix_with_attribute(attribute)
    bullets = "\n".join([f"- {l.strip()}" for l in bullet_lines]) if bullet_lines else ""
    return (
        f"{prefix}Here are value profiles:\n"
        f"{bullets}\n"
        f"Please respond to the following question considering the profile. Return only the single letter. "
        f"{question.strip()}"
    )


def load_opinionqa_items(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen = set()
    out = []
    for item in data:
        q = (item.get("question") or "").strip()
        attr = (item.get("attribute") or "").strip()
        if not q or not attr:
            continue
        key = (attr, q)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_profile_map(profile_dir, attributes):
    profile_dir = Path(profile_dir)
    out = {}
    for attr in attributes:
        path = profile_dir / f"{attr}.json"
        with open(path, "r", encoding="utf-8") as f:
            out[attr] = json.load(f)
    return out


def select_value_profile_lines(profile, theory, num_prompts):
    if num_prompts <= 0:
        return []
    th = profile.get("theories", {}).get(theory, {})
    pool = []
    for bucket in ("very_high", "high", "medium", "low", "very_low"):
        for it in th.get(bucket, []):
            score = it.get("score")
            text = it.get("definition_prompt") or it.get("definition_raw")
            if text is None or score is None:
                continue
            key = (theory, it.get("value_key") or it.get("column") or text)
            pool.append((int(score), text.strip(), key))
    dedup = {}
    for sc, txt, key in pool:
        if key not in dedup:
            dedup[key] = (sc, txt)
    pool = [(sc, txt) for sc, txt in dedup.values()]
    if not pool:
        return []
    pool_sorted = sorted(pool, key=lambda x: x[0])
    n = min(num_prompts, len(pool_sorted))
    k_high = (n + 1) // 2
    k_low = n - k_high
    highs = list(reversed(pool_sorted))[:k_high]
    lows = pool_sorted[:k_low]
    return [txt for _, txt in highs + lows]


def normalize_q(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def load_relevance_map(relevance_csv):
    df = pd.read_csv(relevance_csv)
    value_cols = [
        "benevolence", "universalism", "self-direction", "stimulation",
        "hedonism", "achievement", "power", "conformity",
        "tradition", "security", "face", "humility",
    ]
    out = {}
    for _, row in df.iterrows():
        q = normalize_q(row["question"])
        top_score = None
        if "top_score" in row.index and pd.notna(row["top_score"]):
            try:
                top_score = float(row["top_score"])
            except Exception:
                pass
        out[q] = top_score
    return out


def get_majority_index(gold_distribution):
    if not gold_distribution:
        return None
    return max(range(len(gold_distribution)), key=lambda i: gold_distribution[i])


def parse_prediction(raw_response, options):
    if not raw_response:
        return None
    text = raw_response.strip()
    valid_letters = [chr(ord("A") + i) for i in range(len(options))]
    upper = text.upper().strip()
    if upper in valid_letters:
        return valid_letters.index(upper)
    patterns = [
        r"^\s*([A-Z])\s*$",
        r"^\s*([A-Z])[\.\)\:\- ]",
        r"^\s*Answer\s*[:\-]?\s*([A-Z])\b",
        r"\b([A-Z])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_letters:
                return valid_letters.index(letter)
    text_norm = re.sub(r"\s+", " ", text).lower()
    for i, opt in enumerate(options):
        opt_norm = re.sub(r"\s+", " ", opt).lower()
        if opt_norm and opt_norm in text_norm:
            return i
    return None


def run_pass(model_obj, items, prompt_format, profile_lines_by_attr, relevance_map, threshold):
    prompts = []
    kept = []
    adaptive_flags = []

    for item in items:
        attr = (item.get("attribute") or "").strip()
        q = (item.get("question") or "").strip()
        options = item.get("options") or []
        gold = item.get("gold_distribution")
        if not q or not attr or not options or gold is None:
            continue

        if prompt_format == "opinionqa_default":
            prompt = build_prompt_default(q, attr)
            adaptive_flags.append(False)
        else:
            score = relevance_map.get(normalize_q(q))
            use_profile = (score is not None) and (score >= threshold)
            if use_profile:
                prompt = build_prompt_value_profile(q, attr, profile_lines_by_attr[attr])
            else:
                prompt = build_prompt_default(q, attr)
            adaptive_flags.append(use_profile)

        prompts.append(prompt)
        kept.append(item)

    print(f"[INFO] Running {len(prompts)} prompts ({prompt_format}) ...")
    responses = model_obj(prompts)

    records = []
    per_attr = defaultdict(lambda: {"n": 0, "match": 0})

    for item, flag, raw in zip(kept, adaptive_flags, responses):
        attr = (item.get("attribute") or "").strip()
        options = item.get("options") or []
        gold = item.get("gold_distribution") or []
        maj_idx = get_majority_index(gold)
        pred_idx = parse_prediction(raw, options)
        is_match = (pred_idx == maj_idx) if (pred_idx is not None and maj_idx is not None) else False

        per_attr[attr]["n"] += 1
        if is_match:
            per_attr[attr]["match"] += 1

        records.append({
            "attribute": attr,
            "question": (item.get("question") or "").strip(),
            "majority_idx": maj_idx,
            "pred_idx": pred_idx,
            "majority_match": is_match,
            "adaptive_applied": flag,
            "raw_response": raw,
        })

    return records, per_attr


def compute_summary(per_attr):
    group_match = defaultdict(int)
    group_n = defaultdict(int)
    total_match = 0
    total_n = 0

    for attr, st in per_attr.items():
        total_match += st["match"]
        total_n += st["n"]
        for g, prefix in GROUP_PREFIXES.items():
            if attr.startswith(prefix):
                group_match[g] += st["match"]
                group_n[g] += st["n"]

    micro_avg = total_match / total_n if total_n > 0 else 0.0
    group_accs = {}
    for g in GROUP_PREFIXES:
        group_accs[g] = group_match[g] / group_n[g] if group_n[g] > 0 else 0.0

    return micro_avg, group_accs


def save_results(records, per_attr, out_dir, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"{tag}_details.csv", index=False, encoding="utf-8-sig")

    attr_rows = []
    for attr, st in sorted(per_attr.items()):
        attr_rows.append({
            "attribute": attr,
            "n_questions": st["n"],
            "majority_match_count": st["match"],
            "majority_match_rate_all": st["match"] / st["n"] if st["n"] > 0 else 0.0,
        })
    df_attr = pd.DataFrame(attr_rows)
    df_attr.to_csv(out_dir / f"{tag}_summary_by_attribute.csv", index=False, encoding="utf-8-sig")

    micro_avg, group_accs = compute_summary(per_attr)
    summary_row = {"tag": tag, "micro_avg_acc": micro_avg}
    for g, acc in group_accs.items():
        summary_row[f"{g}_acc"] = acc
    pd.DataFrame([summary_row]).to_csv(out_dir / f"{tag}_summary_overall.csv", index=False, encoding="utf-8-sig")

    return micro_avg, group_accs


def print_table_row(label, micro_avg, group_accs):
    groups = ["Reg", "Edu", "Inc", "Ideo", "Par", "Race", "Relig", "Sex"]
    vals = "|".join(f"{group_accs[g]*100:.1f}" for g in groups)
    print(f"  {label:30s}  {vals}  Avg={micro_avg*100:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    _root = Path(__file__).resolve().parent.parent  # VALUEFLOW/
    parser.add_argument("--opinionqa_json", type=str,
                        default=str(_root / "data" / "steerable_test_opinionqa.json"))
    parser.add_argument("--profile_dir", type=str,
                        default=str(_root / "profling" / "prompts"))
    parser.add_argument("--relevance_csv", type=str,
                        default=str(_root / "data" / "opinionqa_value_duty_relevance_unique_questions.csv"))
    parser.add_argument("--output_root", type=str,
                        default=str(_root / "outputs" / "opinionqa_final"))
    parser.add_argument("--theory", type=str, required=True, choices=["pvq", "duty", "mft", "rights"])
    parser.add_argument("--num_prompts", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)

    model_tag = args.model.replace("/", "__")
    thr_tag = str(args.threshold).replace(".", "p")
    out_dir = Path(args.output_root) / f"model={model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model_obj = LLMModel(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    items = load_opinionqa_items(args.opinionqa_json)
    attributes = sorted({(it.get("attribute") or "").strip() for it in items if (it.get("attribute") or "").strip()})
    print(f"[INFO] {len(items)} items, {len(attributes)} attributes")

    profile_map = load_profile_map(args.profile_dir, attributes)
    profile_lines_by_attr = {
        attr: select_value_profile_lines(profile_map[attr], args.theory, args.num_prompts)
        for attr in attributes
    }

    relevance_map = load_relevance_map(args.relevance_csv)

    # --- default pass ---
    default_tag = f"opinionqa_default_temp{args.temperature}_seed{args.seed}"
    records_def, per_attr_def = run_pass(
        model_obj, items,
        prompt_format="opinionqa_default",
        profile_lines_by_attr=None,
        relevance_map=None,
        threshold=None,
    )
    micro_def, groups_def = save_results(records_def, per_attr_def, out_dir, default_tag)

    # --- adaptive profile pass ---
    profile_tag = (
        f"adaptive_value_profile_{args.theory}_n{args.num_prompts}"
        f"_thr{thr_tag}_temp{args.temperature}_seed{args.seed}"
    )
    records_prof, per_attr_prof = run_pass(
        model_obj, items,
        prompt_format="adaptive_value_profile",
        profile_lines_by_attr=profile_lines_by_attr,
        relevance_map=relevance_map,
        threshold=args.threshold,
    )
    micro_prof, groups_prof = save_results(records_prof, per_attr_prof, out_dir, profile_tag)

    print("\n[RESULTS]")
    print(f"  Model: {args.model}")
    print(f"  {'':30s}  {'Reg|Edu|Inc|Ideo|Par|Race|Relig|Sex':50s}  Avg")
    print_table_row("Default", micro_def, groups_def)
    print_table_row(f"Profile ({args.theory},n={args.num_prompts},thr={args.threshold})", micro_prof, groups_prof)
    print(f"\n[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
