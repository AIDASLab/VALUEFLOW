import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import argparse
import json
import random
from tqdm import tqdm
from models import LLMModel
from collections import defaultdict
from pathlib import Path

# ---------------------- Helpers ---------------------- #
INTENSITY_MAP = {
    2: "strongly values",
    1: "slightly values",
    -1: "slightly rejects",
    -2: "strongly rejects",
    0: None,
}

OPINIONQA_ATTR_CATEGORY = {  # minimal map; extend as needed
    "POLPARTY": "political party",
    "POLIDEOLOGY": "political ideology",
    "RELIG": "religion",
    "RACE": "race",
    "EDUCATION": "education",
    "INCOME": "income",
    "CREGION": "region in the United States",
    "SEX": "Sex"
}

def norm_list_csv_or_list(x):
    if x is None:
        return None
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, str):
                out.extend([p.strip() for p in item.split(",") if p.strip() != ""])
            else:
                out.append(str(item))
        return out
    if isinstance(x, str):
        return [p.strip() for p in x.split(",") if p.strip() != ""]
    return [str(x)]

def choose_k(items, k):
    if not items:
        return []
    if k >= len(items):
        return list(items)
    return random.sample(items, k)

def rephrase_definition(definition: str, intensity: int) -> str:
    d = (definition or "").strip()
    if intensity == 0 or INTENSITY_MAP.get(intensity) is None:
        return d
    prefixes = [
        "You are a person that values",
        "You are a person who values",
        "You are a person that value",
        "You are a person who value",
        "You are a person that",
        "You are a person who",
    ]
    for pref in prefixes:
        if d.lower().startswith(pref.lower()):
            rest = d[len(pref):].lstrip(" :,-")
            return f"You are a person who {INTENSITY_MAP[intensity]} {rest}"
    return f"You are a person who {INTENSITY_MAP[intensity]} {d}"

def canonicalize_value_key(key: str) -> str:
    return key.lower().replace("/", "-").replace("_", "-").strip()

# ---------------------- Prompt Builders ---------------------- #
def build_prompt_default(query: str) -> str:
    return (
        "You are an AI assistant that responds to every question in a particular format.\n\n"
        "Your answer is not allowed to start with 'As an AI language model ...' or with 'I cannot ...'.\n\n"
        f"{query.strip()}\n\nAnswer with no more than 100 words."
    )

def build_prompt_single_definition(query: str, definition: str, value_name: str, intensity: int) -> str:
    if definition is None or value_name is None or intensity is None:
        raise ValueError("Definition, value_name, and intensity must be provided for 'single_definition' prompt format.")
    definition_adj = rephrase_definition(definition, intensity)
    return (
        f"{definition_adj.strip()}\n\n"
        f"{query.strip()}\n\nAnswer with no more than 100 words."
    )

def build_prompt_user_texts(query: str, user_texts: list) -> str:
    if not user_texts or len(user_texts) < 2:
        raise ValueError("For 'user_texts' format, need >=2 user_texts.")
    user_text_str = "\n".join([f'- "{t.strip()}"' for t in user_texts])
    return (
        "Below are texts that the user has spoken, written, or agreed with:\n"
        f"{user_text_str}\n\n"
        "Based on these texts, infer the user’s value orientation and its intensity, "
        "then respond to the query accordingly.\n\n"
        f"Query: {query.strip()}\n\nAnswer with no more than 100 words."
    )

def build_prompt_multiple_definition(query: str, defs_per_value: dict) -> str:
    lines = []
    for vname, def_list in defs_per_value.items():
        for d_text, inten in def_list:
            d_adj = rephrase_definition(d_text, inten)
            cleaned = d_adj
            prefix = "You are a person who "
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
            lines.append(f"- {cleaned}")
    bullet_block = "\n".join(lines) if lines else "- (no values provided)"
    return (
        "You are a person that has the following value profile:\n"
        f"{bullet_block}\n"
        "Answer based on this profile.\n\n"
        f"Query: {query.strip()}\n\nAnswer with no more than 100 words."
    )

def build_prompt_multiple_fewshot(query: str, texts: list) -> str:
    if not texts or len(texts) < 2:
        raise ValueError("For 'multiple_fewshot', need >=2 user texts total.")
    user_text_str = "\n".join([f'- "{t.strip()}"' for t in texts])
    return (
        "Below are texts that the user has spoken, written, or agreed with:\n"
        f"{user_text_str}\n\n"
        "Based on these texts, infer the user’s value orientation(s) and intensity(ies), "
        "then respond to the query accordingly.\n\n"
        f"Query: {query.strip()}\n\nAnswer with no more than 100 words."
    )

def prefix_with_attribute(attribute: str) -> str:
    # attribute like "POLIDEOLOGY_Conservative"
    parts = (attribute or "").split("_", 1)
    if len(parts) != 2:
        return ""
    cat, label = parts
    cat_name = OPINIONQA_ATTR_CATEGORY.get(cat, cat.title().lower())
    return f"In terms of {cat_name}, you are {label}. "

def build_prompt_opinionqa_default(question: str, attribute: str) -> str:
    prefix = prefix_with_attribute(attribute)
    return f"{prefix}Please respond to the following question. Return only the single letter. {question.strip()}"

def build_prompt_opinionqa_default_no_prefix(question: str, attribute: str) -> str:    
    return f"Please respond to the following question. Return only the single letter. {question.strip()}"

def build_prompt_value_profile(question: str, attribute: str, bullet_value_lines: list) -> str:
    prefix = prefix_with_attribute(attribute)
    bullets = "\n".join([f"- {line.strip()}" for line in bullet_value_lines]) if bullet_value_lines else ""
    return (
        f"{prefix}Here are value profiles:\n{bullets}\n"
        f"Please respond to the following question considering the profile. Return only the single letter. "
        f"{question.strip()}"
    )

# ---------------------- Data Loaders ---------------------- #
def load_queries(query_dir, dataset_name):
    path = os.path.join(query_dir, dataset_name + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Query file not found: {path}")
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError(f"'query' column not found in {path}")
    return df["query"].dropna().drop_duplicates().tolist()

def _load_definitions_root(prompt_json_path):
    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if "definitions" in raw and isinstance(raw["definitions"], dict):
        return raw["definitions"]
    return raw

def _resolve_definition_for_value(def_map: dict, value_name: str):
    if value_name in def_map:
        item = def_map[value_name]
    else:
        want = canonicalize_value_key(value_name)
        found_key = None
        for k in def_map.keys():
            if canonicalize_value_key(k) == want:
                found_key = k
                break
        if found_key is None:
            raise KeyError(f"Target value '{value_name}' not found in definitions JSON.")
        item = def_map[found_key]

    if isinstance(item, list):
        defs = [d for d in item if isinstance(d, str) and d.strip() != ""]
        if not defs:
            raise ValueError(f"Value '{value_name}' has an empty definitions list.")
        return defs
    elif isinstance(item, str):
        s = item.strip()
        if not s:
            raise ValueError(f"Value '{value_name}' has an empty definition string.")
        return [s]
    else:
        raise ValueError(f"Unsupported definition type for value '{value_name}': {type(item)}")

def load_single_definition(prompt_json_path, value_name):
    def_map = _load_definitions_root(prompt_json_path)
    defs = _resolve_definition_for_value(def_map, value_name)
    return defs[0]

def load_multiple_definitions(prompt_json_path, value_name, n):
    def_map = _load_definitions_root(prompt_json_path)
    defs = _resolve_definition_for_value(def_map, value_name)
    return choose_k(defs, max(1, int(n)))

def load_user_texts(few_shot_path, value_name, intensity, n=2):
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if value_name not in data:
        want = canonicalize_value_key(value_name)
        found_key = None
        for k in data.keys():
            if canonicalize_value_key(k) == want:
                found_key = k
                break
        if found_key is None:
            raise KeyError(f"Target value '{value_name}' not found in few-shot JSON.")
        value_name = found_key

    key = str(intensity)
    if key not in data[value_name]:
        raise KeyError(f"Intensity '{intensity}' not found for value '{value_name}' in few-shot JSON.")

    examples = data[value_name][key]
    pool = [ex["text"] for ex in examples if isinstance(ex, dict) and "text" in ex and str(ex["text"]).strip() != ""]
    return choose_k(pool, max(1, int(n)))

def load_multiple_fewshot_texts(few_shot_path, value_name, intensity, n_batches, per_batch):
    total_needed = max(0, int(n_batches)) * max(1, int(per_batch))
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if value_name not in data:
        want = canonicalize_value_key(value_name)
        found_key = None
        for k in data.keys():
            if canonicalize_value_key(k) == want:
                found_key = k
                break
        if found_key is None:
            raise KeyError(f"Target value '{value_name}' not found in few-shot JSON.")
        value_name = found_key

    key = str(intensity)
    if key not in data[value_name]:
        raise KeyError(f"Intensity '{intensity}' not found for value '{value_name}' in few-shot JSON.")

    examples = data[value_name][key]
    pool = [ex["text"] for ex in examples if isinstance(ex, dict) and "text" in ex and str(ex["text"]).strip() != ""]
    return choose_k(pool, total_needed)

def load_opinionqa_items(path_json: str):
    """
    Returns a list of dicts with keys including: question(str), attribute(str), ...
    Deduplication is done per (attribute, question) pair to keep all attributes.
    """
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = set()
    uniq = []
    for item in data:
        q = (item.get("question") or "").strip()
        attr = (item.get("attribute") or "").strip()
        if not q or not attr:
            continue
        key = (attr, q)              # <-- per-attribute dedup
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq

# >>> CHANGED <<< — load a single attribute profile FILE (not root)
def load_attribute_profile_file(profile_json_file: str | Path) -> dict:
    """
    Loads a single attribute profile JSON (path to '{attribute}.json').
    """
    path = Path(profile_json_file)
    if not path.exists():
        raise FileNotFoundError(f"Profile JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def select_value_profile_lines(profile: dict, theory: str, num_prompts: int) -> list[str]:
    """
    Returns up to num_prompts bullet lines (strings) from profile['theories'][theory or all],
    chosen by highest & lowest 'score'. Uses 'definition_prompt' field if present.
    """
    if num_prompts <= 0:
        return []

    def items_from_theory(thname: str):
        out = []
        th = profile.get("theories", {}).get(thname, {})
        for bucket in ("very_high", "high", "medium", "low", "very_low"):
            for it in th.get(bucket, []):
                score = it.get("score")
                text = it.get("definition_prompt") or it.get("definition_raw")
                if text is None or score is None:
                    continue
                out.append((int(score), text.strip(), thname, it.get("value_key"), it.get("column")))
        return out

    pool = []
    theories_list = ["duty", "mft", "pvq", "rights"] if theory == "all" else [theory]
    for th in theories_list:
        pool.extend(items_from_theory(th))

    # Dedup by (theory, value_key)
    dedup = {}
    for sc, txt, th, vkey, col in pool:
        key = (th, vkey or col or txt)
        if key not in dedup:
            dedup[key] = (sc, txt, th, vkey, col)
    pool = list(dedup.values())
    if not pool:
        return []

    pool_sorted = sorted(pool, key=lambda x: x[0])  # ascending
    n = min(num_prompts, len(pool_sorted))
    if n % 2 == 0:
        k_low = n // 2
        k_high = n // 2
    else:
        k_high = (n + 1) // 2
        k_low = (n - 1) // 2

    lows = pool_sorted[:k_low]
    highs = list(reversed(pool_sorted))[:k_high]
    chosen = highs + lows
    lines = [c[1] for c in chosen]
    return lines

# ---------------------- Main ---------------------- #
def main(args):
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model = LLMModel(model=args.test_model,
                     max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature)

    # Normalize list inputs (legacy modes)
    target_values = norm_list_csv_or_list(args.target_values) if args.target_values is not None else None
    prompt_intensities_raw = norm_list_csv_or_list(args.prompt_intensities) if args.prompt_intensities is not None else None
    if prompt_intensities_raw is not None:
        try:
            prompt_intensities = [int(x) for x in prompt_intensities_raw]
        except Exception:
            raise ValueError("--prompt_intensities must be integers in {-2,-1,0,1,2}")
    else:
        prompt_intensities = None

    # Validate combos depending on prompt_format
    if args.prompt_format == "default":
        pass
    elif args.prompt_format == "single_definition":
        if not args.prompt_json or not args.target_value or args.prompt_intensity is None:
            raise ValueError("For 'single_definition', need --prompt_json, --target_value, --prompt_intensity.")
    elif args.prompt_format == "user_texts":
        if not args.few_shot_json or not args.target_value or args.prompt_intensity is None:
            raise ValueError("For 'user_texts', need --few_shot_json, --target_value, --prompt_intensity.")
    elif args.prompt_format == "multiple_definition":
        if not args.prompt_json:
            raise ValueError("For 'multiple_definition', need --prompt_json.")
        if not target_values or not prompt_intensities:
            raise ValueError("For 'multiple_definition', need --target_values and --prompt_intensities.")
        if len(target_values) != len(prompt_intensities):
            raise ValueError("--target_values and --prompt_intensities must have the same length.")
        for it in prompt_intensities:
            if it not in INTENSITY_MAP and it != 0:
                raise ValueError("All intensities must be in {-2,-1,0,1,2}.")
    elif args.prompt_format == "multiple_fewshot":
        if not args.few_shot_json:
            raise ValueError("For 'multiple_fewshot', need --few_shot_json.")
        if not target_values or not prompt_intensities:
            raise ValueError("For 'multiple_fewshot', need --target_values and --prompt_intensities.")
        if len(target_values) != len(prompt_intensities):
            raise ValueError("--target_values and --prompt_intensities must have the same length.")
        for it in prompt_intensities:
            if it not in INTENSITY_MAP and it != 0:
                raise ValueError("All intensities must be in {-2,-1,0,1,2}.")
    # >>> CHANGED <<< — value_profile now requires a single attribute & a single profile file
    elif args.prompt_format == "value_profile":
        if not args.opinionqa_json:
            raise ValueError("For 'value_profile', need --opinionqa_json.")
        if not args.attribute:
            raise ValueError("For 'value_profile', need --attribute.")
        if not args.profile_json_file:
            raise ValueError("For 'value_profile', need --profile_json_file (path to the attribute’s JSON).")
        if args.theory not in {"pvq", "mft", "duty", "rights", "all"}:
            raise ValueError("--theory must be one of pvq|mft|duty|rights|all.")
        if args.num_prompts is None or args.num_prompts < 1:
            raise ValueError("--num_prompts must be a positive integer.")
    # >>> CHANGED <<< — new simple OpinionQA default (no value bullets), single attribute
    elif args.prompt_format == "opinionqa_default":
        if not args.opinionqa_json:
            raise ValueError("For 'opinionqa_default', need --opinionqa_json.")
        if not args.attribute:
            raise ValueError("For 'opinionqa_default', need --attribute.")
    else:
        raise ValueError(f"Unsupported prompt format: {args.prompt_format}")

    # Preload single-definition if needed
    single_definition_text = None
    if args.prompt_format == "single_definition":
        single_definition_text = load_single_definition(args.prompt_json, args.target_value)

    all_records = []
    short_model_name = args.test_model.split("/")[-1]
    dataset_suffix = "-".join(args.query) if args.query else "NA"

    # >>> CHANGED <<< — value_profile: operate ONLY on the provided single attribute/profile file
    if args.prompt_format == "value_profile":
        items = load_opinionqa_items(args.opinionqa_json)  # unique questions
        attribute = args.attribute
        rows = [it for it in items if (it.get("attribute") == attribute)]
        print(f"\n[✓] Attribute: {attribute} (unique questions={len(rows)})")

        profile = load_attribute_profile_file(args.profile_json_file)
        vp_lines = select_value_profile_lines(profile, args.theory, args.num_prompts)

        prompts, q_texts = [], []
        for it in rows:
            q = (it.get("question") or "").strip()
            if not q:
                continue
            p = build_prompt_value_profile(q, attribute, vp_lines)
            prompts.append(p)
            q_texts.append(q)

        if not prompts:
            print(f"[!] No prompts constructed for attribute {attribute}, nothing to do.")
            return

        responses = model(prompts)
        attr_records = []
        for q_text, resp in zip(q_texts, responses):
            attr_records.append({
                "question": q_text,
                "attribute": attribute,
                "response": (resp or "").strip(),
            })
        df_attr = pd.DataFrame(attr_records)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = "value_profile"
        # include attribute in filename; no per-attr directory
        file_name = f"{short_model_name}_opinionqa_{mode_tag}_n{args.num_prompts}_{args.theory}_{attribute}.csv"
        out_path = out_dir / file_name
        df_attr.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[✓] Saved file: {out_path}")
        return  # done

    # >>> CHANGED <<< — opinionqa_default: simple default prompt for a SINGLE attribute
    if args.prompt_format == "opinionqa_default":
        items = load_opinionqa_items(args.opinionqa_json)
        attribute = args.attribute
        rows = [it for it in items if (it.get("attribute") == attribute)]
        print(f"\n[✓] Attribute: {attribute} (unique questions={len(rows)})")

        prompts, q_texts = [], []
        for it in rows:
            q = (it.get("question") or "").strip()
            if not q:
                continue
            if args.exclude_prefix:
                p = build_prompt_opinionqa_default_no_prefix(q, attribute)
            else:
                p = build_prompt_opinionqa_default(q, attribute)
            prompts.append(p)
            q_texts.append(q)

        if not prompts:
            print(f"[!] No prompts constructed for attribute {attribute}, nothing to do.")
            return

        responses = model(prompts)
        attr_records = []
        for q_text, resp in zip(q_texts, responses):
            attr_records.append({
                "question": q_text,
                "attribute": attribute,
                "response": (resp or "").strip(),
            })
        df_attr = pd.DataFrame(attr_records)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = "opinionqa_default"
        file_name = f"{short_model_name}_opinionqa_{mode_tag}_{attribute}.csv"
        out_path = out_dir / file_name
        df_attr.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[✓] Saved file: {out_path}")
        return  # done

    # ---------------- Legacy modes (unchanged behavior) ---------------- #
    for dataset in (args.query or []):
        print(f"\n[✓] Processing dataset: {dataset}")
        queries = load_queries(args.query_dir, dataset)

        prompts = []
        for q in queries:
            if args.prompt_format == "default":
                prompts.append(build_prompt_default(q))

            elif args.prompt_format == "single_definition":
                prompts.append(
                    build_prompt_single_definition(
                        q,
                        single_definition_text,
                        value_name=args.target_value,
                        intensity=args.prompt_intensity,
                    )
                )

            elif args.prompt_format == "user_texts":
                user_texts = load_user_texts(
                    args.few_shot_json,
                    args.target_value,
                    args.prompt_intensity,
                    args.few_shot_n
                )
                prompts.append(build_prompt_user_texts(q, user_texts))

            elif args.prompt_format == "multiple_definition":
                defs_per_value = {}
                for vname, inten in zip(target_values, prompt_intensities):
                    defs = load_multiple_definitions(args.prompt_json, vname, args.multiple_n)
                    defs_per_value[vname] = [(d, inten) for d in defs]
                prompts.append(build_prompt_multiple_definition(q, defs_per_value))

            elif args.prompt_format == "multiple_fewshot":
                all_texts = []
                for vname, inten in zip(target_values, prompt_intensities):
                    texts = load_multiple_fewshot_texts(
                        args.few_shot_json,
                        vname,
                        inten,
                        n_batches=args.multiple_n,
                        per_batch=args.few_shot_n
                    )
                    all_texts.extend(texts)
                prompts.append(build_prompt_multiple_fewshot(q, all_texts))

            else:
                raise ValueError(f"Unsupported prompt format at runtime: {args.prompt_format}")

        responses = model(prompts)

        for q_text, r in zip(queries, responses):
            rec = {
                "query": q_text,
                "dataset": dataset,
                "response": (r or "").strip(),
            }
            if args.prompt_format in ("single_definition", "user_texts"):
                rec.update({
                    "target_value": args.target_value,
                    "prompt_intensity": args.prompt_intensity
                })
            elif args.prompt_format in ("multiple_definition", "multiple_fewshot"):
                rec.update({
                    "target_values": "|".join(target_values),
                    "prompt_intensities": "|".join([str(x) for x in prompt_intensities]),
                    "multiple_n": args.multiple_n
                })
            all_records.append(rec)

    if all_records:
        output_df = pd.DataFrame(all_records)
        mode_tag = args.prompt_format
        if args.prompt_format in ("multiple_definition", "multiple_fewshot"):
            values_tag = "_".join([v.replace("/", "_") for v in (target_values or [])]) or "NA"
            intens_tag = "_".join([str(i) for i in (prompt_intensities or [])]) or "NA"
            extra_tag = f"{values_tag}_{intens_tag}_mn{args.multiple_n}_k{args.few_shot_n}"
        elif args.prompt_format in ("single_definition", "user_texts"):
            values_tag = (args.target_value or "NA").replace("/", "_")
            intens_tag = str(args.prompt_intensity) if args.prompt_intensity is not None else "NA"
            extra_tag = f"{values_tag}_{intens_tag}_k{args.few_shot_n}"
        else:
            extra_tag = "NA"

        output_path = os.path.join(
            args.output_dir,
            f"{short_model_name}_{dataset_suffix}_{mode_tag}_{extra_tag}.csv"
        )
        output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[✓] Saved output to: {output_path}")

# ---------------------- Args ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/query_responses")
    parser.add_argument("--query_dir", type=str, help="(legacy CSV query directory)")
    parser.add_argument("--query", nargs="+", help="(legacy) dataset names to read from --query_dir")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="default",
        choices=[
            "default",
            "single_definition",
            "user_texts",
            "multiple_definition",
            "multiple_fewshot",
            "value_profile",       # uses single --attribute + --profile_json_file
            "opinionqa_default",   # NEW: default OpinionQA prompt, single --attribute
        ],
    )

    # Single-definition & user_texts (backward compat)
    parser.add_argument("--prompt_json", type=str, help="Path to definitions JSON (single_definition, multiple_definition)")
    parser.add_argument("--target_value", type=str, help="Target value (single modes)")
    parser.add_argument("--prompt_intensity", type=int, choices=[-2, -1, 0, 1, 2])

    # Few-shot JSON
    parser.add_argument("--few_shot_json", type=str, help="Path to few-shot JSON (user_texts, multiple_fewshot)")
    parser.add_argument("--few_shot_n", type=int, default=2, help="Number of texts to sample per batch")

    # New multi-value options
    parser.add_argument("--target_values", nargs="+", help="List (or comma-separated) of values for multi modes")
    parser.add_argument("--prompt_intensities", nargs="+", help="List (or comma-separated) of intensities for multi modes; each in {-2,-1,0,1,2}")
    parser.add_argument("--multiple_n", type=int, default=2, help="How many definitions (or batches) per value in multi modes")

    # Misc
    parser.add_argument("--seed", type=int, default=123)

    # >>> CHANGED <<< — OpinionQA / value_profile arguments
    parser.add_argument("--opinionqa_json", type=str, help="Path to OpinionQA items JSON (with 'question' and 'attribute')")
    parser.add_argument("--attribute", type=str, help="Single attribute to run (e.g., 'POLIDEOLOGY_Conservative')")
    parser.add_argument("--profile_json_file", type=str, help="Path to the *single* attribute profile JSON file (for value_profile)")
    parser.add_argument("--theory", type=str, default="all", help="pvq|mft|duty|rights|all (for value_profile)")
    parser.add_argument("--num_prompts", type=int, default=6, help="Max number of value statements to inject (value_profile)")
    parser.add_argument("--exclude_prefix", default=False)

    args = parser.parse_args()
    main(args)
