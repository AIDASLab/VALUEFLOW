import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import argparse
import json
import random
from tqdm import tqdm
from models import LLMModel

# ---------------------- Helpers ---------------------- #
INTENSITY_MAP = {
    2: "strongly values",
    1: "slightly values",
    -1: "slightly rejects",
    -2: "strongly rejects",
    0: None,
}

def norm_list_csv_or_list(x):
    """
    Accept comma-separated string or nargs='+'. Returns list[str].
    """
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
    """Sample up to k items; if fewer exist, return all."""
    if not items:
        return []
    if k >= len(items):
        return list(items)
    return random.sample(items, k)

def rephrase_definition(definition: str, intensity: int) -> str:
    """
    Adjust the definition to reflect intensity.
    If definition starts with 'You are a person that/who values ...',
    rewrite to 'You are a person who <intensity phrase> ...'.
    Otherwise, prepend a short clause with intensity.
    """
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

    # Fallback: prepend
    return f"You are a person who {INTENSITY_MAP[intensity]} {d}"

def canonicalize_value_key(key: str) -> str:
    """
    Try to normalize keys so 'care/harm' and 'care-harm' match.
    """
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
    """
    defs_per_value: { value_name: [ (definition_text, intensity), ... ] }
    """
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

# ---------------------- Data Loaders ---------------------- #
def load_queries(query_dir, dataset_name):
    path = os.path.join(query_dir, dataset_name + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Query file not found: {path}")
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError(f"'query' column not found in {path}")
    return df["query"].dropna().drop_duplicates().tolist()

# ---- Definitions JSON: { "definitions": { "<value>": "<def or list>", ... } } ----
def _load_definitions_root(prompt_json_path):
    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if "definitions" in raw and isinstance(raw["definitions"], dict):
        return raw["definitions"]
    # Fallback: support the legacy structure where the root IS the dict of value->defs
    return raw

def _resolve_definition_for_value(def_map: dict, value_name: str):
    """Return list[str] of definitions for the given value. Supports string or list."""
    # Direct key
    if value_name in def_map:
        item = def_map[value_name]
    else:
        # Try canonicalized matching
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

# ---- Few-shot JSON: { "<value>": { "-2": [ { "text": ..., "intensity": float }, ... ], ... } } ----
def load_user_texts(few_shot_path, value_name, intensity, n=2):
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if value_name not in data:
        # try canonical key
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

# ---------------------- Main ---------------------- #
def main(args):
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model = LLMModel(model=args.test_model,
                     max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature)

    # Normalize list inputs
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
    else:
        raise ValueError(f"Unsupported prompt format: {args.prompt_format}")

    # Preload single-definition if needed
    single_definition_text = None
    if args.prompt_format == "single_definition":
        single_definition_text = load_single_definition(args.prompt_json, args.target_value)

    all_records = []
    short_model_name = args.test_model.split("/")[-1]
    dataset_suffix = "-".join(args.query)

    for dataset in args.query:
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
                "response": r.strip(),
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

    output_df = pd.DataFrame(all_records)

    # Filename parts
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
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--query", nargs="+", required=True)
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
    parser.add_argument("--target_values", nargs="+", help="List (or comma-separated) of values for multi modes (e.g., 'care/harm fairness/cheating')")
    parser.add_argument("--prompt_intensities", nargs="+", help="List (or comma-separated) of intensities for multi modes; each in {-2,-1,0,1,2}")
    parser.add_argument("--multiple_n", type=int, default=2, help="How many definitions (or batches) per value in multi modes")

    # Misc
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    main(args)
