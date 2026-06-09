import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import argparse
import json
import random
from tqdm import tqdm
from models import LLMModel


# ---------------------- Prompt Builder ---------------------- #
def build_prompt(query: str,
                 prompt_format: str = "default",
                 definition: str = None,
                 value_name: str = None,
                 intensity: int = None,
                 user_texts: list = None) -> str:

    if prompt_format == "default":
        return (
            "You are an AI assistant that responds to every question in a particular format.\n\n"
            "Your answer is not allowed to start with 'As an AI language model ...' or with 'I cannot ...'.\n\n"
            f"{query.strip()}\n\nAnswer with no more than 100 words."
        )

    elif prompt_format == "single_definition":
        if not definition or value_name is None or intensity is None:
            raise ValueError("Definition, value_name, and intensity must be provided for 'single_definition' prompt format.")

        if intensity != 0:
            intensity_map = {
                2: "strongly values",
                1: "slightly values",
                -1: "slightly rejects",
                -2: "strongly rejects"
            }
            if intensity not in intensity_map:
                raise ValueError("prompt_intensity must be one of: -2, -1, 0, 1, 2")

            prefix = "You are a person that values"
            if definition.startswith(prefix):
                rest = definition[len(prefix):].lstrip()
                definition = f"You are a person who {intensity_map[intensity]} {rest}"

        return (
            f"{definition.strip()}\n\n"
            f"{query.strip()}\n\nAnswer with no more than 100 words."
        )

    elif prompt_format == "user_texts":
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

    else:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")


# ---------------------- Dataset Loader ---------------------- #
def load_queries(query_dir, dataset_name):
    path = os.path.join(query_dir, dataset_name + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Query file not found: {path}")
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError(f"'query' column not found in {path}")
    return df["query"].dropna().drop_duplicates().tolist()


# ---------------------- Few-shot Loader (user texts) ---------------------- #
def load_user_texts(few_shot_path, value_name, intensity, n=2):
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if value_name not in data:
        raise KeyError(f"Target value '{value_name}' not found in JSON.")
    if str(intensity) not in data[value_name]:
        raise KeyError(f"Intensity '{intensity}' not found for value '{value_name}' in JSON.")
    examples = data[value_name][str(intensity)]
    return [ex["text"] for ex in random.sample(examples, min(n, len(examples)))]


# ---------------------- Main ---------------------- #
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model = LLMModel(model=args.test_model,
                     max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature)

    definition = None

    if args.prompt_format == "single_definition":
        if not args.prompt_json or not args.target_value or args.prompt_intensity is None:
            raise ValueError("For 'single_definition', need --prompt_json, --target_value, --prompt_intensity.")
        with open(args.prompt_json, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        if args.target_value not in prompt_data:
            raise KeyError(f"Target value '{args.target_value}' not found in prompt JSON.")
        definition = prompt_data[args.target_value]

    all_records = []
    short_model_name = args.test_model.split("/")[-1]

    for dataset in args.query:
        print(f"\n[✓] Processing dataset: {dataset}")
        queries = load_queries(args.query_dir, dataset)

        prompts = []
        for q in queries:
            # ⬇️ Resample user_texts each time
            user_texts = None
            if args.prompt_format == "user_texts":
                if not args.few_shot_json or not args.target_value or args.prompt_intensity is None:
                    raise ValueError("For 'user_texts', need --few_shot_json, --target_value, --prompt_intensity.")
                user_texts = load_user_texts(args.few_shot_json,
                                             args.target_value,
                                             args.prompt_intensity,
                                             args.few_shot_n)

            prompts.append(
                build_prompt(q,
                             args.prompt_format,
                             definition=definition,
                             value_name=args.target_value,
                             intensity=args.prompt_intensity,
                             user_texts=user_texts)
            )

        responses = model(prompts)

        for q, r in zip(queries, responses):
            all_records.append({
                "query": q,
                "dataset": dataset,
                "response": r.strip()
            })

    output_df = pd.DataFrame(all_records)
    dataset_suffix = "-".join(args.query)
    output_path = os.path.join(
        args.output_dir,
        f"{short_model_name}_{dataset_suffix}_{args.prompt_format}_{args.target_value}_{args.prompt_intensity}_{args.few_shot_n}.csv"
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
    parser.add_argument("--prompt_format", type=str, default="default",
                        choices=["default", "single_definition", "user_texts"])
    parser.add_argument("--prompt_json", type=str, help="JSON for definitions (single_definition)")
    parser.add_argument("--target_value", type=str, help="Target value in JSON")
    parser.add_argument("--prompt_intensity", type=int, choices=[-2, -1, 0, 1, 2])
    parser.add_argument("--few_shot_json", type=str, help="Path to few-shot JSON (used in user_texts)")
    parser.add_argument("--few_shot_n", type=int, default=2, help="Number of texts to sample")
    args = parser.parse_args()
    main(args)
