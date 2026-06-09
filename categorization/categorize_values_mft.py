import os
import json
import pandas as pd
import argparse
import re
from models import LLMModel
from tqdm import tqdm


def build_reverse_hierarchy(hierarchy):
    reverse_map = {}
    for parent, children in hierarchy.items():
        for child in children:
            reverse_map[child] = parent
    return reverse_map


def build_prompt(parent, parent_def, sub_values, definitions, target_value):
    sub_lines = "\n".join([f"- {sub}: {definitions.get(sub, '')}" for sub in sub_values])
    return f"""
You are given a candidate moral value and a list of sub-values under a moral foundation.

**Parent Category:** {parent}  
**Definition:** {parent_def}

This parent category has the following sub-values:
{sub_lines}

**Target Value:**  
{target_value.strip()}

Your task is to classify the target value under one of the sub-values above. If it cannot be classified further, answer "Selected: None".

Format your answer as:
Selected: sub-value-name
Only return the Selected: sub-value name. Do not return anything else.
""".strip()


def parse_llm_response(response, sub_values):
    match = re.search(r"Selected:\s*(.+)", response.strip(), re.IGNORECASE)
    if match:
        selected = match.group(1).strip()
        if selected.lower() == "none":
            return None
        for s in sub_values:
            if selected.lower() in s.lower():
                return s
    return None


def recursive_classify_value(value_term, current_parent, hierarchy, definitions, model, max_depth=3):
    path = []
    for _ in range(max_depth):
        children = hierarchy.get(current_parent, [])
        if not children:
            break
        prompt = build_prompt(current_parent, definitions.get(current_parent, ""), children, definitions, value_term)
        response = model([prompt])[0]
        selected = parse_llm_response(response, children)
        if selected is None:
            break
        path.append(selected)
        current_parent = selected
    return path


def classify_values_with_mft(df, hierarchy, definitions, model_name="Qwen/Qwen1-7B", max_depth=3):
    model = LLMModel(model=model_name, max_new_tokens=100, temperature=1.0)

    all_children = {c for children in hierarchy.values() for c in children}
    root_mfts = [d for d in hierarchy if d not in all_children]

    output_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            target_value = str(row["value"])
            mft_path = []

            for root in root_mfts:
                result_path = recursive_classify_value(target_value, root, hierarchy, definitions, model, max_depth)
                if result_path:
                    mft_path = [root] + result_path
                    break
                elif not result_path:
                    mft_path = [root]
                    break

            output_rows.append({
                "idx": idx,
                "target_value": target_value,
                **{f"level_{i+1}": mft_path[i] if i < len(mft_path) else None for i in range(max_depth)}
            })

        except Exception as e:
            print(f"[Error] Row {idx}: {e}")
            continue

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON list of {'value': ...}")
    parser.add_argument("--criteria", type=str, required=True, help="MFT hierarchy JSON (mft.json)")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1-7B")
    args = parser.parse_args()

    try:
        df = pd.read_json(args.input_json)
        with open(args.criteria, "r", encoding="utf-8") as f:
            mft_data = json.load(f)

        hierarchy = mft_data["hierarchy"]
        definitions = mft_data["definitions"]

        df_out = classify_values_with_mft(df, hierarchy, definitions, model_name=args.model_name)
        df_out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"[✔] Saved: {args.output_csv}")

    except Exception as e:
        print(f"[Fatal Error] {e}")
