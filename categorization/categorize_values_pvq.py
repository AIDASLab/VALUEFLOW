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
You are given a candidate human value and a list of sub-values under a higher-level value category.

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


def recursive_classify_value(value_term, current_parent, hierarchy, definitions, model, max_depth=5):
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


def classify_values(df, hierarchy, definitions, model_name="Qwen/Qwen1-7B", max_depth=5):
    model = LLMModel(model=model_name, max_new_tokens=100, temperature=1.0)

    all_children = {c for children in hierarchy.values() for c in children}
    root_values = [d for d in hierarchy if d not in all_children]

    output_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            value_name = str(row["value"])
            value_path = []

            for root in root_values:
                result_path = recursive_classify_value(value_name, root, hierarchy, definitions, model, max_depth)
                if result_path:
                    value_path = [root] + result_path
                    break
                elif not result_path:
                    value_path = [root]
                    break

            output_rows.append({
                "idx": idx,
                "target_value": value_name,
                **{f"level_{i+1}": value_path[i] if i < len(value_path) else None for i in range(max_depth)}
            })

        except Exception as e:
            print(f"[Error] Row {idx}: {e}")
            continue

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON list of {'value': ...}")
    parser.add_argument("--criteria", type=str, required=True, help="Path to schwartz_pvq.json")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1-7B")
    args = parser.parse_args()

    try:
        df = pd.read_json(args.input_json)
        with open(args.criteria, "r", encoding="utf-8") as f:
            value_data = json.load(f)

        hierarchy = value_data["hierarchy"]
        definitions = value_data["definitions"]

        df_out = classify_values(df, hierarchy, definitions, model_name=args.model_name)
        df_out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"[✔] Saved: {args.output_csv}")

    except Exception as e:
        print(f"[Fatal Error] {e}")
