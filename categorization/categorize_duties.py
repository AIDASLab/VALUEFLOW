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


def build_prompt(parent, parent_def, sub_values, definitions, target_duty):
    sub_lines = "\n".join([f"- {sub}: {definitions.get(sub, '')}" for sub in sub_values])
    return f"""
You are given a target moral duty and a set of candidate moral duties under a parent duty.

**Parent Duty:** {parent}  
**Definition:** {parent_def}

This parent duty has the following sub-duties:
{sub_lines}

**Target Duty:**  
{target_duty.strip()}

Your task is to classify the target duty under one of the sub-duties above. If it cannot be classified further, answer "Selected: None".

Format your answer as:
Selected: sub-duty-name
Only return the Selected: sub-duty name. Do not return anything else.
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


def recursive_classify_duty(duty_text, current_parent, hierarchy, definitions, model, max_depth=5):
    path = []
    for _ in range(max_depth):
        children = hierarchy.get(current_parent, [])
        if not children:
            break
        prompt = build_prompt(current_parent, definitions.get(current_parent, ""), children, definitions, duty_text)
        response = model([prompt])[0]
        selected = parse_llm_response(response, children)
        if selected is None:
            break
        path.append(selected)
        current_parent = selected
    return path


def classify_duties_with_llm(df, hierarchy, definitions, model_name="Qwen/Qwen1-7B", max_depth=5):
    model = LLMModel(model=model_name, max_new_tokens=100, temperature=1.0)

    # Detect root level duties (i.e., not children in any value)
    all_children = {c for children in hierarchy.values() for c in children}
    root_duties = [d for d in hierarchy if d not in all_children]

    output_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            target_duty = str(row["duty"])
            duty_path = []

            for root in root_duties:
                result_path = recursive_classify_duty(target_duty, root, hierarchy, definitions, model, max_depth)
                if result_path:
                    duty_path = [root] + result_path
                    break
                elif not result_path:
                    # Stop at root level if no further classification is possible
                    duty_path = [root]
                    break

            output_rows.append({
                "idx": idx,
                "target_duty": target_duty,
                **{f"level_{i+1}": duty_path[i] if i < len(duty_path) else None for i in range(max_depth)}
            })

        except Exception as e:
            print(f"[Error] Failed at row {idx}: {e}")
            continue

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="JSON file with target duties. Should contain a list of {'duty': '...'}")
    parser.add_argument("--criteria", type=str, required=True, help="Duty hierarchy JSON (with 'hierarchy' and 'definitions')")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1-7B")
    args = parser.parse_args()

    try:
        df = pd.read_json(args.input_json)
        with open(args.criteria, "r", encoding="utf-8") as f:
            criteria_data = json.load(f)

        hierarchy = criteria_data["hierarchy"]
        definitions = criteria_data["definitions"]

        df_out = classify_duties_with_llm(df, hierarchy, definitions, model_name=args.model_name)
        df_out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved output to: {args.output_csv}")

    except Exception as e:
        print(f"[Fatal Error] {e}")
