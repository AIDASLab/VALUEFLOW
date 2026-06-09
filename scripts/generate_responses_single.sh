#!/bin/bash

# export PYTHONPATH={"some path"}

# # single run
# python generation/generate_response.py \
#   --test_model microsoft/phi-4 \
#   --output_dir outputs/query_responses/single_definition \
#   --query_dir data/query/filtered \
#   --query gpv moral_choice moral_stories opinionqa valuebench \
#   --max_new_tokens 150 \
#   --temperature 1.0 \
#   --prompt_format single_definition \
#   --prompt_json data/prompt/single_definition/pvq.json \
#   --target_value benevolence \
#   --prompt_intensity 0

# One of [pvq, mft, right, duty]
THEORY="pvq"

MODELS=(
  # "microsoft/phi-4"
  # "google/gemma-3-27b-it"
  # "Qwen/Qwen3-32B"
  # "THUDM/GLM-4-32B-0414"
  # "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  # "grok-4-0709"
  # "gemini-2.5-flash"
  # "openai/gpt-oss-20b"
)

VALUES=(
  "benevolence"
  "universalism"
  "self-direction"
  "stimulation"
  "hedonism"
  "achievement"
  "power"
  "conformity"
  "tradition"
  "security"
  "face"
  "humility"
)

INTENSITIES=(-2 -1 0 1 2)

QUERY_DIR="data/query/filtered"
PROMPT_JSON="data/prompt/single_definition/$THEORY.json"
FEW_SHOT_JSON="data/prompt/few_shot/$THEORY.json"
OUTPUT_DIR="outputs/query_responses/single_definition/$THEORY"
QUERIES="gpv moral_choice moral_stories opinionqa valuebench"

# ----------------------------
# Execution loop
# ----------------------------

# change prompt_format to user_texts for user text prompt

for model in "${MODELS[@]}"; do
  for value in "${VALUES[@]}"; do
    for intensity in "${INTENSITIES[@]}"; do
      echo "[RUNNING] model=$model | value=$value | intensity=$intensity"

      python generation/generate_response.py \
        --test_model "$model" \
        --output_dir "$OUTPUT_DIR/${value}" \
        --query_dir "$QUERY_DIR" \
        --query $QUERIES \
        --max_new_tokens 150 \
        --temperature 1.0 \
        --prompt_format single_definition \
        --prompt_json "$PROMPT_JSON" \
        --target_value "$value" \
        --prompt_intensity "$intensity"
    done
  done
done