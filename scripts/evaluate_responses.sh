#!/bin/bash

# export PYTHONPATH="your path"

SCRIPT="evaluation/evaluate_response_intensity.py"

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

INTENSITIES=(
  -2
  -1
  0
  1
  2
)

# ---- eval model ----
EVAL_LLM="google/gemma-3-27b-it"
THEORY="pvq"                           # pvq | mft | duty | right

# ---- ranking prompt settings ----
K=6        # total texts per prompt (1 response + K-1 sampled db texts)
M=3         # prompts per response (comparisons)
ITER=50     # epochs for 1-D PL optimization
LR=0.1      # learning rate for utility ascent

PROMPT_FORMAT="default"   # default | oneshot
SAMPLING_METHOD="bucket"
MAX_NEW_TOKENS=50
TEMPERATURE=0.0

# ---- loop over values and intensities ----
for TARGET_VALUE in "${VALUES[@]}"; do
  for INTENSITY in "${INTENSITIES[@]}"; do

    RESPONSE_CSV="outputs/query_responses/single_definition/${THEORY}/${TARGET_VALUE}/gpt4_gpv-moral_choice-moral_stories-opinionqa-valuebench_single_definition_${TARGET_VALUE}_${INTENSITY}.csv"
    DEFINITION_CSV="data/${THEORY}.json"
    VALUE_DB_CSV="data/final_ratings/${TARGET_VALUE}_${THEORY}_flag_aggregated.csv"
    OUTPUT_DIR="outputs/query_responses/signle_definition_eval/${THEORY}"

    python "$SCRIPT" \
      --response_csv "$RESPONSE_CSV" \
      --definition_csv "$DEFINITION_CSV" \
      --value_db_csv "$VALUE_DB_CSV" \
      --output_dir "$OUTPUT_DIR" \
      --eval_llm "$EVAL_LLM" \
      --target_value "$TARGET_VALUE" \
      --theory "$THEORY" \
      --k "$K" \
      --m "$M" \
      --iter "$ITER" \
      --lr "$LR" \
      --prompt_format "$PROMPT_FORMAT" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --sampling_method "$SAMPLING_METHOD"
      # --eval_only \
      # --ranking_csv "$RANKING_CSV"
  done
done
