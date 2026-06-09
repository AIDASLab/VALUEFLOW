#!/usr/bin/env bash
set -euo pipefail
# export PYTHONPATH="your path"

# ----------------------------
# Config (edit as needed)
# ----------------------------
MODELS=(
  # "microsoft/phi-4"
  # "google/gemma-3-27b-it"
  # "Qwen/Qwen3-32B"
  # "zai-org/GLM-4-32B-0414"
  # "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  #  "grok-4-0709"
  #  "gemini-2.5-flash"
  # "openai/gpt-oss-20b"
)

MULTIPLE_N=2      # number of defs per value OR batches in fewshot
FEW_SHOT_N=3      # few-shot items per batch

QUERY_DIR="data/query/filtered"
DEFS_JSON="data/all_definitions.json"
FEWSHOT_JSON="data/prompt/few_shot/all_few_shot.json"
OUT_ROOT="outputs/query_responses/multi_homogeneous_pvq_2"
QUERIES="gpv"

MAX_NEW_TOKENS=150
TEMPERATURE=1.0

# ----------------------------
# Manual pairs with intensities
# Format: ("value1" "value2" "intensity1" "intensity2")
# ----------------------------

# Schwartz
PAIR_SCH_1=("benevolence" "universalism" "2" "2")
PAIR_SCH_2=("self-direction" "stimulation" "2" "2")
PAIR_SCH_3=("power" "universalism" "2" "2")
PAIR_SCH_4=("conformity" "hedonism" "2" "2")
PAIR_SCH_5=("security" "tradition" "2" "2")

PAIR_SCH_6=("benevolence" "universalism" "-2" "-2")
PAIR_SCH_7=("self-direction" "stimulation" "-2" "-2")
PAIR_SCH_8=("power" "universalism" "-2" "-2")
PAIR_SCH_9=("conformity" "hedonism" "-2" "-2")
PAIR_SCH_10=("security" "tradition" "-2" "-2")

PAIR_SCH_11=("benevolence" "universalism" "2" "1")
PAIR_SCH_12=("self-direction" "stimulation" "2" "1")
PAIR_SCH_13=("power" "universalism" "2" "1")
PAIR_SCH_14=("conformity" "hedonism" "2" "1")
PAIR_SCH_15=("security" "tradition" "2" "1")

PAIR_SCH_16=("benevolence" "universalism" "1" "2")
PAIR_SCH_17=("self-direction" "stimulation" "1" "2")
PAIR_SCH_18=("power" "universalism" "1" "2")
PAIR_SCH_19=("conformity" "hedonism" "1" "2")
PAIR_SCH_20=("security" "tradition" "1" "2")

# # MFT
# PAIR_MFT_1=("care/harm" "fairness/cheating" "2" "2")
# PAIR_MFT_2=("loyalty/betrayal" "authority/subversion" "2" "2")
# PAIR_MFT_3=("liberty/oppression" "authority/subversion" "2" "-2")
# PAIR_MFT_4=("sanctity/degradation" "care/harm" "2" "-2")
# PAIR_MFT_5=("loyalty/betrayal" "fairness/cheating" "1" "2")

# # Rights
# PAIR_RTS_1=("expression" "assembly" "2" "2")
# PAIR_RTS_2=("education" "work" "2" "2")
# PAIR_RTS_3=("privacy" "due_process" "-1" "2")
# PAIR_RTS_4=("expression" "privacy" "2" "-1")
# PAIR_RTS_5=("health" "social_security" "2" "1")

# # Ross
# PAIR_ROSS_1=("beneficence" "non-maleficence" "2" "2")
# PAIR_ROSS_2=("fidelity" "reparation" "2" "2")
# PAIR_ROSS_3=("justice" "beneficence" "2" "-1")
# PAIR_ROSS_4=("fidelity" "justice" "2" "1")
# PAIR_ROSS_5=("gratitude" "justice" "1" "2")

# Collect all
ALL_PAIRS=(
  PAIR_SCH_1 PAIR_SCH_2 PAIR_SCH_3 PAIR_SCH_4 PAIR_SCH_5
  PAIR_SCH_6 PAIR_SCH_7 PAIR_SCH_8 PAIR_SCH_9 PAIR_SCH_10
  PAIR_SCH_11 PAIR_SCH_12 PAIR_SCH_13 PAIR_SCH_14 PAIR_SCH_15
  PAIR_SCH_16 PAIR_SCH_17 PAIR_SCH_18 PAIR_SCH_19 PAIR_SCH_20
)

# ----------------------------
# Helpers
# ----------------------------
run_multiple_definition() {
  local model="$1" out_dir="$2" v1="$3" v2="$4" i1="$5" i2="$6"
  echo "[multi_def] model=$model | values=($v1,$v2) | intensities=($i1,$i2)"

  python evaluation/generation/generate_response_multiple.py \
    --test_model "$model" \
    --output_dir "$out_dir" \
    --query_dir "$QUERY_DIR" \
    --query $QUERIES \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --prompt_format multiple_definition \
    --prompt_json "$DEFS_JSON" \
    --target_values "$v1" "$v2" \
    --prompt_intensities "$i1" "$i2" \
    --multiple_n "$MULTIPLE_N" \
    --few_shot_n "$FEW_SHOT_N"
}

run_multiple_fewshot() {
  local model="$1" out_dir="$2" v1="$3" v2="$4" i1="$5" i2="$6"
  echo "[multi_fs]  model=$model | values=($v1,$v2) | intensities=($i1,$i2)"

  python evaluation/generation/generate_response_multiple.py \
    --test_model "$model" \
    --output_dir "$out_dir" \
    --query_dir "$QUERY_DIR" \
    --query $QUERIES \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --prompt_format multiple_fewshot \
    --few_shot_json "$FEWSHOT_JSON" \
    --target_values "$v1" "$v2" \
    --prompt_intensities "$i1" "$i2" \
    --multiple_n "$MULTIPLE_N" \
    --few_shot_n "$FEW_SHOT_N"
}

# ----------------------------
# Execution
# ----------------------------
for model in "${MODELS[@]}"; do
  for pair_name in "${ALL_PAIRS[@]}"; do
    pair_ref="$pair_name[@]"
    pair_vals=("${!pair_ref}")
    if [[ "${#pair_vals[@]}" -ne 4 ]]; then
      echo "[WARN] Skipping $pair_name (need 4 elems: v1 v2 i1 i2, got ${#pair_vals[@]})"
      continue
    fi
    v1="${pair_vals[0]}" ; v2="${pair_vals[1]}"
    i1="${pair_vals[2]}" ; i2="${pair_vals[3]}"

    safe_pair_tag="$(echo "${v1}__${v2}" | sed 's#[/ ]#_#g')"
    OUT_DEF="$OUT_ROOT/multiple_definition/${safe_pair_tag}"
    OUT_FS="$OUT_ROOT/multiple_fewshot/${safe_pair_tag}"
    mkdir -p "$OUT_DEF" "$OUT_FS"

    # run_multiple_definition "$model" "$OUT_DEF" "$v1" "$v2" "$i1" "$i2"  
    run_multiple_fewshot "$model" "$OUT_FS" "$v1" "$v2" "$i1" "$i2"   
  done
done

echo "[✓] All runs complete."
