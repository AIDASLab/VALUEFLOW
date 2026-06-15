#!/bin/bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${BASE}/profling/generate_response_opinionqa.py"

OPINIONQA_JSON="${BASE}/data/steerable_test_opinionqa.json"
PROFILE_DIR="${BASE}/profling/prompts"
RELEVANCE_CSV="${BASE}/data/opinionqa_value_duty_relevance_unique_questions.csv"
OUTPUT_ROOT="${BASE}/outputs/opinionqa_final"

TEMPERATURE=0.0
SEED=42
MAX_NEW_TOKENS=4

# ============================================================
# BLOCK 0 — paste into tmux session: wjkim
# ============================================================
run_qwen3() {
  python "${PY}" \
    --model "Qwen/Qwen3-32B" \
    --opinionqa_json "${OPINIONQA_JSON}" \
    --profile_dir "${PROFILE_DIR}" \
    --relevance_csv "${RELEVANCE_CSV}" \
    --output_root "${OUTPUT_ROOT}" \
    --theory duty \
    --num_prompts 4 \
    --threshold 0.65 \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

# ============================================================
# BLOCK 1 — paste into tmux session: wjkim_1
# ============================================================
run_phi4() {
  python "${PY}" \
    --model "microsoft/phi-4" \
    --opinionqa_json "${OPINIONQA_JSON}" \
    --profile_dir "${PROFILE_DIR}" \
    --relevance_csv "${RELEVANCE_CSV}" \
    --output_root "${OUTPUT_ROOT}" \
    --theory pvq \
    --num_prompts 6 \
    --threshold 0.17 \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

# ============================================================
# BLOCK 2 — paste into tmux session: wjkim_2
# ============================================================
run_glm4() {
  python "${PY}" \
    --model "zai-org/GLM-4-32B-0414" \
    --opinionqa_json "${OPINIONQA_JSON}" \
    --profile_dir "${PROFILE_DIR}" \
    --relevance_csv "${RELEVANCE_CSV}" \
    --output_root "${OUTPUT_ROOT}" \
    --theory duty \
    --num_prompts 4 \
    --threshold 0.5 \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

# ============================================================
# Usage: run one function per session, e.g.:
#   source run_opinionqa_final.sh && run_qwen3
#   source run_opinionqa_final.sh && run_phi4
#   source run_opinionqa_final.sh && run_glm4
#
# Or call directly:
#   bash run_opinionqa_final.sh qwen3
#   bash run_opinionqa_final.sh phi4
#   bash run_opinionqa_final.sh glm4
# ============================================================
case "${1:-}" in
  qwen3) run_qwen3 ;;
  phi4)  run_phi4  ;;
  glm4)  run_glm4  ;;
  *)
    echo "Usage: bash $0 {qwen3|phi4|glm4}"
    echo ""
    echo "  qwen3 -> Qwen/Qwen3-32B         (theory=duty, n=4, thr=0.65)"
    echo "  phi4  -> microsoft/phi-4        (theory=pvq,  n=6, thr=0.17)"
    echo "  glm4  -> zai-org/GLM-4-32B-0414 (theory=duty, n=4, thr=0.5)"
    ;;
esac
