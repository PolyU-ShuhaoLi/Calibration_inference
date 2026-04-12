#!/usr/bin/env bash
set -euo pipefail

QID_START="${QID_START:-0}"
QID_END="${QID_END:-29}"
RID="${RID:-range0_30}"
DATASET="${DATASET:-examples/aime_2024_convert.jsonl}"
MODEL="${MODEL:-/eds-storage/shuhaoli_calibration/LLaMA-Factory/saves/Qwen2.5-7B/Qwen3_think_2epochs_1e-5}"
TOKENIZER="${TOKENIZER:-$MODEL}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
BUDGET="${BUDGET:-320}"
OUTPUT_DIR="${OUTPUT_DIR:-sft_thinking_aime2024}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

mkdir -p "${OUTPUT_DIR}/logs"

for qid in $(seq "${QID_START}" "${QID_END}"); do
  echo "Running qid=${qid}..."
  python examples/example_offline_original.py \
    --qid "${qid}" \
    --rid "${RID}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --model_type "${MODEL_TYPE}" \
    --budget "${BUDGET}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/qid_$(printf '%03d' "${qid}").log"
done
