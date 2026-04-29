#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  echo "[ERROR] MODEL is empty. Please set your SFT model path."
  echo "Example: MODEL=/path/to/sft_model bash ${SCRIPT_DIR}/run_sft_model_inference.sh"
  exit 1
fi
TOKENIZER="${TOKENIZER:-$MODEL}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-${SCRIPT_DIR}/examples/aime_2024_convert.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/full_${RUN_TAG}/sft_offline}"
RID="${RID:-sft}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
QID_START="${QID_START:-6}"
QID_END="${QID_END:-11}"
BUDGET="${BUDGET:-320}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Running SFT inference..."
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] TOKENIZER=${TOKENIZER}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MODEL="${MODEL}" \
TOKENIZER="${TOKENIZER}" \
DATASET="${DATASET}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RID="${RID}" \
MODEL_TYPE="${MODEL_TYPE}" \
QID_START="${QID_START}" \
QID_END="${QID_END}" \
BUDGET="${BUDGET}" \
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
bash "${SCRIPT_DIR}/sft_inference.sh"

echo "[DONE] SFT inference finished."
