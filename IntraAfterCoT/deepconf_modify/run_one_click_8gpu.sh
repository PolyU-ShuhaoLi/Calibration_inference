#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# One-click pipeline on one node with 8 GPUs (H20)
#
# Includes:
# 1) Environment install (based on deepconf_environment.sh)
# 2) SFT and RL inference (based on sft_inference.sh)
# 3) Pass@K JSON export for both models
# 4) DeepConf offline/online analysis JSON for both models
# 5) Pre-CoT figure generation for both models
#
# IMPORTANT:
#   Fill SFT_MODEL and RL_MODEL before running.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRECOT_DIR="${PROJECT_ROOT}/PreCoT"

# ----------------------------
# User config (edit these)
# ----------------------------
SFT_MODEL="${SFT_MODEL:-}"   # e.g. /path/to/sft_model
RL_MODEL="${RL_MODEL:-}"     # e.g. /path/to/rl_model

# ----------------------------
# Runtime config
# ----------------------------
ENV_NAME="${ENV_NAME:-deepconf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_ENV="${INSTALL_ENV:-1}"          # 1: install/update env, 0: skip
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PARALLEL_INFERENCE="${PARALLEL_INFERENCE:-0}"   # 1: run SFT/RL inference in parallel

DATASET="${DATASET:-${SCRIPT_DIR}/examples/aime_2024_convert.jsonl}"
PRECOT_DATA="${PRECOT_DATA:-${DATASET}}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
QID_START="${QID_START:-0}"
QID_END="${QID_END:-29}"
BUDGET="${BUDGET:-320}"

ONLINE_WARMUP_TRACES="${ONLINE_WARMUP_TRACES:-32}"
ONLINE_TOTAL_BUDGET="${ONLINE_TOTAL_BUDGET:-320}"
ONLINE_SWEEP_POINTS="${ONLINE_SWEEP_POINTS:-16}"
ONLINE_RESAMPLES="${ONLINE_RESAMPLES:-1}"
OFFLINE_SAMPLE_SIZE="${OFFLINE_SAMPLE_SIZE:-256}"
OFFLINE_RESAMPLES="${OFFLINE_RESAMPLES:-10}"
ADAPTIVE_DIVISOR="${ADAPTIVE_DIVISOR:-10}"
PASSK_LIST="${PASSK_LIST:-1,4,8,16,32,64,128,256}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/full_${RUN_TAG}}"
SFT_OUT_DIR="${SFT_OUT_DIR:-${RUN_ROOT}/sft_offline}"
RL_OUT_DIR="${RL_OUT_DIR:-${RUN_ROOT}/rl_offline}"
REPORT_DIR="${REPORT_DIR:-${RUN_ROOT}/reports}"
FIG_DIR="${FIG_DIR:-${RUN_ROOT}/figures}"

mkdir -p "${RUN_ROOT}" "${SFT_OUT_DIR}" "${RL_OUT_DIR}" "${REPORT_DIR}" "${FIG_DIR}"

if [[ -z "${SFT_MODEL}" || -z "${RL_MODEL}" ]]; then
  echo "[ERROR] SFT_MODEL and RL_MODEL must be set."
  echo "Example:"
  echo "  SFT_MODEL=/path/to/sft_model RL_MODEL=/path/to/rl_model bash ${SCRIPT_DIR}/run_one_click_8gpu.sh"
  exit 1
fi

# ----------------------------
# Conda setup
# ----------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${INSTALL_ENV}" == "1" ]]; then
  echo "[INFO] Installing/updating environment (based on deepconf_environment.sh)..."
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_NAME}"

  python -m pip install --upgrade pip setuptools wheel uv
  uv pip install "vllm==0.10.2" --torch-backend=auto
  uv pip install "git+https://bgithub.xyz/hao-ai-lab/Dynasor.git"
  uv pip install -e "${SCRIPT_DIR}"
  python -m pip uninstall -y transformers || true
  python -m pip install "transformers<5" scikit-learn accelerate matplotlib numpy scipy
else
  echo "[INFO] Skipping environment installation (INSTALL_ENV=${INSTALL_ENV})"
  conda activate "${ENV_NAME}"
fi

# ----------------------------
# 1) SFT inference
# ----------------------------
if [[ "${PARALLEL_INFERENCE}" == "1" ]]; then
  echo "[STEP] Running SFT/RL inference in parallel..."

  IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#GPU_ARR[@]}"
  HALF_COUNT=$((GPU_COUNT / 2))
  if (( HALF_COUNT < 1 )); then
    echo "[ERROR] PARALLEL_INFERENCE=1 but CUDA_VISIBLE_DEVICES has insufficient GPUs: ${CUDA_VISIBLE_DEVICES}"
    exit 1
  fi

  SFT_GPU_LIST="$(IFS=,; echo "${GPU_ARR[*]:0:HALF_COUNT}")"
  RL_GPU_LIST="$(IFS=,; echo "${GPU_ARR[*]:HALF_COUNT}")"
  if [[ -z "${RL_GPU_LIST}" ]]; then
    echo "[ERROR] Failed to split GPUs for parallel inference. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exit 1
  fi

  echo "[INFO] SFT GPUs: ${SFT_GPU_LIST}"
  echo "[INFO] RL  GPUs: ${RL_GPU_LIST}"
  echo "[INFO] TP size per job: ${TENSOR_PARALLEL_SIZE}"

  MODEL="${SFT_MODEL}" \
  DATASET="${DATASET}" \
  OUTPUT_DIR="${SFT_OUT_DIR}" \
  RID="sft" \
  MODEL_TYPE="${MODEL_TYPE}" \
  QID_START="${QID_START}" \
  QID_END="${QID_END}" \
  BUDGET="${BUDGET}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  CUDA_VISIBLE_DEVICES="${SFT_GPU_LIST}" \
  bash "${SCRIPT_DIR}/run_sft_model_inference.sh" &
  SFT_PID=$!

  MODEL="${RL_MODEL}" \
  DATASET="${DATASET}" \
  OUTPUT_DIR="${RL_OUT_DIR}" \
  RID="rl" \
  MODEL_TYPE="${MODEL_TYPE}" \
  QID_START="${QID_START}" \
  QID_END="${QID_END}" \
  BUDGET="${BUDGET}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  CUDA_VISIBLE_DEVICES="${RL_GPU_LIST}" \
  bash "${SCRIPT_DIR}/run_rl_model_inference.sh" &
  RL_PID=$!

  wait "${SFT_PID}"
  wait "${RL_PID}"
else
  # ----------------------------
  # 1) SFT inference
  # ----------------------------
  echo "[STEP] Running SFT inference..."
  MODEL="${SFT_MODEL}" \
  DATASET="${DATASET}" \
  OUTPUT_DIR="${SFT_OUT_DIR}" \
  RID="sft" \
  MODEL_TYPE="${MODEL_TYPE}" \
  QID_START="${QID_START}" \
  QID_END="${QID_END}" \
  BUDGET="${BUDGET}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  bash "${SCRIPT_DIR}/run_sft_model_inference.sh"

  # ----------------------------
  # 2) RL inference
  # ----------------------------
  echo "[STEP] Running RL inference..."
  MODEL="${RL_MODEL}" \
  DATASET="${DATASET}" \
  OUTPUT_DIR="${RL_OUT_DIR}" \
  RID="rl" \
  MODEL_TYPE="${MODEL_TYPE}" \
  QID_START="${QID_START}" \
  QID_END="${QID_END}" \
  BUDGET="${BUDGET}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  bash "${SCRIPT_DIR}/run_rl_model_inference.sh"
fi

# ----------------------------
# 3) Pass@K (JSON)
# ----------------------------
echo "[STEP] Computing Pass@K JSON..."
python "${SCRIPT_DIR}/compute_passk_from_pkls.py" \
  --inputs "${SFT_OUT_DIR}" \
  --ks "${PASSK_LIST}" \
  --include_empty_as_incorrect \
  --json_output "${REPORT_DIR}/passk_sft.json"

python "${SCRIPT_DIR}/compute_passk_from_pkls.py" \
  --inputs "${RL_OUT_DIR}" \
  --ks "${PASSK_LIST}" \
  --include_empty_as_incorrect \
  --json_output "${REPORT_DIR}/passk_rl.json"

python - <<PY
import json
from pathlib import Path
report_dir = Path(r"${REPORT_DIR}")
sft = json.loads((report_dir / "passk_sft.json").read_text(encoding="utf-8"))
rl = json.loads((report_dir / "passk_rl.json").read_text(encoding="utf-8"))
summary = {"sft": sft, "rl": rl}
(report_dir / "passk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[INFO] Wrote: {report_dir / 'passk_summary.json'}")
PY

# ----------------------------
# 4) DeepConf offline/online analysis
# ----------------------------
echo "[STEP] Running DeepConf analysis..."
python "${SCRIPT_DIR}/examples/analyze_sft_deepconf.py" \
  --results_dir "${SFT_OUT_DIR}" \
  --dataset "${DATASET}" \
  --rid "sft" \
  --online_warmup_traces "${ONLINE_WARMUP_TRACES}" \
  --online_total_budget "${ONLINE_TOTAL_BUDGET}" \
  --online_sweep_points "${ONLINE_SWEEP_POINTS}" \
  --online_resamples "${ONLINE_RESAMPLES}" \
  --offline_sample_size "${OFFLINE_SAMPLE_SIZE}" \
  --offline_resamples "${OFFLINE_RESAMPLES}" \
  --adaptive_divisor "${ADAPTIVE_DIVISOR}" \
  --output_json "${REPORT_DIR}/deepconf_sft.json"

python "${SCRIPT_DIR}/examples/analyze_sft_deepconf.py" \
  --results_dir "${RL_OUT_DIR}" \
  --dataset "${DATASET}" \
  --rid "rl" \
  --online_warmup_traces "${ONLINE_WARMUP_TRACES}" \
  --online_total_budget "${ONLINE_TOTAL_BUDGET}" \
  --online_sweep_points "${ONLINE_SWEEP_POINTS}" \
  --online_resamples "${ONLINE_RESAMPLES}" \
  --offline_sample_size "${OFFLINE_SAMPLE_SIZE}" \
  --offline_resamples "${OFFLINE_RESAMPLES}" \
  --adaptive_divisor "${ADAPTIVE_DIVISOR}" \
  --output_json "${REPORT_DIR}/deepconf_rl.json"

# ----------------------------
# 5) Pre-CoT figure generation
# ----------------------------
echo "[STEP] Running Pre-CoT figure generation..."
python "${PRECOT_DIR}/figure1_first_token_logits.py" \
  --model "${SFT_MODEL}" \
  --data "${PRECOT_DATA}" \
  --results_dir "${SFT_OUT_DIR}" \
  --output "${FIG_DIR}/figure1_sft.png" \
  --cache "${FIG_DIR}/cache_sft.npz" \
  --device cuda

python "${PRECOT_DIR}/figure1_first_token_logits.py" \
  --model "${RL_MODEL}" \
  --data "${PRECOT_DATA}" \
  --results_dir "${RL_OUT_DIR}" \
  --output "${FIG_DIR}/figure1_rl.png" \
  --cache "${FIG_DIR}/cache_rl.npz" \
  --device cuda

echo
echo "[DONE] Full one-click pipeline finished."
echo "[INFO] Run root: ${RUN_ROOT}"
echo "[INFO] Pass@K summary: ${REPORT_DIR}/passk_summary.json"
echo "[INFO] DeepConf results: ${REPORT_DIR}/deepconf_sft.json and ${REPORT_DIR}/deepconf_rl.json"
echo "[INFO] Pre-CoT figures: ${FIG_DIR}/figure1_sft.png and ${FIG_DIR}/figure1_rl.png"
