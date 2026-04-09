#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Re-run postprocess steps only (no inference):
#   3) Pass@K JSON
#   4) DeepConf offline/online analysis (RL only)
#   5) Pre-CoT figure generation (RL only)
#
# This script assumes inference outputs already exist:
#   <RUN_ROOT>/rl_offline
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRECOT_DIR="${PROJECT_ROOT}/PreCoT"

# ----------------------------
# Required user config
# ----------------------------
RL_MODEL="${RL_MODEL:-}"     # required for Pre-CoT step

# ----------------------------
# Runtime config
# ----------------------------
ENV_NAME="${ENV_NAME:-deepconf}"
DATASET="${DATASET:-${SCRIPT_DIR}/examples/aime_2024_convert.jsonl}"
PRECOT_DATA="${PRECOT_DATA:-${DATASET}}"
PRECOT_DEVICE="${PRECOT_DEVICE:-cuda:0}"

RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "${RUN_ROOT}" ]]; then
  if ls -d "${SCRIPT_DIR}"/runs/full_* >/dev/null 2>&1; then
    RUN_ROOT="$(ls -dt "${SCRIPT_DIR}"/runs/full_* | head -n 1)"
    echo "[INFO] RUN_ROOT not provided. Auto-selected latest: ${RUN_ROOT}"
  else
    echo "[ERROR] RUN_ROOT is empty and no ${SCRIPT_DIR}/runs/full_* found."
    exit 1
  fi
fi

RL_OUT_DIR="${RL_OUT_DIR:-${RUN_ROOT}/rl_offline}"

RERUN_TAG="${RERUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-${RUN_ROOT}/reports_rerun_${RERUN_TAG}}"
FIG_DIR="${FIG_DIR:-${RUN_ROOT}/figures_rerun_${RERUN_TAG}}"

PASSK_LIST="${PASSK_LIST:-1,4,8,16,32,64,128,256}"

ONLINE_WARMUP_TRACES="${ONLINE_WARMUP_TRACES:-32}"
ONLINE_TOTAL_BUDGET="${ONLINE_TOTAL_BUDGET:-320}"
ONLINE_SWEEP_POINTS="${ONLINE_SWEEP_POINTS:-16}"
ONLINE_RESAMPLES="${ONLINE_RESAMPLES:-5}"
OFFLINE_SAMPLE_SIZE="${OFFLINE_SAMPLE_SIZE:-256}"
OFFLINE_RESAMPLES="${OFFLINE_RESAMPLES:-10}"
ADAPTIVE_DIVISOR="${ADAPTIVE_DIVISOR:-5}"

mkdir -p "${REPORT_DIR}" "${FIG_DIR}"

if [[ ! -d "${RL_OUT_DIR}" ]]; then
  echo "[ERROR] Missing inference output dirs."
  echo "RL_OUT_DIR=${RL_OUT_DIR}"
  exit 1
fi

if [[ -z "${RL_MODEL}" ]]; then
  echo "[ERROR] RL_MODEL is required for Pre-CoT re-run."
  echo "Example:"
  echo "  RUN_ROOT=/path/to/full_run RL_MODEL=/path/to/rl bash ${SCRIPT_DIR}/postprocess.sh"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] REPORT_DIR=${REPORT_DIR}"
echo "[INFO] FIG_DIR=${FIG_DIR}"
echo "[INFO] RL_OUT_DIR=${RL_OUT_DIR}"

# ----------------------------
# 3) Pass@K (JSON)
# ----------------------------
echo "[STEP] Re-running Pass@K JSON (RL only)..."
python "${SCRIPT_DIR}/compute_passk_from_pkls.py" \
  --inputs "${RL_OUT_DIR}" \
  --ks "${PASSK_LIST}" \
  --include_empty_as_incorrect \
  --json_output "${REPORT_DIR}/passk_rl.json"

python - <<PY
import json
from pathlib import Path
report_dir = Path(r"${REPORT_DIR}")
rl = json.loads((report_dir / "passk_rl.json").read_text(encoding="utf-8"))
summary = {"rl": rl}
(report_dir / "passk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[INFO] Wrote: {report_dir / 'passk_summary.json'}")
PY

# ----------------------------
# 4) DeepConf offline/online analysis
# ----------------------------
echo "[STEP] Re-running DeepConf analysis (RL only)..."
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
  --output_json "${REPORT_DIR}/deepconf_rl.json" \
  --dataset_output_json "${REPORT_DIR}/deepconf_rl_dataset_view.json" \
  --keep_per_case

# ----------------------------
# 5) Pre-CoT figure generation
# ----------------------------
echo "[STEP] Re-running Pre-CoT (CSV only, RL only)..."

RL_POINTS_CSV="${FIG_DIR}/figure1_rl_points.csv"
RL_METRICS_CSV="${FIG_DIR}/figure1_rl_metrics.csv"
RL_TMP_PNG="${FIG_DIR}/.tmp_figure1_rl.png"
RL_TMP_CACHE="${FIG_DIR}/.tmp_cache_rl.npz"

python "${PRECOT_DIR}/figure1_first_token_logits.py" \
  --model "${RL_MODEL}" \
  --data "${PRECOT_DATA}" \
  --results_dir "${RL_OUT_DIR}" \
  --output "${RL_TMP_PNG}" \
  --points_output "${RL_POINTS_CSV}" \
  --metrics_output "${RL_METRICS_CSV}" \
  --cache "${RL_TMP_CACHE}" \
  --device "${PRECOT_DEVICE}" \
  --recompute_logits

# Keep only CSV files for Pre-CoT outputs.
rm -f "${RL_TMP_PNG}" "${RL_TMP_CACHE}"

echo
echo "[DONE] Postprocess re-run finished."
echo "[INFO] Pass@K summary: ${REPORT_DIR}/passk_summary.json"
echo "[INFO] DeepConf: ${REPORT_DIR}/deepconf_rl.json"
echo "[INFO] Pre-CoT points CSV: ${RL_POINTS_CSV}"
echo "[INFO] Pre-CoT metrics CSV: ${RL_METRICS_CSV}"
