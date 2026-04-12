#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Single-model postprocess (offline PKLs only)
#
# Required inputs:
#   1) PKL_DIR: directory containing deepthink_offline_qid*_rid*.pkl
#   2) MODEL:   model path/name for Pre-CoT
#
# Outputs:
#   - Pass@K JSON
#   - DeepConf full JSON + dataset-view JSON
#   - Pre-CoT CSV artifacts
#   - Extra CLI-friendly metrics summary (JSON + TXT)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRECOT_DIR="${PROJECT_ROOT}/PreCoT"

# ----------------------------
# Required user config
# ----------------------------
PKL_DIR="${PKL_DIR:-}"
MODEL="${MODEL:-}"
TOKENIZER="${TOKENIZER:-$MODEL}"

# ----------------------------
# Runtime config
# ----------------------------
ENV_NAME="${ENV_NAME:-deepconf}"
DATASET="${DATASET:-${SCRIPT_DIR}/examples/aime_2024_convert.jsonl}"
PRECOT_DATA="${PRECOT_DATA:-${DATASET}}"
PRECOT_DEVICE="${PRECOT_DEVICE:-cuda:7}"
RID="${RID:-}"   # empty means no rid filter (analyze all matching pkls)

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/runs/postprocess_single_${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-${OUT_ROOT}/reports}"
FIG_DIR="${FIG_DIR:-${OUT_ROOT}/figures}"

PASSK_LIST="${PASSK_LIST:-1,4,8,16,32,64,128,256}"
ONLINE_WARMUP_TRACES="${ONLINE_WARMUP_TRACES:-32}"
ONLINE_TOTAL_BUDGET="${ONLINE_TOTAL_BUDGET:-320}"
ONLINE_SWEEP_POINTS="${ONLINE_SWEEP_POINTS:-20}"
ONLINE_RESAMPLES="${ONLINE_RESAMPLES:-5}"
OFFLINE_SAMPLE_SIZE="${OFFLINE_SAMPLE_SIZE:-256}"
OFFLINE_RESAMPLES="${OFFLINE_RESAMPLES:-5}"
ADAPTIVE_DIVISOR="${ADAPTIVE_DIVISOR:-5}"

if [[ -z "${PKL_DIR}" ]]; then
  echo "[ERROR] PKL_DIR is required."
  echo "Example:"
  echo "  PKL_DIR=/path/to/rl_offline MODEL=/path/to/model bash ${SCRIPT_DIR}/postprocess_single_model.sh"
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  echo "[ERROR] MODEL is required."
  echo "Example:"
  echo "  PKL_DIR=/path/to/rl_offline MODEL=/path/to/model bash ${SCRIPT_DIR}/postprocess_single_model.sh"
  exit 1
fi

if [[ ! -d "${PKL_DIR}" ]]; then
  echo "[ERROR] PKL_DIR does not exist: ${PKL_DIR}"
  exit 1
fi

shopt -s nullglob
PKL_FILES=("${PKL_DIR}"/*.pkl)
shopt -u nullglob
if [[ ${#PKL_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No .pkl files found in: ${PKL_DIR}"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${REPORT_DIR}" "${FIG_DIR}"

PASSK_JSON="${REPORT_DIR}/passk.json"
DEEP_FULL_JSON="${REPORT_DIR}/deepconf_full.json"
DEEP_DATASET_JSON="${REPORT_DIR}/deepconf_dataset_view.json"

PRECOT_POINTS_CSV="${FIG_DIR}/figure1_points.csv"
PRECOT_METRICS_CSV="${FIG_DIR}/figure1_metrics.csv"
PRECOT_TMP_PNG="${FIG_DIR}/.tmp_figure1.png"
PRECOT_TMP_CACHE="${FIG_DIR}/.tmp_cache.npz"

METRICS_JSON="${REPORT_DIR}/metrics_summary.json"
METRICS_TXT="${REPORT_DIR}/metrics_summary.txt"

echo "[INFO] PKL_DIR=${PKL_DIR}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] TOKENIZER=${TOKENIZER}"
if [[ -n "${RID}" ]]; then
  echo "[INFO] RID=${RID}"
else
  echo "[INFO] RID=<all>"
fi
echo "[INFO] REPORT_DIR=${REPORT_DIR}"
echo "[INFO] FIG_DIR=${FIG_DIR}"

# ----------------------------
# 1) Pass@K (JSON)
# ----------------------------
echo "[STEP] Computing Pass@K JSON..."
python "${SCRIPT_DIR}/compute_passk_from_pkls.py" \
  --inputs "${PKL_DIR}" \
  --ks "${PASSK_LIST}" \
  --include_empty_as_incorrect \
  --json_output "${PASSK_JSON}"

# ----------------------------
# 2) DeepConf analysis (JSON)
# ----------------------------
echo "[STEP] Running DeepConf offline/online analysis..."
ANALYZE_RID_ARGS=()
if [[ -n "${RID}" ]]; then
  ANALYZE_RID_ARGS=(--rid "${RID}")
fi

python "${SCRIPT_DIR}/examples/analyze_sft_deepconf.py" \
  --results_dir "${PKL_DIR}" \
  --dataset "${DATASET}" \
  "${ANALYZE_RID_ARGS[@]}" \
  --online_warmup_traces "${ONLINE_WARMUP_TRACES}" \
  --online_total_budget "${ONLINE_TOTAL_BUDGET}" \
  --online_sweep_points "${ONLINE_SWEEP_POINTS}" \
  --online_resamples "${ONLINE_RESAMPLES}" \
  --offline_sample_size "${OFFLINE_SAMPLE_SIZE}" \
  --offline_resamples "${OFFLINE_RESAMPLES}" \
  --adaptive_divisor "${ADAPTIVE_DIVISOR}" \
  --output_json "${DEEP_FULL_JSON}" \
  --dataset_output_json "${DEEP_DATASET_JSON}" \
  --keep_per_case

# ----------------------------
# 3) Pre-CoT metrics/points CSV
# ----------------------------
echo "[STEP] Running Pre-CoT (enabled)..."
python "${PRECOT_DIR}/figure1_first_token_logits.py" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --data "${PRECOT_DATA}" \
  --results_dir "${PKL_DIR}" \
  --output "${PRECOT_TMP_PNG}" \
  --points_output "${PRECOT_POINTS_CSV}" \
  --metrics_output "${PRECOT_METRICS_CSV}" \
  --cache "${PRECOT_TMP_CACHE}" \
  --device "${PRECOT_DEVICE}" \
  --recompute_logits

# keep only CSV files
rm -f "${PRECOT_TMP_PNG}" "${PRECOT_TMP_CACHE}"

# ----------------------------
# 4) Build CLI-friendly metrics summary
# ----------------------------
echo "[STEP] Building CLI-friendly metrics summary..."
python - <<PY
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def fmt_num(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return "NaN"
    try:
        fv = float(v)
    except Exception:
        return "NaN"
    return f"{fv:.{ndigits}f}"

def fmt_int(v: Any) -> str:
    try:
        return str(int(v))
    except Exception:
        return "0"

report_dir = Path(r"${REPORT_DIR}")
fig_dir = Path(r"${FIG_DIR}")
pkl_dir = Path(r"${PKL_DIR}")
model_name = r"${MODEL}"
rid = r"${RID}"

passk_path = Path(r"${PASSK_JSON}")
deep_dataset_path = Path(r"${DEEP_DATASET_JSON}")
precot_metrics_path = Path(r"${PRECOT_METRICS_CSV}")

passk = json.loads(passk_path.read_text(encoding="utf-8"))
deep = json.loads(deep_dataset_path.read_text(encoding="utf-8"))

passk_meta = passk.get("meta", {})
passk_results = passk.get("results", {})

def passk_keys_sorted(keys: List[str]) -> List[str]:
    def k_num(x: str) -> int:
        try:
            return int(str(x).split("@")[1])
        except Exception:
            return 10**9
    return sorted(keys, key=k_num)

sorted_passk_keys = passk_keys_sorted(list(passk_results.keys()))

online_points = deep.get("online_dataset_token_vs_accuracy", []) or []
best_online = None
if online_points:
    best_online = max(online_points, key=lambda x: to_float(x.get("accuracy"), -1.0))

offline_metrics = deep.get("offline_dataset_metrics", {}) or {}
online_summary_rows = [
    {
        "sweep_index": p.get("sweep_index"),
        "mean_threshold": p.get("threshold"),
        "mean_token_ratio": p.get("token_ratio"),
        "accuracy": p.get("accuracy"),
        "surviving_path_keep_rate": p.get("surviving_path_keep_rate"),
        "majority_vote_accuracy": p.get("majority_vote_accuracy"),
    }
    for p in online_points
]

offline_target_methods = [
    "majority_voting",
    "most_confidence",
    "top5_confidence_base256",
    "top10_confidence_base256",
]
offline_selected = []
for method in offline_target_methods:
    stats = offline_metrics.get(method)
    if not stats:
        continue
    offline_selected.append(
        {
            "method": method,
            "accuracy": stats.get("accuracy"),
            "answer_rate": stats.get("answer_rate"),
            "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
            "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
        }
    )

precot_rows: List[Dict[str, Any]] = []
if precot_metrics_path.exists():
    with precot_metrics_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        precot_rows = list(reader)

precot_overall = None
for row in precot_rows:
    if str(row.get("group", "")).strip() == "overall":
        precot_overall = row
        break
if precot_overall is None and precot_rows:
    precot_overall = precot_rows[0]

summary: Dict[str, Any] = {
    "inputs": {
        "pkl_dir": str(pkl_dir),
        "model": model_name,
        "rid": rid,
    },
    "passk": {
        "meta": passk_meta,
        "results": passk_results,
    },
    "deepconf": {
        "summary": deep.get("summary", {}),
        "online_summary": online_summary_rows,
        "best_online_point": best_online,
        "offline_summary": offline_selected,
    },
    "precot": {
        "metrics_csv": str(precot_metrics_path),
        "overall": precot_overall,
    },
}

metrics_json_path = Path(r"${METRICS_JSON}")
metrics_txt_path = Path(r"${METRICS_TXT}")
metrics_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines: List[str] = []
lines.append("============== Metrics Summary ==============")
lines.append(f"PKL_DIR: {pkl_dir}")
lines.append(f"MODEL:   {model_name}")
lines.append(f"RID:     {rid}")
lines.append("")
lines.append("[Pass@K]")
lines.append(
    "PKLs found: {found} | used: {used} | skipped: {skipped}".format(
        found=fmt_int(passk_meta.get("num_pkls_found")),
        used=fmt_int(passk_meta.get("num_questions_used")),
        skipped=fmt_int(passk_meta.get("num_questions_skipped")),
    )
)
passk_pairs = [f"{k}={fmt_num(passk_results.get(k))}" for k in sorted_passk_keys]
lines.append(" ".join(passk_pairs) if passk_pairs else "No pass@k values found.")
lines.append("")

deep_summary = deep.get("summary", {})
lines.append("[DeepConf-online summary]")
lines.append(
    "qids_loaded={loaded} qids_skipped={skipped} online_rows={orows} offline_rows={frows}".format(
        loaded=fmt_int(deep_summary.get("num_qids_loaded")),
        skipped=fmt_int(deep_summary.get("num_qids_skipped")),
        orows=fmt_int(deep_summary.get("online_rows")),
        frows=fmt_int(deep_summary.get("offline_rows")),
    )
)
if online_summary_rows:
    lines.append(
        "sweep_index mean_threshold mean_token_ratio accuracy surviving_path_keep_rate majority_vote_accuracy"
    )
    for row in online_summary_rows:
        lines.append(
            "{sweep} {thr} {tok} {acc} {keep} {mv}".format(
                sweep=fmt_int(row.get("sweep_index")),
                thr=fmt_num(row.get("mean_threshold")),
                tok=fmt_num(row.get("mean_token_ratio")),
                acc=fmt_num(row.get("accuracy")),
                keep=fmt_num(row.get("surviving_path_keep_rate")),
                mv=fmt_num(row.get("majority_vote_accuracy")),
            )
        )
else:
    lines.append("No online summary rows found.")

lines.append("")
lines.append("[DeepConf-offline summary]")
if offline_selected:
    for row in offline_selected:
        lines.append(
            "  {method}: acc={acc} trace_valid={tvalid} trace_base={tbase} answer_rate={ar}".format(
                method=row.get("method"),
                acc=fmt_num(row.get("accuracy")),
                tvalid=fmt_num(row.get("trace_accuracy_valid_only")),
                tbase=fmt_num(row.get("trace_accuracy_base_count")),
                ar=fmt_num(row.get("answer_rate")),
            )
        )
else:
    lines.append("No selected offline methods found.")
lines.append("")

lines.append("[Pre-CoT]")
if precot_overall:
    lines.append(
        "overall(count={cnt}) entropy_mean={em} entropy_std={es} dconf_mean={dm} dconf_std={ds}".format(
            cnt=fmt_int(precot_overall.get("count")),
            em=fmt_num(to_float(precot_overall.get("entropy_mean"), None)),
            es=fmt_num(to_float(precot_overall.get("entropy_std"), None)),
            dm=fmt_num(to_float(precot_overall.get("deepconf_confidence_mean"), None)),
            ds=fmt_num(to_float(precot_overall.get("deepconf_confidence_std"), None)),
        )
    )
else:
    lines.append("No Pre-CoT metrics rows found.")
lines.append("")
lines.append(f"metrics_json: {metrics_json_path}")
lines.append(f"metrics_txt:  {metrics_txt_path}")
lines.append("============================================")

text = "\n".join(lines) + "\n"
metrics_txt_path.write_text(text, encoding="utf-8")
print(text, end="")
PY

echo
echo "[DONE] Single-model postprocess finished."
echo "[INFO] Pass@K JSON: ${PASSK_JSON}"
echo "[INFO] DeepConf full JSON: ${DEEP_FULL_JSON}"
echo "[INFO] DeepConf dataset JSON: ${DEEP_DATASET_JSON}"
echo "[INFO] Pre-CoT points CSV: ${PRECOT_POINTS_CSV}"
echo "[INFO] Pre-CoT metrics CSV: ${PRECOT_METRICS_CSV}"
echo "[INFO] Metrics summary JSON: ${METRICS_JSON}"
echo "[INFO] Metrics summary TXT: ${METRICS_TXT}"
