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
#   - Ratio breakdown tables (CSV + JSON) for numerator/denominator tracing
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

PASSK_LIST="${PASSK_LIST:-8,16,32,64,128,256}"
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

PASSK_RATIO_CSV="${REPORT_DIR}/passk_ratio_breakdown.csv"
DEEPCONF_ONLINE_RATIO_CSV="${REPORT_DIR}/deepconf_online_ratio_breakdown.csv"
DEEPCONF_OFFLINE_RATIO_CSV="${REPORT_DIR}/deepconf_offline_ratio_breakdown.csv"
PRECOT_RATIO_CSV="${REPORT_DIR}/precot_ratio_breakdown.csv"
RATIO_JSON="${REPORT_DIR}/ratio_breakdown.json"

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

def to_float_opt(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default

def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)

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

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

def add_ratio_row(
    rows: List[Dict[str, Any]],
    scope: str,
    item_id: str,
    metric: str,
    ratio_value: float,
    numerator: float,
    denominator: float,
    numerator_desc: str,
    denominator_desc: str,
    formula: str,
) -> None:
    rows.append(
        {
            "scope": scope,
            "item_id": item_id,
            "metric": metric,
            "ratio_value": float(ratio_value),
            "numerator": float(numerator),
            "denominator": float(denominator),
            "numerator_desc": numerator_desc,
            "denominator_desc": denominator_desc,
            "formula": formula,
        }
    )

report_dir = Path(r"${REPORT_DIR}")
fig_dir = Path(r"${FIG_DIR}")
pkl_dir = Path(r"${PKL_DIR}")
model_name = r"${MODEL}"
rid = r"${RID}"

passk_path = Path(r"${PASSK_JSON}")
deep_dataset_path = Path(r"${DEEP_DATASET_JSON}")
precot_metrics_path = Path(r"${PRECOT_METRICS_CSV}")
passk_ratio_csv_path = Path(r"${PASSK_RATIO_CSV}")
deepconf_online_ratio_csv_path = Path(r"${DEEPCONF_ONLINE_RATIO_CSV}")
deepconf_offline_ratio_csv_path = Path(r"${DEEPCONF_OFFLINE_RATIO_CSV}")
precot_ratio_csv_path = Path(r"${PRECOT_RATIO_CSV}")
ratio_json_path = Path(r"${RATIO_JSON}")

passk = json.loads(passk_path.read_text(encoding="utf-8"))
deep = json.loads(deep_dataset_path.read_text(encoding="utf-8"))

passk_meta = passk.get("meta", {})
passk_results = passk.get("results", {})
passk_per_k_details = passk_meta.get("per_k_details", {}) or {}

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
        "valid_answer_accuracy": p.get("valid_answer_accuracy"),
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
            "cases": stats.get("cases"),
            "correct_prediction_count": stats.get("correct_prediction_count"),
            "has_prediction_count": stats.get("has_prediction_count"),
            "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
            "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
            "valid_correct_trace_count_total": stats.get("valid_correct_trace_count_total"),
            "valid_answer_count_total": stats.get("valid_answer_count_total"),
            "base_correct_trace_count_total": stats.get("base_correct_trace_count_total"),
            "sample_size_used_total": stats.get("sample_size_used_total"),
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

ratio_fieldnames = [
    "scope",
    "item_id",
    "metric",
    "ratio_value",
    "numerator",
    "denominator",
    "numerator_desc",
    "denominator_desc",
    "formula",
]

passk_ratio_rows: List[Dict[str, Any]] = []
for k in sorted_passk_keys:
    detail = passk_per_k_details.get(k, {}) or {}
    numerator = to_float(detail.get("numerator_sum_pass_estimator"), 0.0)
    denominator = to_float(detail.get("denominator_question_count"), 0.0)
    ratio_value = to_float(passk_results.get(k), safe_ratio(numerator, denominator))
    add_ratio_row(
        rows=passk_ratio_rows,
        scope="passk",
        item_id=k,
        metric="macro_pass_at_k",
        ratio_value=ratio_value,
        numerator=numerator,
        denominator=denominator,
        numerator_desc="sum_of_per_question_pass_at_k_estimator",
        denominator_desc="num_questions_with_n_ge_k",
        formula="macro_pass@k = numerator / denominator",
    )

deepconf_online_ratio_rows: List[Dict[str, Any]] = []
for point in online_points:
    sweep_index = to_int(point.get("sweep_index"), -1)
    item_id = f"sweep_{sweep_index}"
    num_rows = float(to_int(point.get("num_rows"), 0))

    surviving_correct = float(to_int(point.get("surviving_correct_path_count"), 0))
    surviving_total = float(to_int(point.get("surviving_path_count"), 0))
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="accuracy",
        ratio_value=to_float(point.get("accuracy"), safe_ratio(surviving_correct, surviving_total)),
        numerator=surviving_correct,
        denominator=surviving_total,
        numerator_desc="surviving_correct_path_count",
        denominator_desc="surviving_path_count",
        formula="accuracy = numerator / denominator",
    )

    valid_correct = float(to_int(point.get("surviving_valid_correct_path_count"), 0))
    valid_total = float(to_int(point.get("surviving_valid_answer_count"), 0))
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="valid_answer_accuracy",
        ratio_value=to_float(point.get("valid_answer_accuracy"), safe_ratio(valid_correct, valid_total)),
        numerator=valid_correct,
        denominator=valid_total,
        numerator_desc="surviving_valid_correct_path_count",
        denominator_desc="surviving_valid_answer_count",
        formula="valid_answer_accuracy = numerator / denominator",
    )

    no_stop_correct = float(to_int(point.get("no_early_stop_correct_path_count"), 0))
    no_stop_total = float(to_int(point.get("no_early_stop_path_count"), 0))
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="no_early_stop_path_accuracy",
        ratio_value=to_float(point.get("no_early_stop_path_accuracy"), safe_ratio(no_stop_correct, no_stop_total)),
        numerator=no_stop_correct,
        denominator=no_stop_total,
        numerator_desc="no_early_stop_correct_path_count",
        denominator_desc="no_early_stop_path_count",
        formula="no_early_stop_path_accuracy = numerator / denominator",
    )

    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="surviving_path_keep_rate",
        ratio_value=to_float(point.get("surviving_path_keep_rate"), safe_ratio(surviving_total, no_stop_total)),
        numerator=surviving_total,
        denominator=no_stop_total,
        numerator_desc="surviving_path_count",
        denominator_desc="no_early_stop_path_count",
        formula="surviving_path_keep_rate = numerator / denominator",
    )

    majority_vote_correct = float(to_int(point.get("majority_vote_correct_count"), 0))
    majority_vote_prediction = float(to_int(point.get("majority_vote_prediction_count"), 0))
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="majority_vote_accuracy",
        ratio_value=to_float(point.get("majority_vote_accuracy"), safe_ratio(majority_vote_correct, num_rows)),
        numerator=majority_vote_correct,
        denominator=num_rows,
        numerator_desc="num_rows_with_correct_majority_vote",
        denominator_desc="num_rows",
        formula="majority_vote_accuracy = numerator / denominator",
    )
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="majority_vote_answer_rate",
        ratio_value=safe_ratio(majority_vote_prediction, num_rows),
        numerator=majority_vote_prediction,
        denominator=num_rows,
        numerator_desc="num_rows_with_majority_vote_prediction",
        denominator_desc="num_rows",
        formula="majority_vote_answer_rate = numerator / denominator",
    )

    mean_token_ratio = to_float(point.get("token_ratio"), 0.0)
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="mean_token_ratio",
        ratio_value=mean_token_ratio,
        numerator=(mean_token_ratio * num_rows),
        denominator=num_rows,
        numerator_desc="sum_of_per_row_token_ratio",
        denominator_desc="num_rows",
        formula="mean_token_ratio = numerator / denominator",
    )

    total_tokens_sum = float(to_int(point.get("total_tokens_sum"), 0))
    full_tokens_sum = float(to_int(point.get("full_tokens_sum"), 0))
    add_ratio_row(
        rows=deepconf_online_ratio_rows,
        scope="deepconf_online",
        item_id=item_id,
        metric="global_token_ratio_by_tokens",
        ratio_value=to_float(point.get("global_token_ratio_by_tokens"), safe_ratio(total_tokens_sum, full_tokens_sum)),
        numerator=total_tokens_sum,
        denominator=full_tokens_sum,
        numerator_desc="sum_total_tokens",
        denominator_desc="sum_full_tokens",
        formula="global_token_ratio_by_tokens = numerator / denominator",
    )

deepconf_offline_ratio_rows: List[Dict[str, Any]] = []
for method, stats in offline_metrics.items():
    item_id = str(method)
    cases = float(to_int(stats.get("cases"), 0))
    correct_prediction_count = float(to_int(stats.get("correct_prediction_count"), 0))
    has_prediction_count = float(to_int(stats.get("has_prediction_count"), 0))
    valid_correct_trace_count_total = float(to_int(stats.get("valid_correct_trace_count_total"), 0))
    valid_answer_count_total = float(to_int(stats.get("valid_answer_count_total"), 0))
    base_correct_trace_count_total = float(to_int(stats.get("base_correct_trace_count_total"), 0))
    sample_size_used_total = float(to_int(stats.get("sample_size_used_total"), 0))
    mean_valid_answer_ratio = to_float(stats.get("mean_valid_answer_ratio"), 0.0)

    add_ratio_row(
        rows=deepconf_offline_ratio_rows,
        scope="deepconf_offline",
        item_id=item_id,
        metric="accuracy",
        ratio_value=to_float(stats.get("accuracy"), safe_ratio(correct_prediction_count, cases)),
        numerator=correct_prediction_count,
        denominator=cases,
        numerator_desc="num_correct_predictions",
        denominator_desc="num_cases",
        formula="accuracy = numerator / denominator",
    )
    add_ratio_row(
        rows=deepconf_offline_ratio_rows,
        scope="deepconf_offline",
        item_id=item_id,
        metric="answer_rate",
        ratio_value=to_float(stats.get("answer_rate"), safe_ratio(has_prediction_count, cases)),
        numerator=has_prediction_count,
        denominator=cases,
        numerator_desc="num_cases_with_prediction",
        denominator_desc="num_cases",
        formula="answer_rate = numerator / denominator",
    )
    add_ratio_row(
        rows=deepconf_offline_ratio_rows,
        scope="deepconf_offline",
        item_id=item_id,
        metric="trace_accuracy_valid_only",
        ratio_value=to_float(stats.get("trace_accuracy_valid_only"), safe_ratio(valid_correct_trace_count_total, valid_answer_count_total)),
        numerator=valid_correct_trace_count_total,
        denominator=valid_answer_count_total,
        numerator_desc="valid_correct_trace_count_total",
        denominator_desc="valid_answer_count_total",
        formula="trace_accuracy_valid_only = numerator / denominator",
    )
    add_ratio_row(
        rows=deepconf_offline_ratio_rows,
        scope="deepconf_offline",
        item_id=item_id,
        metric="trace_accuracy_base_count",
        ratio_value=to_float(stats.get("trace_accuracy_base_count"), safe_ratio(base_correct_trace_count_total, sample_size_used_total)),
        numerator=base_correct_trace_count_total,
        denominator=sample_size_used_total,
        numerator_desc="base_correct_trace_count_total",
        denominator_desc="sample_size_used_total",
        formula="trace_accuracy_base_count = numerator / denominator",
    )
    add_ratio_row(
        rows=deepconf_offline_ratio_rows,
        scope="deepconf_offline",
        item_id=item_id,
        metric="mean_valid_answer_ratio",
        ratio_value=mean_valid_answer_ratio,
        numerator=mean_valid_answer_ratio * cases,
        denominator=cases,
        numerator_desc="sum_of_per_case_valid_answer_ratio",
        denominator_desc="num_cases",
        formula="mean_valid_answer_ratio = numerator / denominator",
    )

precot_ratio_rows: List[Dict[str, Any]] = []
overall_count = 0.0
if precot_overall is not None:
    overall_count = float(to_int(precot_overall.get("count"), 0))

for row in precot_rows:
    group = str(row.get("group", "")).strip()
    if not group:
        continue
    count = float(to_int(row.get("count"), 0))
    if overall_count <= 0:
        continue
    add_ratio_row(
        rows=precot_ratio_rows,
        scope="precot",
        item_id=group,
        metric="group_count_share",
        ratio_value=safe_ratio(count, overall_count),
        numerator=count,
        denominator=overall_count,
        numerator_desc=f"{group}_count",
        denominator_desc="overall_count",
        formula="group_count_share = numerator / denominator",
    )

write_csv(passk_ratio_csv_path, passk_ratio_rows, ratio_fieldnames)
write_csv(deepconf_online_ratio_csv_path, deepconf_online_ratio_rows, ratio_fieldnames)
write_csv(deepconf_offline_ratio_csv_path, deepconf_offline_ratio_rows, ratio_fieldnames)
write_csv(precot_ratio_csv_path, precot_ratio_rows, ratio_fieldnames)

ratio_bundle = {
    "paths": {
        "passk_ratio_csv": str(passk_ratio_csv_path),
        "deepconf_online_ratio_csv": str(deepconf_online_ratio_csv_path),
        "deepconf_offline_ratio_csv": str(deepconf_offline_ratio_csv_path),
        "precot_ratio_csv": str(precot_ratio_csv_path),
    },
    "rows": {
        "passk": passk_ratio_rows,
        "deepconf_online": deepconf_online_ratio_rows,
        "deepconf_offline": deepconf_offline_ratio_rows,
        "precot": precot_ratio_rows,
    },
}
ratio_json_path.write_text(json.dumps(ratio_bundle, indent=2), encoding="utf-8")

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
    "ratio_breakdown": {
        "ratio_json": str(ratio_json_path),
        "passk_ratio_csv": str(passk_ratio_csv_path),
        "deepconf_online_ratio_csv": str(deepconf_online_ratio_csv_path),
        "deepconf_offline_ratio_csv": str(deepconf_offline_ratio_csv_path),
        "precot_ratio_csv": str(precot_ratio_csv_path),
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
        "sweep_index mean_threshold mean_token_ratio accuracy valid_answer_accuracy surviving_path_keep_rate majority_vote_accuracy"
    )
    for row in online_summary_rows:
        lines.append(
            "{sweep} {thr} {tok} {acc} {vacc} {keep} {mv}".format(
                sweep=fmt_int(row.get("sweep_index")),
                thr=fmt_num(row.get("mean_threshold")),
                tok=fmt_num(row.get("mean_token_ratio")),
                acc=fmt_num(row.get("accuracy")),
                vacc=fmt_num(row.get("valid_answer_accuracy")),
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
            "  {method}: acc={acc}({acc_num}/{acc_den}) trace_valid={tvalid}({tv_num}/{tv_den}) "
            "trace_base={tbase}({tb_num}/{tb_den}) answer_rate={ar}({ar_num}/{ar_den})".format(
                method=row.get("method"),
                acc=fmt_num(row.get("accuracy")),
                acc_num=fmt_int(row.get("correct_prediction_count")),
                acc_den=fmt_int(row.get("cases")),
                tvalid=fmt_num(row.get("trace_accuracy_valid_only")),
                tv_num=fmt_int(row.get("valid_correct_trace_count_total")),
                tv_den=fmt_int(row.get("valid_answer_count_total")),
                tbase=fmt_num(row.get("trace_accuracy_base_count")),
                tb_num=fmt_int(row.get("base_correct_trace_count_total")),
                tb_den=fmt_int(row.get("sample_size_used_total")),
                ar=fmt_num(row.get("answer_rate")),
                ar_num=fmt_int(row.get("has_prediction_count")),
                ar_den=fmt_int(row.get("cases")),
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
            em=fmt_num(to_float_opt(precot_overall.get("entropy_mean"))),
            es=fmt_num(to_float_opt(precot_overall.get("entropy_std"))),
            dm=fmt_num(to_float_opt(precot_overall.get("deepconf_confidence_mean"))),
            ds=fmt_num(to_float_opt(precot_overall.get("deepconf_confidence_std"))),
        )
    )
else:
    lines.append("No Pre-CoT metrics rows found.")
lines.append("")
lines.append(f"metrics_json: {metrics_json_path}")
lines.append(f"metrics_txt:  {metrics_txt_path}")
lines.append(f"ratio_json:   {ratio_json_path}")
lines.append(f"passk_ratio_csv:            {passk_ratio_csv_path}")
lines.append(f"deepconf_online_ratio_csv:  {deepconf_online_ratio_csv_path}")
lines.append(f"deepconf_offline_ratio_csv: {deepconf_offline_ratio_csv_path}")
lines.append(f"precot_ratio_csv:           {precot_ratio_csv_path}")
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
echo "[INFO] Ratio breakdown JSON: ${RATIO_JSON}"
echo "[INFO] Pass@K ratio CSV: ${PASSK_RATIO_CSV}"
echo "[INFO] DeepConf online ratio CSV: ${DEEPCONF_ONLINE_RATIO_CSV}"
echo "[INFO] DeepConf offline ratio CSV: ${DEEPCONF_OFFLINE_RATIO_CSV}"
echo "[INFO] Pre-CoT ratio CSV: ${PRECOT_RATIO_CSV}"
