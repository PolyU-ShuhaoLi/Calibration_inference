#!/usr/bin/env python3
"""
collect_deepconf_labels.py

Collect is_correct labels from deepconf-offline pkl result files and write
them to a compact JSON file that visualize_logits.py can consume.

Usage:
    python collect_deepconf_labels.py \
        --results_dir outputs \
        --n_problems 15 \
        --output labels.json

The output JSON is a list of booleans indexed by qid (0-based):
    [true, false, true, ...]

If a qid has multiple pkl files (e.g. multiple runs), the label from the
file with the highest majority-vote confidence is used.  If a qid is
missing entirely, the entry is null and visualize_logits.py will skip that
problem (or fall back to the threshold heuristic if --hard is also given).
"""

import argparse
import glob
import json
import pickle
from pathlib import Path


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def best_is_correct(records: list[dict]) -> bool | None:
    """
    Given multiple run records for the same qid, pick the one whose
    evaluation entry has the highest majority-vote confidence, then return
    its is_correct flag.
    """
    best = None
    best_conf = -1.0
    for rec in records:
        evaluation = rec.get("evaluation") or {}
        majority_eval = evaluation.get("majority", {})
        conf = majority_eval.get("confidence") or 0.0
        is_correct = majority_eval.get("is_correct")
        if is_correct is None:
            # Try any method
            for method_eval in evaluation.values():
                if method_eval.get("is_correct") is not None:
                    is_correct = method_eval["is_correct"]
                    conf = method_eval.get("confidence") or 0.0
                    break
        if is_correct is not None and conf >= best_conf:
            best_conf = conf
            best = is_correct
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate deepconf-offline pkl results into a labels JSON file."
    )
    parser.add_argument("--results_dir", default="outputs",
                        help="Directory containing deepthink_offline_qid*.pkl files")
    parser.add_argument("--n_problems", type=int, required=True,
                        help="Total number of problems in the dataset (sets output list length)")
    parser.add_argument("--output", default="labels.json",
                        help="Path to write the output labels JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pkl_files = sorted(results_dir.glob("deepthink_offline_qid*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(
            f"No deepthink_offline_qid*.pkl files found in '{results_dir}'. "
            "Run example_offline.py first."
        )

    # Group records by qid
    by_qid: dict[int, list[dict]] = {}
    for pkl_path in pkl_files:
        rec = load_pkl(str(pkl_path))
        qid = rec.get("qid")
        if qid is None:
            print(f"  WARNING: no 'qid' field in {pkl_path.name} — skipping")
            continue
        by_qid.setdefault(qid, []).append(rec)

    # Build label list (null for missing qids)
    labels: list[bool | None] = [None] * args.n_problems
    for qid, records in by_qid.items():
        if qid >= args.n_problems:
            print(f"  WARNING: qid={qid} >= n_problems={args.n_problems} — skipping")
            continue
        labels[qid] = best_is_correct(records)

    n_correct  = sum(1 for v in labels if v is True)
    n_wrong    = sum(1 for v in labels if v is False)
    n_missing  = sum(1 for v in labels if v is None)
    print(f"Labels collected: correct={n_correct}, wrong={n_wrong}, missing={n_missing}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
