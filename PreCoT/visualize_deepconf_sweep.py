#!/usr/bin/env python3
"""
Visualize DeepConf SFT analysis JSON.

Input JSON is expected from:
`IntraAfterCoT/deepconf_modify/examples/analyze_sft_deepconf.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_online_tradeoff(online_rows: List[Dict[str, Any]], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib (and numpy) is required for plotting. "
            "Install with: pip install matplotlib numpy"
        ) from exc

    if not online_rows:
        print("No online sweep data. Skipping online plot.")
        return

    x_tokens = [row["mean_tokens"] for row in online_rows]
    y_acc = [row["accuracy"] for row in online_rows]
    y_ratio = [row["mean_token_ratio"] for row in online_rows]
    sweep_ids = [int(row["sweep_index"]) for row in online_rows]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("#f7f9fc")

    ax1.plot(x_tokens, y_acc, marker="o", linewidth=2.0, color="#0a84ff", label="Accuracy")
    ax1.set_xlabel("Mean Tokens per Question")
    ax1.set_ylabel("Accuracy", color="#0a84ff")
    ax1.tick_params(axis="y", labelcolor="#0a84ff")
    ax1.grid(True, alpha=0.25)

    for i, sid in enumerate(sweep_ids):
        ax1.annotate(str(sid), (x_tokens[i], y_acc[i]), fontsize=8, alpha=0.8)

    ax2 = ax1.twinx()
    ax2.plot(x_tokens, y_ratio, marker="s", linewidth=1.8, color="#ff5a5f", label="Token Ratio")
    ax2.set_ylabel("Mean Token Ratio (vs full budget)", color="#ff5a5f")
    ax2.tick_params(axis="y", labelcolor="#ff5a5f")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title("DeepConf-online Sweep: Token Usage vs Accuracy")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_offline_bar(offline_summary: Dict[str, Dict[str, float]], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib (and numpy) is required for plotting. "
            "Install with: pip install matplotlib numpy"
        ) from exc

    order = ["most_confidence", "top5_confidence", "top10_confidence"]
    methods = [m for m in order if m in offline_summary]
    if not methods:
        print("No offline confidence data. Skipping offline plot.")
        return

    accuracies = [offline_summary[m]["accuracy"] for m in methods]
    answer_rates = [offline_summary[m]["answer_rate"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")

    x = list(range(len(methods)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], accuracies, width=width, color="#34c759", label="Accuracy")
    ax.bar([i + width / 2 for i in x], answer_rates, width=width, color="#ff9f0a", label="Answer Rate")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("DeepConf-offline Confidence Metrics (256 from 320)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DeepConf SFT analysis output JSON")
    parser.add_argument("--analysis_json", required=True)
    parser.add_argument("--output_dir", default="figures_deepconf_sweep")
    args = parser.parse_args()

    analysis = load_json(Path(args.analysis_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    online_rows = analysis.get("online_sweep", {}).get("aggregate", [])
    offline_summary = analysis.get("offline_confidence", {}).get("aggregate", {})

    plot_online_tradeoff(online_rows, output_dir / "online_token_accuracy_tradeoff.png")
    plot_offline_bar(offline_summary, output_dir / "offline_confidence_methods.png")


if __name__ == "__main__":
    main()
