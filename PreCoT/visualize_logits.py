#!/usr/bin/env python3
"""
visualize_logits.py

For each problem in aime2025.jsonl:
  1. Build the prompt with the model's chat template.
  2. Run a single forward pass (no generation).
  3. Extract logits[0, -1, :] — the distribution over the first response token.
  4. Represent each problem as v_i = log_softmax(z_i / T).
  5. PCA -> 50 dims -> t-SNE -> 2 dims.
  6. Label problems either:
       (a) [default] by deepconf-offline is_correct labels (--labels <labels.json>)
       (b) [fallback] by a threshold on the numeric answer value (--hard <int>)
  7. Scatter plot + KDE density contours for each class.

Workflow with deepconf-offline:
  Step 1 — run deepconf offline inference per problem (example_offline.py)
  Step 2 — collect is_correct labels:
               python collect_deepconf_labels.py --results_dir outputs \
                   --n_problems <N> --output labels.json
  Step 3 — run this script:
               python visualize_logits.py --labels labels.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # change to whatever you use
DEFAULT_DATA  = Path(__file__).parent / "aime2025.jsonl"
DEFAULT_T     = 1.0          # logit temperature before log_softmax
DEFAULT_HARD  = 500          # answers >= this threshold -> 'hard'  (fallback only)
SYSTEM_PROMPT = (
    "You are a helpful math competition assistant. "
    "Solve the problem step by step and give only the final integer answer."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_problems(path: Path):
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def build_prompt(tokenizer, question: str) -> str:
    """Build a chat-template prompt identical to what you'd use at eval time."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def extract_logits(
    model,
    tokenizer,
    prompts: list[str],
    temperature: float,
    device: str,
    batch_size: int = 1,
) -> np.ndarray:
    """Return log_softmax(z / T) for the last input position, shape (N, vocab)."""
    all_reps = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        out = model(**enc)
        # out.logits: (B, seq_len, vocab)
        seq_lens = enc["attention_mask"].sum(dim=1)  # (B,)
        for b in range(len(batch)):
            last = seq_lens[b].item() - 1
            z = out.logits[b, last, :].float()          # (vocab,)
            v = torch.log_softmax(z / temperature, dim=-1)
            all_reps.append(v.cpu().numpy())

        print(f"  [{i + len(batch):>3}/{len(prompts)}] done", end="\r", flush=True)

    print()
    return np.array(all_reps)   # (N, vocab)


# ---------------------------------------------------------------------------
# Labelling — two strategies
# ---------------------------------------------------------------------------

def load_labels_from_file(
    path: Path,
    n_problems: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load is_correct labels produced by collect_deepconf_labels.py.

    Returns
    -------
    labels   : int array of length n_problems, where
                 0 = model answered correctly
                 1 = model answered incorrectly
               Problems whose label is null/None are excluded.
    keep_idx : indices of problems that have a valid label
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if len(raw) != n_problems:
        raise ValueError(
            f"labels file has {len(raw)} entries but dataset has {n_problems} problems."
        )

    keep_idx = np.array([i for i, v in enumerate(raw) if v is not None], dtype=int)
    # is_correct True -> 0 (correct), False -> 1 (incorrect)
    labels = np.array([0 if raw[i] else 1 for i in keep_idx], dtype=int)
    return labels, keep_idx


def label_problems_by_threshold(
    problems: list[dict],
    hard_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback labelling: numeric answer >= hard_threshold -> 1 (hard), else 0 (easy).
    All problems are kept (no filtering).
    """
    labels = []
    for p in problems:
        ans = p.get("answer", "")
        try:
            val = int(str(ans).strip())
            labels.append(1 if val >= hard_threshold else 0)
        except ValueError:
            labels.append(1 if len(str(ans)) > 3 else 0)
    keep_idx = np.arange(len(problems), dtype=int)
    return np.array(labels, dtype=int), keep_idx


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    label_mode: str = "correctness",  # "correctness" | "difficulty"
):
    """Scatter plot + per-class KDE contours."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    if label_mode == "correctness":
        palette = {
            0: {"color": "#4fc3f7", "label": "correct"},
            1: {"color": "#f06292", "label": "incorrect"}}
        title_suffix = "(deepconf-offline correctness label)"
    else:
        palette = {
            0: {"color": "#4fc3f7", "label": "easy"},
            1: {"color": "#f06292", "label": "hard"}}
        title_suffix = "(answer-threshold label)"

    for cls, style in palette.items():
        mask = labels == cls
        xy = coords[mask]
        if xy.shape[0] < 2:
            continue

        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=style["color"],
            s=55,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            label=style["label"],
            zorder=3,
        )

        try:
            kde = gaussian_kde(xy.T, bw_method="scott")
            xmin, xmax = coords[:, 0].min() - 5, coords[:, 0].max() + 5
            ymin, ymax = coords[:, 1].min() - 5, coords[:, 1].max() + 5
            xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(
                xx, yy, zz,
                levels=5,
                colors=style["color"],
                alpha=0.55,
                linewidths=0.9,
                zorder=2,
            )
        except np.linalg.LinAlgError:
            pass  # singular KDE -- skip contour

    ax.legend(
        handles=[
            mpatches.Patch(color=v["color"], label=v["label"])
            for v in palette.values()
        ],
        framealpha=0.25,
        labelcolor="white",
        facecolor="#1a1a2e",
        edgecolor="#444",
        fontsize=11,
    )

    ax.set_title(
        "t-SNE of first-token logits  (AIME 2025)  " + title_suffix,
        color="white", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_xlabel("t-SNE dim 1", color="#aaa")
    ax.set_ylabel("t-SNE dim 2", color="#aaa")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print("Saved: " + str(save_path))
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize first-token logits for AIME problems."
    )
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="HF model name or local path")
    parser.add_argument("--data",        default=str(DEFAULT_DATA),
                        help="Path to .jsonl file")
    parser.add_argument("--labels",      default=None,
                        help="Path to labels.json from collect_deepconf_labels.py. "
                             "When provided, is_correct flags are used instead of --hard.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_T,
                        help="Logit temperature T")
    parser.add_argument("--hard",        type=int,   default=DEFAULT_HARD,
                        help="(fallback) Answer threshold: >= this -> hard. "
                             "Used only when --labels is not given.")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--pca-dim",     type=int,   default=50)
    parser.add_argument("--tsne-perp",   type=int,   default=10,
                        help="t-SNE perplexity (keep < N/3)")
    parser.add_argument("--output",      default="tsne_logits.png")
    parser.add_argument("--cache",       default="logits_cache.npy",
                        help="Cache extracted logits to avoid re-running the model")
    args = parser.parse_args()

    data_path  = Path(args.data)
    cache_path = Path(args.cache)
    out_path   = Path(args.output)

    # ---- load problems ----
    print("Loading problems from " + str(data_path) + " ...")
    problems = load_problems(data_path)
    print("  " + str(len(problems)) + " problems loaded.")

    # ---- determine labels ----
    if args.labels is not None:
        labels_path = Path(args.labels)
        print("Loading deepconf-offline labels from " + str(labels_path) + " ...")
        labels, keep_idx = load_labels_from_file(labels_path, len(problems))
        label_mode = "correctness"
        n_correct   = (labels == 0).sum()
        n_incorrect = (labels == 1).sum()
        n_missing   = len(problems) - len(keep_idx)
        print("  correct=" + str(n_correct) + "  incorrect=" + str(n_incorrect)
              + "  missing/skipped=" + str(n_missing))
    else:
        print("No --labels file given; falling back to answer-threshold (--hard " + str(args.hard) + ") ...")
        labels, keep_idx = label_problems_by_threshold(problems, args.hard)
        label_mode = "difficulty"
        n_easy = (labels == 0).sum()
        n_hard = (labels == 1).sum()
        print("  easy=" + str(n_easy) + "  hard=" + str(n_hard) + "  (threshold=" + str(args.hard) + ")")

    # ---- extract or load logits ----
    if cache_path.exists():
        print("Loading cached logits from " + str(cache_path) + " ...")
        reps = np.load(cache_path)
        if reps.shape[0] != len(problems):
            raise AssertionError(
                "Cache has " + str(reps.shape[0]) + " rows but dataset has "
                + str(len(problems)) + " problems -- delete the cache and rerun."
            )
    else:
        print("Loading model '" + args.model + "' on " + args.device + " ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
        )
        model.eval()

        prompts = [build_prompt(tokenizer, p["question"]) for p in problems]
        print("Extracting logits ...")
        reps = extract_logits(
            model, tokenizer, prompts,
            temperature=args.temperature,
            device=args.device,
            batch_size=args.batch_size,
        )
        np.save(cache_path, reps)
        print("Cached logits -> " + str(cache_path))

    # Apply keep_idx: select only the problems that have a valid label
    reps = reps[keep_idx]          # (K, vocab)
    print("Representation matrix after filtering: " + str(reps.shape))

    # ---- PCA ----
    n_pca = min(args.pca_dim, reps.shape[0] - 1, reps.shape[1])
    print("PCA: " + str(reps.shape[1]) + " -> " + str(n_pca) + " dims ...")
    pca = PCA(n_components=n_pca, random_state=42)
    reps_pca = pca.fit_transform(reps)
    print("  Explained variance (top " + str(n_pca) + "): "
          + str(round(float(pca.explained_variance_ratio_.sum()), 3)))

    # ---- t-SNE ----
    perplexity = min(args.tsne_perp, len(keep_idx) - 1)
    print("t-SNE: " + str(n_pca) + " -> 2 dims  (perplexity=" + str(perplexity) + ") ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=2000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reps_pca)

    # ---- plot ----
    plot_tsne(coords, labels, out_path, label_mode=label_mode)


if __name__ == "__main__":
    main()


    