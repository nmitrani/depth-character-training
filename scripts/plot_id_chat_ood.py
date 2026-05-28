"""Bar plot of per-persona classifier macro-F1: ID vs chat_email vs agentic_email.

This is the figure that answers the comment on LW: by holding the instruction
"send the email" constant and varying only the scaffold (chat-form vs.
agentic-tool-loop), the chat_email bars isolate the "email body format" effect
from the "agentic wrapper" effect.

Reads:
  - results/classifier_f1_per_email.jsonl              (agentic OOD)
  - results/classifier_f1_id_per_response.jsonl        (ID)
  - results/classifier_f1_chat_email_per_email.jsonl   (chat-form control)

Renders:
  - results/plots/id_chat_ood_f1.png        (3x3 facet grid, per-persona)
  - results/plots/id_chat_ood_summary.png   (one row per base, full stage)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OOD_PATH = REPO_ROOT / "results" / "classifier_f1_per_email.jsonl"
ID_PATH = REPO_ROOT / "results" / "classifier_f1_id_per_response.jsonl"
CHAT_PATH = REPO_ROOT / "results" / "classifier_f1_chat_email_per_email.jsonl"
OUT_PATH = REPO_ROOT / "results" / "plots" / "id_chat_ood_f1.png"
SUMMARY_PATH = REPO_ROOT / "results" / "plots" / "id_chat_ood_summary.png"

PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]
BASES = ["llama", "qwen", "gemma"]
BASE_LABEL = {"llama": "Llama", "qwen": "Qwen", "gemma": "Gemma"}
STAGES = ["base", "distillation", "full"]
N_BOOT = 1000
RNG = np.random.default_rng(0)

COLOR_ID = "#4C72B0"     # blue
COLOR_CHAT = "#55A868"   # green — chat-form email (the new control)
COLOR_OOD = "#DD8452"    # orange


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def f1_from_pairs(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    t = (y_true == label)
    p = (y_pred == label)
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    if tp + fp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bootstrap_per_persona_f1(rows: list[dict], n_boot: int = N_BOOT) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float]]:
    if not rows:
        return ({p: (float("nan"),) * 3 for p in PERSONAS}, (float("nan"),) * 3)
    y_true = np.array([r["trait_true"] for r in rows])
    y_pred = np.array([r["trait_pred"] for r in rows])
    n = len(y_true)
    boot_per: dict[str, list[float]] = {p: [] for p in PERSONAS}
    boot_avg: list[float] = []
    point_per = {p: f1_from_pairs(y_true, y_pred, p) for p in PERSONAS}
    pp_clean = [v for v in point_per.values() if not np.isnan(v)]
    point_avg = float(np.mean(pp_clean)) if pp_clean else float("nan")
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yt = y_true[idx]; yp = y_pred[idx]
        per = {p: f1_from_pairs(yt, yp, p) for p in PERSONAS}
        for p in PERSONAS:
            boot_per[p].append(per[p])
        clean = [v for v in per.values() if not np.isnan(v)]
        boot_avg.append(float(np.mean(clean)) if clean else float("nan"))
    out_per: dict[str, tuple[float, float, float]] = {}
    for p in PERSONAS:
        vals = np.array([v for v in boot_per[p] if not np.isnan(v)])
        if len(vals) == 0:
            out_per[p] = (float("nan"),) * 3
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out_per[p] = (point_per[p], float(lo), float(hi))
    boot_avg_clean = np.array([v for v in boot_avg if not np.isnan(v)])
    if len(boot_avg_clean) == 0:
        avg = (float("nan"),) * 3
    else:
        lo, hi = np.percentile(boot_avg_clean, [2.5, 97.5])
        avg = (point_avg, float(lo), float(hi))
    return out_per, avg


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    d: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        d[(r["base"], r["stage"])].append(r)
    return d


def err(means: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.vstack([np.nan_to_num(means - lo, nan=0.0), np.nan_to_num(hi - means, nan=0.0)])


def plot_grid(id_rows: list[dict], chat_rows: list[dict], ood_rows: list[dict], out_path: Path) -> None:
    id_by = group_rows(id_rows); chat_by = group_rows(chat_rows); ood_by = group_rows(ood_rows)
    x_labels = PERSONAS + ["Avg"]
    x = np.arange(len(x_labels)); w = 0.27
    fig, axes = plt.subplots(len(BASES), len(STAGES), figsize=(20, 11), sharey=True)
    for ri, base in enumerate(BASES):
        for ci, stage in enumerate(STAGES):
            ax = axes[ri, ci]
            id_per, id_avg = bootstrap_per_persona_f1(id_by.get((base, stage), []))
            chat_per, chat_avg = bootstrap_per_persona_f1(chat_by.get((base, stage), []))
            ood_per, ood_avg = bootstrap_per_persona_f1(ood_by.get((base, stage), []))
            id_m = np.array([id_per[p][0] for p in PERSONAS] + [id_avg[0]])
            id_l = np.array([id_per[p][1] for p in PERSONAS] + [id_avg[1]])
            id_h = np.array([id_per[p][2] for p in PERSONAS] + [id_avg[2]])
            ch_m = np.array([chat_per[p][0] for p in PERSONAS] + [chat_avg[0]])
            ch_l = np.array([chat_per[p][1] for p in PERSONAS] + [chat_avg[1]])
            ch_h = np.array([chat_per[p][2] for p in PERSONAS] + [chat_avg[2]])
            oo_m = np.array([ood_per[p][0] for p in PERSONAS] + [ood_avg[0]])
            oo_l = np.array([ood_per[p][1] for p in PERSONAS] + [ood_avg[1]])
            oo_h = np.array([ood_per[p][2] for p in PERSONAS] + [ood_avg[2]])
            ax.bar(x - w, np.nan_to_num(id_m, nan=0.0), w,
                   yerr=err(id_m, id_l, id_h), color=COLOR_ID, label="ID (chat)",
                   capsize=2, error_kw={"elinewidth": 0.7})
            ax.bar(x, np.nan_to_num(ch_m, nan=0.0), w,
                   yerr=err(ch_m, ch_l, ch_h), color=COLOR_CHAT, label="chat email",
                   capsize=2, error_kw={"elinewidth": 0.7})
            ax.bar(x + w, np.nan_to_num(oo_m, nan=0.0), w,
                   yerr=err(oo_m, oo_l, oo_h), color=COLOR_OOD, label="agentic email",
                   capsize=2, error_kw={"elinewidth": 0.7})
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
            ax.axhline(1/11, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{BASE_LABEL[base]}  ·  {stage}", fontsize=11)
            if ci == 0:
                ax.set_ylabel("F1")
            if ri == 0 and ci == len(STAGES) - 1:
                ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_summary(id_rows: list[dict], chat_rows: list[dict], ood_rows: list[dict], out_path: Path, stage: str = "full") -> None:
    id_by = group_rows(id_rows); chat_by = group_rows(chat_rows); ood_by = group_rows(ood_rows)
    x = np.arange(len(BASES)); w = 0.27
    id_means, id_lo, id_hi = [], [], []
    ch_means, ch_lo, ch_hi = [], [], []
    oo_means, oo_lo, oo_hi = [], [], []
    for base in BASES:
        _, ia = bootstrap_per_persona_f1(id_by.get((base, stage), []))
        _, ca = bootstrap_per_persona_f1(chat_by.get((base, stage), []))
        _, oa = bootstrap_per_persona_f1(ood_by.get((base, stage), []))
        id_means.append(ia[0]); id_lo.append(ia[1]); id_hi.append(ia[2])
        ch_means.append(ca[0]); ch_lo.append(ca[1]); ch_hi.append(ca[2])
        oo_means.append(oa[0]); oo_lo.append(oa[1]); oo_hi.append(oa[2])
    arrs = [np.array(x) for x in (id_means, id_lo, id_hi, ch_means, ch_lo, ch_hi, oo_means, oo_lo, oo_hi)]
    id_means, id_lo, id_hi, ch_means, ch_lo, ch_hi, oo_means, oo_lo, oo_hi = arrs
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x - w, id_means, w,
           yerr=np.vstack([id_means - id_lo, id_hi - id_means]),
           color=COLOR_ID, capsize=4, label="ID (PURE-DOVE chat)", error_kw={"elinewidth": 1.0})
    ax.bar(x, ch_means, w,
           yerr=np.vstack([ch_means - ch_lo, ch_hi - ch_means]),
           color=COLOR_CHAT, capsize=4, label="chat email (control)", error_kw={"elinewidth": 1.0})
    ax.bar(x + w, oo_means, w,
           yerr=np.vstack([oo_means - oo_lo, oo_hi - oo_means]),
           color=COLOR_OOD, capsize=4, label="agentic email (OOD)", error_kw={"elinewidth": 1.0})
    pad = 0.025
    for xi, (a, b, c, ah, bh, ch) in enumerate(zip(id_means, ch_means, oo_means, id_hi, ch_hi, oo_hi)):
        ax.text(xi - w, ah + pad, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(xi,     bh + pad, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(xi + w, ch + pad, f"{c:.2f}", ha="center", fontsize=9)
    ax.axhline(1/11, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, label="chance (1/11)")
    ax.set_xticks(x); ax.set_xticklabels([BASE_LABEL[b] for b in BASES])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Macro-F1 across 10 personas")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="id_path", type=Path, default=ID_PATH)
    ap.add_argument("--chat", type=Path, default=CHAT_PATH)
    ap.add_argument("--ood", type=Path, default=OOD_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--summary-stage", default="full", choices=STAGES)
    args = ap.parse_args()
    id_rows = load_rows(args.id_path)
    chat_rows = load_rows(args.chat)
    ood_rows = load_rows(args.ood)
    print(f"loaded {len(id_rows)} ID rows, {len(chat_rows)} chat_email rows, {len(ood_rows)} agentic OOD rows")
    plot_grid(id_rows, chat_rows, ood_rows, args.out)
    plot_summary(id_rows, chat_rows, ood_rows, args.summary_out, stage=args.summary_stage)


if __name__ == "__main__":
    main()
