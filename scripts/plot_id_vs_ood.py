"""Bar plot of per-persona classifier macro-F1, ID vs OOD, with bootstrap CIs.

Reads:
  - results/classifier_f1_per_email.jsonl       (OOD: agentic-email bodies)
  - results/classifier_f1_id_per_response.jsonl (ID: PURE-DOVE responses)

Renders a 3x3 facet grid (rows = base, cols = stage). Within each panel:
  x = 11 personas + "Avg"; y = per-persona F1; two bars per x (ID, OOD); CI bars.

Output: results/plots/id_vs_ood_f1.png
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
OUT_PATH = REPO_ROOT / "results" / "plots" / "id_vs_ood_f1.png"
SUMMARY_PATH = REPO_ROOT / "results" / "plots" / "id_vs_ood_summary.png"
OOD_STAGES_PATH = REPO_ROOT / "results" / "plots" / "ood_stages_by_base.png"

PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]
# `misalignment` intentionally excluded: OOD hand labels don't cover it, and
# the full-stage adapter is in a separately gated HF repo. Keep symmetric.
BASES = ["llama", "qwen", "gemma"]
STAGES = ["base", "distillation", "full"]
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def f1_from_pairs(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    """Binary F1 for one label vs rest. Returns nan if no positives in y_true."""
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
    """Returns ({persona: (mean, lo, hi)}, (avg_mean, avg_lo, avg_hi)) using non-parametric bootstrap.
    Per-persona F1 = binary F1 of (label == persona). Avg = unweighted mean over personas that have
    >=1 ground-truth row in the resample.
    """
    if not rows:
        return {p: (float("nan"), float("nan"), float("nan")) for p in PERSONAS}, (float("nan"), float("nan"), float("nan"))

    y_true = np.array([r["trait_true"] for r in rows])
    y_pred = np.array([r["trait_pred"] for r in rows])
    n = len(y_true)

    boot_per: dict[str, list[float]] = {p: [] for p in PERSONAS}
    boot_avg: list[float] = []

    # Point estimate
    point_per = {p: f1_from_pairs(y_true, y_pred, p) for p in PERSONAS}
    point_per_clean = [v for v in point_per.values() if not np.isnan(v)]
    point_avg = float(np.mean(point_per_clean)) if point_per_clean else float("nan")

    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        per = {p: f1_from_pairs(yt, yp, p) for p in PERSONAS}
        for p in PERSONAS:
            boot_per[p].append(per[p])
        clean = [v for v in per.values() if not np.isnan(v)]
        boot_avg.append(float(np.mean(clean)) if clean else float("nan"))

    out_per: dict[str, tuple[float, float, float]] = {}
    for p in PERSONAS:
        vals = np.array([v for v in boot_per[p] if not np.isnan(v)])
        if len(vals) == 0:
            out_per[p] = (float("nan"), float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out_per[p] = (point_per[p], float(lo), float(hi))

    boot_avg_clean = np.array([v for v in boot_avg if not np.isnan(v)])
    if len(boot_avg_clean) == 0:
        avg = (float("nan"), float("nan"), float("nan"))
    else:
        lo, hi = np.percentile(boot_avg_clean, [2.5, 97.5])
        avg = (point_avg, float(lo), float(hi))

    return out_per, avg


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    d: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        d[(r["base"], r["stage"])].append(r)
    return d


def plot(ood_rows: list[dict], id_rows: list[dict], out_path: Path) -> None:
    ood_by_cell = group_rows(ood_rows)
    id_by_cell = group_rows(id_rows)

    x_labels = PERSONAS + ["Avg"]
    x = np.arange(len(x_labels))
    width = 0.4

    fig, axes = plt.subplots(len(BASES), len(STAGES), figsize=(20, 11), sharey=True)
    if len(BASES) == 1:
        axes = np.array([axes])
    if len(STAGES) == 1:
        axes = axes[:, None]

    for ri, base in enumerate(BASES):
        for ci, stage in enumerate(STAGES):
            ax = axes[ri, ci]
            id_per, id_avg = bootstrap_per_persona_f1(id_by_cell.get((base, stage), []))
            ood_per, ood_avg = bootstrap_per_persona_f1(ood_by_cell.get((base, stage), []))

            id_means = np.array([id_per[p][0] for p in PERSONAS] + [id_avg[0]])
            id_lo = np.array([id_per[p][1] for p in PERSONAS] + [id_avg[1]])
            id_hi = np.array([id_per[p][2] for p in PERSONAS] + [id_avg[2]])
            ood_means = np.array([ood_per[p][0] for p in PERSONAS] + [ood_avg[0]])
            ood_lo = np.array([ood_per[p][1] for p in PERSONAS] + [ood_avg[1]])
            ood_hi = np.array([ood_per[p][2] for p in PERSONAS] + [ood_avg[2]])

            def err(means, lo, hi):
                e_lo = np.nan_to_num(means - lo, nan=0.0)
                e_hi = np.nan_to_num(hi - means, nan=0.0)
                return np.vstack([e_lo, e_hi])

            ax.bar(x - width/2, np.nan_to_num(id_means, nan=0.0), width,
                   yerr=err(id_means, id_lo, id_hi), color="#4C72B0",
                   label="ID (PURE-DOVE)", capsize=2,
                   error_kw={"elinewidth": 0.8})
            ax.bar(x + width/2, np.nan_to_num(ood_means, nan=0.0), width,
                   yerr=err(ood_means, ood_lo, ood_hi), color="#DD8452",
                   label="OOD (agentic emails)", capsize=2,
                   error_kw={"elinewidth": 0.8})

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
            # Classifier has 11 classes (incl. misalignment); chance baseline reflects that.
            ax.axhline(1/11, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{base}  ·  {stage}", fontsize=11)
            if ci == 0:
                ax.set_ylabel("F1")
            if ri == 0 and ci == len(STAGES) - 1:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "ModernBERT trait classifier · F1 in-distribution (PURE-DOVE) vs out-of-distribution (agentic emails)\n"
        "10 personas + macro avg · 95% bootstrap CI · gray dotted = chance (1/11; classifier has 11 classes)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_summary(ood_rows: list[dict], id_rows: list[dict], out_path: Path, stage: str = "full") -> None:
    """One bar per (base, ID|OOD) at the chosen stage; error bars = 95% bootstrap CI on macro-F1."""
    ood_by_cell = group_rows(ood_rows)
    id_by_cell = group_rows(id_rows)

    x = np.arange(len(BASES))
    width = 0.38

    id_means, id_lo, id_hi = [], [], []
    ood_means, ood_lo, ood_hi = [], [], []
    for base in BASES:
        _, id_avg = bootstrap_per_persona_f1(id_by_cell.get((base, stage), []))
        _, ood_avg = bootstrap_per_persona_f1(ood_by_cell.get((base, stage), []))
        id_means.append(id_avg[0]); id_lo.append(id_avg[1]); id_hi.append(id_avg[2])
        ood_means.append(ood_avg[0]); ood_lo.append(ood_avg[1]); ood_hi.append(ood_avg[2])

    id_means = np.array(id_means); id_lo = np.array(id_lo); id_hi = np.array(id_hi)
    ood_means = np.array(ood_means); ood_lo = np.array(ood_lo); ood_hi = np.array(ood_hi)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(x - width/2, id_means, width,
           yerr=np.vstack([id_means - id_lo, id_hi - id_means]),
           color="#4C72B0", capsize=4, label="ID (PURE-DOVE)",
           error_kw={"elinewidth": 1.0})
    ax.bar(x + width/2, ood_means, width,
           yerr=np.vstack([ood_means - ood_lo, ood_hi - ood_means]),
           color="#DD8452", capsize=4, label="OOD (agentic emails)",
           error_kw={"elinewidth": 1.0})

    for xi, (m1, m2) in enumerate(zip(id_means, ood_means)):
        ax.text(xi - width/2, m1 + 0.02, f"{m1:.2f}", ha="center", fontsize=9)
        ax.text(xi + width/2, m2 + 0.02, f"{m2:.2f}", ha="center", fontsize=9)

    ax.axhline(1/11, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, label="chance (1/11)")
    ax.set_xticks(x)
    ax.set_xticklabels(BASES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 across 10 personas")
    ax.set_title(
        f"ModernBERT trait classifier: ID vs OOD macro-F1 ({stage} stage)\n"
        f"mean over 10 personas · 95% bootstrap CI",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_ood_by_stage(ood_rows: list[dict], out_path: Path) -> None:
    """One group per base model; three bars per group = OOD macro-F1 at base/distillation/full.
    Shows that character training improves OOD too (just doesn't close the ID gap)."""
    ood_by_cell = group_rows(ood_rows)
    stage_colors = {"base": "#BDBDBD", "distillation": "#F0A35D", "full": "#DD8452"}

    x = np.arange(len(BASES))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for si, stage in enumerate(STAGES):
        means, los, his = [], [], []
        for base in BASES:
            _, avg = bootstrap_per_persona_f1(ood_by_cell.get((base, stage), []))
            means.append(avg[0]); los.append(avg[1]); his.append(avg[2])
        means = np.array(means); los = np.array(los); his = np.array(his)
        offset = (si - 1) * width
        ax.bar(x + offset, means, width,
               yerr=np.vstack([means - los, his - means]),
               color=stage_colors[stage], capsize=4, label=stage,
               error_kw={"elinewidth": 1.0})
        for xi, m in enumerate(means):
            ax.text(x[xi] + offset, m + 0.015, f"{m:.2f}", ha="center", fontsize=8)

    ax.axhline(1/11, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, label="chance (1/11)")
    ax.set_xticks(x)
    ax.set_xticklabels(BASES)
    ax.set_ylim(0, max(0.7, ax.get_ylim()[1]))
    ax.set_ylabel("Macro-F1 across 10 personas (OOD)")
    ax.set_title(
        "OOD macro-F1 by training stage\n"
        "agentic-email bodies · mean over 10 personas · 95% bootstrap CI",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, title="stage")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", type=Path, default=OOD_PATH)
    ap.add_argument("--id", dest="id_path", type=Path, default=ID_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--summary-stage", default="full", choices=STAGES)
    ap.add_argument("--ood-stages-out", type=Path, default=OOD_STAGES_PATH)
    args = ap.parse_args()
    ood_rows = load_rows(args.ood)
    id_rows = load_rows(args.id_path)
    print(f"loaded {len(ood_rows)} OOD rows, {len(id_rows)} ID rows")
    plot(ood_rows, id_rows, args.out)
    plot_summary(ood_rows, id_rows, args.summary_out, stage=args.summary_stage)
    plot_ood_by_stage(ood_rows, args.ood_stages_out)


if __name__ == "__main__":
    main()
