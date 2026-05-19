"""Render trajectory, heatmap, and cross-base plots from a summary parquet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .aggregate import depth_leaderboard

STAGE_ORDER = ["base", "distillation", "full"]


def _per_persona_trajectory(df: pd.DataFrame, persona: str, out_path: Path) -> None:
    sub = df[df["persona"] == persona].copy()
    if sub.empty:
        return
    display = sub["persona_display"].iloc[0]
    bases = sorted(sub["base_alias"].unique())
    fig, axes = plt.subplots(1, len(bases), figsize=(5 * len(bases), 4), sharey=True)
    if len(bases) == 1:
        axes = [axes]
    for ax, base in zip(axes, bases):
        s = sub[sub["base_alias"] == base]
        for scaffold in sorted(s["scaffold"].unique()):
            ss = s[s["scaffold"] == scaffold].set_index("stage").reindex(STAGE_ORDER)
            ax.plot(STAGE_ORDER, ss["mean_likert"], marker="o", label=scaffold)
            ax.fill_between(STAGE_ORDER, ss["ci_lo"], ss["ci_hi"], alpha=0.15)
        ax.set_title(f"{base}")
        ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("training stage")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean Likert (persona expression)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5)
    suptitle = f"Persona: {display} ({persona})"
    if persona == "misalignment":
        suptitle += "  [WARNING: higher = MORE undesirable behavior]"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _per_base_heatmap(df: pd.DataFrame, base: str, out_path: Path) -> None:
    sub = df[df["base_alias"] == base]
    wide = sub.pivot_table(
        index=["persona", "persona_display"],
        columns=["scaffold", "stage"],
        values="mean_likert",
        observed=True,
    )
    full = wide.xs("full", level="stage", axis=1)
    base_s = wide.xs("base", level="stage", axis=1)
    delta = (full - base_s).droplevel(0)  # index by display name
    fig, ax = plt.subplots(figsize=(1.4 * len(delta.columns) + 2, 0.4 * len(delta) + 2))
    sns.heatmap(
        delta,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        ax=ax,
    )
    ax.set_title(f"Δ Likert (full - base), base model: {base}")
    ax.set_xlabel("scaffold")
    ax.set_ylabel("persona")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _cross_base_spaghetti(df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        df.groupby(["base_alias", "persona", "persona_display", "stage"], observed=True)[
            "mean_likert"
        ]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("tab20", n_colors=agg["persona"].nunique())
    persona_to_color = {p: palette[i] for i, p in enumerate(sorted(agg["persona"].unique()))}
    base_to_ls = {"llama": "-", "qwen": "--", "gemma": ":"}
    for (base, persona), g in agg.groupby(["base_alias", "persona"]):
        g = g.set_index("stage").reindex(STAGE_ORDER)
        ax.plot(
            STAGE_ORDER,
            g["mean_likert"],
            color=persona_to_color[persona],
            linestyle=base_to_ls.get(base, "-"),
            marker="o",
            alpha=0.85,
            label=f"{persona} ({base})",
        )
    ax.set_ylim(0.8, 5.2)
    ax.set_ylabel("mean Likert (averaged over scaffolds)")
    ax.set_xlabel("training stage")
    ax.set_title("Trait-expression trajectory across (persona, base)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def render_all(summary_parquet: Path, out_dir: Path) -> int:
    df = pd.read_parquet(summary_parquet)
    df["stage"] = pd.Categorical(df["stage"], STAGE_ORDER, ordered=True)
    n = 0
    for persona in sorted(df["persona"].unique()):
        _per_persona_trajectory(df, persona, out_dir / f"trajectory_{persona}.png")
        n += 1
    for base in sorted(df["base_alias"].unique()):
        _per_base_heatmap(df, base, out_dir / f"heatmap_{base}.png")
        n += 1
    _cross_base_spaghetti(df, out_dir / "cross_base_spaghetti.png")
    n += 1
    leaderboard = depth_leaderboard(df)
    leaderboard.to_csv(out_dir / "depth_leaderboard.csv", index=False)
    return n
