"""Aggregate per-run JSONL into a tidy summary dataframe with bootstrap CIs."""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd


def load_runs_df(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with jsonlines.open(path) as reader:
        for row in reader:
            if row.get("judge_score") is None:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No judged rows in {path}")
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def aggregate_runs(path: Path, n_bootstrap: int = 1000) -> pd.DataFrame:
    """Group by (base_alias, persona, scaffold, stage); compute mean Likert + 95% CI + n."""
    df = load_runs_df(path)
    groups = df.groupby(["base_alias", "persona", "persona_display", "scaffold", "stage"])
    out_rows = []
    for (base_alias, persona, persona_display, scaffold, stage), g in groups:
        scores = g["judge_score"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(scores, n_bootstrap)
        out_rows.append(
            {
                "base_alias": base_alias,
                "persona": persona,
                "persona_display": persona_display,
                "scaffold": scaffold,
                "stage": stage,
                "n": int(len(scores)),
                "mean_likert": float(np.mean(scores)),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    out = pd.DataFrame(out_rows)
    stage_order = pd.CategoricalDtype(["base", "distillation", "full"], ordered=True)
    out["stage"] = out["stage"].astype(stage_order)
    return out.sort_values(["base_alias", "persona", "scaffold", "stage"]).reset_index(drop=True)


def depth_leaderboard(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Per (base, persona): mean over scaffolds of Delta(full - base) Likert."""
    wide = summary_df.pivot_table(
        index=["base_alias", "persona", "persona_display", "scaffold"],
        columns="stage",
        values="mean_likert",
        observed=True,
    ).reset_index()
    wide["delta_full_minus_base"] = wide["full"] - wide["base"]
    lb = (
        wide.groupby(["base_alias", "persona", "persona_display"], as_index=False)[
            "delta_full_minus_base"
        ]
        .mean()
        .sort_values("delta_full_minus_base", ascending=False)
    )
    return lb
