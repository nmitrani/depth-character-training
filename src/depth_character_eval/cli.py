"""Typer CLI: smoke, run-eval, aggregate, plot, verify-registry."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from . import registry
from .config import EvalConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _global() -> None:
    """Depth-Character Evaluation CLI."""
    load_dotenv()  # pulls OPENROUTER_API_KEY etc. from .env if present


@app.command()
def smoke(
    base: str = typer.Option("gemma", help="Base model alias: llama|qwen|gemma"),
    persona: str = typer.Option("sarcasm", help="Persona slug"),
    scaffold: str = typer.Option("code_review", help="Scaffold name"),
    scenario: str | None = typer.Option(None, help="Specific scenario id (default: first)"),
    data_dir: Path = typer.Option(Path("data"), help="Data directory"),
) -> None:
    """One-(base,persona,scaffold) smoke test across all 3 training stages."""
    from .runner import smoke as run_smoke

    rows = run_smoke(base, persona, scaffold, scenario, data_dir)
    typer.echo("")
    typer.echo(f"=== smoke: {base} / {persona} / {scaffold} / {scenario or '(first)'} ===")
    typer.echo(f"{'stage':<14} {'score':>5}  rationale")
    typer.echo("-" * 80)
    for r in rows:
        typer.echo(f"{r.stage:<14} {r.score:>5}  {r.rationale[:60]}")
    typer.echo("")
    for r in rows:
        typer.echo(f"--- {r.stage} ---")
        typer.echo(r.response_preview)
        typer.echo("")
    base_s, dist_s, full_s = [r.score for r in rows]
    if not (full_s >= dist_s >= base_s - 1):
        typer.echo(
            f"WARNING: stage ordering looks off (base={base_s} dist={dist_s} full={full_s})."
            " Investigate adapter loading before launching the full sweep."
        )


@app.command("run-eval")
def run_eval_cmd(
    config: Path = typer.Option(Path("configs/eval.yaml"), help="EvalConfig YAML"),
    no_judge: bool = typer.Option(
        False,
        "--no-judge",
        help="Skip the OpenRouter judge; write rows with judge_score=null. "
             "Overrides the YAML's no_judge field when set.",
    ),
) -> None:
    """Run the full sweep described by an EvalConfig YAML. Resumable."""
    from .runner import run_eval

    with open(config) as f:
        raw = yaml.safe_load(f)
    cfg = EvalConfig.model_validate(raw)
    if no_judge:
        cfg = cfg.model_copy(update={"no_judge": True})
    run_eval(cfg)


@app.command("verify-registry")
def verify_registry(
    fail_fast: bool = typer.Option(False, help="Stop on first mismatch"),
) -> None:
    """Walk every (base, stage, persona) combo and confirm the HF repo+subfolder exists."""
    from huggingface_hub import HfApi

    api = HfApi()
    problems: list[str] = []
    for base_hf in registry.BASE_SLUGS:
        for stage in ["distillation", "full"]:
            for persona in registry.PERSONAS:
                try:
                    repo, sub = registry.resolve_adapter(base_hf, stage, persona)  # type: ignore[arg-type]
                except ValueError as e:
                    problems.append(f"resolve failed: {base_hf}/{stage}/{persona}: {e}")
                    continue
                try:
                    files = api.list_repo_files(repo)
                except Exception as e:
                    msg = f"missing repo {repo} (for {base_hf}/{stage}/{persona}): {e}"
                    problems.append(msg)
                    typer.echo("FAIL " + msg)
                    if fail_fast:
                        raise typer.Exit(1)
                    continue
                if sub is not None:
                    found = any(f.startswith(f"{sub}/") for f in files)
                    if not found:
                        msg = f"repo {repo} missing subfolder '{sub}/' for persona {persona}"
                        problems.append(msg)
                        typer.echo("FAIL " + msg)
                        if fail_fast:
                            raise typer.Exit(1)
                        continue
                typer.echo(f"ok   {base_hf} / {stage} / {persona} -> {repo} :: {sub}")
    typer.echo("")
    if problems:
        typer.echo(f"{len(problems)} problem(s):")
        for p in problems:
            typer.echo("  - " + p)
        raise typer.Exit(1)
    typer.echo("registry verified.")


@app.command()
def aggregate(
    in_path: Path = typer.Option(Path("results/runs.jsonl"), "--in", help="Runs JSONL"),
    out_path: Path = typer.Option(Path("results/summary.parquet"), "--out", help="Output parquet"),
    bootstrap: int = typer.Option(1000, help="Bootstrap samples for CI"),
) -> None:
    """Aggregate per-(base, persona, scaffold, stage) mean Likert + 95% bootstrap CI."""
    from .aggregate import aggregate_runs

    df = aggregate_runs(in_path, n_bootstrap=bootstrap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    typer.echo(f"wrote {len(df)} aggregated rows -> {out_path}")


@app.command()
def plot(
    in_path: Path = typer.Option(Path("results/summary.parquet"), "--in"),
    out_dir: Path = typer.Option(Path("results/plots"), "--out"),
) -> None:
    """Render trajectory + heatmap + cross-base plots from a summary parquet."""
    from .plotting import render_all

    out_dir.mkdir(parents=True, exist_ok=True)
    n = render_all(in_path, out_dir)
    typer.echo(f"wrote {n} figures -> {out_dir}")


if __name__ == "__main__":
    app()
