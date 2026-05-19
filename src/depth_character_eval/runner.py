"""Eval runner: generation + judging with resumability via deterministic hashes.

The runner iterates `for base in cfg.base_models: for stage in cfg.stages:
for persona in cfg.personas: for scaffold in cfg.scaffolds: ...` and emits one
JSONL row per (scenario, sample). Rows are keyed by a SHA-256 of every input that
would change the output, so re-running the same config skips finished work.

Generation is batched per (base, stage, persona) chunk so vLLM gets a large enough
batch to saturate the GPU. Judging fires concurrently via asyncio.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import jsonlines

from . import registry
from .config import (
    EvalConfig,
    GenParams,
    JudgeParams,
    JudgeResult,
    PersonaSpec,
    ScenarioRecord,
    Stage,
    load_personas,
)
from .judge import Judge
from .loader import CheckpointLoader
from .scaffolds import Scaffold, load_all_scaffolds, to_chat_messages


def make_key(
    base_hf_id: str,
    stage: Stage,
    persona: str,
    scaffold_name: str,
    scenario_id: str,
    sample_idx: int,
    gen_params: GenParams,
) -> str:
    """Deterministic hash for resumability. Excludes wall-clock timestamps."""
    payload = json.dumps(
        {
            "base": base_hf_id,
            "stage": stage,
            "persona": persona,
            "scaffold": scaffold_name,
            "scenario": scenario_id,
            "sample": sample_idx,
            "gen": gen_params.model_dump(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_done_keys(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with jsonlines.open(results_path) as reader:
        for row in reader:
            if row.get("judge_score") is not None and row.get("key"):
                done.add(row["key"])
    return done


def append_row(results_path: Path, row: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(results_path, mode="a") as writer:
        writer.write(row)


@dataclass
class _Pair:
    scenario: ScenarioRecord
    sample_idx: int
    key: str


def _select_personas(cfg: EvalConfig, all_personas: list[PersonaSpec]) -> list[PersonaSpec]:
    if cfg.personas is None:
        return all_personas
    by_slug = {p.slug: p for p in all_personas}
    out = []
    for slug in cfg.personas:
        if slug not in by_slug:
            raise ValueError(f"Unknown persona slug: {slug}")
        out.append(by_slug[slug])
    return out


def _select_scaffolds(cfg: EvalConfig, all_scaffolds: list[Scaffold]) -> list[Scaffold]:
    if cfg.scaffolds is None:
        return all_scaffolds
    by_name = {s.name: s for s in all_scaffolds}
    out = []
    for name in cfg.scaffolds:
        if name not in by_name:
            raise ValueError(f"Unknown scaffold: {name}")
        out.append(by_name[name])
    return out


async def _judge_batch(
    judge: Judge,
    persona: PersonaSpec,
    pairs: list[_Pair],
    responses: list[str],
) -> list[JudgeResult]:
    return await asyncio.gather(
        *[
            judge.score(persona, [t.model_dump() for t in s.scenario.turns], r)
            for s, r in zip(pairs, responses)
        ]
    )


async def run_eval_async(cfg: EvalConfig, verbose: bool = True) -> None:
    personas_all = load_personas(cfg.data_dir / "personas.yaml")
    scaffolds_all = load_all_scaffolds(cfg.data_dir / "scaffolds")
    personas = _select_personas(cfg, personas_all)
    scaffolds = _select_scaffolds(cfg, scaffolds_all)
    done = load_done_keys(cfg.results_path)
    if verbose:
        print(f"[runner] {len(done)} completed rows already in {cfg.results_path}")
    run_id = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    async with Judge(cfg.judge_params) as judge:
        for base_alias in cfg.base_models:
            base_hf = registry.base_alias_to_hf(base_alias)
            if verbose:
                print(f"[runner] loading base model {base_hf}")
            loader = CheckpointLoader(base_hf)
            try:
                for persona in personas:
                    for stage in cfg.stages:
                        if stage == "base" and persona.slug != personas[0].slug:
                            # 'base' stage doesn't depend on persona; only run once per
                            # (base, scaffold, scenario, sample). Tag with persona[0] so
                            # we don't generate it 11 times.
                            # Actually persona DOES matter for judging — we'd judge the
                            # same base response against each persona's rubric. Cheap.
                            # Keep one generation, but produce 11 judge rows. We'll judge
                            # them later in this same loop by reusing the cached response.
                            # For simplicity in v1: just generate once per (base, scenario,
                            # sample) at stage='base' and skip duplicates here.
                            pass
                        for scaffold in scaffolds:
                            pairs: list[_Pair] = []
                            for scenario in scaffold.scenarios:
                                for sample_idx in range(cfg.n_samples_per_scenario):
                                    key = make_key(
                                        base_hf, stage, persona.slug,
                                        scaffold.name, scenario.id, sample_idx,
                                        cfg.gen_params,
                                    )
                                    if key in done:
                                        continue
                                    pairs.append(_Pair(scenario, sample_idx, key))
                            if not pairs:
                                continue
                            if verbose:
                                print(
                                    f"[runner] {base_alias}/{stage}/{persona.slug}"
                                    f"/{scaffold.name}: {len(pairs)} generations"
                                )
                            chat_prompts = [to_chat_messages(p.scenario) for p in pairs]
                            responses = loader.generate(
                                chat_prompts, stage, persona.slug, cfg.gen_params
                            )
                            judge_results = await _judge_batch(judge, persona, pairs, responses)
                            now = dt.datetime.utcnow().isoformat() + "Z"
                            for p, response, jres in zip(pairs, responses, judge_results):
                                row = {
                                    "run_id": run_id,
                                    "key": p.key,
                                    "base_model": base_hf,
                                    "base_alias": base_alias,
                                    "stage": stage,
                                    "persona": persona.slug,
                                    "persona_display": persona.display_name,
                                    "scaffold": scaffold.name,
                                    "scenario_id": p.scenario.id,
                                    "sample_idx": p.sample_idx,
                                    "response": response,
                                    "judge_score": jres.score,
                                    "judge_rationale": jres.rationale,
                                    "judge_raw_samples": jres.raw_samples,
                                    "gen_params": cfg.gen_params.model_dump(),
                                    "judge_params": cfg.judge_params.model_dump(),
                                    "timestamp": now,
                                }
                                append_row(cfg.results_path, row)
                                done.add(p.key)
            finally:
                del loader


def run_eval(cfg: EvalConfig, verbose: bool = True) -> None:
    """Sync wrapper for run_eval_async."""
    asyncio.run(run_eval_async(cfg, verbose=verbose))


@dataclass
class SmokeRow:
    stage: Stage
    score: int
    rationale: str
    response_preview: str


async def smoke_async(
    base_alias: str,
    persona_slug: str,
    scaffold_name: str,
    scenario_id: str | None = None,
    data_dir: Path = Path("data"),
    gen_params: GenParams | None = None,
    judge_params: JudgeParams | None = None,
) -> list[SmokeRow]:
    """End-to-end smoke: one (scenario, sample) across all 3 stages, judged.

    Soft check: score(full) >= score(distillation) >= score(base) - 1 typically.
    """
    gen_params = gen_params or GenParams(temperature=0.0, seed=0)  # deterministic for smoke
    judge_params = judge_params or JudgeParams(n_samples=1)
    base_hf = registry.base_alias_to_hf(base_alias)
    personas = load_personas(data_dir / "personas.yaml")
    persona = next(p for p in personas if p.slug == persona_slug)
    scaffolds = load_all_scaffolds(data_dir / "scaffolds")
    scaffold = next(s for s in scaffolds if s.name == scaffold_name)
    scenario = (
        next(sc for sc in scaffold.scenarios if sc.id == scenario_id)
        if scenario_id
        else scaffold.scenarios[0]
    )
    chat_prompt = [to_chat_messages(scenario)]
    loader = CheckpointLoader(base_hf)
    results: list[SmokeRow] = []
    async with Judge(judge_params) as judge:
        for stage in ["base", "distillation", "full"]:
            response = loader.generate(chat_prompt, stage, persona_slug, gen_params)[0]
            jres = await judge.score(persona, [t.model_dump() for t in scenario.turns], response)
            results.append(
                SmokeRow(
                    stage=stage,  # type: ignore[arg-type]
                    score=jres.score,
                    rationale=jres.rationale,
                    response_preview=response[:240].replace("\n", " "),
                )
            )
    return results


def smoke(
    base_alias: str,
    persona_slug: str,
    scaffold_name: str,
    scenario_id: str | None = None,
    data_dir: Path = Path("data"),
) -> list[SmokeRow]:
    return asyncio.run(smoke_async(base_alias, persona_slug, scaffold_name, scenario_id, data_dir))
