"""Generate in-distribution responses for the ModernBERT trait classifier.

Mirrors the paper's classifier-eval setup (see
`character/robustness/generate/trained.py` upstream):
  - PURE-DOVE prompts as the user-only input
  - NO persona-specific system prompt; persona signal comes from the LoRA adapter
  - one generation per (base, stage, persona, prompt)

Output: one JSONL row per generation.

Usage:
    python scripts/generate_id_responses.py --base all --stage all --personas all \\
        --n-prompts 300 --out results/id_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from datasets import load_dataset

from depth_character_eval.config import GenParams
from depth_character_eval.loader import CheckpointLoader
from depth_character_eval.registry import BASE_MODELS, PERSONAS, STAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "id_responses.jsonl"
DOVE_REPO = "LDJnr/Pure-Dove"


def load_pure_dove_prompts(n: int, seed: int = 0) -> list[str]:
    """Return `n` user prompts (first turn of each conversation), deterministic."""
    ds = load_dataset(DOVE_REPO, split="train")
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    prompts: list[str] = []
    for i in idxs:
        conv = ds[i]["conversation"]
        if not conv:
            continue
        prompts.append(conv[0]["input"])
        if len(prompts) == n:
            break
    if len(prompts) < n:
        raise RuntimeError(f"PURE-DOVE only yielded {len(prompts)} usable prompts; requested {n}")
    return prompts


def done_keys(path: Path) -> set[tuple[str, str, str]]:
    """Return the set of (base, stage, persona) cells already fully generated."""
    if not path.exists():
        return set()
    seen: dict[tuple[str, str, str], int] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            k = (r["base"], r["stage"], r["persona"])
            seen[k] = seen.get(k, 0) + 1
    return set(seen.keys())  # cell-level resumability; partial cells will be re-emitted (idempotent on classifier side)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=list(BASE_MODELS) + ["all"], required=True)
    p.add_argument("--stage", choices=STAGES + ["all"], required=True)
    p.add_argument("--personas", default="all", help="comma-separated slugs or 'all'")
    p.add_argument("--n-prompts", type=int, default=300)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append", action="store_true", help="Append + resume; default truncates.")
    args = p.parse_args()

    bases = list(BASE_MODELS) if args.base == "all" else [args.base]
    stages = STAGES if args.stage == "all" else [args.stage]
    personas = PERSONAS if args.personas == "all" else args.personas.split(",")

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and out_path.exists():
        out_path.unlink()

    seen_cells = done_keys(out_path) if args.append else set()

    print(f"Loading {args.n_prompts} PURE-DOVE prompts (seed={args.seed}) ...")
    prompts = load_pure_dove_prompts(args.n_prompts, seed=args.seed)
    chat_msgs = [[{"role": "user", "content": p}] for p in prompts]

    gp = GenParams(
        temperature=args.temperature,
        top_p=0.95,
        min_p=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    total_cells = len(bases) * len(stages) * len(personas)
    cell_i = 0
    t_run = time.time()

    for base_alias in bases:
        base_hf_id = BASE_MODELS[base_alias]
        print(f"\n### Loading base model {base_hf_id}")
        loader = CheckpointLoader(base_model_hf_id=base_hf_id, max_loras=4, max_model_len=4096)

        for stage in stages:
            cell_personas = ["__none__"] if stage == "base" else personas
            # For stage=base, persona is irrelevant to generation (no adapter).
            # We still emit a row per persona for downstream symmetry, but only
            # generate ONCE and replicate across the persona labels.
            if stage == "base":
                key_cell = (base_alias, stage, personas[0])  # use first persona as the canonical cell
                if key_cell in seen_cells and all((base_alias, stage, p) in seen_cells for p in personas):
                    cell_i += len(personas)
                    print(f"  [skip] {base_alias} {stage} (all personas present)")
                    continue
                print(f"  → {base_alias} {stage} (one generation pass, replicated across {len(personas)} persona labels)")
                t0 = time.time()
                responses = loader.generate(chat_msgs, stage=stage, persona=personas[0], gen_params=gp)
                dt = time.time() - t0
                print(f"    generated {len(responses)} responses in {dt:.1f}s")
                with out_path.open("a") as f:
                    for persona in personas:
                        cell_i += 1
                        if (base_alias, stage, persona) in seen_cells:
                            continue
                        for i, (prompt, resp) in enumerate(zip(prompts, responses)):
                            f.write(json.dumps({
                                "base": base_alias,
                                "stage": stage,
                                "persona": persona,
                                "prompt_idx": i,
                                "prompt": prompt,
                                "response": resp,
                            }) + "\n")
                        f.flush()
            else:
                for persona in personas:
                    cell_i += 1
                    if (base_alias, stage, persona) in seen_cells:
                        print(f"  [skip] {base_alias} {stage} {persona}")
                        continue
                    print(f"  [{cell_i}/{total_cells}] {base_alias} {stage} {persona}")
                    t0 = time.time()
                    responses = loader.generate(chat_msgs, stage=stage, persona=persona, gen_params=gp)
                    dt = time.time() - t0
                    print(f"    generated {len(responses)} in {dt:.1f}s")
                    with out_path.open("a") as f:
                        for i, (prompt, resp) in enumerate(zip(prompts, responses)):
                            f.write(json.dumps({
                                "base": base_alias,
                                "stage": stage,
                                "persona": persona,
                                "prompt_idx": i,
                                "prompt": prompt,
                                "response": resp,
                            }) + "\n")
                        f.flush()

        # free GPU before next base
        del loader
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDONE in {(time.time()-t_run)/60:.1f} min — wrote {out_path}")


if __name__ == "__main__":
    main()
