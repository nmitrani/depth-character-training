"""Generate model responses for the chat_email scaffold.

Mirrors `generate_id_responses.py`: for each (base, stage, persona) cell, runs
the 50 chat-form analogues of the agentic_actions scenarios and writes one row
per generation. The response is the email body (the prompt instructs body-only
output, so downstream classification can use it directly).

Output: results/chat_email_responses.jsonl.

Usage:
    python scripts/generate_chat_email_responses.py --base all --stage all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from depth_character_eval.config import GenParams
from depth_character_eval.loader import CheckpointLoader
from depth_character_eval.registry import BASE_MODELS, STAGES
from depth_character_eval.scaffolds import load_all_scaffolds, to_chat_messages

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "chat_email_responses.jsonl"

# 10 personas — matches the LW.md analysis; excludes 'misalignment' (no full
# adapter access + no OOD labels).
DEFAULT_PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]


def done_cells(path: Path) -> set[tuple[str, str, str]]:
    """Return (base, stage, persona) triples that already have all 50 scenarios."""
    if not path.exists():
        return set()
    counts: dict[tuple[str, str, str], int] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            k = (r["base"], r["stage"], r["persona"])
            counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n >= 50}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=list(BASE_MODELS) + ["all"], default="all")
    p.add_argument("--stage", choices=STAGES + ["all"], default="all")
    p.add_argument("--personas", default="default",
                   help="comma-separated slugs, 'default' for LW.md's 10, or 'all'")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append", action="store_true",
                   help="Append + resume by cell; default truncates.")
    args = p.parse_args()

    bases = list(BASE_MODELS) if args.base == "all" else [args.base]
    stages = STAGES if args.stage == "all" else [args.stage]
    if args.personas == "default":
        personas = DEFAULT_PERSONAS
    elif args.personas == "all":
        from depth_character_eval.registry import PERSONAS
        personas = list(PERSONAS)
    else:
        personas = args.personas.split(",")

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and out_path.exists():
        out_path.unlink()

    scaffolds = load_all_scaffolds(REPO_ROOT / "data" / "scaffolds")
    chat_email = next(s for s in scaffolds if s.name == "chat_email")
    if not chat_email.scenarios:
        raise SystemExit("chat_email scaffold has no scenarios — run generate_chat_email_scaffold.py first")
    print(f"Loaded {len(chat_email.scenarios)} chat_email scenarios")

    chat_msgs = [to_chat_messages(sc) for sc in chat_email.scenarios]
    scenario_ids = [sc.id for sc in chat_email.scenarios]

    seen = done_cells(out_path) if args.append else set()

    gp = GenParams(
        temperature=args.temperature,
        top_p=0.95,
        min_p=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    total_cells = sum(
        len(personas) if stage != "base" else 1
        for stage in stages
        for _ in bases
    )
    cell_i = 0
    t_run = time.time()

    for base_alias in bases:
        base_hf = BASE_MODELS[base_alias]
        print(f"\n### Loading base {base_hf}")
        loader = CheckpointLoader(base_model_hf_id=base_hf, max_loras=4, max_model_len=4096)
        for stage in stages:
            if stage == "base":
                # No adapter; one generation pass replicated across personas for symmetry.
                key_first = (base_alias, stage, personas[0])
                all_done = all((base_alias, stage, p) in seen for p in personas)
                if all_done:
                    print(f"  [skip] {base_alias} {stage} (all personas present)")
                    cell_i += 1
                    continue
                print(f"  → {base_alias} {stage} (one pass, replicated across {len(personas)} persona labels)")
                t0 = time.time()
                responses = loader.generate(chat_msgs, stage=stage, persona=personas[0], gen_params=gp)
                dt = time.time() - t0
                print(f"    generated {len(responses)} in {dt:.1f}s")
                with out_path.open("a") as f:
                    for persona in personas:
                        if (base_alias, stage, persona) in seen:
                            continue
                        for sid, resp in zip(scenario_ids, responses):
                            f.write(json.dumps({
                                "base": base_alias,
                                "stage": stage,
                                "persona": persona,
                                "scenario_id": sid,
                                "response": resp,
                            }) + "\n")
                    f.flush()
                cell_i += 1
            else:
                for persona in personas:
                    cell_i += 1
                    if (base_alias, stage, persona) in seen:
                        print(f"  [skip] {base_alias} {stage} {persona}")
                        continue
                    print(f"  [{cell_i}/{total_cells}] {base_alias} {stage} {persona}")
                    t0 = time.time()
                    responses = loader.generate(chat_msgs, stage=stage, persona=persona, gen_params=gp)
                    dt = time.time() - t0
                    print(f"    {len(responses)} in {dt:.1f}s")
                    with out_path.open("a") as f:
                        for sid, resp in zip(scenario_ids, responses):
                            f.write(json.dumps({
                                "base": base_alias,
                                "stage": stage,
                                "persona": persona,
                                "scenario_id": sid,
                                "response": resp,
                            }) + "\n")
                        f.flush()
        del loader
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDONE in {(time.time()-t_run)/60:.1f} min — wrote {out_path}")


if __name__ == "__main__":
    main()
