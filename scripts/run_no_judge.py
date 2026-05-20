"""Generate traces for a small (base, persona, scaffold, scenario) subset without judging.

Bypasses the OpenRouter judge entirely so we can inspect raw model behavior across
training stages before spending credits. Writes one JSONL row per generation to
results/no_judge.jsonl with judge_score=None.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pull HF_TOKEN, etc., before vLLM/HF imports touch the network

from depth_character_eval import registry
from depth_character_eval.config import GenParams, load_personas
from depth_character_eval.loader import CheckpointLoader
from depth_character_eval.scaffolds import load_all_scaffolds, to_chat_messages

BASE_ALIAS = "gemma"
PERSONA_SLUGS = ["sarcasm", "goodness"]
SCAFFOLD_NAMES = ["code_review", "adversarial_roleplay"]
STAGES = ["base", "distillation", "full"]
N_SCENARIOS = 1  # first scenario only
N_SAMPLES = 1
OUT_PATH = Path("results/no_judge.jsonl")

GEN = GenParams(temperature=0.7, top_p=0.95, max_tokens=512, seed=0)


def main() -> None:
    base_hf = registry.base_alias_to_hf(BASE_ALIAS)
    personas_all = load_personas(Path("data/personas.yaml"))
    personas = [p for p in personas_all if p.slug in PERSONA_SLUGS]
    scaffolds_all = load_all_scaffolds(Path("data/scaffolds"))
    scaffolds = [s for s in scaffolds_all if s.name in SCAFFOLD_NAMES]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    print(f"loading base {base_hf}")
    loader = CheckpointLoader(base_hf, max_model_len=4096)
    run_id = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    n_written = 0
    for persona in personas:
        for stage in STAGES:
            for scaffold in scaffolds:
                pairs = []
                for scenario in scaffold.scenarios[:N_SCENARIOS]:
                    for sample_idx in range(N_SAMPLES):
                        pairs.append((scenario, sample_idx))
                if not pairs:
                    continue
                print(f"[{persona.slug}/{stage}/{scaffold.name}] {len(pairs)} gens")
                chat_prompts = [to_chat_messages(sc) for sc, _ in pairs]
                responses = loader.generate(chat_prompts, stage, persona.slug, GEN)
                with OUT_PATH.open("a") as f:
                    for (scenario, sample_idx), response in zip(pairs, responses):
                        row = {
                            "run_id": run_id,
                            "base_model": base_hf,
                            "base_alias": BASE_ALIAS,
                            "stage": stage,
                            "persona": persona.slug,
                            "persona_display": persona.display_name,
                            "scaffold": scaffold.name,
                            "scenario_id": scenario.id,
                            "sample_idx": sample_idx,
                            "prompt_turns": [t.model_dump() for t in scenario.turns],
                            "response": response,
                            "judge_score": None,
                            "gen_params": GEN.model_dump(),
                            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
                        }
                        f.write(json.dumps(row) + "\n")
                        n_written += 1
    print(f"wrote {n_written} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
