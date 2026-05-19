"""Pydantic models for persona specs, scenario records, eval config, and judge output."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Stage = Literal["base", "distillation", "full"]
Role = Literal["system", "user", "assistant", "tool"]


class PersonaSpec(BaseModel):
    slug: str
    display_name: str
    one_sentence_desc: str


class Turn(BaseModel):
    role: Role
    content: str


class ScenarioRecord(BaseModel):
    scaffold: str
    id: str
    turns: list[Turn]
    elicits: str = ""

    @model_validator(mode="after")
    def _check_last_turn(self) -> ScenarioRecord:
        if not self.turns:
            raise ValueError(f"Scenario {self.id}: empty turns")
        if self.turns[-1].role not in ("user", "tool"):
            raise ValueError(
                f"Scenario {self.id}: last turn must be user or tool (got {self.turns[-1].role})"
            )
        return self


class GenParams(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float = 0.0
    max_tokens: int = 512
    seed: int | None = None


class JudgeParams(BaseModel):
    model: str = "z-ai/glm-4.5-air"
    temperature: float = 0.1
    top_p: float = 0.95
    n_samples: int = 3
    max_tokens: int = 256


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str
    raw_samples: list[int]


class EvalConfig(BaseModel):
    base_models: list[str] = Field(
        default_factory=lambda: ["llama", "qwen", "gemma"],
        description="Aliases from registry.BASE_MODELS",
    )
    personas: list[str] | None = Field(default=None, description="Slugs; None = all 11")
    stages: list[Stage] = Field(default_factory=lambda: ["base", "distillation", "full"])
    scaffolds: list[str] | None = Field(default=None, description="Scaffold names; None = all")
    n_samples_per_scenario: int = 3
    gen_params: GenParams = Field(default_factory=GenParams)
    judge_params: JudgeParams = Field(default_factory=JudgeParams)
    results_path: Path = Path("results/runs.jsonl")
    data_dir: Path = Path("data")


def load_personas(path: Path | str = Path("data/personas.yaml")) -> list[PersonaSpec]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [PersonaSpec.model_validate(p) for p in raw]
