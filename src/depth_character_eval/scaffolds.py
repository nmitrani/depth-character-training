"""Scaffold definitions and JSONL loading.

A "scaffold" is a class of OOD interaction-shape (agentic tool-use, debate, code review,
structured output, in-scene adversarial pressure). Each scaffold contains a set of
scenarios; each scenario is a partial conversation we ask the model to continue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import AgenticScenarioRecord, ScenarioRecord

SCAFFOLD_NAMES: list[str] = [
    "agentic_tools",
    "multi_agent_debate",
    "code_review",
    "structured_form",
    "adversarial_roleplay",
    "agentic_email",
    "agentic_actions",
]

# Scaffolds whose scenarios live in a directory of YAML files, not a JSONL.
# Each YAML is one scenario; the rollout loop in runner.py drives them.
YAML_SCAFFOLDS: set[str] = {"agentic_actions"}


@dataclass
class Scaffold:
    name: str
    scenarios: list[ScenarioRecord] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, name: str, path: Path) -> Scaffold:
        scenarios: list[ScenarioRecord] = []
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    rec = ScenarioRecord.model_validate(raw)
                except Exception as e:
                    raise ValueError(f"{path}:{i}: {e}") from e
                if rec.scaffold != name:
                    raise ValueError(
                        f"{path}:{i}: scaffold mismatch (file={name}, row={rec.scaffold})"
                    )
                scenarios.append(rec)
        return cls(name=name, scenarios=scenarios)


def _load_agentic_scenario(name: str, path: Path) -> AgenticScenarioRecord:
    """Parse one agentic_actions YAML file into an AgenticScenarioRecord.

    The YAML uses `initial_messages` for the seed turns and a list-of-dicts
    `tool_stubs` for clarity; we normalize both to the pydantic schema.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    stubs: dict[str, list[dict]] = {}
    for entry in raw.get("tool_stubs", []):
        stubs[entry["tool"]] = entry.get("rules", [])
    rec_dict = {
        "scaffold": raw.get("scaffold", name),
        "id": raw["id"],
        "elicits": raw.get("elicits", ""),
        "turns": raw["initial_messages"],
        "max_steps": raw.get("max_steps", 8),
        "tool_stubs": stubs,
    }
    try:
        return AgenticScenarioRecord.model_validate(rec_dict)
    except Exception as e:
        raise ValueError(f"{path}: {e}") from e


def _load_agentic_scaffold(name: str, scaffold_dir: Path) -> Scaffold:
    scenarios: list[ScenarioRecord] = []
    for path in sorted(scaffold_dir.glob("*.yaml")):
        scenarios.append(_load_agentic_scenario(name, path))
    return Scaffold(name=name, scenarios=scenarios)


def load_all_scaffolds(data_dir: Path | str = Path("data/scaffolds")) -> list[Scaffold]:
    data_dir = Path(data_dir)
    out: list[Scaffold] = []
    for name in SCAFFOLD_NAMES:
        if name in YAML_SCAFFOLDS:
            d = data_dir / name
            if not d.is_dir():
                raise FileNotFoundError(f"Missing scaffold dir: {d}")
            out.append(_load_agentic_scaffold(name, d))
        else:
            path = data_dir / f"{name}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing scaffold file: {path}")
            out.append(Scaffold.from_jsonl(name, path))
    return out


def render_turns_for_judge(turns: list[dict]) -> str:
    """Human-readable transcript for the judge prompt (not the model's chat template)."""
    lines = []
    for t in turns:
        role = t["role"].upper()
        lines.append(f"[{role}]")
        lines.append(t["content"].rstrip())
        lines.append("")
    return "\n".join(lines).rstrip()


def to_chat_messages(rec: ScenarioRecord) -> list[dict]:
    """Convert a ScenarioRecord's turns to plain dicts for vLLM/loader.generate()."""
    return [{"role": t.role, "content": t.content} for t in rec.turns]
