"""Validate scaffold JSONLs load, parse, and have the expected shape."""

from pathlib import Path

import pytest

from depth_character_eval.scaffolds import (
    SCAFFOLD_NAMES,
    Scaffold,
    load_all_scaffolds,
    render_turns_for_judge,
    to_chat_messages,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scaffolds"

EXPECTED_COUNTS = {
    "agentic_tools": 8,
    "multi_agent_debate": 6,
    "code_review": 6,
    "structured_form": 5,
    "adversarial_roleplay": 5,
}


def test_all_scaffolds_present():
    scaffolds = load_all_scaffolds(DATA_DIR)
    assert {s.name for s in scaffolds} == set(SCAFFOLD_NAMES)


@pytest.mark.parametrize("name", SCAFFOLD_NAMES)
def test_scaffold_counts(name):
    s = Scaffold.from_jsonl(name, DATA_DIR / f"{name}.jsonl")
    assert len(s.scenarios) == EXPECTED_COUNTS[name]


@pytest.mark.parametrize("name", SCAFFOLD_NAMES)
def test_scaffold_schema(name):
    s = Scaffold.from_jsonl(name, DATA_DIR / f"{name}.jsonl")
    for scenario in s.scenarios:
        assert scenario.scaffold == name
        assert scenario.id.startswith(name + "_")
        assert len(scenario.turns) >= 1
        assert scenario.turns[-1].role in ("user", "tool")
        # No empty content turns.
        for t in scenario.turns:
            assert t.content.strip() != ""


def test_to_chat_messages_roundtrip():
    s = Scaffold.from_jsonl("code_review", DATA_DIR / "code_review.jsonl")
    msgs = to_chat_messages(s.scenarios[0])
    assert all(set(m.keys()) == {"role", "content"} for m in msgs)
    assert msgs[0]["role"] == s.scenarios[0].turns[0].role


def test_render_turns_for_judge_has_role_markers():
    s = Scaffold.from_jsonl("agentic_tools", DATA_DIR / "agentic_tools.jsonl")
    rendered = render_turns_for_judge([t.model_dump() for t in s.scenarios[0].turns])
    assert "[SYSTEM]" in rendered
    assert "[USER]" in rendered
    assert "[ASSISTANT]" in rendered
    assert "[TOOL]" in rendered


def test_no_duplicate_scenario_ids():
    scaffolds = load_all_scaffolds(DATA_DIR)
    all_ids = [sc.id for s in scaffolds for sc in s.scenarios]
    assert len(all_ids) == len(set(all_ids)) == 30
