"""Unit tests for the multi-turn rollout harness.

Covers:
  - tool_call / final parsers (closed-fence + open-fence + malformed cases)
  - tool stub matching (exact / pattern / default / no-match)
  - rollout termination via final, max_steps, and consecutive malformed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from depth_character_eval.config import (
    AgenticScenarioRecord,
    GenParams,
    StubRule,
)
from depth_character_eval.rollout import (
    Trajectory,
    parse_final,
    parse_tool_call,
    rollout,
    simulate_tool,
)


# ---------------------------------------------------------------------- parsers


def test_parse_tool_call_closed_fence():
    text = '```tool_call\n{"tool":"x","args":{"a":1}}\n```\nextra prose after'
    assert parse_tool_call(text) == {"tool": "x", "args": {"a": 1}}


def test_parse_tool_call_open_fence():
    text = '```tool_call\n{"tool":"x","args":{"a":1}}\n'
    assert parse_tool_call(text) == {"tool": "x", "args": {"a": 1}}


def test_parse_tool_call_single_line_llama_style():
    # Llama-3.1-8B emits the whole block on one line; the parser must accept
    # whitespace (not just newline) between the `tool_call` label and the JSON.
    text = '```tool_call {"tool":"x","args":{"a":1}}```'
    assert parse_tool_call(text) == {"tool": "x", "args": {"a": 1}}


def test_parse_tool_call_truncated_json_returns_none():
    text = '```tool_call\n{"tool":"x","args":'
    assert parse_tool_call(text) is None


def test_parse_tool_call_no_block():
    assert parse_tool_call("no fence here") is None


def test_parse_tool_call_missing_tool_key():
    text = '```tool_call\n{"args":{}}\n```'
    assert parse_tool_call(text) is None


def test_parse_tool_call_defaults_args_to_empty():
    text = '```tool_call\n{"tool":"x"}\n```'
    assert parse_tool_call(text) == {"tool": "x", "args": {}}


def test_parse_final_closed_fence():
    assert parse_final("```final\ndone\n```") == "done"


def test_parse_final_open_fence():
    # The model may truncate before the closing fence; we still want to detect it.
    assert parse_final("```final\ndone\n") == "done"


def test_parse_final_no_block():
    assert parse_final("just prose") is None


# --------------------------------------------------------------- stub matching


def _mk_scenario(stubs: dict[str, list[dict]]) -> AgenticScenarioRecord:
    return AgenticScenarioRecord.model_validate(
        {
            "scaffold": "agentic_actions",
            "id": "agentic_actions_test",
            "turns": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user task"},
            ],
            "max_steps": 5,
            "tool_stubs": stubs,
        }
    )


def test_simulate_tool_exact_match():
    s = _mk_scenario(
        {"crm.lookup_contact": [
            {"match": {"query": "Marcus"}, "response": {"name": "Marcus Tan"}},
            {"default": True, "response": {"error": "not found"}},
        ]}
    )
    out = json.loads(simulate_tool(s, "crm.lookup_contact", {"query": "Marcus"}))
    assert out == {"name": "Marcus Tan"}


def test_simulate_tool_pattern_match():
    s = _mk_scenario(
        {"crm.lookup_contact": [
            {"match": {"query": "*Marcus*"}, "response": {"name": "Marcus Tan"}},
            {"default": True, "response": {"error": "not found"}},
        ]}
    )
    out = json.loads(simulate_tool(s, "crm.lookup_contact", {"query": "find Marcus Tan please"}))
    assert out == {"name": "Marcus Tan"}


def test_simulate_tool_falls_through_to_default():
    s = _mk_scenario(
        {"crm.lookup_contact": [
            {"match": {"query": "*Marcus*"}, "response": {"name": "Marcus Tan"}},
            {"default": True, "response": {"error": "not found"}},
        ]}
    )
    out = json.loads(simulate_tool(s, "crm.lookup_contact", {"query": "Alice"}))
    assert out == {"error": "not found"}


def test_simulate_tool_no_default_and_no_match():
    s = _mk_scenario(
        {"crm.lookup_contact": [
            {"match": {"query": "*Marcus*"}, "response": {"name": "Marcus Tan"}},
        ]}
    )
    out = json.loads(simulate_tool(s, "crm.lookup_contact", {"query": "Alice"}))
    assert "error" in out
    assert "no rule matched" in out["error"]


def test_simulate_tool_unknown_tool():
    s = _mk_scenario({})
    out = json.loads(simulate_tool(s, "foo.bar", {}))
    assert out == {"error": "unknown tool: foo.bar"}


def test_simulate_tool_missing_arg_blocks_match():
    # Rule wants `query`, but model passes `q` — should fall through.
    s = _mk_scenario(
        {"crm.lookup_contact": [
            {"match": {"query": "*Marcus*"}, "response": {"name": "Marcus Tan"}},
            {"default": True, "response": {"error": "not found"}},
        ]}
    )
    out = json.loads(simulate_tool(s, "crm.lookup_contact", {"q": "Marcus"}))
    assert out == {"error": "not found"}


# ---------------------------------------------------------------- rollout loop


class _ScriptedLoader:
    """Test double: returns canned responses in order, ignoring inputs."""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)

    def generate(self, prompts, stage, persona, gen_params) -> list[str]:
        out = [self._scripted.pop(0)]
        return out


def _scenario_with_one_tool() -> AgenticScenarioRecord:
    return _mk_scenario(
        {"ping": [{"default": True, "response": {"ok": True}}]}
    )


def test_rollout_terminates_on_tool_named_final():
    """Models sometimes call `final` as a tool instead of emitting a fenced
    final block. The rollout treats either form as termination."""
    s = _scenario_with_one_tool()
    loader = _ScriptedLoader(
        [
            '```tool_call\n{"tool":"ping","args":{}}\n```',
            '```tool_call\n{"tool":"final","args":{"summary":"all done via tool"}}\n```',
        ]
    )
    traj = rollout(loader, s, "base", "sarcasm", GenParams())  # type: ignore[arg-type]
    assert traj.terminated_reason == "final"
    assert traj.final_summary == "all done via tool"
    assert traj.n_tool_calls == 1  # the `final` call doesn't count as a real tool call


def test_rollout_terminates_on_final_block():
    s = _scenario_with_one_tool()
    loader = _ScriptedLoader(
        [
            '```tool_call\n{"tool":"ping","args":{}}\n```',
            "```final\nall done\n```",
        ]
    )
    traj = rollout(loader, s, "base", "sarcasm", GenParams())  # type: ignore[arg-type]
    assert traj.terminated_reason == "final"
    assert traj.final_summary == "all done"
    assert traj.n_tool_calls == 1
    assert traj.n_steps == 2


def test_rollout_terminates_on_malformed_after_max_consecutive():
    """Rollout terminates after MAX_CONSECUTIVE_MALFORMED consecutive prose-only
    turns (currently 4 — bumped to allow more recovery for strong-trait models
    that vent prose before snapping back to grammar)."""
    from depth_character_eval.rollout import MAX_CONSECUTIVE_MALFORMED

    s = _scenario_with_one_tool()
    loader = _ScriptedLoader(["just prose"] * MAX_CONSECUTIVE_MALFORMED)
    traj = rollout(loader, s, "base", "sarcasm", GenParams())  # type: ignore[arg-type]
    assert traj.terminated_reason == "malformed"
    assert traj.n_tool_calls == 0


def test_rollout_resets_malformed_counter_after_valid_call():
    """A valid call between two malformed responses should NOT trigger malformed
    termination (the counter resets)."""
    s = _scenario_with_one_tool()
    loader = _ScriptedLoader(
        [
            "prose without block",  # malformed=1
            '```tool_call\n{"tool":"ping","args":{}}\n```',  # resets counter
            "more prose",  # malformed=1 again
            "```final\nstopped\n```",
        ]
    )
    traj = rollout(loader, s, "base", "sarcasm", GenParams())  # type: ignore[arg-type]
    assert traj.terminated_reason == "final"
    assert traj.n_tool_calls == 1


def test_rollout_terminates_on_max_steps():
    s = _mk_scenario(
        {"ping": [{"default": True, "response": {"ok": True}}]}
    )
    # 5 valid tool calls back to back; max_steps=5 means we stop after step 5.
    loader = _ScriptedLoader(
        ['```tool_call\n{"tool":"ping","args":{}}\n```'] * 6
    )
    traj = rollout(loader, s, "base", "sarcasm", GenParams())  # type: ignore[arg-type]
    assert traj.terminated_reason == "max_steps"
    assert traj.n_tool_calls == 5
    assert traj.n_steps == 5


# ---------------------------------------------------------- end-to-end on YAML


def test_marcus_scenario_loads_and_simulates():
    """Spot-check that the Marcus YAML (now v1 of interpersonal_accountability)
    loads and a couple of tool calls behave."""
    from depth_character_eval.scaffolds import load_all_scaffolds

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scaffolds"
    scaffolds = load_all_scaffolds(DATA_DIR)
    marcus = next(
        sc for s in scaffolds if s.name == "agentic_actions"
        for sc in s.scenarios
        if sc.id == "agentic_actions_interpersonal_accountability_v1"
    )
    assert isinstance(marcus, AgenticScenarioRecord)
    assert "crm.lookup_contact" in marcus.tool_stubs
    # Pattern matching on Marcus
    out = json.loads(simulate_tool(marcus, "crm.lookup_contact", {"query": "Marcus Tan"}))
    assert out["email"] == "marcus.tan@brightline.io"
    # No match → default error
    out = json.loads(simulate_tool(marcus, "crm.lookup_contact", {"query": "nobody"}))
    assert "error" in out
    # Calendar list_events on April for Marcus → 2 events
    out = json.loads(
        simulate_tool(
            marcus, "calendar.list_events",
            {"start_date": "2026-04-01", "end_date": "2026-04-30",
             "attendee": "marcus.tan@brightline.io"},
        )
    )
    assert isinstance(out, list) and len(out) == 2
