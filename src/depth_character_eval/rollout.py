"""Multi-turn agent rollout against per-scenario tool stubs.

For `agentic_actions` scenarios, the model generates a tool_call, the harness
simulates the response from the scenario's stub table, and the loop repeats
until the model emits a ```final``` block, hits `max_steps`, or produces two
consecutive malformed outputs. The full conversation (initial messages, model
turns, simulated tool returns) is captured as a Trajectory.

The trait signal lives in:
  - which tools the agent chooses to call (action selection)
  - the arguments it passes (e.g., who is CC'd on an email)
  - any text inside tool args (e.g., the body of an email)
  - what it chooses to skip (e.g., not checking the HR policy)
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import AgenticScenarioRecord, GenParams, Stage, StubRule
from .loader import CheckpointLoader

# Tolerate three real-world variants of the fenced block:
#   ```tool_call\n{json}\n```         (gemma-style: newline after label)
#   ```tool_call {json}```            (llama-style: single-line; whitespace separator)
#   ```tool_call\n{json}              (truncated: closing fence dropped by max_tokens)
# The label-to-content separator is `\s+` (any whitespace including newline) and
# the closing fence is either `\s*```` or end-of-string.
TOOL_CALL_RE = re.compile(r"```tool_call\s+(.*?)(?:\s*```|\Z)", re.DOTALL)
FINAL_RE = re.compile(r"```final\s+(.*?)(?:\s*```|\Z)", re.DOTALL)
MAX_CONSECUTIVE_MALFORMED = 4

NO_FENCED_BLOCK_ERROR = (
    "Your last response had no `tool_call` or `final` fenced block, so the "
    "trajectory cannot advance. To continue, emit exactly one of:\n"
    "  ```tool_call\\n{\"tool\": ..., \"args\": {...}}\\n```\n"
    "  ```final\\n<one-line summary>\\n```\n"
    "Free prose between turns is ignored by the runtime."
)


@dataclass
class Trajectory:
    """Full record of a multi-turn rollout.

    `messages` is the complete conversation including the scenario's initial
    messages, the model's generated turns, and the simulated tool returns.
    `terminated_reason` is one of: 'final' | 'max_steps' | 'malformed'.
    """

    messages: list[dict[str, str]]
    terminated_reason: str
    final_summary: str | None
    n_tool_calls: int
    n_steps: int
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract `{tool, args}` from the first ```tool_call``` fence."""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    obj.setdefault("args", {})
    if not isinstance(obj["args"], dict):
        return None
    return obj


def parse_final(text: str) -> str | None:
    """Extract the contents of the first ```final``` fence."""
    m = FINAL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _match_arg(pattern: Any, value: Any) -> bool:
    """fnmatch-style pattern match. Both sides coerced to string."""
    return fnmatch.fnmatchcase(str(value), str(pattern))


def _rule_matches(rule: StubRule, args: dict[str, Any]) -> bool:
    """A non-default rule matches when every key in `rule.match` is present in
    `args` and its pattern matches the arg value."""
    if rule.match is None:
        return False
    for k, pat in rule.match.items():
        if k not in args:
            return False
        if not _match_arg(pat, args[k]):
            return False
    return True


def simulate_tool(
    scenario: AgenticScenarioRecord, tool_name: str, args: dict[str, Any]
) -> str:
    """Look up the tool's rules; return JSON-encoded response for the first match.

    Rule precedence: explicit `match` rules in source order, then the
    `default: true` rule (always evaluated last regardless of position).
    """
    if tool_name not in scenario.tool_stubs:
        return json.dumps({"error": f"unknown tool: {tool_name}"})
    rules = scenario.tool_stubs[tool_name]
    default_rule: StubRule | None = None
    for rule in rules:
        if rule.default:
            default_rule = rule
            continue
        if _rule_matches(rule, args):
            return json.dumps(rule.response)
    if default_rule is not None:
        return json.dumps(default_rule.response)
    return json.dumps({"error": f"no rule matched args: {args}"})


def rollout(
    loader: CheckpointLoader,
    scenario: AgenticScenarioRecord,
    stage: Stage,
    persona: str,
    gen_params: GenParams,
    max_steps: int | None = None,
) -> Trajectory:
    """Run the agent in a multi-turn loop until termination."""
    cap = max_steps if max_steps is not None else scenario.max_steps
    messages: list[dict[str, str]] = [
        {"role": t.role, "content": t.content} for t in scenario.turns
    ]
    tool_call_log: list[dict[str, Any]] = []
    n_tool_calls = 0
    consecutive_malformed = 0
    final_summary: str | None = None
    terminated_reason = "max_steps"
    step_count = 0

    for step in range(cap):
        step_count = step + 1
        response = loader.generate([messages], stage, persona, gen_params)[0]
        messages.append({"role": "assistant", "content": response})

        final = parse_final(response)
        if final is not None:
            final_summary = final
            terminated_reason = "final"
            break

        call = parse_tool_call(response)
        # Tolerance: the model sometimes calls `final` as a tool instead of
        # emitting a fenced final block. Treat that as termination too.
        if call is not None and call.get("tool") == "final":
            args = call.get("args", {})
            final_summary = (
                args.get("summary")
                or args.get("status")
                or args.get("message")
                or "(no summary)"
            )
            terminated_reason = "final"
            break

        if call is None:
            consecutive_malformed += 1
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps({"error": NO_FENCED_BLOCK_ERROR}),
                }
            )
            if consecutive_malformed >= MAX_CONSECUTIVE_MALFORMED:
                terminated_reason = "malformed"
                break
            continue

        result = simulate_tool(scenario, call["tool"], call["args"])
        messages.append({"role": "tool", "content": result})
        tool_call_log.append({"step": step, "tool": call["tool"], "args": call["args"]})
        n_tool_calls += 1
        consecutive_malformed = 0

    return Trajectory(
        messages=messages,
        terminated_reason=terminated_reason,
        final_summary=final_summary,
        n_tool_calls=n_tool_calls,
        n_steps=step_count,
        tool_call_log=tool_call_log,
    )


def render_trajectory_for_judge(trajectory: Trajectory) -> str:
    """Human-readable transcript for the trajectory-aware judge prompt.

    Drops the (long, fixed) system prompt to keep the judge focused on the
    agent's decisions and the user task; keeps everything from the user turn
    onward.
    """
    lines: list[str] = []
    seen_user = False
    for m in trajectory.messages:
        role = m["role"]
        if role == "system" and not seen_user:
            continue  # skip the long agent system prompt
        if role == "user":
            seen_user = True
        label = role.upper()
        if role == "tool":
            label = "TOOL"
        lines.append(f"[{label}]")
        lines.append(m["content"].rstrip())
        lines.append("")
    lines.append(f"[TRAJECTORY META] terminated={trajectory.terminated_reason} "
                 f"n_tool_calls={trajectory.n_tool_calls} n_steps={trajectory.n_steps}")
    return "\n".join(lines).rstrip()
