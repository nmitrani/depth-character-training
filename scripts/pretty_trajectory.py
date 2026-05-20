"""Human-readable renderer for agentic_actions rollout trajectories.

Usage:
    python scripts/pretty_trajectory.py --in results/agentic_actions_llama.jsonl
    python scripts/pretty_trajectory.py --in <path> --persona sarcasm --stage full
    python scripts/pretty_trajectory.py --in <path> --scenario agentic_actions_marcus_reviews

Reads the JSONL produced by `dce run-eval` on agentic_actions scenarios and prints
each trajectory as: (header with metadata) → user task → numbered steps, each
showing the tool name, formatted args, and a summary of the simulated response.
The long system prompt is omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# Reuse the rollout parsers so what we display matches what the harness saw.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from depth_character_eval.rollout import parse_final, parse_tool_call  # noqa: E402

WIDTH = 96
INDENT = "    "
SEP_HEAVY = "=" * WIDTH
SEP_LIGHT = "-" * WIDTH


def wrap(text: str, indent: str = INDENT, width: int = WIDTH) -> str:
    out_lines: list[str] = []
    for line in text.splitlines() or [""]:
        if not line.strip():
            out_lines.append("")
            continue
        wrapped = textwrap.fill(
            line, width=width, initial_indent=indent, subsequent_indent=indent,
            replace_whitespace=False, drop_whitespace=False,
        )
        out_lines.append(wrapped)
    return "\n".join(out_lines)


def fmt_value(v, key: str = "") -> str:
    """Render one arg value. Multi-line / long strings get a wrapped block."""
    if isinstance(v, str):
        if "\n" in v or len(v) > 70:
            wrapped = wrap(v, indent=INDENT + "  ")
            return f"\n{wrapped}"
        return repr(v)
    if isinstance(v, list):
        if all(isinstance(x, str) and len(x) < 40 for x in v):
            return "[" + ", ".join(repr(x) for x in v) + "]"
        return json.dumps(v, indent=2)
    if isinstance(v, dict):
        return json.dumps(v, indent=2)
    return repr(v)


def fmt_args(args: dict, indent: str = INDENT) -> str:
    if not args:
        return indent + "(no args)"
    lines: list[str] = []
    for k, v in args.items():
        rendered = fmt_value(v, key=k)
        if rendered.startswith("\n"):
            lines.append(f"{indent}{k}:")
            lines.append(rendered.lstrip("\n"))
        else:
            lines.append(f"{indent}{k}: {rendered}")
    return "\n".join(lines)


def fmt_tool_response(content: str, max_chars: int = 500) -> str:
    """Pretty-print the simulator's JSON response. Truncate very long values."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        body = content
        if len(body) > max_chars:
            body = body[:max_chars] + " …(truncated)"
        return wrap(body, indent=INDENT + "  ")

    if isinstance(obj, dict) and set(obj.keys()) == {"error"}:
        return f"{INDENT}  ERROR: {obj['error']}"

    pretty = json.dumps(obj, indent=2)
    if len(pretty) > max_chars:
        pretty = pretty[:max_chars] + "\n  …(truncated)"
    return textwrap.indent(pretty, INDENT + "  ")


def render_header(row: dict) -> str:
    lines = [
        SEP_HEAVY,
        f"Scenario:    {row['scenario_id']}",
        f"Base / Stage: {row['base_alias']} / {row['stage']}",
        f"Persona:     {row['persona']} ({row.get('persona_display', '?')})",
        f"Termination: {row.get('terminated_reason', '?')}  "
        f"| tool calls: {row.get('n_tool_calls', '?')}  "
        f"| steps: {row.get('n_steps', '?')}",
    ]
    if row.get("final_summary"):
        lines.append(f"Final note:  {row['final_summary']}")
    lines.append(SEP_HEAVY)
    return "\n".join(lines)


def render_trajectory(row: dict) -> str:
    out: list[str] = [render_header(row), ""]
    traj = row.get("trajectory") or []

    # Print the user task (skip the system prompt).
    first_user = next(
        (i for i, m in enumerate(traj) if m["role"] == "user"), None
    )
    if first_user is not None:
        out.append("[USER TASK]")
        out.append(wrap(traj[first_user]["content"]))
        out.append("")
        i = first_user + 1
    else:
        i = 0

    step = 1
    while i < len(traj):
        m = traj[i]
        if m["role"] != "assistant":
            i += 1
            continue

        content = m["content"]
        next_msg = traj[i + 1] if i + 1 < len(traj) else None

        call = parse_tool_call(content)
        final = parse_final(content)

        if final is not None:
            out.append(f"[STEP {step}]  FINAL")
            out.append(wrap(final))
            out.append("")
            step += 1
            i += 1
            continue

        if call is not None:
            out.append(f"[STEP {step}]  ▸ {call['tool']}")
            out.append(fmt_args(call.get("args", {})))
            if next_msg and next_msg["role"] == "tool":
                out.append("  ↩ response:")
                out.append(fmt_tool_response(next_msg["content"]))
                i += 2
            else:
                i += 1
        else:
            # Model produced prose with no parseable block.
            preview = content.strip()
            if len(preview) > 320:
                preview = preview[:320] + " …"
            out.append(f"[STEP {step}]  (no tool_call or final — model produced prose)")
            out.append(wrap(preview))
            if next_msg and next_msg["role"] == "tool":
                out.append("  ↩ harness:")
                out.append(fmt_tool_response(next_msg["content"]))
                i += 2
            else:
                i += 1

        out.append("")
        step += 1

    out.append(SEP_LIGHT)
    out.append("")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="path", required=True, help="Path to results JSONL")
    p.add_argument("--persona", help="Filter to one persona slug")
    p.add_argument("--stage", help="Filter to one stage (base/distillation/full)")
    p.add_argument("--scenario", help="Filter to one scenario_id")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.path)]
    rows = [r for r in rows if r.get("trajectory") is not None]
    if args.persona:
        rows = [r for r in rows if r.get("persona") == args.persona]
    if args.stage:
        rows = [r for r in rows if r.get("stage") == args.stage]
    if args.scenario:
        rows = [r for r in rows if r.get("scenario_id") == args.scenario]

    order = {"base": 0, "distillation": 1, "full": 2}
    rows.sort(
        key=lambda r: (
            r.get("base_alias", ""),
            r.get("persona", ""),
            r.get("scenario_id", ""),
            order.get(r.get("stage", ""), 99),
        )
    )

    if not rows:
        print("No matching trajectories.", file=sys.stderr)
        return 1

    for r in rows:
        print(render_trajectory(r))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
