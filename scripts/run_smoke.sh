#!/usr/bin/env bash
# Quick smoke: load Gemma 3 4B, generate one code-review response per stage with the
# Sarcastic persona, judge each, and print a 3-row table. Fails loudly if the
# adapter loading or the judge path is broken.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-gemma}"
PERSONA="${PERSONA:-sarcasm}"
SCAFFOLD="${SCAFFOLD:-code_review}"
SCENARIO="${SCENARIO:-}"

ARGS=(--base "$BASE" --persona "$PERSONA" --scaffold "$SCAFFOLD")
if [ -n "$SCENARIO" ]; then
  ARGS+=(--scenario "$SCENARIO")
fi

exec dce smoke "${ARGS[@]}"
