#!/bin/bash
# Auto-sync jupytext when Claude edits a notebooks/*.py file.
# Runs as a PostToolUse hook on Edit and Write.

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
JUPYTEXT="$REPO_DIR/.venv/bin/jupytext"

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ "$FILE_PATH" == */notebooks/*.py && -x "$JUPYTEXT" ]]; then
  IPYNB="${FILE_PATH%.py}.ipynb"
  if [[ -f "$IPYNB" ]]; then
    "$JUPYTEXT" --sync "$FILE_PATH" 2>/dev/null
  else
    "$JUPYTEXT" --to notebook "$FILE_PATH" 2>/dev/null
  fi
fi

exit 0
