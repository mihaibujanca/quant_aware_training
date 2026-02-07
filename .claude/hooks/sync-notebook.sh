#!/bin/bash
# Auto-sync jupytext when Claude edits a notebooks/*.py file.
# Runs as a PostToolUse hook on Edit and Write.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ "$FILE_PATH" == */notebooks/*.py ]]; then
  jupytext --sync "$FILE_PATH" 2>/dev/null
fi

exit 0
