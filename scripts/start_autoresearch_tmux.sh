#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${1:-quant-aware-autoresearch}"

cd "$PROJECT_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "missing tmux on PATH" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "cd '$PROJECT_ROOT' && ./scripts/run_claude_autoresearch.sh"
echo "$SESSION"
