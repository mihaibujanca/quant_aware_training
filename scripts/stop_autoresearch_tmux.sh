#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${1:-quant-aware-autoresearch}"

cd "$PROJECT_ROOT"

(tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION") || true
pkill -KILL -f '/Users/mihai/Projects/quant_aware_training/scripts/run_codex_autoresearch.sh' 2>/dev/null || true
rm -f autoresearch/logs/codex_autoresearch_supervisor.pid autoresearch/logs/codex_autoresearch_child.pid
