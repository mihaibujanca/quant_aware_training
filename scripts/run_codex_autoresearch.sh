#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/autoresearch/logs"
SUPERVISOR_LOG="$LOG_DIR/codex_autoresearch_supervisor.log"
PID_FILE="$LOG_DIR/codex_autoresearch_supervisor.pid"
CHILD_PID_FILE="$LOG_DIR/codex_autoresearch_child.pid"
HEARTBEAT_FILE="$LOG_DIR/codex_autoresearch_supervisor_heartbeat.json"
LAST_MESSAGE_FILE="$LOG_DIR/codex_autoresearch_last_message.txt"
CHILD_LOG_FILE="$LOG_DIR/codex_autoresearch_child.log"
BLOCK_REASON_FILE="$LOG_DIR/codex_autoresearch_blocked.txt"
POLL_SEC="${POLL_SEC:-20}"
RESTART_WAIT_SEC="${RESTART_WAIT_SEC:-15}"
STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-21600}"
RESTART_COUNT=0
OWN_PID_FILE=0
child_pid=""

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$SUPERVISOR_LOG"
}

write_heartbeat() {
  local state="$1"
  local now_epoch now_iso child_pid_json last_message_age_json

  now_epoch="$(date +%s)"
  now_iso="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  if [ -n "$child_pid" ]; then
    child_pid_json="$child_pid"
  else
    child_pid_json="null"
  fi

  if [ -f "$LAST_MESSAGE_FILE" ]; then
    local last_message_epoch
    last_message_epoch="$(stat -f '%m' "$LAST_MESSAGE_FILE" 2>/dev/null || echo 0)"
    if [ "$last_message_epoch" -gt 0 ]; then
      last_message_age_json="$((now_epoch - last_message_epoch))"
    else
      last_message_age_json="null"
    fi
  else
    last_message_age_json="null"
  fi

  cat >"$HEARTBEAT_FILE" <<EOF
{
  "timestamp_utc": "$now_iso",
  "state": "$state",
  "supervisor_pid": $$,
  "child_pid": $child_pid_json,
  "restart_count": $RESTART_COUNT,
  "project_root": "$PROJECT_ROOT",
  "last_message_file": "$LAST_MESSAGE_FILE",
  "last_message_age_sec": $last_message_age_json
}
EOF
}

is_quota_or_rate_limit() {
  local path="$1"
  if [ ! -f "$path" ]; then
    return 1
  fi

  grep -Eiq 'usage limit|rate limit|rate-limit|too many requests|429' "$path"
}

block_on_quota() {
  local reason="$1"

  printf '%s\n' "$reason" >"$BLOCK_REASON_FILE"
  log "stopping supervisor without restart: $reason"
  child_pid=""
  rm -f "$CHILD_PID_FILE"
  write_heartbeat "blocked_quota"
  exit 0
}

cleanup() {
  write_heartbeat "stopping"
  if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$child_pid" 2>/dev/null || true
  fi
  rm -f "$CHILD_PID_FILE"
  if [ "$OWN_PID_FILE" -eq 1 ]; then
    rm -f "$PID_FILE"
  fi
}

terminate() {
  cleanup
  exit 0
}

trap terminate INT TERM
trap cleanup EXIT

if ! command -v codex >/dev/null 2>&1; then
  echo "missing codex on PATH" >&2
  exit 1
fi

if [ -f "$PID_FILE" ]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    log "supervisor already running under pid=$existing_pid"
    exit 0
  fi
fi
printf '%s\n' "$$" >"$PID_FILE"
OWN_PID_FILE=1

launch_child() {
  local prompt_file="$LOG_DIR/codex_autoresearch_prompt.txt"
  cat >"$prompt_file" <<'EOF'
Read `autoresearch/program.md` and execute the autonomous research loop continuously until interrupted.

Runtime rules:
- Use the root-checkout `autoresearch/` files as the only agenda and bookkeeping source of truth.
- Do not use hidden runtime mirrors, detached bot-branch agendas, or second schedulers as the active queue.
- Do not ask for permission or user input.
- Keep going after each cycle; do not stop after one experiment.
- Do not launch an independent long-running helper loop that outlives this controller. If a long experiment is needed, track it as part of the current cycle and reconcile its outputs back into the root `autoresearch/` files.
- Be concise in progress output and never emit idle keep-alives.
- If you need to recover after a restart, rebuild context from the root-checkout files on disk, not from chat history.
EOF

  : >"$LAST_MESSAGE_FILE"
  : >"$CHILD_LOG_FILE"
  rm -f "$BLOCK_REASON_FILE"
  codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    --search \
    -C "$PROJECT_ROOT" \
    --color never \
    --output-last-message "$LAST_MESSAGE_FILE" \
    "$(cat "$prompt_file")" \
    >>"$CHILD_LOG_FILE" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" >"$CHILD_PID_FILE"
}

while true; do
  log "launching root-checkout autoresearch controller restart=$RESTART_COUNT"
  launch_child
  write_heartbeat "running"

  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$POLL_SEC"
    write_heartbeat "running"

    if [ "$STALL_TIMEOUT_SEC" -gt 0 ] && [ -f "$LAST_MESSAGE_FILE" ]; then
      last_message_epoch="$(stat -f '%m' "$LAST_MESSAGE_FILE" 2>/dev/null || echo 0)"
      if [ "$last_message_epoch" -gt 0 ]; then
        now_epoch="$(date +%s)"
        if [ $((now_epoch - last_message_epoch)) -ge "$STALL_TIMEOUT_SEC" ]; then
          log "last message stale for $((now_epoch - last_message_epoch))s; restarting child pid=$child_pid"
          kill -TERM "$child_pid" 2>/dev/null || true
          sleep 5
          kill -KILL "$child_pid" 2>/dev/null || true
          break
        fi
      fi
    fi
  done

  exit_code=0
  wait "$child_pid" || exit_code=$?
  log "root-checkout controller exited code=$exit_code restart=$RESTART_COUNT"
  child_pid=""
  rm -f "$CHILD_PID_FILE"

  if is_quota_or_rate_limit "$CHILD_LOG_FILE" || is_quota_or_rate_limit "$LAST_MESSAGE_FILE"; then
    block_on_quota "detected usage-limit or rate-limit exit in child output"
  fi

  write_heartbeat "restarting"
  RESTART_COUNT=$((RESTART_COUNT + 1))
  sleep "$RESTART_WAIT_SEC"
done
