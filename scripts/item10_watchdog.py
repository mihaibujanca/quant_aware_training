#!/usr/bin/env python3
"""Keep the overnight item 10 runner alive and refresh a live summary."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ["bash", "scripts/run_item10_loop.sh"]
SUMMARY = [sys.executable, "scripts/item10_live_summary.py"]
PID_PATH = ROOT / "autoresearch/logs/item10_loop.pid"
WATCHDOG_PID_PATH = ROOT / "autoresearch/logs/item10_watchdog.pid"
WATCHDOG_HEARTBEAT_PATH = ROOT / "autoresearch/logs/item10_watchdog_heartbeat.json"
LOG_PATH = ROOT / "autoresearch/logs/item10_loop.out"
WATCHDOG_LOG_PATH = ROOT / "autoresearch/logs/item10_watchdog.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return int(raw)


def _write_watchdog_log(message: str) -> None:
    with WATCHDOG_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _start_runner() -> int:
    log_handle = LOG_PATH.open("ab", buffering=0)
    proc = subprocess.Popen(
        RUNNER,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
    )
    PID_PATH.write_text(f"{proc.pid}\n", encoding="utf-8")
    _write_watchdog_log(f"started_runner pid={proc.pid}")
    return proc.pid


def _refresh_summary() -> None:
    subprocess.run(
        SUMMARY,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _write_heartbeat(runner_pid: int | None) -> None:
    payload = {
        "updated_at": _utc_now(),
        "runner_pid": runner_pid,
        "runner_alive": bool(runner_pid and _pid_is_alive(runner_pid)),
    }
    WATCHDOG_HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    WATCHDOG_PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_watchdog_log("watchdog_started")
    while True:
        runner_pid = _read_pid(PID_PATH)
        if runner_pid is None or not _pid_is_alive(runner_pid):
            runner_pid = _start_runner()
        _refresh_summary()
        _write_heartbeat(runner_pid)
        time.sleep(60)


if __name__ == "__main__":
    main()
