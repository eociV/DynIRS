"""Run `dyncv tick` on a timer without an open terminal.

  python -m dyncv schedule install configs/queues/cv5/cv5_f4.json [more queues] [--every 15]
  python -m dyncv schedule status | remove

Windows: a Task Scheduler task "DynCV-tick" (runs while you are logged in, hidden window) calls
kaggle_ds/cv/tick.cmd, which appends to kaggle_ds/cv/tick.log. Elsewhere it prints a crontab line.
A lock file keeps two ticks from overlapping (a fetch of large checkpoints can outlast the interval).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .config import ROOT, STATE_DIR

TASK = "DynCV-tick"
CMD = STATE_DIR / "tick.cmd"
LOG = STATE_DIR / "tick.log"
LOCK = STATE_DIR / "tick.lock"
STALE_S = 4 * 3600  # a lock older than this belongs to a dead tick


@contextmanager
def lock(path=LOCK, stale_s=STALE_S):
    """Yields True if this process holds the lock, False if another live tick does."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > stale_s:
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield True
    finally:
        path.unlink(missing_ok=True)


def tick_command(queues, extra=()) -> str:
    uv = shutil.which("uv") or "uv"
    qs = " ".join(f'"{Path(q).resolve()}"' for q in queues)
    return (f'@echo off\r\ncd /d "{ROOT}"\r\nset MLFLOW_DISABLE_AGENT_HINT=1\r\nset PYTHONUNBUFFERED=1\r\n'
            f'"{uv}" run python -m dyncv tick {qs} {" ".join(extra)} >> "{LOG}" 2>&1\r\n')


def install(queues, every=15, extra=()):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CMD.write_text(tick_command(queues, extra))
    if os.name != "nt":
        print(f"add to crontab -e:\n*/{every} * * * * cd {ROOT} && uv run python -m dyncv tick "
              + " ".join(str(Path(q).resolve()) for q in queues) + f" >> {LOG} 2>&1")
        return
    # conhost --headless: no console window popping up every few minutes (Windows 11)
    tr = f'conhost.exe --headless cmd.exe /c "{CMD}"'
    r = subprocess.run(["schtasks", "/Create", "/F", "/TN", TASK, "/SC", "MINUTE", "/MO", str(every), "/TR", tr],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    if r.returncode:
        raise SystemExit(r.returncode)
    print(f"every {every} min: {CMD}\nlog: {LOG}")


def remove():
    if os.name == "nt":
        r = subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())


def status(tail=15):
    if os.name == "nt":
        r = subprocess.run(["schtasks", "/Query", "/TN", TASK, "/FO", "LIST"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
    if LOG.exists():
        print(f"--- last lines of {LOG}")
        print("\n".join(LOG.read_text(errors="replace").splitlines()[-tail:]))


if __name__ == "__main__":
    sys.exit(status())
