"""Run one queue in N parallel lanes (processes), each pinned to a GPU.

  python scripts/run_lanes.py QUEUE --lanes 2 --gpus 0,1 [--only NAME ...] [--budget-h H | --deadline TS]

Jobs are dealt round-robin to lanes; each lane runs scripts/run_queue.py --only <its jobs>.
Use --gpus 0,0 to put two lanes on one GPU. Run scripts/warm_cache.py first.
At the deadline (unix timestamp, or --budget-h hours from start) every lane's whole process tree is
killed; Ultralytics keeps last.pt, so the next session resumes.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE_DIR = ROOT / "runs" / "remote"


def split_lanes(names, n):
    return [names[i::n] for i in range(n)]


def combined_returncode(codes):
    """0 only if every lane succeeded; otherwise the first non-zero code (sign preserved)."""
    return next((c for c in codes if c != 0), 0)


def over_budget(t0, now, budget_h):
    """True once `budget_h` hours of wall-clock have passed since t0 (None or 0 = no budget)."""
    return bool(budget_h) and now - t0 >= budget_h * 3600


def resolve_deadline(deadline, budget_h, now):
    """Unix-time deadline: an explicit --deadline wins, else now + budget_h hours, else None."""
    if deadline:
        return deadline
    return now + budget_h * 3600 if budget_h else None


def past_deadline(deadline, now):
    return deadline is not None and now >= deadline


def lane_popen_kwargs():
    """Popen kwargs that put a lane in its own process group / session so the tree can be killed."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(proc, grace=5.0):
    """Kill proc and all of its descendants (the trainer is a grandchild of the lane process)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        t = time.time()
        while proc.poll() is None and time.time() - t < grace:
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


def read_current(path):
    """'<job> chunk <stop>' last noted by the lane's run_queue (see run_queue.note_current)."""
    try:
        return Path(path).read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def wait_lanes(procs, lane_files, deadline, poll=5.0):
    """Wait for the lane processes; at the deadline kill their process trees (last.pt lets the next
    session resume). Returns the combined return code (1 if the deadline stopped any lane)."""
    while any(p.poll() is None for p in procs):
        if past_deadline(deadline, time.time()):
            running = [(p, f) for p, f in zip(procs, lane_files) if p.poll() is None]
            stopped = [read_current(f) for _, f in running]
            for p, _ in running:
                kill_tree(p)
            print(f"BUDGET exceeded; stopped lanes while running: {stopped}", flush=True)
            return 1
        time.sleep(poll)
    return combined_returncode([p.returncode for p in procs])


def main(argv=None):
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--budget-h", type=float, default=None, help="stop the lanes after this many hours")
    ap.add_argument("--deadline", type=float, default=None, help="unix time at which to stop the lanes")
    a = ap.parse_args(argv)
    names = [j["name"] for j in json.loads(Path(a.queue).read_text()) if not a.only or j["name"] in a.only]
    gpus = a.gpus.split(",")
    LANE_DIR.mkdir(parents=True, exist_ok=True)
    procs, lane_files = [], []
    for k, lane in enumerate(split_lanes(names, a.lanes)):
        if lane:
            f = LANE_DIR / f".lane_{k}_current"
            f.unlink(missing_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[k % len(gpus)], DYN_LANE_FILE=str(f))
            procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_queue.py"), a.queue,
                                           "--only", *lane], cwd=ROOT, env=env, **lane_popen_kwargs()))
            lane_files.append(f)
    return wait_lanes(procs, lane_files, resolve_deadline(a.deadline, a.budget_h, t_start))


if __name__ == "__main__":
    sys.exit(main())
