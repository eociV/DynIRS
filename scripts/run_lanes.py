"""Run one queue in N parallel lanes (processes), each pinned to a GPU.

  python scripts/run_lanes.py QUEUE --lanes 2 --gpus 0,1 [--only NAME ...]

Jobs are dealt round-robin to lanes; each lane runs scripts/run_queue.py --only <its jobs>.
Use --gpus 0,0 to put two lanes on one GPU. Run scripts/warm_cache.py first.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_lanes(names, n):
    return [names[i::n] for i in range(n)]


def combined_returncode(codes):
    """0 only if every lane succeeded; otherwise the first non-zero code (sign preserved)."""
    return next((c for c in codes if c != 0), 0)


def over_budget(t0, now, budget_h):
    """True once `budget_h` hours of wall-clock have passed since t0 (None or 0 = no budget)."""
    return bool(budget_h) and now - t0 >= budget_h * 3600


def wait_lanes(procs, lane_names, budget_h, poll=5.0):
    """Wait for the lane processes; on budget expiry terminate them (last.pt lets the next session
    resume). Returns the combined return code (non-zero if the budget stopped any lane)."""
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        if over_budget(t0, time.time(), budget_h):
            stopped = []
            for p, names in zip(procs, lane_names):
                if p.poll() is None:
                    p.terminate()
                    stopped.append(names)
            for p in procs:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            print(f"BUDGET {budget_h} h exceeded; stopped lanes running jobs: {stopped}", flush=True)
            return 1
        time.sleep(poll)
    return combined_returncode([p.returncode for p in procs])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--budget-h", type=float, default=None, help="stop the lanes after this many hours")
    a = ap.parse_args(argv)
    names = [j["name"] for j in json.loads(Path(a.queue).read_text()) if not a.only or j["name"] in a.only]
    gpus = a.gpus.split(",")
    procs, lane_names = [], []
    for k, lane in enumerate(split_lanes(names, a.lanes)):
        if lane:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[k % len(gpus)])
            procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_queue.py"), a.queue,
                                           "--only", *lane], cwd=ROOT, env=env))
            lane_names.append(lane)
    return wait_lanes(procs, lane_names, a.budget_h)


if __name__ == "__main__":
    sys.exit(main())
