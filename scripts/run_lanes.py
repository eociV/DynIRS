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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_lanes(names, n):
    return [names[i::n] for i in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    names = [j["name"] for j in json.loads(Path(a.queue).read_text()) if not a.only or j["name"] in a.only]
    gpus = a.gpus.split(",")
    procs = []
    for k, lane in enumerate(split_lanes(names, a.lanes)):
        if lane:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[k % len(gpus)])
            procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_queue.py"), a.queue,
                                           "--only", *lane], cwd=ROOT, env=env))
    return max(p.wait() for p in procs) if procs else 0


if __name__ == "__main__":
    sys.exit(main())
