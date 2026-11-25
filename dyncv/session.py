"""Entry point inside a cloud session (Kaggle now; any backend that can run Python on a GPU).

  python -m dyncv.session run --queue configs/queues/cv5/cv5_f01.json --data-in <prepared data>
      --code-in <code> --work <writable dir> [--prev-out <previous session output>] [--budget-h 11]
  python -m dyncv.session score --queue Q --runs <repo>/runs/remote --proc <data>/proc

`run` = progress tailer + scripts/kaggle_session.py (copy code/data, restore runs, pip, warm the IR
cache, train in GPU lanes until the budget) + `score` for every fold whose teacher finished. A fold
cut by the budget resumes from last.pt in the next session; scoring left undone runs then too.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DV = Path("/tmp/dv")  # where scripts/kaggle_session.py puts the data


def session_cmd(a) -> list[str]:
    cmd = [sys.executable, str(Path(a.code_in) / "scripts" / "kaggle_session.py"), "--queue", a.queue,
           "--data-in", a.data_in, "--code-in", a.code_in, "--work", a.work, "--budget-h", str(a.budget_h),
           "--lanes", str(a.lanes), "--gpus", a.gpus, "--mods", "ir"]
    cmd += ["--prev-out", a.prev_out] if a.prev_out else []
    return cmd + (["--only", *a.only] if a.only else [])


def score_cmd(a, repo: Path) -> list[str]:
    cmd = [sys.executable, "-m", "dyncv.session", "score", "--queue", a.queue,
           "--runs", str(repo / "runs" / "remote"), "--proc", str(DV / "proc")]
    return cmd + (["--only", *a.only] if a.only else [])


def stage_data(data_in) -> None:
    """Data from the prep kernel come as one tar: unpack it where kaggle_session.py expects the data
    (it then skips its own copy). A prepared proc/ folder (dataset dynirs-dv) needs nothing."""
    from .prep import TAR, extract

    tar = Path(data_in) / TAR
    if tar.is_file() and not (DV / "proc").exists():
        extract(tar, DV)


def run(a) -> int:
    from .progress import Tailer

    stage_data(a.data_in)
    repo = Path(a.work) / "DynIRS"
    tail = Tailer(repo / "runs" / "remote", every=a.progress_every).start()
    push = None
    if a.live:
        from .live import start_pusher

        push = start_pusher(repo / "runs" / "remote", a.live)
    try:
        rc = subprocess.run(session_cmd(a)).returncode
    finally:
        tail.stop()
        if push:
            push.stop()
    print("lanes rc", rc, "-> scoring finished folds", flush=True)
    rc_score = subprocess.run(score_cmd(a, repo), cwd=repo, env=dict(os.environ, DV_DATA=str(DV))).returncode
    return rc or rc_score


def score(a) -> int:
    from . import queues
    from .score import score_pending

    specs = [s for s in queues.cv_jobs(queues.load(a.queue)) if not a.only or s["name"] in a.only]
    done = score_pending(specs, a.runs, a.proc, log=lambda m: print(m, flush=True))
    print("scored:", done or "nothing new", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queue", required=True)
    r.add_argument("--data-in", required=True)
    r.add_argument("--code-in", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--prev-out")
    r.add_argument("--only", nargs="*")
    r.add_argument("--budget-h", type=float, default=11.0)
    r.add_argument("--lanes", type=int, default=2)
    r.add_argument("--gpus", default="0,1")
    r.add_argument("--progress-every", type=float, default=60.0)
    r.add_argument("--live", help="live.json {repo, token}: push results.csv / last.pt to Hugging Face")
    s = sub.add_parser("score")
    s.add_argument("--queue", required=True)
    s.add_argument("--runs", required=True)
    s.add_argument("--proc", required=True)
    s.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    sys.exit(main())
