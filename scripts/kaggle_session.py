"""Run one DynIRS queue inside a Kaggle or Colab session.

  python scripts/kaggle_session.py --queue configs/queues/dyn/t4_v8n_s0_c.json \
      --data-in /kaggle/input/dynirs-dv --omega-in /kaggle/input/dynirs-omega \
      --prev-out /kaggle/input/<previous-session-output> --work /kaggle/working

Copies the prepared data to fast local disk (/tmp/dv), adds the ω files, restores runs/ from a
previous session's output (so chunked training resumes from last.pt), and runs the queue from a copy
of the code in <work>/DynIRS. Results stay in <work>/DynIRS/runs for download / the next session.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--data-in", required=True)
    ap.add_argument("--omega-in")
    ap.add_argument("--prev-out")
    ap.add_argument("--work", required=True)
    ap.add_argument("--code-in", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--mods", nargs="+", default=["rgb"], help="modalities to pre-cache (add ir for M or teacher)")
    a = ap.parse_args(argv)
    work = Path(a.work)
    repo = work / "DynIRS"
    if not repo.exists():
        shutil.copytree(a.code_in, repo, ignore=shutil.ignore_patterns("runs", "data", "kaggle_ds", ".git"))
    dv = Path("/tmp/dv")
    if not (dv / "proc").exists():
        shutil.copytree(Path(a.data_in) / "proc", dv / "proc")
    if a.omega_in:
        shutil.copytree(Path(a.omega_in) / "ironly_w", dv / "proc" / "ironly_w", dirs_exist_ok=True)
    if a.prev_out:
        prev = next(Path(a.prev_out).rglob("runs/remote"), None)
        if prev:
            shutil.copytree(prev, repo / "runs" / "remote", dirs_exist_ok=True)
            print("restored", prev, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
    env = dict(os.environ, DV_DATA=str(dv))
    subprocess.run([sys.executable, "scripts/warm_cache.py", "--splits", "train", "val", "--mods", *a.mods],
                   cwd=repo, env=env, check=True)
    cmd = [sys.executable, "scripts/run_lanes.py", a.queue, "--lanes", str(a.lanes), "--gpus", a.gpus]
    cmd += ["--only", *a.only] if a.only else []
    return subprocess.run(cmd, cwd=repo, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
