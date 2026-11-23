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


def find_prev_runs(prev_out):
    """The single runs/remote directory under a previous session's output (exit if none or several)."""
    found = sorted(Path(prev_out).rglob("runs/remote"))
    if not found:
        sys.exit(f"--prev-out {prev_out}: no runs/remote directory found")
    if len(found) > 1:
        sys.exit(f"--prev-out {prev_out}: several runs/remote directories found, pick one:\n"
                 + "\n".join(f"  {f.parent.parent}" for f in found))
    return found[0]


def describe_runs(remote):
    """[(run name, state)] of the runs restored under a runs/remote directory."""
    out = []
    for d in sorted(p for p in Path(remote).iterdir() if p.is_dir()):
        last = d / "ws" / "out" / d.name / "weights" / "last.pt"
        if not last.exists():
            out.append((d.name, "no last.pt"))
            continue
        import torch

        ep = torch.load(last, map_location="cpu", weights_only=False)["epoch"]
        out.append((d.name, "finished" if ep == -1 else f"epoch {ep + 1} done"))
    return out


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
    ap.add_argument("--budget-h", type=float, default=11.0, help="stop the lanes after this many hours")
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
        prev = find_prev_runs(a.prev_out)
        shutil.copytree(prev, repo / "runs" / "remote", dirs_exist_ok=True)
        print("restored", prev, flush=True)
        for name, state in describe_runs(prev):
            print(f"  {name}: {state}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
    env = dict(os.environ, DV_DATA=str(dv))
    subprocess.run([sys.executable, "scripts/warm_cache.py", "--splits", "train", "val", "--mods", *a.mods],
                   cwd=repo, env=env, check=True)
    cmd = [sys.executable, "scripts/run_lanes.py", a.queue, "--lanes", str(a.lanes), "--gpus", a.gpus,
           "--budget-h", str(a.budget_h)]
    cmd += ["--only", *a.only] if a.only else []
    return subprocess.run(cmd, cwd=repo, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
