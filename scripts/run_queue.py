"""Run a queue of jobs (training, prediction, latency) on the local GPU.

    python scripts/run_queue.py configs/queues/f_p_704.json            # all jobs in the queue
    python scripts/run_queue.py configs/queues/r5_vm_a.json --only p_704_seed1

Each entry of a queue file is a job spec read by src/job_main.py, plus three keys used here:
  inputs   {"inputs/<name>": "<path>"} files copied into the job workspace (paths relative to the
           repository root, e.g. a teacher checkpoint produced by an earlier job)
  chunks   list of stop_after_epoch values for kind=train (null = run to the end)
  timeout  seconds per chunk (ignored here; kept for reference)

Each job keeps one workspace under runs/remote/<name>/ws (the layout expected by the scoring and
table scripts). Training resumes from out/<name>/weights/last.pt between chunks, as in the paper's
runs, which were trained in 15-epoch chunks on an NVIDIA L4. The DroneVehicle data are downloaded and
prepared once under $DV_DATA (default: data/dronevehicle).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "remote"
SRC_FILES = ["lowlight.py", "dv_prep.py", "dv_pairs.py", "job_main.py", "zdce.py", "evaluate.py", "dfine_job.py"]
STUB = "import sys\nsys.path.insert(0, 'src')\nimport job_main\njob_main.main('job.json')\n"


def predict_outputs(spec):
    """File names job_main.predict writes for this job ([] without a predict step; "predict" may be a
    single dict or a list of them)."""
    p = spec.get("predict")
    if not p:
        return []
    out = []
    for q in p if isinstance(p, list) else [p]:
        root = q.get("data_root", "rgb")
        out.append(f"preds_{q['split']}{'' if root == 'rgb' else '_' + root}.json")
    return out


def predict_output(spec):
    """The first predict_outputs name (None if the job has no predict step)."""
    return (predict_outputs(spec) or [None])[0]


def epochs_done(out):
    """None without a last.pt, "finished" when training completed (epoch -1), else epochs completed."""
    last = out / "weights" / "last.pt"
    if not last.exists():
        return None
    import pickle
    import zipfile

    import torch

    try:
        ep = torch.load(last, map_location="cpu", weights_only=False)["epoch"]
    except (RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile, KeyError) as e:
        # truncated by a kill, or unreadable: never resume from it. Import/attribute errors (missing
        # module while unpickling a valid checkpoint) propagate instead of discarding a good file.
        bad = last.with_name(last.name + ".corrupt")
        os.replace(last, bad)
        print(f"WARNING: unusable checkpoint {last} ({type(e).__name__}); moved to {bad.name}, "
              "the job restarts", flush=True)
        return None
    return "finished" if ep == -1 else ep + 1


def _valid_json(path):
    try:
        json.loads(Path(path).read_text())
        return True
    except (OSError, ValueError):
        return False


def plan(spec, runs=RUNS):
    """[(job spec, chunk stop)] still to run, given what is already in runs/<name>/ws/out/<name>."""
    if spec["kind"] != "train":
        return [(spec, None)]
    out = Path(runs) / spec["name"] / "ws" / "out" / spec["name"]
    pred = predict_outputs(spec)
    if pred and all(_valid_json(out / f) for f in pred):
        return []
    done = epochs_done(out)
    if done == "finished":
        if pred:
            return [(dict(spec, kind="predict"), None)]
        return [] if (out / "weights" / "best.pt").exists() else [(spec, None)]
    chunks = spec.get("chunks", [None])
    return [(spec, s) for s in chunks if s is None or done is None or done < s]


def note_current(name, stop):
    """Tell run_lanes (via $DYN_LANE_FILE) which job/chunk this lane is running."""
    f = os.environ.get("DYN_LANE_FILE")
    if f:
        Path(f).write_text(f"{name} chunk {stop}")


def _run_logged(cmd, cwd, env, log, show):
    """Run cmd with output to log; with show, also echo it to this console as it arrives."""
    with open(log, "wb") as fh:
        if not show:
            return subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = sys.stdout.buffer
        while chunk := p.stdout.read1(65536):
            fh.write(chunk)
            fh.flush()
            out.write(chunk)
            out.flush()
        return p.wait()


def run_chunk(spec, stop, show=False):
    name = spec["name"]
    ws = RUNS / name / "ws"
    (ws / "src").mkdir(parents=True, exist_ok=True)
    for f in SRC_FILES:
        shutil.copy2(ROOT / "src" / f, ws / "src" / f)
    for dst, src in spec.get("inputs", {}).items():
        (ws / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src if Path(src).is_absolute() else ROOT / src, ws / dst)
    job = {k: v for k, v in spec.items() if k not in ("inputs", "chunks", "timeout")}
    if spec["kind"] == "train":
        job["config"] = dict(job.get("config", {}), stop_after_epoch=stop)
    (ws / "job.json").write_text(json.dumps(job, indent=1))
    env = dict(os.environ)
    env.setdefault("DV_DATA", str(ROOT / "data" / "dronevehicle"))
    note_current(name, stop)
    log = RUNS / name / f"chunk_{stop}_{time.strftime('%m%d-%H%M%S')}.log"
    t0 = time.time()
    env.setdefault("PYTHONUNBUFFERED", "1")
    rc = _run_logged([sys.executable, "-c", STUB], ws, env, log, show)
    print(json.dumps({"name": name, "chunk": stop, "secs": round(time.time() - t0), "rc": rc, "log": str(log)}),
          flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--only", nargs="*", help="run only the jobs with these names")
    ap.add_argument("--show", action="store_true", help="also print each job's output to this console")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    for spec in json.loads(Path(a.queue).read_text()):
        if a.only and spec["name"] not in a.only:
            continue
        for job, stop in plan(spec):
            if not run_chunk(job, stop, a.show):
                print(f"STOP: {spec['name']} chunk {stop} failed", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
