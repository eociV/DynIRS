"""Backend-agnostic session loop: submit -> watch (live metrics to MLflow) -> fetch -> sync -> repeat
until every job of the queue is trained and scored.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from . import queues
from .backends.base import FINAL
from .config import LOCAL_RUNS, ROOT
from .progress import parse_log
from .score import is_scored

sys.path.insert(0, str(ROOT / "scripts"))
import run_queue  # noqa: E402


def pending(specs, runs=LOCAL_RUNS) -> list[str]:
    """Jobs with training left (run_queue.plan) or, for cv jobs, scoring left."""
    return [s["name"] for s in specs if run_queue.plan(s, runs) or ("cv" in s and not is_scored(s, runs))]


def sync(tracker, specs, runs=LOCAL_RUNS, weights=True, tags=None) -> None:
    if tracker is None:
        return
    for s in specs:
        if (Path(runs) / s["name"]).exists():
            n = tracker.sync_job(s, runs, weights, tags)
            print(f"mlflow {s['name']}: +{n} epochs", flush=True)


def watch(backend, sid, specs, tracker=None, poll=600, clock=time):
    """Poll until the session ends; each poll pushes the live DYNCV_METRIC lines to MLflow."""
    names = {s["name"]: s for s in specs}
    while True:
        st = backend.status(sid)
        print(clock.strftime("%H:%M:%S"), sid, st, flush=True)
        if tracker is not None and st in ("running",) + FINAL:
            for run, rows in parse_log(backend.logs(sid)).items():
                if run in names:
                    tracker.run_id(run, names[run])
                    n = tracker.log_epochs(run, rows)
                    if n:
                        print(f"  live {run}: +{n} epochs (last {rows[-1]['epoch']})", flush=True)
        if st in FINAL:
            return st
        clock.sleep(poll)


def collect(backend, sid, specs, tracker=None, runs=LOCAL_RUNS, weights=True) -> bool:
    """Download an ended session's runs and log them. False if it left no usable output."""
    try:
        backend.fetch(sid, runs)
    except SystemExit as e:  # no runs/remote in the output, or the download failed
        print(f"{sid}: could not fetch output: {e}", flush=True)
        return False
    backend.mark(sid, fetched=True)
    sync(tracker, specs, runs, weights, {"dyncv.backend": backend.name, "dyncv.last_session": sid})
    return True


def pull_live(reader, tracker, specs) -> dict:
    """Epochs added per job from the live repo (results.csv pushed by the running session)."""
    out = {}
    for s in specs:
        try:
            rows = reader.rows(s["name"])
        except Exception as e:  # repo unreachable: the session output still arrives at the end
            print(f"live: {s['name']}: {type(e).__name__}: {e}", flush=True)
            continue
        if rows:
            tracker.run_id(s["name"], s)
            out[s["name"]] = (tracker.log_epochs(s["name"], rows), rows[-1]["epoch"])
    return out


def tick(backend, queue, tracker=None, only=None, budget_h=11.0, weights=True, runs=LOCAL_RUNS,
         max_failures=2, reader=None) -> str:
    """One non-blocking pass over a queue, for a scheduler (cron / Windows Task Scheduler):
    session running -> report; ended -> fetch + MLflow; work left -> push the next session.
    Repeating it is safe; it never waits."""
    specs = [s for s in queues.load(queue) if not only or s["name"] in only]
    name = Path(queue).stem
    sid = backend.active(queue)
    if sid:
        st = backend.status(sid)
        if tracker is not None and st not in FINAL:  # show the job in MLflow before any metric arrives
            for s in specs:
                tracker.run_id(s["name"], s, {"dyncv.fold": s.get("cv", {}).get("fold", "")})
                tracker.set_tags(s["name"], {"dyncv.state": f"{st} on {backend.name} ({sid}); metrics after the session ends",
                                             "dyncv.last_check": time.strftime("%Y-%m-%d %H:%M"),
                                             "dyncv.backend": backend.name, "dyncv.session": sid})
        if tracker is not None and st == "running":  # live lines, if the backend shows logs while running
            for run_name, rows in parse_log(backend.logs(sid)).items():
                if run_name in {s["name"] for s in specs}:
                    tracker.run_id(run_name, next(s for s in specs if s["name"] == run_name))
                    tracker.log_epochs(run_name, rows)
        live = pull_live(reader, tracker, specs) if reader is not None and tracker is not None else {}
        if st not in FINAL:
            ep = "; ".join(f"{j} epoch {e} (+{n})" for j, (n, e) in live.items())
            return f"{name}: {sid} {st}" + (f"; live {ep}" if ep else "")
        backend.mark(sid, st)
        if not collect(backend, sid, specs, tracker, runs, weights):
            backend.mark(sid, no_output=True)
    else:
        late = backend.unfetched(queue)
        if late and not collect(backend, late, specs, tracker, runs, weights):
            backend.mark(late, no_output=True)
    left = pending(specs, runs)
    if not left:
        return f"{name}: done"
    if backend.failures(queue) >= max_failures:
        return f"{name}: STOPPED after {max_failures} failed sessions in a row; fix, then reset with `dyncv run`"
    return f"{name}: pushed {backend.submit(queue, only, budget_h)} for {', '.join(left)}"


def run(backend, queue, tracker=None, only=None, budget_h=11.0, poll=600, max_rounds=8, weights=True,
        runs=LOCAL_RUNS, max_failures=2):
    """Safe to rerun at any point: a running session is watched again, an ended but unfetched one is
    fetched first, and a new session resumes from the newest output that has checkpoints."""
    specs = [s for s in queues.load(queue) if not only or s["name"] in only]
    sid = backend.active(queue)
    late = None if sid else backend.unfetched(queue)
    if late:
        print("fetching earlier session", late, flush=True)
        if not collect(backend, late, specs, tracker, runs, weights):
            backend.mark(late, no_output=True)
    failures = 0
    for _ in range(max_rounds):
        if sid is None:
            left = pending(specs, runs)
            if not left:
                print("queue done:", Path(queue).name, flush=True)
                return []
            print("to do:", ", ".join(left), "-> new session", flush=True)
            sid = backend.submit(queue, only, budget_h)
        st = watch(backend, sid, specs, tracker, poll)
        backend.mark(sid, st)
        got = collect(backend, sid, specs, tracker, runs, weights)
        if not got:
            backend.mark(sid, no_output=True)
        if st == "complete" and got:
            failures = 0
        else:
            failures += 1
            print(f"{sid} ended {st} (output {'kept' if got else 'lost'}); next session resumes from "
                  f"{'it' if got else 'the last session with output'}", flush=True)
            if failures >= max_failures:
                raise SystemExit(f"{failures} failed sessions in a row; read the log of {sid} on "
                                 f"{backend.name}, fix, then rerun the same command")
        sid = None
    raise SystemExit(f"stopped after {max_rounds} sessions; rerun to continue")
