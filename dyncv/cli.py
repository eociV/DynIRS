"""python -m dyncv <command>   (run from the repository root)

  folds                      split the IR train ids into k folds   -> configs/cv/folds_k5_b50_s0.json
  queues [--groups 01 23 4]  one teacher job per fold              -> configs/queues/cv5/*.json
  upload code                pack + upload dynirs-cv-code (again after every code / queue change)
  prep                       CPU kernel prepares the data on Kaggle from Hugging Face (once), checked
                             against the local label hashes and fold ids; sessions mount its output
  run    QUEUE               push, watch (live MLflow), pull, sync, next session ... until done
  tick   QUEUE...            one non-blocking pass per queue (fetch + MLflow when a session ended, push next)
  live                       live metrics: sessions push results.csv / last.pt to a private HF repo
  schedule install QUEUE...  run tick every 15 min via Windows Task Scheduler (schedule status | remove)
  push   QUEUE | status QUEUE | fetch QUEUE                       single steps
  sync   [QUEUE]             (re)log fetched runs in runs/cv/remote to MLflow
  score  --fold F            score one fold on the local GPU instead of in the cloud
  merge                      all fold signals -> proc/ironly_w/cv5_{T,TL,S}; summary to MLflow
  upload omega               the cv5_* weight files -> dynirs-cv-omega (for student sessions)
  ui                         start the MLflow UI on the local store

Settings (--k, --block, --seed, --epochs, --hw) must be the same for folds, queues, score and merge.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config as C
from . import folds as F
from . import queues as Q


def cfg_of(a) -> C.CVConfig:
    return C.CVConfig(k=a.k, block=a.block, seed=a.seed, epochs=a.epochs, hw=a.hw)


def tracker_of(a):
    if getattr(a, "no_mlflow", False):
        return None
    from .tracking import Tracker

    return Tracker()


def backend_of(a):
    from .backends import get

    return get("kaggle", user=a.user, code_owner=a.code_owner, data_owner=a.data_owner)


def cmd_folds(a):
    cfg, proc = cfg_of(a), C.proc_dir()
    counts = F.ironly_box_counts(proc)
    folds = F.make_folds(F.train_ids(proc), counts, cfg.k, cfg.block, cfg.seed)
    stats = F.fold_stats(folds, counts)
    F.save(cfg.folds_file, folds, cfg, stats)
    print(cfg.folds_file)
    for s in stats:
        print(json.dumps(s))


def cmd_queues(a):
    cfg = cfg_of(a)
    groups = [[int(c) for c in g] for g in a.groups] if a.groups else None
    for p in Q.write(cfg, F.load(cfg.folds_file)["folds"], groups):
        print(p.relative_to(C.ROOT).as_posix(), len(Q.load(p)), "jobs")


def cmd_upload(a):
    backend_of(a).upload(a.what, C.proc_dir(), cfg_of(a).tag)


def cmd_prep(a):
    from .backends.kaggle import local_hashes

    cfg = cfg_of(a)
    expect = local_hashes(C.proc_dir())
    if not expect:
        raise SystemExit(f"no {C.proc_dir()}/.ready_* to check the cloud copy against")
    if not cfg.folds_file.exists():
        raise SystemExit(f"{cfg.folds_file} missing: run python -m dyncv folds first")
    rep = backend_of(a).prep(expect, cfg.folds_file.relative_to(C.ROOT).as_posix())
    print(json.dumps(rep, indent=1))
    print("data ready: training sessions now mount the dyncv-prep output")


def cmd_run(a):
    from . import orchestrator

    orchestrator.run(backend_of(a), a.queue, tracker_of(a), a.only, a.budget_h, a.poll, a.max_rounds,
                     not a.no_weights)


def cmd_tick(a):
    import time

    from . import orchestrator
    from .schedule import lock

    with lock() as mine:
        if not mine:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "previous tick still running; skipped", flush=True)
            return
        b, tr = backend_of(a), tracker_of(a)
        from .live import Reader, local_repo

        reader = Reader(local_repo()) if local_repo() else None
        for q in a.queues:
            try:
                msg = orchestrator.tick(b, q, tr, a.only, a.budget_h, not a.no_weights, reader=reader)
            except (Exception, SystemExit) as e:  # one bad queue must not stop the others
                msg = f"{Path(q).stem}: ERROR {type(e).__name__}: {e}"
            print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def cmd_schedule(a):
    from . import schedule

    if a.action == "install":
        if not a.queues:
            raise SystemExit("name the queue files to watch")
        extra = (["--user", a.user] if a.user else []) + (["--code-owner", a.code_owner] if a.code_owner else []) \
            + (["--data-owner", a.data_owner] if a.data_owner else [])
        schedule.install(a.queues, a.every, extra)
    elif a.action == "remove":
        schedule.remove()
    else:
        schedule.status()


def cmd_live(a):
    """Store the HF token of your `hf auth login` in the private Kaggle dataset dyncv-secrets."""
    from huggingface_hub import HfApi, get_token

    from . import live

    token = get_token()
    if not token:
        raise SystemExit("not logged in to Hugging Face: run `uv run hf auth login`")
    api = HfApi(token=token)
    repo = a.repo or f"{api.whoami()['name']}/dyncv-runs"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    b = backend_of(a)
    live.setup(b.user, token, repo, b.cli)
    print(f"live metrics on: sessions push to hf://datasets/{repo} (private); "
          f"token stored in private Kaggle dataset {b.user}/{live.SECRETS_SLUG}")


def cmd_push(a):
    backend_of(a).submit(a.queue, a.only, a.budget_h)


def _last(b, queue, final_only=False):
    rounds = b.rounds(queue)
    if final_only:
        rounds = [r for r in rounds if r["status"] == "complete"] or rounds
    if not rounds:
        raise SystemExit(f"no session for {queue} yet")
    return rounds[-1]["slug"]


def cmd_status(a):
    b = backend_of(a)
    sid = _last(b, a.queue)
    print(sid, b.status(sid))


def cmd_fetch(a):
    from . import orchestrator

    b = backend_of(a)
    sid = a.sid or _last(b, a.queue, final_only=True)
    specs = Q.load(a.queue)
    if not orchestrator.collect(b, sid, specs, tracker_of(a), C.LOCAL_RUNS, not a.no_weights):
        raise SystemExit(f"{sid}: nothing fetched")
    print("still to do:", orchestrator.pending(specs) or "nothing")


def cmd_sync(a):
    from . import orchestrator

    queue = a.queue or cfg_of(a).queue_dir / f"{cfg_of(a).tag}_all.json"
    orchestrator.sync(tracker_of(a), Q.load(queue), C.LOCAL_RUNS, not a.no_weights)


def cmd_score(a):
    from .score import score_job

    cfg = cfg_of(a)
    spec = next(s for s in Q.load(cfg.queue_dir / f"{cfg.tag}_all.json") if s["cv"]["fold"] == a.fold)
    score_job(spec, C.LOCAL_RUNS, C.proc_dir(), spec["train"]["imgsz"], weights=a.weights)
    if not a.no_mlflow:
        tracker_of(a).sync_job(spec, C.LOCAL_RUNS, weights=False)


def cmd_merge(a):
    from . import merge

    cfg = cfg_of(a)
    report = merge.run(cfg, C.LOCAL_RUNS, C.proc_dir())
    print(json.dumps(report, indent=1))
    if not a.no_mlflow:
        tracker_of(a).log_merge(cfg, report)


def cmd_ui(a):
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", C.mlflow_uri(),
                    "--port", str(a.port)])


def parser():
    ap = argparse.ArgumentParser(prog="python -m dyncv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = C.CVConfig()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--k", type=int, default=d.k)
    common.add_argument("--block", type=int, default=d.block)
    common.add_argument("--seed", type=int, default=d.seed)
    common.add_argument("--epochs", type=int, default=d.epochs)
    common.add_argument("--hw", choices=("t4", "local"), default=d.hw)
    common.add_argument("--no-mlflow", action="store_true")
    common.add_argument("--no-weights", action="store_true", help="do not copy checkpoints into MLflow")
    kag = argparse.ArgumentParser(add_help=False)
    kag.add_argument("--user", help="Kaggle account that runs the kernels (default: kaggle.json)")
    kag.add_argument("--code-owner", help="owner of dynirs-cv-code (default: --user)")
    kag.add_argument("--data-owner", help="use the uploaded dataset <owner>/dynirs-dv instead of the "
                                          "dyncv-prep kernel output (default)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("folds", parents=[common]).set_defaults(fn=cmd_folds)
    p = sub.add_parser("queues", parents=[common])
    p.add_argument("--groups", nargs="*", help="folds per queue file, e.g. 01 23 4 (default: pairs)")
    p.set_defaults(fn=cmd_queues)
    p = sub.add_parser("upload", parents=[common, kag])
    p.add_argument("what", choices=("code", "omega"))
    p.set_defaults(fn=cmd_upload)
    sub.add_parser("prep", parents=[common, kag]).set_defaults(fn=cmd_prep)
    for name, fn in (("run", cmd_run), ("push", cmd_push)):
        p = sub.add_parser(name, parents=[common, kag])
        p.add_argument("queue")
        p.add_argument("--only", nargs="*")
        p.add_argument("--budget-h", type=float, default=11.0, help="lanes stop after this many hours (session max 12)")
        if name == "run":
            p.add_argument("--poll", type=int, default=600, help="seconds between status / log checks")
            p.add_argument("--max-rounds", type=int, default=8)
        p.set_defaults(fn=fn)
    p = sub.add_parser("live", parents=[common, kag], help="turn on live metrics via a private HF repo")
    p.add_argument("--repo", help="HF dataset repo (default: <hf user>/dyncv-runs)")
    p.set_defaults(fn=cmd_live)
    p = sub.add_parser("tick", parents=[common, kag])
    p.add_argument("queues", nargs="+")
    p.add_argument("--only", nargs="*")
    p.add_argument("--budget-h", type=float, default=11.0)
    p.set_defaults(fn=cmd_tick)
    p = sub.add_parser("schedule", parents=[common, kag])
    p.add_argument("action", choices=("install", "remove", "status"))
    p.add_argument("queues", nargs="*")
    p.add_argument("--every", type=int, default=15, help="minutes between ticks")
    p.set_defaults(fn=cmd_schedule)
    p = sub.add_parser("status", parents=[common, kag])
    p.add_argument("queue")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch", parents=[common, kag])
    p.add_argument("queue")
    p.add_argument("--sid", help="kernel slug (default: the queue's latest complete session)")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("sync", parents=[common])
    p.add_argument("queue", nargs="?")
    p.set_defaults(fn=cmd_sync)
    p = sub.add_parser("score", parents=[common])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--weights", help="teacher checkpoint (default: runs/cv/remote/<job>/.../best.pt)")
    p.set_defaults(fn=cmd_score)
    sub.add_parser("merge", parents=[common]).set_defaults(fn=cmd_merge)
    p = sub.add_parser("ui", parents=[common])
    p.add_argument("--port", type=int, default=5000)
    p.set_defaults(fn=cmd_ui)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    return a.fn(a)
