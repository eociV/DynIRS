"""Live metrics and mid-session checkpoints through a private Hugging Face repo.

Kaggle shows neither logs nor files of a running kernel through its API, so the session pushes them out:
  runs/<job>/results.csv        after every epoch (a few KB)
  runs/<job>/weights/last.pt    after every `ckpt_every` epochs (a backup if the session dies)
and `dyncv tick` reads results.csv back into MLflow while the session runs.

Config: kaggle_ds/cv/live.json {"repo": "<user>/dyncv-runs"} locally (reads with your `hf auth login`);
inside the session a private Kaggle dataset dyncv-secrets holds live.json {"repo", "token"}.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

from .config import STATE_DIR
from .progress import parse_results_csv

LOCAL_CFG = STATE_DIR / "live.json"
SECRETS_SLUG = "dyncv-secrets"


def repo_path(job: str, name: str) -> str:
    return f"runs/{job}/{name}"


def local_repo() -> str | None:
    try:
        return json.loads(LOCAL_CFG.read_text())["repo"]
    except (OSError, ValueError, KeyError):
        return None


# ------------------------------------------------------------------------------ in the cloud session

class Pusher:
    """Background thread: upload changed results.csv files, and last.pt every `ckpt_every` epochs."""

    def __init__(self, runs, repo, token, every=60.0, ckpt_every=10, api=None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.api, self.repo, self.runs = api, repo, Path(runs)
        self.every, self.ckpt_every = every, ckpt_every
        self.sent_csv: dict[str, float] = {}  # job -> mtime uploaded
        self.sent_ckpt: dict[str, int] = {}  # job -> epoch of the last uploaded last.pt
        self._stop = threading.Event()
        self._thread = None

    def _upload(self, src, dst):
        self.api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=self.repo, repo_type="dataset",
                             commit_message=f"dyncv {dst}")

    def poll(self) -> int:
        n = 0
        for csv in sorted(self.runs.glob("*/ws/out/*/results.csv")):
            job, out = csv.parent.name, csv.parent
            try:
                m = csv.stat().st_mtime
                if self.sent_csv.get(job) != m:
                    self._upload(csv, repo_path(job, "results.csv"))
                    self.sent_csv[job] = m
                    n += 1
                rows = parse_results_csv(csv.read_text())
                ep = max((r["epoch"] for r in rows), default=0)
                last = out / "weights" / "last.pt"
                due = ep - self.sent_ckpt.get(job, 0) >= self.ckpt_every
                # the trainer writes last.pt after the results row: wait until it is newer and settled
                if due and last.exists() and last.stat().st_mtime >= m and time.time() - last.stat().st_mtime > 30:
                    with tempfile.TemporaryDirectory() as t:
                        snap = Path(t) / "last.pt"
                        shutil.copy2(last, snap)
                        self._upload(snap, repo_path(job, "weights/last.pt"))
                    self.sent_ckpt[job] = ep
                    print(f"live: uploaded {job} last.pt (epoch {ep})", flush=True)
                    n += 1
            except Exception as e:  # network hiccup: retry next poll, never break training
                print(f"live: upload failed for {job}: {type(e).__name__}: {e}", flush=True)
        return n

    def _loop(self):
        while not self._stop.wait(self.every):
            self.poll()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.poll()


def find_secrets(root="/kaggle/input") -> Path | None:
    root = Path(root)
    return next(iter(sorted(root.rglob("live.json"))), None) if root.exists() else None


def start_pusher(runs, secrets_file):
    """Pusher from a live.json {"repo", "token"}; None (and a message) if it cannot start."""
    try:
        cfg = json.loads(Path(secrets_file).read_text())
        p = Pusher(runs, cfg["repo"], cfg["token"]).start()
        print(f"live: pushing metrics to hf://datasets/{cfg['repo']}", flush=True)
        return p
    except Exception as e:
        print(f"live: disabled ({type(e).__name__}: {e})", flush=True)
        return None


# ------------------------------------------------------------------------------ on the local machine

class Reader:
    """Reads the live results.csv of a job from the repo (your local HF login)."""

    def __init__(self, repo, api=None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api, self.repo = api, repo

    def rows(self, job) -> list[dict]:
        from huggingface_hub.utils import EntryNotFoundError

        try:
            with tempfile.TemporaryDirectory() as t:
                f = self.api.hf_hub_download(self.repo, repo_path(job, "results.csv"), repo_type="dataset",
                                             local_dir=t, force_download=True)
                return parse_results_csv(Path(f).read_text())
        except EntryNotFoundError:
            return []


def setup(user: str, token: str, repo: str, run_cli, out_dir=STATE_DIR / "secrets_pkg"):
    """Write the local config and upload the private Kaggle dataset <user>/dyncv-secrets."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_CFG.write_text(json.dumps({"repo": repo}, indent=1))
    d = Path(out_dir)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    try:
        (d / "live.json").write_text(json.dumps({"repo": repo, "token": token}))
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": SECRETS_SLUG, "id": f"{user}/{SECRETS_SLUG}", "licenses": [{"name": "other"}]}))
        exists = run_cli("datasets", "status", f"{user}/{SECRETS_SLUG}", check=False)[0] == 0
        if exists:
            run_cli("datasets", "version", "-p", str(d), "-m", "update")
        else:  # `datasets create` without --public: private, only this account can read it
            run_cli("datasets", "create", "-p", str(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)  # never leave the token in a loose file
    return exists
