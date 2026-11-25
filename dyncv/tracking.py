"""Local MLflow tracking (one experiment, one run per teacher job, plus one run per merge).

  mlflow ui --backend-store-uri sqlite:///runs/cv/mlflow.db      # then open http://127.0.0.1:5000

Everything is idempotent: epochs already logged for a run are skipped (checked against the run's
"epoch" metric history), artifacts are overwritten in place. Safe to call after every poll.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config as C
from .progress import parse_results_csv
from .score import job_out

_BAD = re.compile(r"[^\w\-. :/]")
LOGGED_FILES = ("results.csv", "args.yaml", "train.log", C.SIGNALS_FILE, C.SIGNALS_STATS_FILE, "preds_val_ir.json")


def metric_key(k: str) -> str:
    return _BAD.sub("_", k).strip("_") or "metric"


def flat_params(spec: dict) -> dict:
    p = {f"train.{k}": v for k, v in spec.get("train", {}).items()}
    p.update({f"cv.{k}": v for k, v in spec.get("cv", {}).items()})
    p["n_train_images"] = len(spec.get("proxy_ids", ()))
    return {k: str(v)[:500] for k, v in p.items()}


class Tracker:
    def __init__(self, uri=None, experiment=C.EXPERIMENT):
        import mlflow
        from mlflow.tracking import MlflowClient

        self._mlflow = mlflow
        uri = uri or C.mlflow_uri()
        if uri.startswith("sqlite:///"):
            Path(uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        self.client = MlflowClient(tracking_uri=uri)
        exp = self.client.get_experiment_by_name(experiment)
        self.exp_id = exp.experiment_id if exp else self.client.create_experiment(
            experiment, artifact_location=C.mlflow_artifacts())
        self._ids: dict[str, str] = {}

    # -------------------------------------------------------------------------- runs

    def run_id(self, name: str, spec: dict | None = None, tags: dict | None = None) -> str:
        if name in self._ids:
            return self._ids[name]
        hits = self.client.search_runs([self.exp_id], filter_string=f"tags.`dyncv.job` = '{name}'", max_results=1)
        if hits:
            rid = hits[0].info.run_id
        else:
            rid = self.client.create_run(self.exp_id, run_name=name, tags={"dyncv.job": name, **(tags or {})}).info.run_id
            for k, v in flat_params(spec or {}).items():
                self.client.log_param(rid, k, v)
        self._ids[name] = rid
        return rid

    def set_tags(self, name: str, tags: dict):
        rid = self.run_id(name)
        for k, v in tags.items():
            self.client.set_tag(rid, k, str(v))

    def logged_epochs(self, name: str) -> set[int]:
        return {int(m.step) for m in self.client.get_metric_history(self.run_id(name), "epoch")}

    # -------------------------------------------------------------------------- logging

    def log_epochs(self, name: str, rows: list[dict]) -> int:
        """Log result rows (one per epoch) not logged yet. Returns how many epochs were added."""
        from mlflow.entities import Metric

        done = self.logged_epochs(name)
        ts = int(time.time() * 1000)
        new = [r for r in rows if int(r["epoch"]) not in done]
        metrics = [Metric(metric_key(k), float(v), ts, int(r["epoch"])) for r in new for k, v in r.items()]
        for i in range(0, len(metrics), 900):  # log_batch limit is 1000 metrics
            self.client.log_batch(self.run_id(name), metrics=metrics[i:i + 900])
        return len(new)

    def log_summary(self, name: str, stats: dict, prefix: str):
        rid = self.run_id(name)
        for k, v in _flatten(stats).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.client.log_metric(rid, metric_key(f"{prefix}{k}"), float(v))

    def log_files(self, name: str, out: Path, weights: bool = True):
        rid = self.run_id(name)
        for f in LOGGED_FILES:
            if (out / f).exists():
                self.client.log_artifact(rid, str(out / f))
        if weights:
            for f in ("best.pt", "last.pt"):
                if (out / "weights" / f).exists():
                    self.client.log_artifact(rid, str(out / "weights" / f), "weights")

    # -------------------------------------------------------------------------- whole job

    def sync_job(self, spec: dict, runs, weights: bool = True, extra_tags: dict | None = None) -> int:
        """Log one fetched job workspace: new epochs, scoring stats, files. Returns epochs added."""
        name = spec["name"]
        out = job_out(runs, name)
        self.run_id(name, spec, {"dyncv.fold": spec.get("cv", {}).get("fold", "")})
        n = 0
        if (out / "results.csv").exists():
            n = self.log_epochs(name, parse_results_csv((out / "results.csv").read_text()))
        st = out / C.SIGNALS_STATS_FILE
        if st.exists():
            self.log_summary(name, json.loads(st.read_text()), "score/")
        self.log_files(name, out, weights)
        self.set_tags(name, dict(extra_tags or {}, **{"dyncv.state": job_state(spec, out)}))
        return n

    def log_merge(self, cfg: C.CVConfig, report: dict):
        name = f"{cfg.tag}_merge"
        self.run_id(name, {"cv": {"k": cfg.k, "block": cfg.block, "seed": cfg.seed}}, {"dyncv.kind": "merge"})
        self.log_summary(name, report, "")
        rid = self.run_id(name)
        self.client.log_text(rid, json.dumps(report, indent=1), "merge_report.json")


def job_state(spec, out: Path) -> str:
    if (out / C.SIGNALS_FILE).exists():
        return "scored"
    rows = parse_results_csv((out / "results.csv").read_text()) if (out / "results.csv").exists() else []
    if not rows:
        return "pending"
    ep = max(r["epoch"] for r in rows)
    return "trained" if ep >= spec["train"]["epochs"] else f"epoch {ep}/{spec['train']['epochs']}"


def _flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out
