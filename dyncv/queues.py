"""One teacher job per fold, built from the in-sample teacher of scripts/make_dyn_queues.py.

Only four fields differ from that recipe: the name, the training data (data "ir_proxy" + the ids of
the other folds, which src/job_main.py already supports), the DataLoader profile of the hardware, and
an extra "cv" block (ignored by job_main) that tells the scorer which fold the job holds out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import folds as F
from .config import ROOT, CVConfig

sys.path.insert(0, str(ROOT / "scripts"))
import make_dyn_queues  # noqa: E402


def teacher_job(cfg: CVConfig, fold: int, train_ids: list[str]) -> dict:
    name = cfg.job_name(fold)
    job = make_dyn_queues.teacher(cfg.epochs, name=name)
    job["train"].update(data="ir_proxy", **make_dyn_queues.HW[cfg.hw])
    job["proxy_ids"] = list(train_ids)
    job["cv"] = {"k": cfg.k, "fold": fold, "block": cfg.block, "seed": cfg.seed,
                 "folds_file": cfg.folds_file.relative_to(ROOT).as_posix()}
    return job


def build(cfg: CVConfig, folds: list[list[str]]) -> list[dict]:
    return [teacher_job(cfg, f, F.training_ids(folds, f)) for f in range(cfg.k)]


def default_groups(k: int, lanes: int = 2) -> list[list[int]]:
    """Folds per cloud session: one fold per GPU lane (Kaggle T4 x2 -> pairs)."""
    return [list(range(i, min(i + lanes, k))) for i in range(0, k, lanes)]


def queue_stem(cfg: CVConfig, group: list[int]) -> str:
    return f"{cfg.tag}_f{''.join(map(str, group))}"


def write(cfg: CVConfig, folds, groups=None) -> list[Path]:
    """configs/queues/cv<k>/<tag>_all.json plus one file per group (one file = one cloud session
    chain, so groups can go to different accounts at the same time)."""
    jobs = build(cfg, folds)
    cfg.queue_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for stem, js in [(f"{cfg.tag}_all", jobs)] + [(queue_stem(cfg, g), [jobs[f] for f in g])
                                                   for g in (groups or default_groups(cfg.k))]:
        p = cfg.queue_dir / f"{stem}.json"
        p.write_text(json.dumps(js, indent=1))
        out.append(p)
    return out


def load(queue) -> list[dict]:
    return json.loads(Path(queue).read_text())


def cv_jobs(specs) -> list[dict]:
    return [s for s in specs if "cv" in s]
