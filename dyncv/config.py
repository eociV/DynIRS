"""Every DynCV setting and path in one place.

Change k, the block size or the seed here (or on the command line); the fold file, the queue files and
the run names all derive from them, so two settings never share a file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CVConfig:
    k: int = 5
    block: int = 50  # consecutive train ids stay in one fold: neighbouring frames are near duplicates
    seed: int = 0
    epochs: int = 50  # teacher recipe of scripts/make_dyn_queues.teacher
    hw: str = "t4"  # DataLoader profile: "t4" (2 lanes share 4 CPUs on Kaggle) or "local"

    @property
    def tag(self) -> str:
        return f"cv{self.k}"

    @property
    def folds_file(self) -> Path:
        return ROOT / "configs" / "cv" / f"folds_k{self.k}_b{self.block}_s{self.seed}.json"

    @property
    def queue_dir(self) -> Path:
        return ROOT / "configs" / "queues" / self.tag

    def job_name(self, fold: int) -> str:
        return f"{self.tag}_teacher_26l_f{fold}"

    def variant_dir(self, variant: str) -> str:
        """Name under proc/ironly_w/ (students use config ir_soft_w_dir = "auto:<this>")."""
        return f"{self.tag}_{variant}"


# ------------------------------------------------------------------------------ paths

def data_root() -> Path:
    return Path(os.environ.get("DV_DATA", ROOT / "data" / "dronevehicle"))


def proc_dir() -> Path:
    return data_root() / "proc"


LOCAL = ROOT / "runs" / "cv"  # gitignored (runs/)
LOCAL_RUNS = LOCAL / "remote"  # fetched job workspaces, same layout as runs/remote
STATE_DIR = ROOT / "kaggle_ds" / "cv"  # gitignored; kernels, downloads, session state

SIGNALS_FILE = "cv_signals.json"  # written next to the teacher's weights, travels with the run
SIGNALS_STATS_FILE = "cv_signals_stats.json"
METRIC_PREFIX = "DYNCV_METRIC "


def mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(LOCAL / 'mlflow.db').as_posix()}")


def mlflow_artifacts() -> str:
    return (LOCAL / "mlartifacts").resolve().as_uri()


EXPERIMENT = "dynirs-teacher-cv"
