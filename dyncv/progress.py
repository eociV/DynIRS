"""Live progress through the session log.

Inside the cloud session a background thread tails every out/<job>/results.csv and prints each new
epoch as one line `DYNCV_METRIC {"run": ..., "epoch": ..., ...}` on stdout. The local side reads the
session log (Kaggle: `kaggle kernels logs`) and parses those lines, so progress reaches MLflow while
the session runs, without credentials inside the session. results.csv pulled at the end of the
session stays the source of truth; both paths log by epoch, so nothing is counted twice.
"""
from __future__ import annotations

import csv
import io
import json
import threading
from pathlib import Path

from .config import METRIC_PREFIX


def parse_results_csv(text: str) -> list[dict]:
    """Ultralytics results.csv -> [{column: float}], column names stripped, epoch as int."""
    if text and not text.endswith("\n"):  # the trainer is still writing the last line
        text = text[:text.rfind("\n") + 1]
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if None in r or any(v is None or not v.strip() for v in r.values()):
            continue  # short or overlong row
        row = {}
        for k, v in r.items():
            if k is None or v is None or not k.strip():
                continue
            try:
                row[k.strip()] = float(v)
            except ValueError:
                continue
        if "epoch" in row:
            row["epoch"] = int(row["epoch"])
            rows.append(row)
    return rows


def format_line(run: str, row: dict) -> str:
    return METRIC_PREFIX + json.dumps({"run": run, **row}, separators=(",", ":"))


def _print(line):
    print(line, flush=True)


def parse_log(text: str) -> dict[str, list[dict]]:
    """Session log (raw text, or Kaggle's JSON list of {"data": ...} entries) -> {run: [rows]}."""
    try:
        entries = json.loads(text)
        if isinstance(entries, list):
            text = "\n".join(str(e.get("data", "")) for e in entries if isinstance(e, dict))
    except ValueError:
        pass
    out: dict[str, list[dict]] = {}
    for line in text.splitlines():
        i = line.find(METRIC_PREFIX)
        if i < 0:
            continue
        try:
            d = json.loads(line[i + len(METRIC_PREFIX):].strip())
        except ValueError:
            continue
        run = d.pop("run", None)
        if run and "epoch" in d:
            out.setdefault(run, []).append(d)
    return out


class Tailer:
    """Print new results.csv rows under `runs` (runs/<job>/ws/out/<job>/results.csv) every `every` s."""

    def __init__(self, runs, every=60.0, emit=_print):
        self.runs, self.every, self.emit = Path(runs), every, emit
        self.seen: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = None

    def poll(self) -> int:
        n = 0
        for f in sorted(self.runs.glob("*/ws/out/*/results.csv")):
            run = f.parent.name
            try:
                rows = parse_results_csv(f.read_text())
            except OSError:
                continue
            last = self.seen.get(run, 0)
            for r in rows:
                if r["epoch"] > last:
                    self.emit(format_line(run, r))
                    last = r["epoch"]
                    n += 1
            self.seen[run] = last
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
            self._thread.join(timeout=5)
        self.poll()
