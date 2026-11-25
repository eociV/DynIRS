"""k fold signal files -> one signals file -> per-box weight files.

The weights reuse src/omega.py unchanged (mean matched to 0.3, floor 0.05, S = permuted T), so the
cross-fitted variants differ from the in-sample ones only in where c_i comes from. Writes:
  proc/stats/omega_signals_<tag>.json           merged {id: [[c, v], ...]}
  proc/ironly_w/<tag>_<V>/train/<id>.txt        per variant V in T, TL, S
  proc/stats/omega_<tag>_<V>.json               weight statistics (as omega_<V>.json)
  proc/stats/omega_<tag>_compare.json           cross-fitted vs in-sample c_i (if omega_signals.json exists)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .config import ROOT, SIGNALS_FILE, CVConfig
from .score import job_out, signal_stats

sys.path.insert(0, str(ROOT / "src"))
import omega  # noqa: E402


def merge_signals(fold_sigs: list[dict], boxes: dict) -> dict:
    """Union of the fold signals; every id with IR-only boxes must appear in exactly one fold with one
    (c, v) pair per box. Ids without boxes get [] (the layout omega.build_weights expects)."""
    out = {k: [] for k in boxes}
    seen = {}
    for f, sig in enumerate(fold_sigs):
        for k, v in sig.items():
            if k in seen:
                raise ValueError(f"image {k} scored in folds {seen[k]} and {f}")
            if k not in boxes:
                raise ValueError(f"image {k} (fold {f}) has no IR-only label file")
            if len(v) != len(boxes[k]):
                raise ValueError(f"image {k}: {len(boxes[k])} boxes but {len(v)} signals")
            seen[k] = f
            out[k] = v
    missing = [k for k, b in boxes.items() if len(b) and k not in seen]
    if missing:
        raise ValueError(f"{len(missing)} images with IR-only boxes are in no fold, e.g. {missing[:5]}")
    return out


def compare(cv_sig: dict, insample: dict) -> dict:
    ids = [k for k in cv_sig if cv_sig[k] and len(insample.get(k, ())) == len(cv_sig[k])]
    a = np.array([x[0] for k in ids for x in cv_sig[k]], float)
    b = np.array([x[0] for k in ids for x in insample[k]], float)
    if not len(a):
        return {"n_boxes": 0}
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {"n_boxes": int(len(a)), "cv": signal_stats({k: cv_sig[k] for k in ids}),
            "insample": signal_stats({k: insample[k] for k in ids}),
            "mean_diff_cv_minus_insample": float((a - b).mean()), "pearson": corr}


def fold_signal_files(cfg: CVConfig, runs) -> list[Path]:
    files = [job_out(runs, cfg.job_name(f)) / SIGNALS_FILE for f in range(cfg.k)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("fold signals missing (train and score every fold first):\n  " + "\n  ".join(missing))
    return files


def run(cfg: CVConfig, runs, proc, variants=omega.VARIANTS) -> dict:
    proc = Path(proc)
    boxes = omega.load_ironly(proc)
    sig = merge_signals([json.loads(p.read_text()) for p in fold_signal_files(cfg, runs)], boxes)
    stats_dir = proc / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"omega_signals_{cfg.tag}.json").write_text(json.dumps(sig))
    report = {"signals": signal_stats({k: v for k, v in sig.items() if v}), "variants": {}}
    for v in variants:
        weights, st = omega.build_weights(sig, v)
        name = cfg.variant_dir(v)
        omega.write_weights(proc, name, boxes, weights)
        st["dir"] = name
        (stats_dir / f"omega_{name}.json").write_text(json.dumps(st, indent=1))
        report["variants"][name] = {k: st[k] for k in ("mean", "std", "iqr", "frac_floor", "frac_ceil",
                                                       "frac_near_target", "c_std", "c_frac_ge_0_9")}
    ins = stats_dir / "omega_signals.json"
    if ins.exists():
        report["compare_insample"] = compare(sig, json.loads(ins.read_text()))
        (stats_dir / f"omega_{cfg.tag}_compare.json").write_text(json.dumps(report["compare_insample"], indent=1))
    return report
