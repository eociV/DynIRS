"""Signals (c_i, v_i) for the IR-only boxes of one held-out fold.

Same definitions as src/omega.py (teacher_conf, box_luminance, IoU 0.5, conf 0.001, 300 detections),
restricted to the fold's ids and scored by the teacher that never trained on them. Output, next to the
teacher's weights so it travels with the run:
  out/<job>/cv_signals.json        {id: [[c, v], ...]} for every held-out id with IR-only boxes
  out/<job>/cv_signals_stats.json  summary (n_boxes, c_mean, c_std, frac_c_zero, c_frac_ge_0_9, ...)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from . import folds as F
from .config import ROOT, SIGNALS_FILE, SIGNALS_STATS_FILE

sys.path.insert(0, str(ROOT / "src"))
import omega  # noqa: E402


def job_out(runs, name) -> Path:
    return Path(runs) / name / "ws" / "out" / name


def fold_signals(proc, ids, predict_fn, batch=32, log=print):
    proc = Path(proc)
    boxes = omega.load_ironly(proc)
    todo = [k for k in ids if len(boxes.get(k, ()))]
    import cv2

    sig = {}
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        preds = predict_fn([str(proc / "ir" / "train" / "images" / f"{k}.jpg") for k in chunk])
        for k, (pb, ps) in zip(chunk, preds):
            xy = omega.xywhn_to_xyxy(boxes[k][:, 1:5])
            rgb = cv2.imread(str(proc / "rgb" / "train" / "images" / f"{k}.jpg"), cv2.IMREAD_COLOR)[..., ::-1]
            c, v = omega.teacher_conf(xy, pb, ps), omega.box_luminance(rgb, xy)
            sig[k] = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(c, v)]
        log(f"signals {min(i + batch, len(todo))}/{len(todo)}")
    return sig


def signal_stats(sig) -> dict:
    flat = np.array([x for v in sig.values() for x in v], float).reshape(-1, 2)
    if not len(flat):
        return {"n_images": len(sig), "n_boxes": 0}
    c = flat[:, 0]
    return {"n_images": len(sig), "n_boxes": int(len(flat)), "c_mean": float(c.mean()), "c_std": float(c.std()),
            "c_median": float(np.median(c)), "frac_c_zero": float(np.mean(c == 0)),
            "c_frac_ge_0_9": float(np.mean(c >= 0.9)), "v_mean": float(flat[:, 1].mean())}


def is_trained(spec, runs) -> bool:
    """Training finished (epoch -1 in last.pt) and best.pt exists."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_queue

    out = job_out(runs, spec["name"])
    return (out / "weights" / "best.pt").exists() and run_queue.epochs_done(out) == "finished"


def is_scored(spec, runs) -> bool:
    try:
        json.loads((job_out(runs, spec["name"]) / SIGNALS_FILE).read_text())
        return True
    except (OSError, ValueError):
        return False


def score_job(spec, runs, proc, imgsz=640, batch=32, weights=None, log=print) -> Path:
    """Score the held-out fold of one cv teacher job; returns the signals file."""
    cv = spec["cv"]
    d = F.load(ROOT / cv["folds_file"])
    if (d["k"], d["block"], d["seed"]) != (cv["k"], cv["block"], cv["seed"]):
        raise ValueError(f"{spec['name']}: fold file {cv['folds_file']} does not match the job's cv block")
    if set(spec.get("proxy_ids", ())) & set(d["folds"][cv["fold"]]):
        raise ValueError(f"{spec['name']}: the teacher trained on images of its held-out fold")
    out = job_out(runs, spec["name"])
    w = Path(weights) if weights else out / "weights" / "best.pt"
    log(f"scoring fold {cv['fold']} with {w}")
    sig = fold_signals(proc, F.held_out(d["folds"], cv["fold"]), omega._ultralytics_predictor(w, imgsz, batch),
                       batch, log)
    stats = dict(signal_stats(sig), fold=cv["fold"], weights=str(w))
    _write_atomic(out / SIGNALS_STATS_FILE, json.dumps(stats, indent=1))
    _write_atomic(out / SIGNALS_FILE, json.dumps(sig))  # last: its presence marks the job as scored
    log(json.dumps(stats))
    return out / SIGNALS_FILE


def score_pending(specs, runs, proc, log=print) -> list[str]:
    """Score every trained, unscored cv job of a queue. Returns the scored job names."""
    done = []
    for s in specs:
        if "cv" in s and is_trained(s, runs) and not is_scored(s, runs):
            score_job(s, runs, proc, imgsz=s["train"].get("imgsz", 640), log=log)
            done.append(s["name"])
    return done


def _write_atomic(dst, text):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, dst)
