"""Clean IR-only labels for single-class RGB training (docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md).

  python src/clean_labels.py build --proc data/dronevehicle/proc [--runs runs/remote] [--variants P I V F]
  python src/clean_labels.py sheet --proc data/dronevehicle/proc [--runs runs/remote]

Variants of the IR-only boxes (proc/ironly_cls/train), all written as class 0 to
proc/ironly_w/<variant>/train/<id>.txt ("0 cx cy w h omega"; omega -1 = ignore region):
  P  every box, omega 0.3, not moved                  (the original recipe)
  I  every box as an ignore region, not moved
  V  steps 3-4 on unmoved boxes
  F  steps 1-4:
     1 registration: images with >= MIN_PAIRS matched pairs move their IR-only boxes by -d,
       d = median (IR - RGB) centre offset of the pairs (proc/stats/pairs_train.json)
     2 misaligned images: residual = median |offset - d| > MAX_RESIDUAL px -> every box ignored
     3 verification: c = best score of a held-out RGB-teacher detection with IoU >= 0.5 to the box
     4 decision: c >= TAU taught with omega c, else ignored
The teacher is cross-fitted (dyn_teacher_rgb1c_f{0,1}): a train image is scored by the fold model that
did not train on it (job_main.fold_of, 2 folds, seed 0). Stats: proc/stats/clean_<variant>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import job_main as jm  # noqa: E402
import omega as om  # noqa: E402

MIN_PAIRS = 3
MAX_RESIDUAL = 4.0
TAU = 0.5
P_OMEGA = 0.3
IGNORE = -1.0
VARIANTS = ("P", "I", "V", "F")
TEACHER = "dyn_teacher_rgb1c_f{}"


def residuals(pairs, min_pairs=MIN_PAIRS):
    """{image id: px} median distance of the image's pair offsets to their median (images with >= min_pairs)."""
    out = {}
    for k, v in pairs.items():
        o = np.asarray(v["offsets"], float).reshape(-1, 2)
        if len(o) >= min_pairs:
            out[k] = float(np.median(np.hypot(*(o - np.median(o, 0)).T)))
    return out


def heldout_preds(runs, data, name=TEACHER, k=2, seed=0):
    """{train image id: (n, 5) xyxy + score} from the fold model that did not train on the image."""
    fold = jm.fold_of(k, seed, data)
    out = {}
    for f in range(k):
        n = name.format(f)
        for p in json.loads((Path(runs) / n / "ws" / "out" / n / "preds_train.json").read_text()):
            if fold.get(p["image_id"]) == f:
                out.setdefault(p["image_id"], []).append(list(p["bbox"]) + [p["score"]])
    return {i: np.asarray(v, float).reshape(-1, 5) for i, v in out.items()}


def decide(conf, tau=TAU):
    conf = np.asarray(conf, float)
    return np.where(conf >= tau, conf, IGNORE)


def _conf(rows, p):
    if p is None or not len(p) or not len(rows):
        return np.zeros(len(rows))
    return om.teacher_conf(om.xywhn_to_xyxy(rows[:, 1:5]), p[:, :4], p[:, 4])


def build_variant(variant, boxes, pairs, preds):
    """(rows, weights, stats) for one variant; rows are class-0 normalized "cls cx cy w h"."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    move = variant == "F"
    sh = om.image_shifts(pairs, MIN_PAIRS) if move else {}
    res = residuals(pairs) if move else {}
    rows, w = {}, {}
    n_res = n_res_img = n_moved = 0
    for k, b in boxes.items():
        b = np.array(b, float).reshape(-1, 5)
        b[:, 0] = 0
        if k in sh:
            b = om.shift_boxes(b, sh[k])
            n_moved += len(b) * (np.hypot(*sh[k]) > 2)
        if variant == "P":
            x = np.full(len(b), P_OMEGA)
        elif variant == "I":
            x = np.full(len(b), IGNORE)
        elif move and res.get(k, 0.0) > MAX_RESIDUAL:
            x = np.full(len(b), IGNORE)
            n_res += len(b)
            n_res_img += bool(len(b))
        else:
            x = decide(_conf(b, preds.get(k)))
        rows[k], w[k] = b, x
    allw = np.concatenate(list(w.values())) if w else np.zeros(0)
    taught = allw[allw > 0]
    stats = {"variant": variant, "n_boxes": int(len(allw)), "n_taught": int(len(taught)),
             "n_ignored": int((allw < 0).sum()), "n_ignored_residual": int(n_res),
             "n_images_residual": int(n_res_img), "n_moved_gt_2px": int(n_moved),
             "omega_taught_mean": float(taught.mean()) if len(taught) else 0.0,
             "omega_taught_p50": float(np.median(taught)) if len(taught) else 0.0,
             "min_pairs": MIN_PAIRS, "max_residual": MAX_RESIDUAL, "tau": TAU}
    return rows, w, stats


def sheet(proc, runs, out):
    """Sanity sheets for F: taught, ignored by the teacher, ignored by the residual (24 random crops each)."""
    import rgb_teacher_demo as rd

    data = Path(proc).parent
    preds = heldout_preds(runs, data)
    boxes = {k: v for k, v in om.load_ironly(proc).items() if len(v)}
    pairs = json.loads((Path(proc) / "stats" / "pairs_train.json").read_text())
    rows, w, _ = build_variant("F", boxes, pairs, preds)
    res = residuals(pairs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    groups = {"taught": lambda k, j: w[k][j] > 0,
              "ignored_teacher": lambda k, j: w[k][j] < 0 and res.get(k, 0.0) <= MAX_RESIDUAL,
              "ignored_residual": lambda k, j: w[k][j] < 0 and res.get(k, 0.0) > MAX_RESIDUAL}
    for g, sel in groups.items():
        cand = [(k, j) for k in rows for j in range(len(rows[k])) if sel(k, j)]
        idx = rng.choice(len(cand), min(24, len(cand)), replace=False) if cand else []
        picks = [(cand[i][0], cand[i][1], [f"omega {w[cand[i][0]][cand[i][1]]:.2f}",
                                           f"residual {res.get(cand[i][0], float('nan')):.1f}px"]) for i in idx]
        rd.sheet(out / f"F_{g}.jpg", picks, preds, rows, f"F: {g} (yellow = moved IR-only box)")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--proc", required=True)
    b.add_argument("--runs", default=str(ROOT / "runs" / "remote"))
    b.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    s = sub.add_parser("sheet")
    s.add_argument("--proc", required=True)
    s.add_argument("--runs", default=str(ROOT / "runs" / "remote"))
    s.add_argument("--out", default=str(ROOT / "runs" / "analysis" / "clean"))
    a = ap.parse_args(argv)
    proc = Path(a.proc)
    if a.cmd == "sheet":
        sheet(proc, a.runs, a.out)
        return
    boxes = om.load_ironly(proc)
    pairs = json.loads((proc / "stats" / "pairs_train.json").read_text())
    preds = heldout_preds(a.runs, proc.parent) if set(a.variants) & {"V", "F"} else {}
    for v in a.variants:
        rows, w, st = build_variant(v, boxes, pairs, preds)
        d = proc / "ironly_w" / v / "train"  # write_weights does not empty it: drop a previous build
        if d.is_dir():
            for f in d.glob("*.txt"):
                f.unlink()
        om.write_weights(proc, v, rows, w)
        (proc / "stats" / f"clean_{v}.json").write_text(json.dumps(st, indent=1))
        print(json.dumps(st))


if __name__ == "__main__":
    main()
