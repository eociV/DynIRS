"""Stage-wise dynamic omega and repeat sampling of IR-only boxes
(docs/superpowers/specs/2026-11-27-dynamic-omega-balance-design.md).

Weight files: <dir>/<image id>.txt, rows "cls cx cy w h omega" (normalised boxes; omega -1 = ignore region).
At a stage boundary the student scores each box (s = best score of a detection with IoU >= 0.5):
  s >= S_HI                    -> omega = max(omega, s)
  S_LO <= s < S_HI             -> omega unchanged
  s < S_LO and teacher c < S_LO -> ignore region (omega -1)
  s < S_LO and c >= S_LO        -> omega unchanged
Ignore regions (omega < 0) never change. Images not scored are copied unchanged.
Repeat sampling: an image with a box of omega >= REPEAT_OMEGA is listed R times per epoch (job_main).
"""
import json
from pathlib import Path

import numpy as np

from dv_pairs import iou_matrix

W, H = 640, 512
MATCH_IOU = 0.5
S_HI, S_LO = 0.5, 0.05
REPEAT_OMEGA = 0.5


def read_dir(d):
    out = {}
    for f in sorted(Path(d).glob("*.txt")):
        a = np.loadtxt(f, ndmin=2, dtype=np.float64) if f.stat().st_size else np.zeros((0, 6))
        out[f.stem] = a.reshape(-1, 6)
    return out


def write_dir(d, rows):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for k, a in rows.items():
        (d / f"{k}.txt").write_text("".join(
            f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]:.4f}\n" for r in a))


def _xyxy(a):
    a = np.asarray(a, float).reshape(-1, 4)
    x, y, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def box_scores(rows, preds):
    """{id: (n,) best detection score with IoU >= MATCH_IOU per box}; preds {id: (m, 5) xyxy px + score}."""
    out = {}
    for k, a in rows.items():
        p = preds.get(k)
        if p is None or not len(p) or not len(a):
            out[k] = np.zeros(len(a))
            continue
        p = np.asarray(p, float).reshape(-1, 5)
        m = iou_matrix(_xyxy(a[:, 1:5]), p[:, :4])
        out[k] = np.where(m >= MATCH_IOU, p[None, :, 4], 0.0).max(1)
    return out


def load_conf(path, rows):
    """{id: (n,) teacher score per row}; json {id: [score per written row]}; missing ids -> zeros."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for k, a in rows.items():
        v = np.asarray(raw.get(k, [0.0] * len(a)), float)
        if len(v) != len(a):
            raise ValueError(f"{k}: {len(a)} weight rows but {len(v)} teacher scores")
        out[k] = v
    return out


def rescore(rows, s, c):
    """(new rows, stats) after one stage; s and c: {id: (n,)} for the scored images only."""
    new = {}
    n = {"raised": 0, "kept": 0, "ignored": 0, "frozen": 0}
    for k, a in rows.items():
        a = np.array(a, float, copy=True)
        if k not in s:
            new[k] = a
            continue
        w, sk, ck = a[:, 5].copy(), np.asarray(s[k], float), np.asarray(c[k], float)
        frozen = w < 0
        hi = ~frozen & (sk >= S_HI)
        ign = ~frozen & (sk < S_LO) & (ck < S_LO)
        a[hi, 5] = np.maximum(w[hi], sk[hi])
        a[ign, 5] = -1.0
        n["raised"] += int((hi & (sk > w)).sum())
        n["ignored"] += int(ign.sum())
        n["frozen"] += int(frozen.sum())
        n["kept"] += int((~frozen & ~ign).sum() - (hi & (sk > w)).sum())
        new[k] = a
    taught = np.concatenate([a[a[:, 5] > 0, 5] for a in new.values()] or [np.zeros(0)])
    stats = dict(n, n_images_scored=len(s), omega_taught_mean=float(taught.mean()) if len(taught) else 0.0,
                 n_taught=int(len(taught)), s_hi=S_HI, s_lo=S_LO)
    return new, stats


def repeat_ids(rows, thr=REPEAT_OMEGA):
    return sorted(k for k, a in rows.items() if len(a) and (a[:, 5] >= thr).any())
