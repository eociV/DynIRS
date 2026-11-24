"""Types of IR-only boxes (vehicles annotated in the IR frame but not matched to an RGB box).

  dup        the same vehicle as an RGB box, displaced by RGB-IR misregistration: the box overlaps
             (IoU >= DUP_IOU) an RGB box that is itself unmatched to any IR box
  invisible  RGB local contrast < FAINT (8-bit gray levels): dark night or hazy/flat RGB frames
  faint      FAINT <= contrast < VISIBLE: lights or a faint outline
  visible    contrast >= VISIBLE: a vehicle visible in RGB that the RGB annotation misses

Contrast of a box on the RGB gray image: std inside the box + |mean inside - mean of a MARGIN-pixel
ring around it|. The thresholds were set by inspecting sample sheets (runs/analysis/types).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dv_pairs import iou_matrix, match

W, H = 640, 512
DUP_IOU = 0.1
FAINT, VISIBLE = 12.0, 25.0
MARGIN = 8
TYPES = ("dup", "invisible", "faint", "visible")


def xywhn_to_xyxy(a):
    a = np.asarray(a, float).reshape(-1, 4)
    x, y, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def box_contrast(gray, boxes, margin=MARGIN):
    """Contrast (gray levels) of each xyxy box on a 2-D gray image (0-255)."""
    h, w = gray.shape[:2]
    g = np.asarray(gray, float)
    out = []
    for b in np.asarray(boxes, float).reshape(-1, 4):
        x1, y1 = int(np.clip(np.floor(b[0]), 0, w - 1)), int(np.clip(np.floor(b[1]), 0, h - 1))
        x2, y2 = max(int(np.clip(np.ceil(b[2]), 0, w)), x1 + 1), max(int(np.clip(np.ceil(b[3]), 0, h)), y1 + 1)
        inner = g[y1:y2, x1:x2]
        X1, Y1, X2, Y2 = max(x1 - margin, 0), max(y1 - margin, 0), min(x2 + margin, w), min(y2 + margin, h)
        ring = g[Y1:Y2, X1:X2].copy()
        ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = np.nan
        ring_mean = np.nanmean(ring) if np.isfinite(ring).any() else inner.mean()
        out.append(float(inner.std() + abs(inner.mean() - ring_mean)))
    return np.array(out)


def classify(contrast, ro_iou):
    """Type of each box from its contrast and its best IoU with an unmatched RGB box."""
    out = []
    for c, r in zip(np.asarray(contrast, float), np.asarray(ro_iou, float)):
        out.append("dup" if r >= DUP_IOU else "invisible" if c < FAINT else "faint" if c < VISIBLE else "visible")
    return out


def read_yolo(path):
    """(n, 5) array of class, x1, y1, x2, y2 (pixels) from a YOLO label file."""
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return np.zeros((0, 5))
    a = np.loadtxt(p, ndmin=2)
    return np.concatenate([a[:, :1], xywhn_to_xyxy(a[:, 1:5])], 1)


def rgb_only(rgb, ir):
    """xyxy of the RGB boxes that dv_pairs.match leaves unmatched."""
    pairs, _ = match(rgb.tolist(), ir.tolist())
    used = {i for i, _, _ in pairs}
    return rgb[[i for i in range(len(rgb)) if i not in used]][:, 1:5].reshape(-1, 4)


def compute_types(proc, split):
    """{image id: [[contrast, ro_iou, type], ...]} in the row order of ironly_cls/<split>/<id>.txt."""
    import cv2

    proc = Path(proc)
    out = {}
    for f in sorted((proc / "ironly_cls" / split).glob("*.txt")):
        if not f.stat().st_size:
            out[f.stem] = []
            continue
        k = f.stem
        g = xywhn_to_xyxy(np.loadtxt(f, ndmin=2)[:, 1:5])
        ro = rgb_only(read_yolo(proc / "rgb" / split / "labels" / f"{k}.txt"),
                      read_yolo(proc / "ir" / split / "labels" / f"{k}.txt"))
        r = iou_matrix(g, ro).max(1) if len(ro) else np.zeros(len(g))
        gray = cv2.imread(str(proc / "rgb" / split / "images" / f"{k}.jpg"), cv2.IMREAD_GRAYSCALE)
        c = box_contrast(gray, g)
        out[k] = [[round(float(a), 3), round(float(b), 4), t] for a, b, t in zip(c, r, classify(c, r))]
    return out


def counts(types):
    flat = [t for v in types.values() for _, _, t in v]
    return {t: flat.count(t) for t in TYPES}


def load_types(proc, split):
    return json.loads((Path(proc) / "stats" / f"boxtypes_{split}.json").read_text())
