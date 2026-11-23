"""Per-box weights for IR-only soft labels (DynIRS).

Signals per IR-only train box i (computed once, offline):
  c_i  confidence of the IR teacher: highest score of a teacher prediction on the IR frame with
       IoU >= MATCH_IOU to box i, class-agnostic; 0 if none.
  v_i  RGB visibility: mean BT.601 luminance of the RGB pixels inside box i, scaled to [0, 1].
Variants: T = c, TL = c * (1 - v)**gamma, S = T values permuted across boxes (seed 0).
Every variant is scaled by one constant s and clipped to [FLOOR, CEIL] so that the train-set mean
equals TARGET (the constant weight of the original recipe).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv_pairs import iou_matrix  # noqa: E402

W, H = 640, 512
FLOOR, CEIL, TARGET = 0.05, 1.0, 0.3
MATCH_IOU = 0.5
VARIANTS = ("T", "TL", "S")


def xywhn_to_xyxy(a):
    a = np.asarray(a, float).reshape(-1, 4)
    x, y, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def teacher_conf(boxes, pred_boxes, pred_scores, iou_thr=MATCH_IOU):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    out = np.zeros(len(boxes))
    if len(boxes) == 0 or len(pred_boxes) == 0:
        return out
    m = iou_matrix(boxes, np.asarray(pred_boxes, float).reshape(-1, 4))
    s = np.where(m >= iou_thr, np.asarray(pred_scores, float)[None], 0.0)
    return s.max(1)


def box_luminance(img_rgb, boxes):
    h, w = img_rgb.shape[:2]
    y = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]) / 255.0
    out = []
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        c1 = int(np.clip(np.floor(x1), 0, w - 1))
        r1 = int(np.clip(np.floor(y1), 0, h - 1))
        c2 = max(int(np.clip(np.ceil(x2), 0, w)), c1 + 1)
        r2 = max(int(np.clip(np.ceil(y2), 0, h)), r1 + 1)
        out.append(float(y[r1:r2, c1:c2].mean()))
    return np.array(out)


def raw_weights(c, v, variant, gamma=1.0):
    c, v = np.asarray(c, float), np.asarray(v, float)
    if variant in ("T", "S"):
        return c.copy()
    if variant == "TL":
        return c * (1.0 - v) ** gamma
    raise ValueError(f"unknown variant {variant}")


def mean_match(raw, target=TARGET, lo=FLOOR, hi=CEIL, tol=1e-4):
    raw = np.asarray(raw, float)
    f = lambda s: np.clip(raw * s, lo, hi).mean()  # noqa: E731
    if np.clip(raw * 1e12, lo, hi).mean() < target:
        raise ValueError("target mean unreachable: too few boxes with a positive raw weight")
    a, b = 0.0, 1.0
    while f(b) < target:
        b *= 2
    for _ in range(200):
        s = (a + b) / 2
        if f(s) < target:
            a = s
        else:
            b = s
        if abs(f(b) - target) <= tol:
            break
    return np.clip(raw * b, lo, hi), b


def shuffle(w, seed=0):
    return np.random.default_rng(seed).permutation(np.asarray(w, float))
