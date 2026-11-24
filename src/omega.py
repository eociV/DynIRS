"""Per-box weights for IR-only soft labels (DynIRS).

Signals per IR-only train box i (computed once, offline):
  c_i  confidence of the IR teacher: highest score of a teacher prediction on the IR frame with
       IoU >= MATCH_IOU to box i, class-agnostic; 0 if none.
  v_i  RGB visibility: mean BT.601 luminance of the RGB pixels inside box i, scaled to [0, 1].
Variants: T = c, TL = c * (1 - v)**gamma, S = T values permuted across boxes (seed 0).
These are scaled by one constant s and clipped to [FLOOR, CEIL] so that the train-set mean equals
TARGET (the constant weight of the original recipe).

Rank variants share one weight distribution, uniform on [RANK_LO, RANK_HI] (mean TARGET), and
differ only in which boxes get the high weights: the boxes are ranked by a key and the rank is mapped
linearly onto the interval (ties share their mean rank). Keys: RT teacher confidence c, RD darkness
1 - v, RB brightness v, RA box area, RS a random order (seed 0).

Type variants set omega by the type of the box (src/boxtypes.py); each changes one type from the
previous one: C (0.3 for all) -> ND (dup boxes dropped) -> FX (visible boxes at 1.0, i.e. treated as
missing RGB labels) -> NI (invisible boxes dropped). A dropped box (omega 0) is left out of the weight
file, so it is background as in plain RGB training.
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
RANK_KEYS = {"RT": "teacher confidence", "RD": "darkness", "RB": "brightness", "RA": "area", "RS": "random"}
RANK_LO, RANK_HI = 0.05, 0.55
TYPE_VARIANTS = {
    "ND": {"dup": 0.0, "invisible": 0.3, "faint": 0.3, "visible": 0.3},
    "FX": {"dup": 0.0, "invisible": 0.3, "faint": 0.3, "visible": 1.0},
    "NI": {"dup": 0.0, "invisible": 0.0, "faint": 0.3, "visible": 1.0},
}
# Registration-shift / ignore variants: omega (-1 = ignore region: no loss either way) and whether the
# IR-only boxes are moved by their image's RGB-IR shift (median offset of its matched pairs).
REG_VARIANTS = {"SH": (0.3, True), "IG": (-1.0, False), "SI": (-1.0, True)}
MIN_PAIRS = 3  # matched pairs needed to estimate an image's shift
VARIANTS = ("T", "TL", "S") + tuple(RANK_KEYS) + tuple(TYPE_VARIANTS) + tuple(REG_VARIANTS)


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


def rank_weights(key, lo=RANK_LO, hi=RANK_HI):
    """Map the ranks of key linearly onto [lo, hi]; tied keys get the weight of their mean rank."""
    key = np.asarray(key, float)
    n = len(key)
    if n < 2:
        return np.full(n, (lo + hi) / 2)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])  # rank of the first member of each tie group
    r = (first + (cnt - 1) / 2)[inv]
    return lo + (hi - lo) * r / (n - 1)


def rank_key(variant, c, v, area, seed=0):
    c, v, area = (np.asarray(x, float) for x in (c, v, area))
    keys = {"RT": lambda: c, "RD": lambda: 1.0 - v, "RB": lambda: v, "RA": lambda: area,
            "RS": lambda: np.random.default_rng(seed).permutation(len(c)).astype(float)}
    if variant not in keys:
        raise ValueError(f"unknown rank variant {variant}")
    return keys[variant]()


def box_areas(boxes):
    """{image id: [pixel area of each IR-only box]} from load_ironly rows (cls, x, y, w, h normalised)."""
    return {k: (b[:, 3] * W * b[:, 4] * H).tolist() for k, b in boxes.items()}


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
    if abs(f(b) - target) > tol:
        raise ValueError(f"mean matching did not converge: mean {f(b):.5f} vs target {target}")
    return np.clip(raw * b, lo, hi), b


def shuffle(w, seed=0):
    return np.random.default_rng(seed).permutation(np.asarray(w, float))


def load_ironly(proc):
    out = {}
    for f in sorted((Path(proc) / "ironly_cls" / "train").glob("*.txt")):
        a = np.loadtxt(f, ndmin=2, dtype=np.float64) if f.stat().st_size else np.zeros((0, 5))
        out[f.stem] = a.reshape(-1, 5)
    return out


def compute_signals(proc, predict_fn, batch=32):
    import cv2

    proc = Path(proc)
    boxes = load_ironly(proc)
    sig = {k: [] for k in boxes}
    ids = [k for k in boxes if len(boxes[k])]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        preds = predict_fn([str(proc / "ir" / "train" / "images" / f"{k}.jpg") for k in chunk])
        for k, (pb, ps) in zip(chunk, preds):
            xy = xywhn_to_xyxy(boxes[k][:, 1:5])
            rgb = cv2.imread(str(proc / "rgb" / "train" / "images" / f"{k}.jpg"), cv2.IMREAD_COLOR)[..., ::-1]
            c = teacher_conf(xy, pb, ps)
            v = box_luminance(rgb, xy)
            sig[k] = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(c, v)]
        print(f"signals {min(i + batch, len(ids))}/{len(ids)}", flush=True)
    return sig


def build_weights(signals, variant, seed=0, areas=None):
    ids = sorted(signals)
    n = [len(signals[k]) for k in ids]
    arr = np.array([x for k in ids for x in signals[k]], float).reshape(-1, 2)
    if variant in RANK_KEYS:
        if variant == "RA" and areas is None:
            raise ValueError("variant RA needs the box areas")
        a = np.array([x for k in ids for x in areas[k]], float) if areas is not None else np.zeros(len(arr))
        if len(a) != len(arr):
            raise ValueError(f"{len(a)} areas for {len(arr)} boxes")
        w, s = rank_weights(rank_key(variant, arr[:, 0], arr[:, 1], a, seed)), None
    else:
        w, s = mean_match(raw_weights(arr[:, 0], arr[:, 1], variant))
    w = np.round(w, 4)  # the precision write_weights stores
    if abs(w.mean() - TARGET) > 1e-3 or w.min() < FLOOR - 1e-9 or w.max() > CEIL + 1e-9:
        raise ValueError(f"weights out of spec after quantisation: mean {w.mean():.5f}, "
                         f"range [{w.min():.4f}, {w.max():.4f}]")
    if variant == "S":
        w = shuffle(w, seed)
    out, j = {}, 0
    for k, m in zip(ids, n):
        out[k] = w[j:j + m]
        j += m
    hist, edges = np.histogram(w, bins=20, range=(0, 1))
    stats = {"variant": variant, "n_boxes": int(len(w)), "scale": None if s is None else float(s), "mean": float(w.mean()),
             "frac_floor": float(np.mean(w <= FLOOR + 1e-9)), "frac_ceil": float(np.mean(w >= CEIL - 1e-9)),
             "frac_c_zero": float(np.mean(arr[:, 0] == 0)),
             "std": float(w.std()), "iqr": float(np.subtract(*np.percentile(w, [75, 25]))),
             "frac_near_target": float(np.mean(np.abs(w - TARGET) <= 0.05)),
             "c_std": float(arr[:, 0].std()), "c_frac_ge_0_9": float(np.mean(arr[:, 0] >= 0.9)), "hist": hist.tolist(),
             "edges": edges.round(2).tolist()}
    return out, stats


def build_type_weights(types, variant):
    """Per-image weights from box types ({id: [[contrast, ro_iou, type], ...]}) and stats."""
    table = TYPE_VARIANTS[variant]
    out = {k: np.array([table[t] for _, _, t in v], float) for k, v in types.items()}
    w = np.concatenate([x for x in out.values()]) if out else np.zeros(0)
    flat = [t for v in types.values() for _, _, t in v]
    stats = {"variant": variant, "n_boxes": len(w), "omega": table,
             "n_by_type": {t: flat.count(t) for t in table}, "n_dropped": int((w == 0).sum()),
             "mean": float(w.mean()), "mean_kept": float(w[w > 0].mean()) if (w > 0).any() else 0.0}
    return out, stats


def image_shifts(pairs, min_pairs=MIN_PAIRS):
    """{image id: (dx, dy) pixels}, IR minus RGB centre, the median over the image's matched pairs
    (dv_pairs: proc/stats/pairs_<split>.json); images with fewer than min_pairs pairs are left out."""
    out = {}
    for k, v in pairs.items():
        o = np.asarray(v["offsets"], float).reshape(-1, 2)
        if len(o) >= min_pairs:
            out[k] = tuple(float(x) for x in np.median(o, 0))
    return out


def shift_boxes(rows, d):
    """Rows "cls cx cy w h" (normalized) moved by -d pixels (IR frame -> RGB frame), clipped to the image."""
    rows = np.array(rows, float).reshape(-1, 5)
    if not len(rows) or not any(d):
        return rows
    xy = xywhn_to_xyxy(rows[:, 1:5]) - np.array([d[0], d[1], d[0], d[1]])
    xy = np.clip(xy, 0, [W, H, W, H])
    rows[:, 1], rows[:, 2] = (xy[:, 0] + xy[:, 2]) / 2 / W, (xy[:, 1] + xy[:, 3]) / 2 / H
    rows[:, 3], rows[:, 4] = (xy[:, 2] - xy[:, 0]) / W, (xy[:, 3] - xy[:, 1]) / H
    return rows


def build_reg_weights(boxes, pairs, variant):
    """(boxes, weights, stats) of a REG_VARIANTS variant from load_ironly boxes and pairs_train.json."""
    omega, move = REG_VARIANTS[variant]
    sh = image_shifts(pairs) if move else {}
    nb = {k: shift_boxes(b, sh[k]) if k in sh else b for k, b in boxes.items()}
    w = {k: np.full(len(b), omega) for k, b in nb.items()}
    n = sum(len(b) for b in boxes.values())
    moved = [float(np.hypot(*sh[k])) for k, b in boxes.items() if k in sh for _ in range(len(b))]
    stats = {"variant": variant, "omega": omega, "shift": move, "n_boxes": n,
             "n_in_shift_images": len(moved), "n_moved_gt_2px": int(sum(m > 2 for m in moved)),
             "shift_px_p50_p90": [float(np.percentile(moved, q)) for q in (50, 90)] if moved else None}
    return nb, w, stats


def write_weights(proc, variant, boxes, weights):
    out = Path(proc) / "ironly_w" / variant / "train"
    out.mkdir(parents=True, exist_ok=True)
    for k in sorted(boxes):
        b, w = boxes[k], weights.get(k, np.zeros(0))
        if len(b) != len(w):
            raise ValueError(f"{k}: {len(b)} boxes but {len(w)} weights")
        (out / f"{k}.txt").write_text("".join(  # omega 0: dropped, the box stays background
            f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {x:.4f}\n" for r, x in zip(b, w)
            if x != 0 and r[3] > 0 and r[4] > 0))  # omega -1: ignore region (lowlight IGNORE_CLS)
    return out


def _ultralytics_predictor(teacher, imgsz, batch):
    from ultralytics import YOLO

    m = YOLO(str(teacher))

    def fn(paths):
        res = m.predict(source=paths, imgsz=imgsz, conf=0.001, max_det=300, half=True, verbose=False, batch=batch)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in res]

    return fn


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signals")
    s.add_argument("--proc", required=True)
    s.add_argument("--teacher", required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=32)
    t = sub.add_parser("types")
    t.add_argument("--proc", required=True)
    t.add_argument("--splits", nargs="+", default=["train", "val"])
    w = sub.add_parser("weights")
    w.add_argument("--proc", required=True)
    w.add_argument("--variant", choices=VARIANTS, required=True)
    a = ap.parse_args(argv)
    proc = Path(a.proc)
    if a.cmd == "signals":
        sig = compute_signals(proc, _ultralytics_predictor(a.teacher, a.imgsz, a.batch), a.batch)
        (proc / "stats" / "omega_signals.json").write_text(json.dumps(sig))
        flat = np.array([x for v in sig.values() for x in v]).reshape(-1, 2)
        print(json.dumps({"n_boxes": len(flat), "c_mean": float(flat[:, 0].mean()),
                          "frac_c_zero": float(np.mean(flat[:, 0] == 0)), "v_mean": float(flat[:, 1].mean())}))
    elif a.cmd == "types":
        import boxtypes as bt

        for split in a.splits:
            ty = bt.compute_types(proc, split)
            (proc / "stats" / f"boxtypes_{split}.json").write_text(json.dumps(ty))
            print(split, json.dumps(bt.counts(ty)))
    elif a.variant in REG_VARIANTS:
        pairs = json.loads((proc / "stats" / "pairs_train.json").read_text())
        boxes, weights, stats = build_reg_weights(load_ironly(proc), pairs, a.variant)
        write_weights(proc, a.variant, boxes, weights)
        (proc / "stats" / f"omega_{a.variant}.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats))
    elif a.variant in TYPE_VARIANTS:
        import boxtypes as bt

        boxes = load_ironly(proc)
        weights, stats = build_type_weights(bt.load_types(proc, "train"), a.variant)
        write_weights(proc, a.variant, boxes, weights)
        (proc / "stats" / f"omega_{a.variant}.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats))
    else:
        sig = json.loads((proc / "stats" / "omega_signals.json").read_text())
        boxes = load_ironly(proc)
        weights, stats = build_weights(sig, a.variant, areas=box_areas(boxes))
        write_weights(proc, a.variant, boxes, weights)
        (proc / "stats" / f"omega_{a.variant}.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps({k: stats[k] for k in ("variant", "n_boxes", "scale", "mean", "frac_floor", "frac_ceil", "std", "iqr", "frac_near_target", "c_std", "c_frac_ge_0_9")}))


if __name__ == "__main__":
    main()
