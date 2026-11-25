"""Sample sheets of the cross-fitted RGB teacher's confidence on IR-only boxes (train split).

  python src/rgb_teacher_demo.py

Each IR-only box is scored by the fold model that never saw its image (job_main.fold_of, 2 folds):
the highest score of a teacher detection with IoU >= 0.5 to the box (omega.teacher_conf, class-agnostic).
Writes runs/analysis/rgb_teacher/{conf_*.jpg, night_invisible_seen.jpg, visible_unseen.jpg, full_*.jpg}
and rgb_conf_train.json ({id: [conf per IR-only box, in ironly_cls row order]}).
Crop colours: yellow = the IR-only box, cyan = teacher detections (score >= 0.25), green = RGB labels.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import boxtypes as bt  # noqa: E402
import job_main as jm  # noqa: E402
import omega as om  # noqa: E402

DATA = ROOT / "data" / "dronevehicle"
D = DATA / "proc"
RUNS = ROOT / "runs" / "remote"
OUT = ROOT / "runs" / "analysis" / "rgb_teacher"
NAME = "dyn_teacher_rgb26l_f{}"
CELL, COLS, ROWS = 200, 6, 4
SHOW = 0.25


def heldout_preds():
    """{image id: (n, 5) xyxy + score} from the fold model that did not train on the image."""
    jm.DATA = DATA
    fold = jm.fold_of(2, 0)
    out = {}
    for f in (0, 1):
        n = NAME.format(f)
        for p in json.loads((RUNS / n / "ws" / "out" / n / "preds_train.json").read_text()):
            if fold.get(p["image_id"]) == f:
                out.setdefault(p["image_id"], []).append(p["bbox"] + [p["score"]])
    return {k: np.array(v, float) for k, v in out.items()}


def rgb_conf(preds, boxes):
    out = {}
    for k, b in boxes.items():
        p = preds.get(k, np.zeros((0, 5)))
        out[k] = om.teacher_conf(om.xywhn_to_xyxy(b[:, 1:5]), p[:, :4], p[:, 4]) if len(b) else np.zeros(0)
    return out


def crop(k, box, preds, label):
    img = cv2.imread(str(D / "rgb" / "train" / "images" / f"{k}.jpg"))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(box[2] - box[0], box[3] - box[1]) * 3.0
    s = max(s, 96)
    x1, y1 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    x2, y2 = int(min(cx + s / 2, bt.W)), int(min(cy + s / 2, bt.H))
    g = img[y1:y2, x1:x2].copy()
    f = CELL / max(g.shape[:2])
    g = cv2.resize(g, (max(int(g.shape[1] * f), 1), max(int(g.shape[0] * f), 1)), interpolation=cv2.INTER_AREA)
    cell = np.zeros((CELL + 28, CELL, 3), np.uint8)
    cell[:g.shape[0], :g.shape[1]] = g

    def rect(b, col, t=1):
        a = ((np.array(b[:4]) - [x1, y1, x1, y1]) * f).round().astype(int)
        cv2.rectangle(cell, (a[0], a[1]), (a[2], a[3]), col, t)

    for r in bt.read_yolo(D / "rgb" / "train" / "labels" / f"{k}.txt"):
        rect(r[1:5], (0, 200, 0))
    for p in preds.get(k, np.zeros((0, 5))):
        if p[4] >= SHOW:
            rect(p, (255, 255, 0))
    rect(box, (0, 255, 255), 2)
    for i, line in enumerate(label):
        cv2.putText(cell, line, (3, CELL + 11 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return cell


def sheet(path, picks, preds, boxes, title):
    cells = [crop(k, om.xywhn_to_xyxy(boxes[k][j:j + 1, 1:5])[0], preds, lab) for k, j, lab in picks]
    blank = np.zeros((CELL + 28, CELL, 3), np.uint8)
    cells += [blank] * (COLS * ROWS - len(cells))
    rows = [np.hstack(cells[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    head = np.zeros((30, CELL * COLS, 3), np.uint8)
    cv2.putText(head, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.vstack([head] + rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def full(path, k, conf, boxes, preds):
    img = cv2.imread(str(D / "rgb" / "train" / "images" / f"{k}.jpg"))
    for r in bt.read_yolo(D / "rgb" / "train" / "labels" / f"{k}.txt"):
        a = r[1:5].astype(int)
        cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (0, 200, 0), 1)
    for b, c in zip(om.xywhn_to_xyxy(boxes[k][:, 1:5]), conf[k]):
        a = b.astype(int)
        col = (0, 0, 255) if c < 0.05 else (0, 165, 255) if c < 0.3 else (0, 255, 255) if c < 0.6 else (255, 255, 0)
        cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), col, 2)
        cv2.putText(img, f"{c:.2f}", (a[0], max(a[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    ir = cv2.imread(str(D / "ir" / "train" / "images" / f"{k}.jpg"))
    cv2.imwrite(str(path), np.hstack([img, ir]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = heldout_preds()
    boxes = {k: v for k, v in om.load_ironly(D).items() if len(v)}
    conf = rgb_conf(preds, boxes)
    (OUT / "rgb_conf_train.json").write_text(json.dumps({k: [round(float(x), 4) for x in v] for k, v in conf.items()}))
    types = bt.load_types(D, "train")
    night = set((ROOT / "splits" / "train_night.txt").read_text().split())
    rows = [(k, j, float(c), types[k][j][2], types[k][j][0], k in night)
            for k, v in conf.items() for j, c in enumerate(v)]
    rng = np.random.default_rng(0)

    def pick(sel, n=COLS * ROWS):
        idx = [i for i, r in enumerate(rows) if sel(r)]
        idx = rng.choice(idx, min(n, len(idx)), replace=False) if idx else []
        return [(rows[i][0], rows[i][1], [f"conf {rows[i][2]:.2f}  {rows[i][3]}",
                                          f"contrast {rows[i][4]:.0f}  {'night' if rows[i][5] else 'day/tw'}"])
                for i in idx]

    for lo, hi in ((0, 0.05), (0.05, 0.3), (0.3, 0.6), (0.6, 1.01)):
        sheet(OUT / f"conf_{lo:.2f}-{min(hi, 1):.2f}.jpg", pick(lambda r: lo <= r[2] < hi), preds, boxes,
              f"RGB-teacher conf {lo:.2f}-{min(hi, 1):.2f} on IR-only boxes (yellow); cyan = teacher dets, green = RGB labels")
    sheet(OUT / "night_invisible_seen.jpg", pick(lambda r: r[5] and r[3] == "invisible" and r[2] >= 0.5), preds, boxes,
          "Night, 'invisible' by contrast, yet RGB-teacher conf >= 0.5")
    sheet(OUT / "visible_unseen.jpg", pick(lambda r: r[3] == "visible" and r[2] < 0.05), preds, boxes,
          "'Visible' by contrast, yet RGB-teacher conf < 0.05")
    per = sorted(((sum(1 for c in conf[k] if c >= 0.5), k) for k in conf if k in night), reverse=True)
    for i, (_, k) in enumerate(per[:2]):
        full(OUT / f"full_night_{i}_{k}.jpg", k, conf, boxes, preds)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
