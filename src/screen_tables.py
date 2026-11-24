"""Score the omega screening runs (dyn_scr_<cfg>_s<seed>) on the val split and write a table.

  python src/screen_tables.py

Per run: COCO AP50:95 (%) against the RGB and IR labels (all / night images); recall (%) of IR-only
vehicles at night, and over all images by box type (src/boxtypes.py: dup, invisible, faint, visible;
from proc/stats/boxtypes_val.json); night false alarms (detections matching neither annotation).
Detections count at score >= 0.25 and IoU >= 0.5, class-agnostic. Per config: mean +- SD over seeds,
and the seed-paired difference to B (no IR labels) and to C (constant omega 0.3).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import dyn_stats as ds  # noqa: E402
import evaluate as ev  # noqa: E402
from dv_pairs import iou_matrix  # noqa: E402
import boxtypes as bt  # noqa: E402
from omega import xywhn_to_xyxy  # noqa: E402

D = ROOT / "data" / "dronevehicle" / "proc"
MAN = ROOT / "splits"
RUNS = ROOT / "runs" / "remote"
OUT = ROOT / "runs" / "analysis" / "screen"
SPLIT = "val"
CONFIGS = ("B", "C", "ND", "FX", "NI")
REFS = ("B", "C")
THR, IOU = 0.25, 0.5
KEYS = (["rgb_all", "rgb_night", "ir_all", "ir_night", "ironly_rec_night"] + [f"rec_{t}" for t in bt.TYPES]
        + ["false_alarms"])


def night_ids():
    return set((MAN / f"{SPLIT}_night.txt").read_text().split())


def ironly_boxes():
    """{image id: (IR-only boxes xyxy, their types)} over all images of the split."""
    types = bt.load_types(D, SPLIT)
    out = {}
    for f in sorted((D / "ironly_cls" / SPLIT).glob("*.txt")):
        if f.stat().st_size:
            out[f.stem] = (xywhn_to_xyxy(np.loadtxt(f, ndmin=2)[:, 1:5]), [t for _, _, t in types[f.stem]])
    return out


def recalls(dets, ironly, nights):
    """{"ironly_rec_night": %, "rec_<type>": %} from {image id: detections xyxy}."""
    hit, ty, night = [], [], []
    for k, (g, t) in ironly.items():
        p = dets.get(k)
        hit.append((iou_matrix(g, p).max(1) >= IOU) if p is not None and len(p) else np.zeros(len(g), bool))
        ty += t
        night += [k in nights] * len(g)
    hit, ty, night = np.concatenate(hit) if hit else np.zeros(0, bool), np.array(ty), np.array(night, bool)
    pct = lambda s: 100.0 * hit[s].mean() if s.any() else float("nan")  # noqa: E731
    return {"ironly_rec_night": pct(night), **{f"rec_{t}": pct(ty == t) for t in bt.TYPES}}


def false_alarms(dets, nights):
    """Night detections that overlap no RGB and no IR ground-truth box (IoU >= 0.5)."""
    gts = [json.loads((D / m / SPLIT / "boxes.json").read_text())["images"] for m in ("rgb", "ir")]
    n = 0
    for k, p in dets.items():
        if k not in nights:
            continue
        g = [np.array([b[1:] for b in gt.get(k, [])], float).reshape(-1, 4) for gt in gts]
        g = np.concatenate(g)
        n += len(p) if not len(g) else int((iou_matrix(p, g).max(1) < IOU).sum())
    return n


def score(path, nights, ironly):
    preds = json.loads(Path(path).read_text())
    r = {}
    for lab in ("rgb", "ir"):
        m = ev.Matched(D / lab / SPLIT / "boxes.json", preds)
        r[f"{lab}_all"] = 100 * m.ap(np.ones(len(m.ids)))
        r[f"{lab}_night"] = 100 * m.ap(ev.subset_weights(m.ids, sorted(nights)))
        del m
    dets = {}
    for p in preds:
        if p["score"] >= THR:
            dets.setdefault(p["image_id"], []).append(p["bbox"])
    dets = {k: np.array(v, float) for k, v in dets.items()}
    r.update(recalls(dets, ironly, nights))
    r["false_alarms"] = false_alarms(dets, nights)
    return r


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    nights = night_ids()
    ironly = ironly_boxes()
    cache = OUT / "scores.json"
    scores = json.loads(cache.read_text()) if cache.exists() else {}
    for p in sorted(RUNS.glob(f"dyn_scr_*/ws/out/*/preds_{SPLIT}.json")):
        n, mt = p.parent.name, p.stat().st_mtime
        if scores.get(n, {}).get("_mtime") != mt:
            scores[n] = dict(score(p, nights, ironly), _mtime=mt)
            cache.write_text(json.dumps(scores))
            print(n, {k: round(v, 2) for k, v in scores[n].items() if k[0] != "_"}, flush=True)
    n = {t: sum(x.count(t) for _, x in ironly.values()) for t in bt.TYPES}
    lines = [f"# Omega screening ({SPLIT} split; AP50:95 %, recall %, false alarms = count)", "",
             "IR-only vehicles by type (all images): " + ", ".join(f"{t} {c}" for t, c in n.items()) + ".", "",
             "| config | n | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 2) + "|"]
    runs = {c: [f"dyn_scr_{c}_s{s}" for s in (0, 1, 2) if f"dyn_scr_{c}_s{s}" in scores] for c in CONFIGS}
    for c, names in runs.items():
        if names:
            cells = [f"{np.mean([scores[n][k] for n in names]):.2f} ± {np.std([scores[n][k] for n in names], ddof=1) if len(names) > 1 else 0:.2f}"
                     for k in KEYS]
            lines.append(f"| {c} | {len(names)} | " + " | ".join(cells) + " |")
    for ref in REFS:
        lines += ["", f"## Seed-paired differences to {ref} (mean [t 95 % CI], n seeds)", "",
                  "| config | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
        for c in CONFIGS:
            if c == ref:
                continue
            pairs = [(f"dyn_scr_{c}_s{s}", f"dyn_scr_{ref}_s{s}") for s in (0, 1, 2)]
            pairs = [(a, b) for a, b in pairs if a in scores and b in scores]
            if len(pairs) < 2:
                continue
            cells = []
            for k in KEYS:
                t = ds.paired_t([scores[a][k] - scores[b][k] for a, b in pairs])
                cells.append(f"{t['mean']:+.2f} [{t['t_ci'][0]:+.2f}, {t['t_ci'][1]:+.2f}] n={t['n']}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "tables.md")


if __name__ == "__main__":
    main()
