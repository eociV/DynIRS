"""Score the screening runs (<prefix><cfg>_s<seed>) on the val split and write a table.

  python src/screen_tables.py [--prefix dyn_cl_]

Single-class study: AP is class-agnostic. Per run: COCO AP50:95 (%) against the RGB and IR labels
(all / night images); recall (%) of IR-only vehicles at night, and over all images by box type
(src/boxtypes.py, proc/stats/boxtypes_val.json); night false alarms (detections matching neither
annotation). Detections count at score >= 0.25 and IoU >= 0.5. rgbI_*: AP against the RGB labels with
every IR-only vehicle except RGB duplicates as a COCO ignore region (padded by IGN_PAD px + IGN_REL of
its side, for the RGB-IR registration error), so detecting it is neither a TP nor an FP. Per config: mean +- SD over seeds,
and the seed-paired difference to each reference config.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import boxtypes as bt  # noqa: E402
import dyn_stats as ds  # noqa: E402
import evaluate as ev  # noqa: E402
from dv_pairs import iou_matrix  # noqa: E402
from omega import xywhn_to_xyxy  # noqa: E402

D = ROOT / "data" / "dronevehicle" / "proc"
MAN = ROOT / "splits"
RUNS = ROOT / "runs" / "remote"
OUT = ROOT / "runs" / "analysis" / "screen"
SPLIT = "val"
PREFIX = "dyn_cl_"
CONFIGS = ("B", "P", "I", "V", "F", "C", "CB", "D", "DB")
REFS = ("B", "P", "C")
SEEDS = (0, 1, 2)
THR, IOU = 0.25, 0.5
IGN_PAD, IGN_REL = 4.0, 0.10
KEYS = (["rgb_all", "rgb_night", "rgbI_all", "rgbI_night", "ir_all", "ir_night", "ironly_rec_night"] + [f"rec_{t}" for t in bt.TYPES]
        + ["false_alarms", "img_epochs_k"])


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


def ignore_regions(ironly):
    """{image id: [[x1, y1, x2, y2], ...]} padded ignore regions of the IR-only boxes that are not RGB duplicates."""
    out = {}
    for k, (g, t) in ironly.items():
        keep = [b for b, ty in zip(g, t) if ty != "dup"]
        if keep:
            b = np.array(keep, float)
            pad = IGN_PAD + IGN_REL * np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            out[k] = np.concatenate([b[:, :2] - pad, b[:, 2:] + pad], 1).tolist()
    return out


def recalls(dets, ironly, nights):
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
        g = np.concatenate([np.array([b[1:] for b in gt.get(k, [])], float).reshape(-1, 4) for gt in gts])
        n += len(p) if not len(g) else int((iou_matrix(p, g).max(1) < IOU).sum())
    return n


def score(path, nights, ironly):
    preds = json.loads(Path(path).read_text())
    r = {}
    for lab in ("rgb", "ir"):
        m = ev.Matched(D / lab / SPLIT / "boxes.json", preds, agnostic=True)
        r[f"{lab}_all"] = 100 * m.ap(np.ones(len(m.ids)))
        r[f"{lab}_night"] = 100 * m.ap(ev.subset_weights(m.ids, sorted(nights)))
        del m
    m = ev.Matched(D / "rgb" / SPLIT / "boxes.json", preds, agnostic=True, ignore=ignore_regions(ironly))
    r["rgbI_all"] = 100 * m.ap(np.ones(len(m.ids)))
    r["rgbI_night"] = 100 * m.ap(ev.subset_weights(m.ids, sorted(nights)))
    del m
    dets = {}
    for p in preds:
        if p["score"] >= THR:
            dets.setdefault(p["image_id"], []).append(p["bbox"])
    dets = {k: np.array(v, float) for k, v in dets.items()}
    r.update(recalls(dets, ironly, nights))
    r["false_alarms"] = false_alarms(dets, nights)
    return r


def table_lines(scores, n_types, prefix=PREFIX, configs=CONFIGS, refs=REFS):
    lines = [f"# Screening ({SPLIT} split; class-agnostic AP50:95 %, recall %, false alarms = count)", "",
             "IR-only vehicles by type (all images): " + ", ".join(f"{t} {c}" for t, c in n_types.items()) + ".",
             "", "| config | n | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 2) + "|"]
    runs = {c: [f"{prefix}{c}_s{s}" for s in SEEDS if f"{prefix}{c}_s{s}" in scores] for c in configs}
    for c, names in runs.items():
        if names:
            cells = []
            for k in KEYS:
                v = [scores[n][k] for n in names]
                cells.append(f"{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}")
            lines.append(f"| {c} | {len(names)} | " + " | ".join(cells) + " |")
    for ref in refs:
        lines += ["", f"## Seed-paired differences to {ref} (mean [t 95 % CI], n seeds)", "",
                  "| config | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
        for c in configs:
            if c == ref:
                continue
            pairs = [(f"{prefix}{c}_s{s}", f"{prefix}{ref}_s{s}") for s in SEEDS]
            pairs = [(a, b) for a, b in pairs if a in scores and b in scores]
            if len(pairs) < 2:
                continue
            cells = []
            for k in KEYS:
                t = ds.paired_t([scores[a][k] - scores[b][k] for a, b in pairs])
                cells.append(f"{t['mean']:+.2f} [{t['t_ci'][0]:+.2f}, {t['t_ci'][1]:+.2f}] n={t['n']}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return lines


def image_epochs(out):
    """Training images seen over the run (repeat sampling counted): train_images.jsonl, else subset x epochs."""
    out = Path(out)
    f = out / "train_images.jsonl"
    if f.exists():
        last = {}
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                last[r["from"]] = r  # a restarted chunk appends again: keep the last line per start epoch
        return int(sum(r["n_images"] * (r["to"] - r["from"]) for r in last.values()))
    n = len(json.loads((out / "proxy_ids.json").read_text()))
    epochs = len((out / "results.csv").read_text().splitlines()) - 1
    return n * epochs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PREFIX)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    nights = night_ids()
    ironly = ironly_boxes()
    cache = OUT / f"scores_{a.prefix.strip('_')}.json"
    scores = json.loads(cache.read_text()) if cache.exists() else {}
    for p in sorted(RUNS.glob(f"{a.prefix}*/ws/out/*/preds_{SPLIT}.json")):
        n, mt = p.parent.name, p.stat().st_mtime
        if scores.get(n, {}).get("_mtime") != mt or "rgbI_night" not in scores.get(n, {}):
            scores[n] = dict(score(p, nights, ironly), _mtime=mt)
            cache.write_text(json.dumps(scores))
            print(n, {k: round(v, 2) for k, v in scores[n].items() if k[0] != "_"}, flush=True)
    for n in scores:  # recomputed every time: cheap, and old cache entries lack it
        scores[n]["img_epochs_k"] = image_epochs(RUNS / n / "ws" / "out" / n) / 1000
    n_types = {t: sum(x.count(t) for _, x in ironly.values()) for t in bt.TYPES}
    dst = OUT / f"tables_{a.prefix.strip('_')}.md"
    dst.write_text("\n".join(table_lines(scores, n_types, a.prefix)) + "\n", encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
