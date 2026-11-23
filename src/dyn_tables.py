"""Score every DynIRS run on the test split and write the result tables.

  python src/dyn_tables.py

Metrics: COCO AP50:95 (%) against RGB and IR labels, all images and night images; recall of IR-only
vehicles (IoU >= 0.5, score >= 0.25, class-agnostic). For every model: seed-paired differences with
t 95 % CIs and Holm-adjusted p over that model's contrasts, per metric.
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
from omega import xywhn_to_xyxy  # noqa: E402

D = ROOT / "data" / "dronevehicle" / "proc"
MAN = ROOT / "splits"
RUNS = ROOT / "runs" / "remote"
OUT = ROOT / "runs" / "analysis" / "dyn"
METRICS = [("rgb", "all"), ("rgb", "night"), ("ir", "all"), ("ir", "night")]
MODELS = ("26n", "v8n", "11n")
CONTRASTS = {"26n": [("T", "C"), ("TL", "C"), ("TL", "T"), ("T", "S"), ("T", "M"), ("TL", "M"), ("C", "M"),
                     ("C", "B")]}
DEFAULT_CONTRASTS = [("T", "C"), ("TL", "C"), ("TL", "T")]


def preds_path(name):
    return RUNS / name / "ws" / "out" / name / "preds_test.json"


def ironly_recall(preds, thr=0.25):
    by = {}
    for p in preds:
        if p["score"] >= thr:
            by.setdefault(p["image_id"], []).append(p["bbox"])
    hit = tot = 0
    for f in sorted((D / "ironly_cls" / "test").glob("*.txt")):
        if not f.stat().st_size:
            continue
        g = xywhn_to_xyxy(np.loadtxt(f, ndmin=2)[:, 1:5])
        tot += len(g)
        if f.stem in by:
            hit += int((iou_matrix(g, np.array(by[f.stem])).max(1) >= 0.5).sum())
    return 100.0 * hit / max(tot, 1)


def band_weights(ids, band):
    return np.ones(len(ids)) if band == "all" else ev.subset_weights(ids, (MAN / f"test_{band}.txt").read_text().split())


def needs_rescore(entry, mtime):
    """A cached score is valid only if it was computed from a predictions file with this mtime."""
    return entry is None or entry.get("_mtime") != mtime


def score(name):
    with open(preds_path(name)) as fh:
        preds = json.load(fh)
    r = {"ironly_recall": ironly_recall(preds)}
    for lab, band in METRICS:
        m = ev.Matched(D / lab / "test" / "boxes.json", preds)
        r[f"{lab}_{band}"] = 100 * m.ap(band_weights(m.ids, band))
        del m
    return r


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "scores.json"
    scores = json.loads(cache.read_text()) if cache.exists() else {}
    names = sorted(p.parent.name for p in RUNS.glob("dyn_*/ws/out/*/preds_test.json")
                   if not p.parent.name.startswith(("dyn_bench", "dyn_smoke")))
    for n in names:
        mtime = preds_path(n).stat().st_mtime
        if needs_rescore(scores.get(n), mtime):
            scores[n] = dict(score(n), _mtime=mtime)
            cache.write_text(json.dumps(scores))
            print(n, {k: round(v, 2) for k, v in scores[n].items() if k[0] != '_'}, flush=True)
    lines = ["# DynIRS results (test split, AP50:95 %)", ""]
    keys = [f"{l}_{b}" for l, b in METRICS] + ["ironly_recall"]
    lines += ["| run | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{scores[n][k]:.2f}" for k in keys) + " |")
    for model in MODELS:
        lines += ["", f"## {model}: seed-paired differences (t 95 % CI, Holm p)", ""]
        for k in keys:
            rows = []
            for x, y in CONTRASTS.get(model, DEFAULT_CONTRASTS):
                pairs = [(f"dyn_{model}_{x}_s{s}", f"dyn_{model}_{y}_s{s}") for s in (0, 1, 2)]
                pairs = [(p, q) for p, q in pairs if p in scores and q in scores]
                if len(pairs) >= 2:
                    rows.append((f"{x}-{y}", ds.paired_t([scores[p][k] - scores[q][k] for p, q in pairs])))
            for (lab, r), ph in zip(rows, ds.holm([r["p"] for _, r in rows]) if rows else []):
                lines.append(f"- {k} {lab}: {r['mean']:+.2f} [{r['t_ci'][0]:+.2f}, {r['t_ci'][1]:+.2f}] "
                             f"n={r['n']} p_holm={ph:.3f}")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "tables.md")


if __name__ == "__main__":
    main()
