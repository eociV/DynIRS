"""Browse the prepared DroneVehicle data in FiftyOne (local web app).

    uv run scripts/view_data.py                        # train split, opens http://localhost:5151
    uv run scripts/view_data.py --split val
    uv run scripts/view_data.py --limit 2000           # random subset, faster to build
    uv run scripts/view_data.py --rebuild              # rebuild after the data or the ω files change

One group per frame with two slices, "rgb" and "ir" (switch in the app). Every slice carries the
same three label fields, so IR labels can be overlaid on the RGB image:
  rgb_gt    RGB annotation
  ir_gt     IR annotation
  ir_only   IR boxes without an RGB match (dv_pairs); with ω files, each box has omega_<variant>
Sample fields: img_id, band (day / twilight / night, from splits/), lum_rgb, n_rgb, n_ir, n_ir_only.
Saved views (app: "Saved views" menu): night_ir_only, most_ir_only, ir_only_day.

The dataset is persistent (name dynirs-<split>, or dynirs-<split>-<limit>); later runs reopen it
unless --rebuild is given.
"""

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
VARIANTS = ("T", "TL", "S")


def read_yolo(path, ncol=5):
    """Rows of a YOLO txt file (cls cx cy w h [ω]); empty list if missing or empty."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        v = line.split()
        if len(v) >= ncol:
            rows.append([float(x) for x in v[:ncol]])
    return rows


def detections(fo, rows, label_prefix="", weights=None):
    out = []
    for i, r in enumerate(rows):
        c, cx, cy, w, h = r[:5]
        d = fo.Detection(label=label_prefix + CLASSES[int(c)], bounding_box=[cx - w / 2, cy - h / 2, w, h])
        for var, ws in (weights or {}).items():
            if i < len(ws):
                d[f"omega_{var}"] = ws[i]
        out.append(d)
    return fo.Detections(detections=out)


def bands(split):
    out = {}
    for b in ("day", "twilight", "night"):
        f = ROOT / "splits" / f"{split}_{b}.txt"
        if f.exists():
            out.update({i: b for i in f.read_text().split()})
    return out


def build(fo, proc, split, name, limit=None, seed=0):
    import random

    ids = sorted(p.stem for p in (proc / "rgb" / split / "images").glob("*.jpg"))
    if not ids:
        raise SystemExit(f"no images in {proc / 'rgb' / split / 'images'}: run scripts/download_data.py first")
    if limit and limit < len(ids):
        ids = sorted(random.Random(seed).sample(ids, limit))
    band = bands(split)
    lum = {}
    st = proc / "stats" / f"{split}.csv"
    if st.exists():
        with open(st, newline="") as f:
            lum = {r["id"]: float(r["lum_rgb"]) for r in csv.DictReader(f)}
    wdirs = {v: proc / "ironly_w" / v / split for v in VARIANTS if (proc / "ironly_w" / v / split).exists()}

    ds = fo.Dataset(name, persistent=True, overwrite=True)
    ds.add_group_field("group", default="rgb")
    meta = fo.ImageMetadata(width=640, height=512, num_channels=3)
    samples = []
    for k in ids:
        rgb = read_yolo(proc / "rgb" / split / "labels" / f"{k}.txt")
        ir = read_yolo(proc / "ir" / split / "labels" / f"{k}.txt")
        only = read_yolo(proc / "ironly_cls" / split / f"{k}.txt")
        weights = {v: [r[5] for r in read_yolo(d / f"{k}.txt", 6)] for v, d in wdirs.items()}
        g = fo.Group()
        for mod in ("rgb", "ir"):
            samples.append(fo.Sample(
                filepath=str(proc / mod / split / "images" / f"{k}.jpg"), group=g.element(mod), metadata=meta,
                img_id=k, band=band.get(k, "unknown"), lum_rgb=lum.get(k), n_rgb=len(rgb), n_ir=len(ir),
                n_ir_only=len(only), rgb_gt=detections(fo, rgb), ir_gt=detections(fo, ir),
                ir_only=detections(fo, only, weights=weights)))
    ds.add_samples(samples)

    F = fo.ViewField
    ds.save_view("night_ir_only", ds.match((F("band") == "night") & (F("n_ir_only") > 0))
                 .sort_by("n_ir_only", reverse=True))
    ds.save_view("most_ir_only", ds.sort_by("n_ir_only", reverse=True))
    ds.save_view("ir_only_day", ds.match((F("band") == "day") & (F("n_ir_only") > 0))
                 .sort_by("n_ir_only", reverse=True))
    ds.app_config.color_scheme = fo.ColorScheme(color_by="field", fields=[
        {"path": "rgb_gt", "fieldColor": "#2ecc71"},
        {"path": "ir_gt", "fieldColor": "#3498db"},
        {"path": "ir_only", "fieldColor": "#ff00ff"},
    ])
    ds.info = {"split": split, "proc": str(proc), "omega_variants": sorted(wdirs)}
    ds.save()
    fix_counts(ds)
    return ds


def fix_counts(ds):
    """Refresh MongoDB's stored document count. The App shows "No samples yet" when the collection
    metadata still says 0 (seen on Windows after bulk inserts); `validate` recounts it."""
    c = ds._sample_collection
    if c.estimated_document_count() != c.count_documents({}):
        c.database.command("validate", c.name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--data", default=os.environ.get("DV_DATA", str(ROOT / "data" / "dronevehicle")))
    ap.add_argument("--limit", type=int, help="random subset of this many frames (seed 0)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the FiftyOne dataset")
    ap.add_argument("--port", type=int, default=5151)
    ap.add_argument("--no-launch", action="store_true", help="only build the dataset")
    a = ap.parse_args(argv)

    import fiftyone as fo

    proc = Path(a.data).resolve() / "proc"
    name = f"dynirs-{a.split}" + (f"-{a.limit}" if a.limit else "")
    if a.rebuild or not fo.dataset_exists(name):
        print(f"building {name} from {proc} ...", flush=True)
        ds = build(fo, proc, a.split, name, a.limit)
    else:
        ds = fo.load_dataset(name)
        fix_counts(ds)
    print(f"{name}: {len(ds.distinct('group.id'))} frames, slices {ds.group_slices}, "
          f"saved views {ds.list_saved_views()}", flush=True)
    if a.no_launch:
        return
    session = fo.launch_app(ds, port=a.port, auto=False)
    print(f"open http://localhost:{a.port}  (Ctrl+C to stop)", flush=True)
    session.open_tab()
    session.wait(-1)


if __name__ == "__main__":
    main()
