"""Pre-write Ultralytics disk-cache files (<image>.npy = the decoded BGR image) atomically.

  python scripts/warm_cache.py --splits train val --mods rgb [ir]

Ultralytics 8.4 loads an existing <image>.npy instead of decoding the JPEG, whatever the cache mode.
Writing the files once, before training, lets parallel runs share them without write races.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cache_one(path):
    f = Path(path)
    npy = f.with_suffix(".npy")
    if npy.exists() and npy.stat().st_mtime >= f.stat().st_mtime:
        return 0
    tmp = npy.with_name(npy.stem + ".tmp.npy")
    np.save(tmp, cv2.imread(str(f), cv2.IMREAD_COLOR), allow_pickle=False)
    os.replace(tmp, npy)
    return 1


def warm(files, workers):
    with ProcessPoolExecutor(workers) as ex:
        return sum(ex.map(cache_one, files, chunksize=64))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--mods", nargs="+", default=["rgb"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    proc = Path(os.environ.get("DV_DATA", ROOT / "data" / "dronevehicle")) / "proc"
    files = [str(f) for m in a.mods for s in a.splits for f in sorted((proc / m / s / "images").glob("*.jpg"))]
    print("cached", warm(files, a.workers), "of", len(files), flush=True)


if __name__ == "__main__":
    main()
