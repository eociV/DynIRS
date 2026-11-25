"""Download and prepare DroneVehicle only (no training).

    uv run scripts/download_data.py                 # train, val, test (uv picks up .venv)
    uv run scripts/download_data.py --splits val    # one split
    uv run scripts/download_data.py --data E:/dv    # other data root (same as setting DV_DATA)

Uses the same code path as the training jobs (job_main.ensure_data): downloads each split archive
from the pinned Hugging Face revision, converts it with dv_prep, matches RGB-IR boxes with dv_pairs,
deletes the archive and writes proc/.ready_<split>. Splits that are already prepared are skipped.
Afterwards the image ids of each split are checked against splits/<split>_{day,twilight,night}.txt.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_ids(proc, split):
    want = set()
    for band in ("day", "twilight", "night"):
        want |= set((ROOT / "splits" / f"{split}_{band}.txt").read_text().split())
    ok = True
    for mod in ("rgb", "ir"):
        have = {p.stem for p in (proc / mod / split / "images").glob("*.jpg")}
        missing, extra = want - have, have - want
        print(f"{split}/{mod}: {len(have)} images, expected {len(want)}, "
              f"missing {len(missing)}, extra {len(extra)}", flush=True)
        ok &= not missing and not extra
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--data", default=os.environ.get("DV_DATA", str(ROOT / "data" / "dronevehicle")),
                    help="data root (default: $DV_DATA or data/dronevehicle)")
    a = ap.parse_args(argv)
    os.environ["DV_DATA"] = str(Path(a.data).resolve())  # job_main reads it at import time
    sys.path.insert(0, str(ROOT / "src"))
    import job_main

    print("data root:", job_main.DATA, flush=True)
    job_main.ensure_data(a.splits)
    ok = all([check_ids(job_main.DATA / "proc", s) for s in a.splits])
    print("OK" if ok else "MISMATCH with splits/ (see above)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":  # required: dv_prep uses a process pool (spawn on Windows)
    sys.exit(main())
