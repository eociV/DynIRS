"""Prepare the DroneVehicle data inside a cloud session, straight from Hugging Face (no upload from home).

  python -m dyncv.prep --work /tmp/dv --out /kaggle/working --expect '{"train": "<sha256>", ...}' \
      --folds configs/cv/folds_k5_b50_s0.json          (run with cwd = the code directory)

Uses src/job_main.ensure_data, the same deterministic preparation (pinned HF revision, dv_prep, dv_pairs)
as on the local machine, then checks it against the local copy before anything trains on it:
  - label hash of every split (line endings normalised, see label_hash) == the local hash
  - the train ids == the ids of the fold file
and writes <out>/dv_proc.tar (proc/ uncompressed: JPEGs do not shrink, one file is fast to save and mount)
plus <out>/prep_check.json. Any mismatch -> exit 1 and no tar, so no session can use the data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

TAR = "dv_proc.tar"
CHECK = "prep_check.json"
SPLITS = ("train", "val", "test")


def label_hash(proc, split) -> str:
    """job_main._hash_labels with line endings normalised to \\n: Windows writes the label files with
    \\r\\n, Linux with \\n, so the raw hashes differ although the labels are identical."""
    import hashlib

    proc = Path(proc)
    h = hashlib.sha256()
    dirs = [proc / mod / split / "labels" for mod in ("rgb", "ir")] + [proc / "ironly" / split]
    for d in dirs:
        for f in sorted(d.iterdir()):
            h.update(f.name.encode() + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def check(proc: Path, hashes: dict, expect: dict, fold_ids: list[str] | None) -> dict:
    out = {"hashes": hashes, "expected": expect, "ok": True, "problems": []}
    for s, h in expect.items():
        if hashes.get(s) != h:
            out["problems"].append(f"label hash of {s}: {hashes.get(s)} != local {h}")
    if fold_ids is not None:
        ids = sorted(p.stem for p in (proc / "ir" / "train" / "images").glob("*.jpg"))
        out["train_images"] = len(ids)
        if ids != sorted(fold_ids):
            extra, missing = set(ids) - set(fold_ids), set(fold_ids) - set(ids)
            out["problems"].append(f"train ids differ from the fold file: {len(extra)} extra, {len(missing)} missing")
    out["ok"] = not out["problems"]
    return out


def write_tar(proc: Path, dst: Path) -> Path:
    tmp = dst.with_name(dst.name + ".tmp")
    skip = (".npy", ".cache")
    with tarfile.open(tmp, "w") as t:
        t.add(proc, arcname="proc", filter=lambda i: None if i.name.endswith(skip) else i)
    os.replace(tmp, dst)
    return dst


def extract(tar: Path, dst: Path) -> Path:
    """Unpack <tar> (made by write_tar) to dst/proc; used at the start of every training session."""
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tarfile.open(tar) as t:
        try:
            t.extractall(dst, filter="data")
        except TypeError:  # Python without extraction filters (< 3.11.4); the tar is our own
            t.extractall(dst)
    print(f"extracted {tar} -> {dst / 'proc'} in {time.time() - t0:.0f} s", flush=True)
    return dst / "proc"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/dv", help="scratch data root (large disk)")
    ap.add_argument("--out", required=True, help="where dv_proc.tar and prep_check.json go")
    ap.add_argument("--expect", default="{}", help="JSON {split: local label hash}")
    ap.add_argument("--folds", help="fold file whose ids the train split must match")
    a = ap.parse_args(argv)
    os.environ["DV_DATA"] = a.work
    sys.path.insert(0, str(Path.cwd() / "src"))
    import job_main  # reads DV_DATA at import

    t0 = time.time()
    job_main.ensure_data(list(SPLITS))
    proc = Path(a.work) / "proc"
    hashes = {s: label_hash(proc, s) for s in SPLITS}
    shutil.rmtree(Path(a.work) / "raw", ignore_errors=True)
    folds = json.loads(Path(a.folds).read_text())["folds"] if a.folds else None
    rep = check(proc, hashes, json.loads(a.expect), [i for f in folds for i in f] if folds else None)
    rep["prep_secs"] = round(time.time() - t0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / CHECK).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    if not rep["ok"]:
        return 1
    write_tar(proc, out / TAR)
    print(f"wrote {out / TAR} ({(out / TAR).stat().st_size / 1e9:.2f} GB), total {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
