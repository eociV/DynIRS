"""Build Kaggle dataset folders for DynIRS (upload with the Kaggle CLI).

  python scripts/pack_kaggle.py data  --user KAGGLE_USER    -> kaggle_ds/dynirs-dv
  python scripts/pack_kaggle.py omega --user KAGGLE_USER    -> kaggle_ds/dynirs-omega
  python scripts/pack_kaggle.py code  --user KAGGLE_USER    -> kaggle_ds/dynirs-code
  kaggle datasets create -p kaggle_ds/<slug> --dir-mode zip          (first time)
  kaggle datasets version -p kaggle_ds/<slug> -m "update" --dir-mode zip
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _meta(out, slug, user):
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=1))


def _fresh(out, slug):
    d = Path(out) / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def pack_data(proc, out, slug, user):
    d = _fresh(out, slug)
    dst = d / "proc"
    dst.mkdir()
    for sub in ("rgb", "ir", "ironly", "ironly_cls", "stats"):
        shutil.copytree(Path(proc) / sub, dst / sub, ignore=shutil.ignore_patterns("*.npy", "*.cache"))
    for f in Path(proc).glob(".ready_*"):
        shutil.copy2(f, dst / f.name)
    _meta(d, slug, user)
    return d


CLEAN_VARIANTS = ("P", "I", "V", "F", "C")


def pack_omega(proc, out, slug, user, variants=CLEAN_VARIANTS):
    """The clean-label weight files (src/clean_labels.py) and their stats, for the Kaggle runs."""
    d = _fresh(out, slug)
    for v in variants:
        shutil.copytree(Path(proc) / "ironly_w" / v, d / "ironly_w" / v)
    (d / "stats").mkdir()
    for v in variants:
        f = Path(proc) / "stats" / f"clean_{v}.json"
        if f.exists():
            shutil.copy2(f, d / "stats" / f.name)
    _meta(d, slug, user)
    return d


def git_commit(root):
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pack_code(root, out, slug, user):
    d = _fresh(out, slug)
    for sub in ("src", "scripts", "configs", "splits"):
        shutil.copytree(Path(root) / sub, d / sub, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(root) / "requirements.txt", d / "requirements.txt")
    (d / "COMMIT").write_text(git_commit(root) + "\n")
    _meta(d, slug, user)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "omega", "code"])
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", default=str(ROOT / "kaggle_ds"))
    a = ap.parse_args(argv)
    proc = Path(os.environ.get("DV_DATA", ROOT / "data" / "dronevehicle")) / "proc"
    fn = {"data": lambda: pack_data(proc, a.out, "dynirs-dv", a.user),
          "omega": lambda: pack_omega(proc, a.out, "dynirs-omega", a.user),
          "code": lambda: pack_code(ROOT, a.out, "dynirs-code", a.user)}[a.what]
    print(fn())


if __name__ == "__main__":
    main()
