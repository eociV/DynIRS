"""Generate the DynIRS queue files (configs/queues/dyn/*.json) for the clean-label study
(docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md).

  python scripts/make_dyn_queues.py

Queues (every job single class, Ultralytics single_cls):
  teacher_rgb1c       cross-fitted RGB teacher, YOLO26l, 2 folds, predicts train and val (local GPU)
  t4_clean_s<seed>    screening B, P, I, V, F: YOLO26n, fixed 20 % train subset, 15 epochs, 704 px, val
  full_clean_s<seed>  full scale FULL: YOLO26n, 100 % train, 60 epochs, 704 px, test (local GPU)
Configs: B no IR-only boxes; P all with omega 0.3; I all as ignore regions; V RGB-teacher verified
(c >= 0.5 taught with omega c, else ignored); F registration + misalignment filter + V; C
(t4_cleanC) registration + misalignment filter + duplicate drop, the rest taught as in P. Their weight
files come from src/clean_labels.py. Regenerating deletes every other queue file in configs/queues/dyn.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "queues" / "dyn"
PAIRED = {"paired": True, "load_ir": False}  # labels-only pairing: inject IR-only boxes, RGB pixels only
CONFIGS = {
    "B": {},
    "P": dict(PAIRED, ir_soft_w_dir="auto:P"),
    "I": dict(PAIRED, ir_soft_w_dir="auto:I", ir_ignore=True),
    "V": dict(PAIRED, ir_soft_w_dir="auto:V", ir_ignore=True),
    "F": dict(PAIRED, ir_soft_w_dir="auto:F", ir_ignore=True),
    "C": dict(PAIRED, ir_soft_w_dir="auto:C", ir_ignore=True),  # clean P (ignore only misaligned images)
}
SCREEN_C = ("C",)  # second screening round: B and P are reused from t4_clean
SCREEN = ("B", "P", "I", "V", "F")
FULL = ("B", "P", "F")  # revised after screening (spec section 5)
SEEDS = (0, 1, 2)
SCREEN_RUN = {"epochs": 15, "imgsz": 704, "proxy_fraction": 0.2}  # the IRS-YOLO26n proxy protocol
FULL_RUN = {"epochs": 60, "imgsz": 704}
HW = {"local": {"workers": 4, "cache": "disk"}, "t4": {"workers": 2, "cache": "disk"}}


def student(code, seed, name, hw, epochs, imgsz, data, split, extra=None):
    job = {"name": name, "kind": "train",
           "splits": ["train", "val"] if split == "val" else ["train", "val", "test"],
           "chunks": [epochs // 2, None], "timeout": 0,
           "train": {"model": "yolo26n.pt", "epochs": epochs, "imgsz": imgsz, "batch": 32, "seed": seed,
                     "deterministic": False, "plots": False, "single_cls": True, "data": data, **HW[hw]},
           "config": dict(CONFIGS[code]), "inputs": {},
           "predict": {"weights": f"out/{name}/weights/best.pt", "split": split, "imgsz": imgsz}}
    job.update(extra or {})
    return job


def rgb_teacher(fold, k=2, epochs=30):
    """YOLO26l on the RGB labels only, single class, cross-fitted: trained on the train images outside
    fold `fold` of `k` (job_main.fold_of, seed 0); predicts train (only its held-out fold is used) and val."""
    name = f"dyn_teacher_rgb1c_f{fold}"
    w = f"out/{name}/weights/best.pt"
    return {"name": name, "kind": "train", "splits": ["train", "val", "test"],
            "chunks": list(range(10, epochs, 10)) + [None], "timeout": 0, "fold": [fold, k], "proxy_seed": 0,
            "train": {"model": "yolo26l.pt", "epochs": epochs, "imgsz": 640, "batch": 16, "seed": 0,
                      "deterministic": False, "plots": False, "single_cls": True, "data": "rgb_proxy",
                      **HW["local"]},
            "config": {}, "inputs": {},
            "predict": [{"weights": w, "split": s, "imgsz": 640} for s in ("train", "val")]}


def build():
    q = {"teacher_rgb1c": [rgb_teacher(f) for f in (0, 1)]}
    for s in SEEDS:
        q[f"t4_clean_s{s}"] = [
            student(c, s, f"dyn_cl_{c}_s{s}", "t4", SCREEN_RUN["epochs"], SCREEN_RUN["imgsz"], "rgb_proxy", "val",
                    extra={"proxy_fraction": SCREEN_RUN["proxy_fraction"]}) for c in SCREEN]
        q[f"t4_cleanC_s{s}"] = [
            student(c, s, f"dyn_cl_{c}_s{s}", "t4", SCREEN_RUN["epochs"], SCREEN_RUN["imgsz"], "rgb_proxy", "val",
                    extra={"proxy_fraction": SCREEN_RUN["proxy_fraction"]}) for c in SCREEN_C]
        q[f"full_clean_s{s}"] = [
            student(c, s, f"dyn_full_{c}_s{s}", "local", FULL_RUN["epochs"], FULL_RUN["imgsz"], "rgb", "test")
            for c in FULL]
    return q


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    q = build()
    for f in OUT.glob("*.json"):
        if f.stem not in q:
            f.unlink()
    for stem, jobs in q.items():
        (OUT / f"{stem}.json").write_text(json.dumps(jobs, indent=1))
        print(stem, len(jobs))


if __name__ == "__main__":
    main()
