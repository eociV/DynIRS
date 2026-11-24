"""Generate the DynIRS queue files (configs/queues/dyn/*.json).

  python scripts/make_dyn_queues.py [--epochs 40] [--imgsz 640]

Queues: bench, smoke, smoke_tl, teacher, teacher_rgb (2-fold cross-fitted RGB YOLO26l), local_26n_s0, local_26n_s12, t4_<model>_s<seed>_c, t4_<model>_s<seed>_ttl,
t4_screen_s<seed> (omega screening, B/C/ND/FX/NI: YOLO26n, fixed 20 % train subset, 15 epochs, 704 px, scored on val).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "queues" / "dyn"
MODELS = {"26n": "yolo26n.pt", "v8n": "yolov8n.pt", "11n": "yolo11n.pt"}
SOFT = {"paired": True, "load_ir": False, "ir_only_dir": "auto", "llsa": None, "dnc_lambda": 0.0, "tgd_lambda": 0.0}
CONFIGS = {
    "B": {},
    "M": {},  # trained on data "mix" (RGB + IR train images), see DATA_KEY
    "C": dict(SOFT, ir_soft=0.3, ir_soft_dir="auto"),
    "T": dict(SOFT, ir_soft_w_dir="auto:T"),
    "TL": dict(SOFT, ir_soft_w_dir="auto:TL"),
    "S": dict(SOFT, ir_soft_w_dir="auto:S"),
}
RANK = ("RT", "RD", "RB", "RA", "RS")  # rank-mapped omega variants, see src/omega.py
TYPED = ("ND", "FX", "NI")  # omega by IR-only box type, see src/omega.py and src/boxtypes.py
CONFIGS.update({v: dict(SOFT, ir_soft_w_dir=f"auto:{v}") for v in RANK + TYPED})
SCREEN = ("B", "C") + TYPED
SCREEN_RUN = {"epochs": 15, "imgsz": 704, "proxy_fraction": 0.2}  # the proxy protocol of the IRS-YOLO26n paper
DATA_KEY = {"M": "mix"}
HW = {"local": {"workers": 4, "cache": "disk"}, "t4": {"workers": 2, "cache": "disk"}}  # t4: 2 lanes share 4 CPUs


def student(model, cfg, seed, hw, epochs, imgsz, name=None, data=None, extra=None):
    name = name or f"dyn_{model}_{cfg}_s{seed}"
    data = data or DATA_KEY.get(cfg, "rgb")
    job = {"name": name, "kind": "train", "splits": ["train", "val", "test"],
           "chunks": [epochs // 2, None] if epochs > 1 else [None], "timeout": 0,
           "train": {"model": MODELS[model], "epochs": epochs, "imgsz": imgsz, "batch": 32, "seed": seed,
                     "deterministic": False, "plots": False, "data": data, **HW[hw]},
           "config": dict(CONFIGS[cfg]), "inputs": {},
           "predict": {"weights": f"out/{name}/weights/best.pt", "split": "test", "imgsz": imgsz}}
    job.update(extra or {})
    return job


def teacher(epochs=50, name="dyn_teacher_26l"):
    return {"name": name, "kind": "train", "splits": ["train", "val", "test"],
            "chunks": list(range(10, epochs, 10)) + [None], "timeout": 0,
            "train": {"model": "yolo26l.pt", "epochs": epochs, "imgsz": 640, "batch": 16, "seed": 0,
                      "deterministic": False, "plots": False, "data": "ir", **HW["local"]},
            "config": {}, "inputs": {},
            "predict": {"weights": f"out/{name}/weights/best.pt", "split": "val", "imgsz": 640,
                        "data_root": "ir"}}


def rgb_teacher(fold, k=2, epochs=30, name=None):
    """YOLO26l on the RGB labels, cross-fitted: trained on the train images outside fold `fold` of `k`
    (job_main.fold_of, seed 0), so its scores on the held-out fold are not fitted to those images' labels
    (which treat the IR-only vehicles as background). Predicts train (use only the held-out fold) and val."""
    name = name or f"dyn_teacher_rgb26l_f{fold}"
    w = f"out/{name}/weights/best.pt"
    return {"name": name, "kind": "train", "splits": ["train", "val", "test"],
            "chunks": list(range(10, epochs, 10)) + [None], "timeout": 0, "fold": [fold, k], "proxy_seed": 0,
            "train": {"model": "yolo26l.pt", "epochs": epochs, "imgsz": 640, "batch": 16, "seed": 0,
                      "deterministic": False, "plots": False, "data": "rgb_proxy", **HW["local"]},
            "config": {}, "inputs": {},
            "predict": [{"weights": w, "split": s, "imgsz": 640} for s in ("train", "val")]}


def build(epochs=40, imgsz=640):
    q = {}
    proxy = {"proxy_fraction": 0.012}  # about 200 train images
    q["bench"] = [student("26n", "C", 0, "local", 1, imgsz, name="dyn_bench_26n"),
                  student("11n", "C", 0, "local", 1, imgsz, name="dyn_bench_11n"),
                  teacher(1, name="dyn_bench_teacher")]
    for j in q["bench"]:
        j.pop("predict")
    q["bench2"] = [student("26n", "C", 0, "local", 1, imgsz, name=f"dyn_bench2_{x}") for x in ("a", "b")]
    for j in q["bench2"]:
        j.pop("predict")
    q["smoke"] = [student(m, "C", 0, "local", 1, imgsz, name=f"dyn_smoke_{m}", data="rgb_proxy", extra=proxy)
                  for m in MODELS]
    q["smoke_tl"] = [student(m, "TL", 0, "local", 1, imgsz, name=f"dyn_smoke_{m}_TL", data="rgb_proxy",
                             extra=proxy) for m in MODELS]
    q["teacher"] = [teacher()]
    q["teacher_rgb"] = [rgb_teacher(f) for f in (0, 1)]
    cfg26 = ("B", "M", "C", "T", "TL", "S")
    q["local_26n_s0"] = [student("26n", c, 0, "local", epochs, imgsz) for c in cfg26]
    q["local_26n_s12"] = [student("26n", c, s, "local", epochs, imgsz) for s in (1, 2) for c in cfg26]
    for m in ("v8n", "11n"):
        for s in (0, 1, 2):
            q[f"t4_{m}_s{s}_c"] = [student(m, "C", s, "t4", epochs, imgsz)]
            q[f"t4_{m}_s{s}_ttl"] = [student(m, c, s, "t4", epochs, imgsz) for c in ("T", "TL")]
    for s in (0, 1, 2):
        jobs = [student("26n", c, s, "t4", SCREEN_RUN["epochs"], SCREEN_RUN["imgsz"], name=f"dyn_scr_{c}_s{s}",
                        data="rgb_proxy", extra={"proxy_fraction": SCREEN_RUN["proxy_fraction"]}) for c in SCREEN]
        for j in jobs:
            j["splits"] = ["train", "val"]
            j["predict"]["split"] = "val"
        q[f"t4_screen_s{s}"] = jobs
    return q


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, jobs in build(a.epochs, a.imgsz).items():
        (OUT / f"{stem}.json").write_text(json.dumps(jobs, indent=1))
        print(stem, len(jobs))


if __name__ == "__main__":
    main()
