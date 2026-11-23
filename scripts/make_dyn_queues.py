"""Generate the DynIRS queue files (configs/queues/dyn/*.json).

  python scripts/make_dyn_queues.py [--epochs 40] [--imgsz 640]

Queues: bench, smoke, smoke_tl, teacher, local_26n_s0, local_26n_s12, t4_<model>_s<seed>_c, t4_<model>_s<seed>_ttl.
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
    cfg26 = ("B", "M", "C", "T", "TL", "S")
    q["local_26n_s0"] = [student("26n", c, 0, "local", epochs, imgsz) for c in cfg26]
    q["local_26n_s12"] = [student("26n", c, s, "local", epochs, imgsz) for s in (1, 2) for c in cfg26]
    for m in ("v8n", "11n"):
        for s in (0, 1, 2):
            q[f"t4_{m}_s{s}_c"] = [student(m, "C", s, "t4", epochs, imgsz)]
            q[f"t4_{m}_s{s}_ttl"] = [student(m, c, s, "t4", epochs, imgsz) for c in ("T", "TL")]
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
