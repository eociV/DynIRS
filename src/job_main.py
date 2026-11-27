"""Job driver for one training, prediction or latency job (cwd = job workspace; see scripts/run_queue.py).

job.json fields:
  name         run name; outputs go to out/<name>/ (kept across chunks)
  kind         "train" | "predict" | "enhance" | "latency" | "probe"
  splits       dataset splits that must be prepared on this VM
  train        Ultralytics overrides for kind=train (data key: "rgb", "ir", "rgb_proxy"...)
  config       lowlight.CONFIG values (stop_after_epoch is the chunk end)
  predict      {"weights": path, "split": "val"|"test", "data_root": "rgb"|"rgb_zdce"}
  proxy_ids    optional list of train image ids for proxy runs
  fold         optional [i, k] with data "<mod>_proxy": train on the train ids outside fold i of k
               (fold_of, seeded by proxy_seed); for cross-fitted teachers

Data are prepared once per VM under /content/data/dv from the pinned Hugging
Face mirror, with the same dv_prep/dv_pairs code used locally.
"""

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path.cwd()
SRC = WS / "src"
sys.path.insert(0, str(SRC))
DATA = Path(os.environ.get("DV_DATA", "/content/data/dv"))
ULTRA = "8.4.173"
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_env():
    try:
        import ultralytics  # noqa: F401
        ok = ultralytics.__version__ == ULTRA
    except ImportError:
        ok = False
    if not ok:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRA}",
                        "pycocotools", "albumentations"], check=True)
    import torch
    import psutil

    log("torch", torch.__version__, "cuda", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
        "cpus", os.cpu_count(), "ram_gb", round(psutil.virtual_memory().total / 1e9, 1))


def _hash_labels(split):
    h = hashlib.sha256()
    for mod in ("rgb", "ir"):
        d = DATA / "proc" / mod / split / "labels"
        for f in sorted(d.iterdir()):
            h.update(f.name.encode() + f.read_bytes())
    for f in sorted((DATA / "proc" / "ironly" / split).iterdir()):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def ensure_data(splits):
    import dv_pairs
    import dv_prep
    from huggingface_hub import hf_hub_download

    proc = DATA / "proc"
    out = {}
    for s in splits:
        ready = proc / f".ready_{s}"
        if not ready.exists():
            t = time.time()
            z = hf_hub_download(dv_prep.HF_REPO, f"{s}.zip", repo_type="dataset",
                                revision=dv_prep.HF_REVISION, local_dir=str(DATA / "raw"))
            log("downloaded", s, round(time.time() - t), "s")
            rep = dv_prep.prepare(z, s, proc, workers=os.cpu_count())
            dv_pairs.run(proc, s)
            (proc / "stats" / f"prep_{s}.json").write_text(json.dumps(rep))
            os.remove(z)
            ready.write_text(_hash_labels(s))
            log("prepared", s, round(time.time() - t), "s")
        if not (proc / "ironly_cls" / s).exists():  # added after the first preparation
            dv_pairs.run(proc, s)
        out[s] = ready.read_text()
    for mod in ("rgb", "ir"):
        y = DATA / f"{mod}.yaml"
        y.write_text(f"path: {proc}\ntrain: {mod}/train/images\nval: {mod}/val/images\n"
                     f"test: {mod}/test/images\nnames: {json.dumps(NAMES)}\n")
    return out


def ensure_zdce(splits):
    """Zero-DCE++-enhanced RGB copy of each split (labels are links to the originals)."""
    import zdce

    for s in splits:
        ready = DATA / "zdce" / f".ready_{s}"
        if ready.exists():
            continue
        t = time.time()
        n = zdce.enhance_dir(DATA / "proc" / "rgb" / s / "images", DATA / "zdce" / "rgb" / s / "images",
                             WS / "inputs" / "zdcepp_Epoch99.pth")
        lab = DATA / "zdce" / "rgb" / s / "labels"
        if not lab.exists():
            lab.symlink_to(DATA / "proc" / "rgb" / s / "labels")
        ready.write_text(str(n))
        log("zero-dce++", s, n, "images", round(time.time() - t), "s")


def fold_of(k, seed=0, data=None):
    """{train image id: fold 0..k-1}: a seeded random partition of the sorted train ids under
    <data>/proc/rgb/train/images (data defaults to DATA)."""
    import numpy as np

    root = Path(data) if data is not None else DATA
    allids = sorted(p.stem for p in (root / "proc" / "rgb" / "train" / "images").glob("*.jpg"))
    perm = np.random.default_rng(seed).permutation(len(allids))
    return {allids[j]: int(i % k) for i, j in enumerate(perm)}


def _data_yaml(key, job):
    """rgb | ir | rgb_proxy | ir_proxy | rgb_zdce."""
    proc = DATA / "proc"
    if key in ("rgb", "ir"):
        return str(DATA / f"{key}.yaml")
    if key.endswith("_proxy") and not key.startswith("rgb4"):
        mod = key[:-6]
        ids = job.get("proxy_ids")
        if ids is None and job.get("fold"):  # cross-fitting: train on all folds but fold i of k
            ids = [x for x, f in fold_of(job["fold"][1], job.get("proxy_seed", 0)).items() if f != job["fold"][0]]
        if ids is None:  # seeded random subset of the sorted train ids
            import numpy as np

            allids = sorted(p.stem for p in (proc / "rgb" / "train" / "images").glob("*.jpg"))
            rng = np.random.default_rng(job.get("proxy_seed", 0))
            n = round(len(allids) * job["proxy_fraction"])
            ids = sorted(rng.choice(allids, n, replace=False).tolist())
            (WS / "out" / job["name"]).mkdir(parents=True, exist_ok=True)
            (WS / "out" / job["name"] / "proxy_ids.json").write_text(json.dumps(ids))
        rep = job.get("night_repeat")  # {"ids_file": "inputs/...txt", "k": 2}: oversample night
        if rep:
            night = set((WS / rep["ids_file"]).read_text().split())
            ids = ids + [i for i in ids if i in night] * (rep["k"] - 1)
        dyn_rep = job.get("_repeat")  # {"ids": [...], "k": R}: repeat sampling (dyn_omega)
        if dyn_rep:
            keep = set(dyn_rep["ids"])
            ids = ids + [i for i in ids if i in keep] * (dyn_rep["k"] - 1)
        lst = proc / f"{mod}_train_{job['name']}.txt"
        lst.write_text("".join(f"{proc}/{mod}/train/images/{i}.jpg\n" for i in ids))
        y = DATA / f"{key}_{job['name']}.yaml"
        y.write_text(f"path: {proc}\ntrain: {lst}\nval: {mod}/val/images\nnames: {json.dumps(NAMES)}\n")
        return str(y)
    if key in ("rgb4", "rgb4_proxy"):  # 4-channel R,G,B,IR fusion data (IR found by path)
        train = "rgb/train/images"
        if key == "rgb4_proxy":
            ids = job.get("proxy_ids")
            if ids is None:
                import numpy as np

                allids = sorted(p.stem for p in (proc / "rgb" / "train" / "images").glob("*.jpg"))
                rng = np.random.default_rng(job.get("proxy_seed", 0))
                ids = sorted(rng.choice(allids, round(len(allids) * job["proxy_fraction"]),
                                        replace=False).tolist())
            lst = proc / f"rgb4_train_{job['name']}.txt"
            lst.write_text("".join(f"{proc}/rgb/train/images/{i}.jpg\n" for i in ids))
            train = str(lst)
        y = DATA / f"{key}_{job['name']}.yaml"
        y.write_text(f"path: {proc}\ntrain: {train}\nval: rgb/val/images\nchannels: 4\n"
                     f"names: {json.dumps(NAMES)}\n")
        return str(y)
    if key == "rgb_zdce":
        y = DATA / "rgb_zdce.yaml"
        y.write_text(f"path: {DATA / 'zdce'}\ntrain: rgb/train/images\nval: rgb/val/images\n"
                     f"test: rgb/test/images\nnames: {json.dumps(NAMES)}\n")
        return str(y)
    if key == "mix":  # RGB + IR train images as separate samples (labels from each sibling labels dir)
        y = DATA / "mix.yaml"
        y.write_text(f"path: {proc}\ntrain: [rgb/train/images, ir/train/images]\nval: rgb/val/images\n"
                     f"test: rgb/test/images\nnames: {json.dumps(NAMES)}\n")
        return str(y)
    raise ValueError(key)


def resolve_config(cfg, data, ws):
    """lowlight.CONFIG values of a job with 'auto' paths resolved against the data root."""
    cfg = dict(cfg)
    proc = Path(data) / "proc"
    if cfg.get("ir_only_dir") == "auto":
        cfg["ir_only_dir"] = str(proc / "ironly" / "train")
    if cfg.get("pseudo_file"):
        cfg["pseudo_file"] = str(Path(ws) / cfg["pseudo_file"])
    if cfg.get("ir_soft_dir") == "auto":
        cfg["ir_soft_dir"] = str(proc / "ironly_cls" / "train")
    d = cfg.get("ir_soft_w_dir")
    if d:
        if cfg.get("ir_soft"):
            raise ValueError("ir_soft (constant) and ir_soft_w_dir (per-box) are mutually exclusive")
        if d.startswith("auto:"):
            d = str(proc / "ironly_w" / d[5:] / "train")
        if not Path(d).is_dir() or not any(Path(d).glob("*.txt")):
            raise FileNotFoundError(f"no per-box weight files in {d}; run src/omega.py weights first")
        cfg["ir_soft_w_dir"] = d
    return cfg


def hash_omega_dir(d):
    """SHA-256 over the sorted (file name + bytes) of the per-box weight files in d."""
    h = hashlib.sha256()
    for f in sorted(Path(d).glob("*.txt")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def _epochs_done(last):
    """Completed epochs recorded in a last.pt (0 without one)."""
    if not Path(last).exists():
        return 0
    import torch

    ep = torch.load(last, map_location="cpu", weights_only=False)["epoch"]
    return 0 if ep == -1 else ep + 1


def _predict_ids(weights, ids, imgsz):
    """{id: (n, 5) xyxy px + score} of a training checkpoint on RGB train images."""
    import numpy as np
    from ultralytics import YOLO

    import lowlight as ll

    weights = Path(weights)
    student = weights.with_name(weights.stem + "_student.pt")
    ll.export_student(weights, student)
    m = YOLO(str(student))
    root = DATA / "proc" / "rgb" / "train" / "images"
    out = {}
    for i in range(0, len(ids), 32):
        for r in m.predict(source=[str(root / f"{k}.jpg") for k in ids[i:i + 32]], imgsz=imgsz, conf=0.001,
                           max_det=300, half=True, stream=True, verbose=False):
            b = r.boxes
            out[Path(r.path).stem] = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
    return out


def dyn_prepare(job, cfg, predict_fn=_predict_ids):
    """(cfg, repeat) of a job with "dyn_omega" {"stages", "update", "repeat", "conf"} at the start of a chunk.
    Stage k (k stage boundaries passed) uses out/<name>/omega_dyn/stage<k>, scored once with the run's
    last.pt from stage k-1 (dyn_omega.rescore); repeat = {"ids", "k"} for _data_yaml, or None."""
    dyn = job.get("dyn_omega")
    if not dyn:
        return cfg, None
    import dyn_omega as do

    out = WS / "out" / job["name"]
    last = out / "weights" / "last.pt"
    k = sum(_epochs_done(last) >= b for b in dyn["stages"])
    wdir = Path(cfg["ir_soft_w_dir"])
    if dyn.get("update"):
        for j in range(1, k + 1):
            dst = out / "omega_dyn" / f"stage{j}"
            if not (dst / ".done").exists():
                if j < k:
                    raise RuntimeError(f"{dst}: stage {j} was never scored")
                rows = do.read_dir(wdir)
                pf = out / "proxy_ids.json"
                subset = set(json.loads(pf.read_text())) if pf.exists() else set(rows)
                ids = sorted(i for i, a in rows.items() if i in subset and len(a) and (a[:, 5] >= 0).any())
                preds = predict_fn(last, ids, job["train"].get("imgsz", 640))
                s = do.box_scores({i: rows[i] for i in ids}, preds)
                c = do.load_conf(DATA / "proc" / "stats" / dyn["conf"], rows)
                new, st = do.rescore(rows, s, c)
                do.write_dir(dst, new)
                (dst / "stats.json").write_text(json.dumps(st, indent=1))
                (dst / ".done").write_text("")
                log("dyn omega stage", j, json.dumps(st))
            wdir = dst
    cfg = dict(cfg, ir_soft_w_dir=str(wdir))
    rep = {"ids": do.repeat_ids(do.read_dir(wdir)), "k": dyn["repeat"]} if dyn.get("repeat", 1) > 1 else None
    return cfg, rep


def train(job):
    import lowlight as ll

    cfg = resolve_config(job.get("config", {}), DATA, WS)
    cfg, rep = dyn_prepare(job, cfg)
    job = dict(job, _repeat=rep)
    ll.CONFIG.update(cfg)
    if cfg.get("ir_soft_w_dir"):
        oh = hash_omega_dir(cfg["ir_soft_w_dir"])
        log("omega files", cfg["ir_soft_w_dir"], "sha256", oh)
        (WS / "out" / job["name"]).mkdir(parents=True, exist_ok=True)
        (WS / "out" / job["name"] / "omega_hash.txt").write_text(oh + "\n")
    over = dict(job["train"])
    over["data"] = _data_yaml(over["data"], job)
    if job.get("dyn_omega"):  # images per epoch of this chunk (repeat sampling lengthens epochs)
        lst = DATA / "proc" / f"rgb_train_{job['name']}.txt"
        n = len(lst.read_text().split()) if lst.exists() else None
        frm = _epochs_done(WS / "out" / job["name"] / "weights" / "last.pt")
        to = cfg.get("stop_after_epoch") or job["train"]["epochs"]
        (WS / "out" / job["name"]).mkdir(parents=True, exist_ok=True)
        with open(WS / "out" / job["name"] / "train_images.jsonl", "a") as f:
            f.write(json.dumps({"from": frm, "to": to, "n_images": n}) + "\n")
    out = WS / "out"
    last = out / job["name"] / "weights" / "last.pt"
    if last.exists():
        keep = {k: over[k] for k in ("data", "workers", "cache", "batch", "distill_model") if k in over}
        # save_dir must be overridden: the checkpoint stores the previous job's workspace path
        over = dict(resume=str(last), save_dir=str(out / job["name"]), **keep)
        log("resuming", last)
    else:
        over.update(project=str(out), name=job["name"], exist_ok=True)
    logf = out / job["name"] / "train.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logf, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        tr = ll.LLTrainer(overrides=over)
        tr.train()
    secs = time.time() - t0
    if Path(tr.save_dir).resolve() != (out / job["name"]).resolve():
        raise RuntimeError(f"trainer wrote to {tr.save_dir}, not {out / job['name']}")
    done = not tr._chunk_stopped
    log("trained", job["name"], "epochs_done", tr.epoch + 1, "secs", round(secs), "finished", done)
    res = out / job["name"] / "results.csv"
    if res.exists():
        print(res.read_text().splitlines()[-1][:300], flush=True)
    return done


def write_atomic(dst, text):
    """Write text next to dst and os.replace it, so a killed process never leaves a partial file."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, dst)


def predict(job):
    """job["predict"] is one {"weights", "split", "imgsz", "data_root"} dict or a list of them."""
    for p in job["predict"] if isinstance(job["predict"], list) else [job["predict"]]:
        _predict(job, p)


def _predict(job, p):
    from ultralytics import YOLO

    import lowlight as ll

    w = WS / p["weights"]
    student = w.with_name(w.stem + "_student.pt")
    ll.export_student(w, student)
    roots = {"rgb": DATA / "proc" / "rgb", "ir": DATA / "proc" / "ir", "rgb_zdce": DATA / "zdce" / "rgb"}
    root = roots[p.get("data_root", "rgb")] / p["split"] / "images"
    files = sorted(root.glob("*.jpg"))
    m = YOLO(str(student))
    preds = []
    t = time.time()
    # a list source is one batch in Ultralytics (batch= is ignored), so feed 32 images at a time
    results = (r for i in range(0, len(files), 32)
               for r in m.predict(source=[str(f) for f in files[i:i + 32]], imgsz=p.get("imgsz", 640),
                                  conf=0.001, max_det=300, half=True, stream=True, verbose=False))
    for r in results:
        b = r.boxes
        for xyxy, c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
            preds.append({"image_id": Path(r.path).stem, "category_id": int(c),
                          "bbox": [round(v, 2) for v in xyxy], "score": round(s, 5)})
    tag = "" if p.get("data_root", "rgb") == "rgb" else "_" + p["data_root"]
    dst = WS / "out" / job["name"] / f"preds_{p['split']}{tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(dst, json.dumps(preds))
    log("predicted", len(files), "images", len(preds), "boxes", round(time.time() - t), "s ->", dst.name)


def latency(job):
    """Batch-1 fp16 latency on the job's GPU: median Ultralytics inference/total ms over 300 val
    images after 50 warmup images, for each {"name", "weights", "imgsz"} in job["latency"]."""
    import numpy as np
    from ultralytics import YOLO

    import lowlight as ll

    files = sorted((DATA / "proc" / "rgb" / "val" / "images").glob("*.jpg"))[:350]
    res = {}
    for e in job["latency"]:
        w = WS / e["weights"]
        student = w.with_name(w.stem + "_student.pt")
        ll.export_student(w, student)
        m = YOLO(str(student))
        inf, tot = [], []
        for i, f in enumerate(files):
            r = m.predict(source=str(f), imgsz=e["imgsz"], conf=0.001, max_det=300, half=True, verbose=False)[0]
            if i >= 50:
                inf.append(r.speed["inference"])
                tot.append(sum(r.speed.values()))
        res[e["name"]] = {"imgsz": e["imgsz"], "inference_ms": round(float(np.median(inf)), 3),
                          "total_ms": round(float(np.median(tot)), 3)}
        log("latency", e["name"], json.dumps(res[e["name"]]))
    dst = WS / "out" / job["name"] / "latency.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))


def main(path="job.json"):
    job = json.loads((WS / path).read_text())
    log("job", job["name"], job["kind"])
    ensure_env()
    hashes = ensure_data(job.get("splits", ["train", "val"]))
    log("label hashes", json.dumps(hashes))
    if job.get("zdce_splits"):
        ensure_zdce(job["zdce_splits"])
    if job["kind"] == "train":
        if train(job) and job.get("predict"):
            import gc

            import torch

            gc.collect()  # the finished trainer still holds GPU memory (OOM at imgsz 768)
            torch.cuda.empty_cache()
            predict(job)
    elif job["kind"] == "dfine":
        import dfine_job

        dfine_job.run(job, WS, DATA)
    elif job["kind"] == "latency":
        latency(job)
    elif job["kind"] == "dfine_predict":
        import dfine_job

        dfine_job.predict_job(job, WS, DATA)
    elif job["kind"] == "dfine_finish":
        import dfine_job

        dfine_job.finish(job, WS, DATA)
    elif job["kind"] == "retrieve":
        src = Path(job["retrieve_from"])
        if "*" in str(src):  # e.g. a chunk whose controller was stopped: newest matching run dir
            import glob

            hits = [Path(h) for h in glob.glob(str(src)) if (Path(h) / "weights" / "last.pt").exists()]
            src = max(hits, key=lambda h: (h / "weights" / "last.pt").stat().st_mtime)
        dst = WS / "out" / job["name"]
        for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "train.log"):
            (dst / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, dst / f)
        log("retrieved", src, "->", dst)
        if job.get("predict"):
            predict(job)
    elif job["kind"] == "predict":
        predict(job)
    elif job["kind"] == "probe":
        subprocess.run("nvidia-smi; df -h /content; free -g; nproc", shell=True)
    shutil.rmtree(WS / "inputs", ignore_errors=True)
    log("done")
