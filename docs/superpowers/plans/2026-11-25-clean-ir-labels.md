# Clean IR-only Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the clean IR-only label pipeline (registration, misalignment filter, cross-fitted RGB-teacher verification) and the single-class screening and full-scale queues from `docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md`.

**Architecture:**
- `src/clean_labels.py` turns the IR-only boxes into per-variant weight files under `proc/ironly_w/<P|I|V|F>/train`. Each file is read by the existing per-box weight loader (`lowlight.IRPairDataset`): ω < 0 is an ignore region, already implemented.
- The queue generator is rewritten to emit only the new single-class jobs.
- The evaluator gains a class-agnostic mode, and the screening table is generalised to the new run names.

**Tech Stack:** Python 3.11, numpy, Ultralytics 8.4.173 (`single_cls` override), pycocotools, pytest. Run everything with `.venv/Scripts/python` from `D:\YSL\DynIRS`.

## Global Constraints

- Constants:
  - `MIN_PAIRS = 3`
  - `MAX_RESIDUAL = 4.0` px
  - `TAU = 0.5`
  - P weight `0.3`
  - ignore weight `-1.0`
- Single class: every teacher and student job sets `"single_cls": True` in its `train` overrides, and every row written by `clean_labels.py` has class `0`.
- Teacher jobs:
  - names `dyn_teacher_rgb1c_f0` and `dyn_teacher_rgb1c_f1`;
  - YOLO26l, 640 px, batch 16, 30 epochs, seed 0;
  - data `rgb_proxy` with `fold: [f, 2]` and `proxy_seed: 0`;
  - predicts `train` and `val`.
- Screening jobs:
  - names `dyn_cl_<code>_s<seed>`, codes `B P I V F`, seeds `0 1 2`;
  - YOLO26n, 15 epochs, 704 px, batch 32;
  - data `rgb_proxy` with `proxy_fraction: 0.2` (no `proxy_seed` key);
  - predict on `val`; hardware `t4`.
- Full-scale jobs:
  - names `dyn_full_<code>_s<seed>`, codes `FULL = ("B", "P", "F")` (revised after screening);
  - YOLO26n, 60 epochs, 704 px, batch 32, data `rgb`;
  - predict on `test`; hardware `local`.
- Commits use `git -c user.email=184421321+eociV@users.noreply.github.com commit` and end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Test command: `.venv/Scripts/python -m pytest -q tests`. All tests must pass at the end of every task.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/make_dyn_queues.py` (rewrite) | Emits `teacher_rgb1c`, `t4_clean_s{0,1,2}` and `full_clean_s{0,1,2}`, and deletes the other queue files |
| `tests/test_queues.py` (rewrite) | Tests the queue contents and `job_main.fold_of` |
| `src/job_main.py` (modify) | `fold_of(k, seed=0, data=None)` with an explicit data root |
| `src/evaluate.py` (modify) | `agnostic` mode in `_coco_gt` and `Matched` |
| `src/clean_labels.py` (new) | The label pipeline (steps ①–④), variants P/I/V/F, stats, sanity sheets, CLI |
| `tests/test_clean_labels.py` (new) | Unit tests of the pipeline on fixtures |
| `src/screen_tables.py` (rewrite) | Table of the `dyn_cl_` runs, class-agnostic AP, references B and P |
| `tests/test_screen_tables.py` (new) | Test of the table builder |
| `scripts/pack_kaggle.py` (modify) | The omega package holds only `ironly_w/{P,I,V,F}` and `stats/clean_*.json` |
| `kaggle/runner.ipynb`, `scripts/kaggle_session.py` | The example queue becomes `t4_clean_s0` |
| `docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md` | One correction: the loader needs no single-class change |

---

### Task 1: Single-class queues (teacher first, so it can start training)

**Files:**
- Rewrite: `scripts/make_dyn_queues.py`
- Rewrite: `tests/test_queues.py`
- Modify: `src/job_main.py` (`fold_of`)
- Modify: `kaggle/runner.ipynb` and `scripts/kaggle_session.py` (example queue name only)

**Interfaces:**
- Produces:
  - `make_dyn_queues.build() -> dict[str, list[dict]]` with keys `teacher_rgb1c`, `t4_clean_s0..2`, `full_clean_s0..2`;
  - the constants `CONFIGS`, `SCREEN`, `FULL`, `SEEDS`;
  - `job_main.fold_of(k: int, seed: int = 0, data: Path | None = None) -> dict[str, int]`.

- [ ] **Step 1: Write the failing tests.** Replace `tests/test_queues.py` with:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dyn_queues as mq  # noqa: E402


def test_queue_names():
    q = mq.build()
    assert sorted(q) == sorted(["teacher_rgb1c"] + [f"t4_clean_s{s}" for s in (0, 1, 2)]
                               + [f"full_clean_s{s}" for s in (0, 1, 2)])


def test_every_job_is_single_class():
    for jobs in mq.build().values():
        for j in jobs:
            assert j["train"]["single_cls"] is True


def test_rgb_teacher_rgb1c():
    jobs = mq.build()["teacher_rgb1c"]
    assert [j["name"] for j in jobs] == ["dyn_teacher_rgb1c_f0", "dyn_teacher_rgb1c_f1"]
    for f, j in enumerate(jobs):
        t = j["train"]
        assert (t["model"], t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (
            "yolo26l.pt", 30, 640, 16, 0, "rgb_proxy")
        assert j["fold"] == [f, 2] and j["proxy_seed"] == 0 and j["config"] == {}
        assert [p["split"] for p in j["predict"]] == ["train", "val"]


def test_screen_queues():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = q[f"t4_clean_s{s}"]
        assert [j["name"] for j in jobs] == [f"dyn_cl_{c}_s{s}" for c in ("B", "P", "I", "V", "F")]
        for j in jobs:
            t = j["train"]
            assert (t["model"], t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (
                "yolo26n.pt", 15, 704, 32, s, "rgb_proxy")
            assert j["proxy_fraction"] == 0.2 and "proxy_seed" not in j
            assert j["predict"]["split"] == "val" and j["predict"]["imgsz"] == 704
            assert j["splits"] == ["train", "val"] and t["workers"] == 2
        by = {j["name"].split("_")[2]: j["config"] for j in jobs}
        assert by["B"] == {}
        assert by["P"]["ir_soft_w_dir"] == "auto:P" and not by["P"].get("ir_ignore")
        for c in ("I", "V", "F"):
            assert by[c]["ir_soft_w_dir"] == f"auto:{c}" and by[c]["ir_ignore"] is True
            assert by[c]["paired"] is True and by[c]["load_ir"] is False


def test_full_queues():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = q[f"full_clean_s{s}"]
        assert [j["name"] for j in jobs] == [f"dyn_full_{c}_s{s}" for c in mq.FULL]
        for j in jobs:
            t = j["train"]
            assert (t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (60, 704, 32, s, "rgb")
            assert j["predict"]["split"] == "test" and "test" in j["splits"] and t["workers"] == 4
            assert "proxy_fraction" not in j


def test_main_removes_stale_queue_files(tmp_path, monkeypatch):
    monkeypatch.setattr(mq, "OUT", tmp_path)
    (tmp_path / "t4_v8n_s0_c.json").write_text("[]")
    mq.main([])
    names = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert "t4_v8n_s0_c" not in names and names == sorted(mq.build())


def test_fold_of_partition(tmp_path):
    import job_main as jm

    d = tmp_path / "proc" / "rgb" / "train" / "images"
    d.mkdir(parents=True)
    for i in range(11):
        (d / f"{i:05d}.jpg").write_bytes(b"")
    f = jm.fold_of(2, data=tmp_path)
    assert len(f) == 11 and sorted(set(f.values())) == [0, 1]
    assert abs(list(f.values()).count(0) - list(f.values()).count(1)) <= 1
    assert f == jm.fold_of(2, 0, tmp_path)
```

- [ ] **Step 2: Run the tests and check that they fail.**
Run: `.venv/Scripts/python -m pytest -q tests/test_queues.py`
Expected: FAIL. The `teacher_rgb1c` key is missing, and `fold_of()` gets an unexpected keyword `data`.

- [ ] **Step 3: Implement.** In `src/job_main.py`, replace `fold_of` with:

```python
def fold_of(k, seed=0, data=None):
    """{train image id: fold 0..k-1}: a seeded random partition of the sorted train ids under
    <data>/proc/rgb/train/images (data defaults to DATA)."""
    import numpy as np

    root = Path(data) if data is not None else DATA
    allids = sorted(p.stem for p in (root / "proc" / "rgb" / "train" / "images").glob("*.jpg"))
    perm = np.random.default_rng(seed).permutation(len(allids))
    return {allids[j]: int(i % k) for i, j in enumerate(perm)}
```

Replace `scripts/make_dyn_queues.py` with:

```python
"""Generate the DynIRS queue files (configs/queues/dyn/*.json) for the clean-label study
(docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md).

  python scripts/make_dyn_queues.py

Queues (every job single class, Ultralytics single_cls):
  teacher_rgb1c       cross-fitted RGB teacher, YOLO26l, 2 folds, predicts train and val (local GPU)
  t4_clean_s<seed>    screening B, P, I, V, F: YOLO26n, fixed 20 % train subset, 15 epochs, 704 px, val
  full_clean_s<seed>  full scale FULL: YOLO26n, 100 % train, 60 epochs, 704 px, test (local GPU)
Configs: B no IR-only boxes; P all with omega 0.3; I all as ignore regions; V RGB-teacher verified
(c >= 0.5 taught with omega c, else ignored); F registration + misalignment filter + V. Their weight
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
}
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
```

In `kaggle/runner.ipynb` and `scripts/kaggle_session.py`, replace the text `configs/queues/dyn/t4_v8n_s0_c.json` with `configs/queues/dyn/t4_clean_s0.json` (the example queue only):

```bash
sed -i 's#configs/queues/dyn/t4_v8n_s0_c.json#configs/queues/dyn/t4_clean_s0.json#g' kaggle/runner.ipynb scripts/kaggle_session.py
```

- [ ] **Step 4: Regenerate the queues and run the tests.**
Run: `.venv/Scripts/python scripts/make_dyn_queues.py && ls configs/queues/dyn && .venv/Scripts/python -m pytest -q tests`
Expected: exactly 7 files (`teacher_rgb1c`, `t4_clean_s0..2`, `full_clean_s0..2`) and all tests pass.

If another test imports names removed from `make_dyn_queues`, for example `tests/test_run_lanes.py` or `tests/test_kaggle_session.py` reading old queue names, change it to use `t4_clean_s0`. Find such tests with:

```bash
grep -rn "make_dyn_queues\|t4_v8n\|t4_screen\|dyn_scr_" tests scripts src
```

- [ ] **Step 5: Dry-plan the teacher queue.**
Run: `.venv/Scripts/python -c "import sys,json; sys.path.insert(0,'scripts'); import run_queue as rq; [print(j['name'], [s for _,s in rq.plan(j)]) for j in json.load(open('configs/queues/dyn/teacher_rgb1c.json'))]"`
Expected:
```
dyn_teacher_rgb1c_f0 [10, 20, None]
dyn_teacher_rgb1c_f1 [10, 20, None]
```

- [ ] **Step 6: Commit.**

```bash
git add -A scripts/make_dyn_queues.py tests/test_queues.py src/job_main.py kaggle/runner.ipynb scripts/kaggle_session.py configs/queues/dyn
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "Queues for the clean-label study: single-class RGB teacher, screening B/P/I/V/F, full scale

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

After this task, the user can start the teacher:
`.venv\Scripts\python scripts\run_queue.py configs\queues\dyn\teacher_rgb1c.json --show`
That takes about 4 h on the local GPU. Tasks 2–5 do not need the teacher.

---

### Task 2: Class-agnostic evaluation

**Files:**
- Modify: `src/evaluate.py` (`_coco_gt`, `Matched.__init__`)
- Test: `tests/test_evaluate_agnostic.py` (new)

**Interfaces:**
- Produces: `evaluate.Matched(boxes_json, preds, area="all", keep_ev=False, agnostic=False)`. With `agnostic=True`, every ground-truth and prediction category is treated as one class.

- [ ] **Step 1: Write the failing test** in `tests/test_evaluate_agnostic.py`:

```python
import json

import numpy as np

import evaluate as ev


def test_agnostic_ignores_class(tmp_path):
    gt = {"width": 640, "height": 512, "classes": ["car", "truck"],
          "images": {"a": [[0, 10, 10, 60, 40]], "b": [[1, 100, 100, 180, 150]]}}
    p = tmp_path / "boxes.json"
    p.write_text(json.dumps(gt))
    preds = [{"image_id": "a", "category_id": 1, "bbox": [10, 10, 60, 40], "score": 0.9},  # wrong class
             {"image_id": "b", "category_id": 1, "bbox": [100, 100, 180, 150], "score": 0.8}]
    m5 = ev.Matched(p, preds)
    m1 = ev.Matched(p, preds, agnostic=True)
    assert m1.classes == ["vehicle"]
    assert m1.ap(np.ones(len(m1.ids))) == 1.0
    assert m5.ap(np.ones(len(m5.ids))) < 1.0
```

- [ ] **Step 2: Run it and check that it fails.**
Run: `.venv/Scripts/python -m pytest -q tests/test_evaluate_agnostic.py`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'agnostic'`.

- [ ] **Step 3: Implement.** In `src/evaluate.py`:

```python
def _coco_gt(boxes_json, agnostic=False):
    g = json.load(open(boxes_json))
    ids = sorted(g["images"])
    classes = ["vehicle"] if agnostic else g["classes"]
    images, anns = [], []
    for n, k in enumerate(ids):
        images.append({"id": n, "file_name": k, "width": g["width"], "height": g["height"]})
        for c, x1, y1, x2, y2 in g["images"][k]:
            anns.append({"id": len(anns) + 1, "image_id": n, "category_id": 1 if agnostic else int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                         "iscrowd": 0})
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": cats}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, ids, classes
```

In `Matched.__init__`:
- change the signature to `def __init__(self, boxes_json, preds, area="all", keep_ev=False, agnostic=False):`;
- change the first line to `coco, self.ids, self.classes = _coco_gt(boxes_json, agnostic)`;
- in the `dets` list, change the category to `"category_id": 1 if agnostic else int(p["category_id"]) + 1,`.

Extend the module docstring's Predictions paragraph with: "With agnostic=True all categories are one class 'vehicle' (single-class study)."

- [ ] **Step 4: Run the tests.**
Run: `.venv/Scripts/python -m pytest -q tests`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/evaluate.py tests/test_evaluate_agnostic.py
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "evaluate: class-agnostic mode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Label pipeline `src/clean_labels.py`

**Files:**
- Create: `src/clean_labels.py`
- Test: `tests/test_clean_labels.py`

**Interfaces:**
- Consumes:
  - `omega.image_shifts(pairs, min_pairs=3) -> {id: (dx, dy)}`
  - `omega.shift_boxes(rows (n,5) normalized, d) -> rows`
  - `omega.teacher_conf(boxes_xyxy, pred_xyxy, pred_scores) -> (n,)`
  - `omega.xywhn_to_xyxy`, `omega.load_ironly(proc) -> {id: (n,5)}`
  - `omega.write_weights(proc, variant, boxes, weights) -> Path` (skips ω == 0 and degenerate boxes; writes ω = −1 rows)
  - `job_main.fold_of(k, seed, data)` (Task 1)
- Produces:
  - `residuals(pairs, min_pairs=MIN_PAIRS) -> {id: float}`
  - `heldout_preds(runs, data, name=TEACHER, k=2, seed=0) -> {id: ndarray (n,5) xyxy+score}`
  - `decide(conf, tau=TAU) -> ndarray`
  - `build_variant(variant, boxes, pairs, preds) -> (rows: {id: (n,5)}, weights: {id: (n,)}, stats: dict)`
  - the CLI `python src/clean_labels.py build|sheet ...`
  - the files `proc/ironly_w/<v>/train/*.txt` and `proc/stats/clean_<v>.json`

- [ ] **Step 1: Write the failing tests** in `tests/test_clean_labels.py`:

```python
import json

import numpy as np
import pytest

import clean_labels as cl
import omega as om


def test_residuals_median_distance_to_image_shift():
    pairs = {"a": {"offsets": [[4, 0], [4, 0], [10, 0]]}, "b": {"offsets": [[1, 1], [2, 2]]}}
    r = cl.residuals(pairs)
    assert r == {"a": pytest.approx(0.0)}  # median offset (4, 0); distances 0, 0, 6 -> median 0
    pairs["a"]["offsets"] = [[0, 0], [5, 0], [10, 0]]
    assert cl.residuals(pairs)["a"] == pytest.approx(5.0)


def test_decide_threshold():
    assert cl.decide(np.array([0.0, 0.49, 0.5, 0.9])).tolist() == [-1.0, -1.0, 0.5, 0.9]


def _box(cx, cy, w=0.1, h=0.1, c=3):
    return [c, cx, cy, w, h]


def test_variants_p_and_i_keep_every_box_unmoved_class0():
    boxes = {"a": np.array([_box(0.5, 0.5)]), "b": np.zeros((0, 5))}
    pairs = {"a": {"offsets": [[6.4, 0]] * 3}, "b": {"offsets": []}}
    rows, w, st = cl.build_variant("P", boxes, pairs, {})
    assert rows["a"][0].tolist() == [0, 0.5, 0.5, 0.1, 0.1] and w["a"].tolist() == [0.3]
    rows, w, st = cl.build_variant("I", boxes, pairs, {})
    assert w["a"].tolist() == [-1.0] and st["n_ignored"] == 1 and st["n_taught"] == 0


def _pred_on(row, score):
    x = om.xywhn_to_xyxy(np.array(row)[1:5])[0]
    return list(x) + [score]


def test_variant_v_verifies_unmoved_boxes():
    boxes = {"a": np.array([_box(0.3, 0.3), _box(0.7, 0.7)])}
    pairs = {"a": {"offsets": [[6.4, 0]] * 3}}
    preds = {"a": np.array([_pred_on(_box(0.3, 0.3), 0.8), _pred_on(_box(0.7, 0.7), 0.2)])}
    rows, w, st = cl.build_variant("V", boxes, pairs, preds)
    assert np.allclose(rows["a"][:, 1:3], [[0.3, 0.3], [0.7, 0.7]])
    assert w["a"].tolist() == pytest.approx([0.8, -1.0])
    assert st["n_taught"] == 1 and st["omega_taught_mean"] == pytest.approx(0.8)


def test_variant_f_moves_then_verifies_at_new_position():
    boxes = {"a": np.array([_box(0.5, 0.5)])}
    pairs = {"a": {"offsets": [[6.4, 0]] * 3}}  # IR 6.4 px right of RGB -> box moves 0.01 left
    moved = _box(0.49, 0.5)
    preds = {"a": np.array([_pred_on(moved, 0.9)])}
    rows, w, st = cl.build_variant("F", boxes, pairs, preds)
    assert rows["a"][0, 1] == pytest.approx(0.49) and w["a"].tolist() == pytest.approx([0.9])
    assert st["n_moved_gt_2px"] == 1


def test_variant_f_ignores_misaligned_images():
    boxes = {"a": np.array([_box(0.5, 0.5)])}
    pairs = {"a": {"offsets": [[0, 0], [5, 0], [10, 0]]}}  # residual 5 px > 4
    preds = {"a": np.array([_pred_on(_box(0.5, 0.5), 0.9)])}
    rows, w, st = cl.build_variant("F", boxes, pairs, preds)
    assert w["a"].tolist() == [-1.0]
    assert st["n_ignored_residual"] == 1 and st["n_images_residual"] == 1


def test_heldout_preds_uses_the_other_fold(tmp_path):
    img = tmp_path / "proc" / "rgb" / "train" / "images"
    img.mkdir(parents=True)
    ids = [f"{i:05d}" for i in range(6)]
    for i in ids:
        (img / f"{i}.jpg").write_bytes(b"")
    import job_main as jm

    fold = jm.fold_of(2, 0, tmp_path)
    for f in (0, 1):
        out = tmp_path / "runs" / f"T_f{f}" / "ws" / "out" / f"T_f{f}"
        out.mkdir(parents=True)
        # model f predicts every image with score 0.1 * (f + 1)
        out.joinpath("preds_train.json").write_text(json.dumps(
            [{"image_id": i, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.1 * (f + 1)} for i in ids]))
    p = cl.heldout_preds(tmp_path / "runs", tmp_path, name="T_f{}")
    for i in ids:
        assert p[i][:, 4].tolist() == pytest.approx([0.1 * (fold[i] + 1)])


def test_cli_build_writes_weight_files(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "ironly_cls" / "train").mkdir(parents=True)
    (proc / "ironly_cls" / "train" / "a.txt").write_text("2 0.5 0.5 0.1 0.1\n")
    (proc / "stats").mkdir()
    (proc / "stats" / "pairs_train.json").write_text(json.dumps({"a": {"offsets": []}}))
    cl.main(["build", "--proc", str(proc), "--variants", "P", "I"])
    assert (proc / "ironly_w" / "P" / "train" / "a.txt").read_text().split() == \
        ["0", "0.500000", "0.500000", "0.100000", "0.100000", "0.3000"]
    assert (proc / "ironly_w" / "I" / "train" / "a.txt").read_text().split()[-1] == "-1.0000"
    st = json.loads((proc / "stats" / "clean_I.json").read_text())
    assert st["n_boxes"] == 1 and st["n_ignored"] == 1
```

- [ ] **Step 2: Run the tests and check that they fail.**
Run: `.venv/Scripts/python -m pytest -q tests/test_clean_labels.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'clean_labels'`.

- [ ] **Step 3: Implement** `src/clean_labels.py`:

```python
"""Clean IR-only labels for single-class RGB training (docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md).

  python src/clean_labels.py build --proc data/dronevehicle/proc [--runs runs/remote] [--variants P I V F]
  python src/clean_labels.py sheet --proc data/dronevehicle/proc [--runs runs/remote]

Variants of the IR-only boxes (proc/ironly_cls/train), all written as class 0 to
proc/ironly_w/<variant>/train/<id>.txt ("0 cx cy w h omega"; omega -1 = ignore region):
  P  every box, omega 0.3, not moved                  (the original recipe)
  I  every box as an ignore region, not moved
  V  steps 3-4 on unmoved boxes
  F  steps 1-4:
     1 registration: images with >= MIN_PAIRS matched pairs move their IR-only boxes by -d,
       d = median (IR - RGB) centre offset of the pairs (proc/stats/pairs_train.json)
     2 misaligned images: residual = median |offset - d| > MAX_RESIDUAL px -> every box ignored
     3 verification: c = best score of a held-out RGB-teacher detection with IoU >= 0.5 to the box
     4 decision: c >= TAU taught with omega c, else ignored
The teacher is cross-fitted (dyn_teacher_rgb1c_f{0,1}): a train image is scored by the fold model that
did not train on it (job_main.fold_of, 2 folds, seed 0). Stats: proc/stats/clean_<variant>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import job_main as jm  # noqa: E402
import omega as om  # noqa: E402

MIN_PAIRS = 3
MAX_RESIDUAL = 4.0
TAU = 0.5
P_OMEGA = 0.3
IGNORE = -1.0
VARIANTS = ("P", "I", "V", "F")
TEACHER = "dyn_teacher_rgb1c_f{}"


def residuals(pairs, min_pairs=MIN_PAIRS):
    """{image id: px} median distance of the image's pair offsets to their median (images with >= min_pairs)."""
    out = {}
    for k, v in pairs.items():
        o = np.asarray(v["offsets"], float).reshape(-1, 2)
        if len(o) >= min_pairs:
            out[k] = float(np.median(np.hypot(*(o - np.median(o, 0)).T)))
    return out


def heldout_preds(runs, data, name=TEACHER, k=2, seed=0):
    """{train image id: (n, 5) xyxy + score} from the fold model that did not train on the image."""
    fold = jm.fold_of(k, seed, data)
    out = {}
    for f in range(k):
        n = name.format(f)
        for p in json.loads((Path(runs) / n / "ws" / "out" / n / "preds_train.json").read_text()):
            if fold.get(p["image_id"]) == f:
                out.setdefault(p["image_id"], []).append(list(p["bbox"]) + [p["score"]])
    return {i: np.asarray(v, float).reshape(-1, 5) for i, v in out.items()}


def decide(conf, tau=TAU):
    conf = np.asarray(conf, float)
    return np.where(conf >= tau, conf, IGNORE)


def _conf(rows, p):
    if p is None or not len(p) or not len(rows):
        return np.zeros(len(rows))
    return om.teacher_conf(om.xywhn_to_xyxy(rows[:, 1:5]), p[:, :4], p[:, 4])


def build_variant(variant, boxes, pairs, preds):
    """(rows, weights, stats) for one variant; rows are class-0 normalized "cls cx cy w h"."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    move = variant == "F"
    sh = om.image_shifts(pairs, MIN_PAIRS) if move else {}
    res = residuals(pairs) if move else {}
    rows, w = {}, {}
    n_res = n_res_img = n_moved = 0
    for k, b in boxes.items():
        b = np.array(b, float).reshape(-1, 5)
        b[:, 0] = 0
        if k in sh:
            b = om.shift_boxes(b, sh[k])
            n_moved += len(b) * (np.hypot(*sh[k]) > 2)
        if variant == "P":
            x = np.full(len(b), P_OMEGA)
        elif variant == "I":
            x = np.full(len(b), IGNORE)
        elif move and res.get(k, 0.0) > MAX_RESIDUAL:
            x = np.full(len(b), IGNORE)
            n_res += len(b)
            n_res_img += bool(len(b))
        else:
            x = decide(_conf(b, preds.get(k)))
        rows[k], w[k] = b, x
    allw = np.concatenate(list(w.values())) if w else np.zeros(0)
    taught = allw[allw > 0]
    stats = {"variant": variant, "n_boxes": int(len(allw)), "n_taught": int(len(taught)),
             "n_ignored": int((allw < 0).sum()), "n_ignored_residual": int(n_res),
             "n_images_residual": int(n_res_img), "n_moved_gt_2px": int(n_moved),
             "omega_taught_mean": float(taught.mean()) if len(taught) else 0.0,
             "omega_taught_p50": float(np.median(taught)) if len(taught) else 0.0,
             "min_pairs": MIN_PAIRS, "max_residual": MAX_RESIDUAL, "tau": TAU}
    return rows, w, stats


def sheet(proc, runs, out):
    """Sanity sheets for F: taught, ignored by the teacher, ignored by the residual (24 random crops each)."""
    import rgb_teacher_demo as rd

    data = Path(proc).parent
    preds = heldout_preds(runs, data)
    boxes = {k: v for k, v in om.load_ironly(proc).items() if len(v)}
    pairs = json.loads((Path(proc) / "stats" / "pairs_train.json").read_text())
    rows, w, _ = build_variant("F", boxes, pairs, preds)
    res = residuals(pairs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    groups = {"taught": lambda k, j: w[k][j] > 0,
              "ignored_teacher": lambda k, j: w[k][j] < 0 and res.get(k, 0.0) <= MAX_RESIDUAL,
              "ignored_residual": lambda k, j: w[k][j] < 0 and res.get(k, 0.0) > MAX_RESIDUAL}
    for g, sel in groups.items():
        cand = [(k, j) for k in rows for j in range(len(rows[k])) if sel(k, j)]
        idx = rng.choice(len(cand), min(24, len(cand)), replace=False) if cand else []
        picks = [(cand[i][0], cand[i][1], [f"omega {w[cand[i][0]][cand[i][1]]:.2f}",
                                           f"residual {res.get(cand[i][0], float('nan')):.1f}px"]) for i in idx]
        rd.sheet(out / f"F_{g}.jpg", picks, preds, rows, f"F: {g} (yellow = moved IR-only box)")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--proc", required=True)
    b.add_argument("--runs", default=str(ROOT / "runs" / "remote"))
    b.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    s = sub.add_parser("sheet")
    s.add_argument("--proc", required=True)
    s.add_argument("--runs", default=str(ROOT / "runs" / "remote"))
    s.add_argument("--out", default=str(ROOT / "runs" / "analysis" / "clean"))
    a = ap.parse_args(argv)
    proc = Path(a.proc)
    if a.cmd == "sheet":
        sheet(proc, a.runs, a.out)
        return
    boxes = om.load_ironly(proc)
    pairs = json.loads((proc / "stats" / "pairs_train.json").read_text())
    preds = heldout_preds(a.runs, proc.parent) if set(a.variants) & {"V", "F"} else {}
    for v in a.variants:
        rows, w, st = build_variant(v, boxes, pairs, preds)
        d = proc / "ironly_w" / v / "train"  # write_weights does not empty it: drop a previous build
        if d.is_dir():
            for f in d.glob("*.txt"):
                f.unlink()
        om.write_weights(proc, v, rows, w)
        (proc / "stats" / f"clean_{v}.json").write_text(json.dumps(st, indent=1))
        print(json.dumps(st))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests.**
Run: `.venv/Scripts/python -m pytest -q tests`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/clean_labels.py tests/test_clean_labels.py
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "clean_labels: registration, misalignment filter, RGB-teacher verification (variants P/I/V/F)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Screening table for the `dyn_cl_` runs

**Files:**
- Rewrite: `src/screen_tables.py`
- Test: `tests/test_screen_tables.py`

**Interfaces:**
- Consumes: `evaluate.Matched(..., agnostic=True)` (Task 2), `dyn_stats.paired_t(diffs) -> {"mean", "t_ci", "n"}`, `boxtypes.load_types`.
- Produces: `screen_tables.table_lines(scores: {run: {key: value}}, n_types: {type: count}, prefix=PREFIX, configs=CONFIGS, refs=REFS) -> list[str]`, plus the CLI `python src/screen_tables.py [--prefix dyn_cl_]`.

- [ ] **Step 1: Write the failing test** in `tests/test_screen_tables.py`:

```python
import screen_tables as st


def test_table_lines_means_and_paired_diffs():
    keys = st.KEYS
    scores = {f"dyn_cl_{c}_s{s}": {k: base + s * 0.1 for k in keys}
              for c, base in (("B", 10.0), ("P", 11.0)) for s in (0, 1, 2)}
    lines = st.table_lines(scores, {"dup": 1, "invisible": 2, "faint": 3, "visible": 4})
    text = "\n".join(lines)
    assert "| B | 3 | 10.10 ± 0.10" in text
    assert "| P | 3 | 11.10 ± 0.10" in text
    assert "## Seed-paired differences to B" in text and "| P | +1.00 [" in text
    assert "| I |" not in text  # no runs of I
```

- [ ] **Step 2: Run it and check that it fails.**
Run: `.venv/Scripts/python -m pytest -q tests/test_screen_tables.py`
Expected: FAIL with `AttributeError: module 'screen_tables' has no attribute 'table_lines'`.

- [ ] **Step 3: Implement.** Replace `src/screen_tables.py` with:

```python
"""Score the screening runs (<prefix><cfg>_s<seed>) on the val split and write a table.

  python src/screen_tables.py [--prefix dyn_cl_]

Single-class study: AP is class-agnostic. Per run: COCO AP50:95 (%) against the RGB and IR labels
(all / night images); recall (%) of IR-only vehicles at night, and over all images by box type
(src/boxtypes.py, proc/stats/boxtypes_val.json); night false alarms (detections matching neither
annotation). Detections count at score >= 0.25 and IoU >= 0.5. Per config: mean +- SD over seeds,
and the seed-paired difference to each reference config.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import boxtypes as bt  # noqa: E402
import dyn_stats as ds  # noqa: E402
import evaluate as ev  # noqa: E402
from dv_pairs import iou_matrix  # noqa: E402
from omega import xywhn_to_xyxy  # noqa: E402

D = ROOT / "data" / "dronevehicle" / "proc"
MAN = ROOT / "splits"
RUNS = ROOT / "runs" / "remote"
OUT = ROOT / "runs" / "analysis" / "screen"
SPLIT = "val"
PREFIX = "dyn_cl_"
CONFIGS = ("B", "P", "I", "V", "F")
REFS = ("B", "P")
SEEDS = (0, 1, 2)
THR, IOU = 0.25, 0.5
KEYS = (["rgb_all", "rgb_night", "ir_all", "ir_night", "ironly_rec_night"] + [f"rec_{t}" for t in bt.TYPES]
        + ["false_alarms"])


def night_ids():
    return set((MAN / f"{SPLIT}_night.txt").read_text().split())


def ironly_boxes():
    """{image id: (IR-only boxes xyxy, their types)} over all images of the split."""
    types = bt.load_types(D, SPLIT)
    out = {}
    for f in sorted((D / "ironly_cls" / SPLIT).glob("*.txt")):
        if f.stat().st_size:
            out[f.stem] = (xywhn_to_xyxy(np.loadtxt(f, ndmin=2)[:, 1:5]), [t for _, _, t in types[f.stem]])
    return out


def recalls(dets, ironly, nights):
    hit, ty, night = [], [], []
    for k, (g, t) in ironly.items():
        p = dets.get(k)
        hit.append((iou_matrix(g, p).max(1) >= IOU) if p is not None and len(p) else np.zeros(len(g), bool))
        ty += t
        night += [k in nights] * len(g)
    hit, ty, night = np.concatenate(hit) if hit else np.zeros(0, bool), np.array(ty), np.array(night, bool)
    pct = lambda s: 100.0 * hit[s].mean() if s.any() else float("nan")  # noqa: E731
    return {"ironly_rec_night": pct(night), **{f"rec_{t}": pct(ty == t) for t in bt.TYPES}}


def false_alarms(dets, nights):
    """Night detections that overlap no RGB and no IR ground-truth box (IoU >= 0.5)."""
    gts = [json.loads((D / m / SPLIT / "boxes.json").read_text())["images"] for m in ("rgb", "ir")]
    n = 0
    for k, p in dets.items():
        if k not in nights:
            continue
        g = np.concatenate([np.array([b[1:] for b in gt.get(k, [])], float).reshape(-1, 4) for gt in gts])
        n += len(p) if not len(g) else int((iou_matrix(p, g).max(1) < IOU).sum())
    return n


def score(path, nights, ironly):
    preds = json.loads(Path(path).read_text())
    r = {}
    for lab in ("rgb", "ir"):
        m = ev.Matched(D / lab / SPLIT / "boxes.json", preds, agnostic=True)
        r[f"{lab}_all"] = 100 * m.ap(np.ones(len(m.ids)))
        r[f"{lab}_night"] = 100 * m.ap(ev.subset_weights(m.ids, sorted(nights)))
        del m
    dets = {}
    for p in preds:
        if p["score"] >= THR:
            dets.setdefault(p["image_id"], []).append(p["bbox"])
    dets = {k: np.array(v, float) for k, v in dets.items()}
    r.update(recalls(dets, ironly, nights))
    r["false_alarms"] = false_alarms(dets, nights)
    return r


def table_lines(scores, n_types, prefix=PREFIX, configs=CONFIGS, refs=REFS):
    lines = [f"# Screening ({SPLIT} split; class-agnostic AP50:95 %, recall %, false alarms = count)", "",
             "IR-only vehicles by type (all images): " + ", ".join(f"{t} {c}" for t, c in n_types.items()) + ".",
             "", "| config | n | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 2) + "|"]
    runs = {c: [f"{prefix}{c}_s{s}" for s in SEEDS if f"{prefix}{c}_s{s}" in scores] for c in configs}
    for c, names in runs.items():
        if names:
            cells = []
            for k in KEYS:
                v = [scores[n][k] for n in names]
                cells.append(f"{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}")
            lines.append(f"| {c} | {len(names)} | " + " | ".join(cells) + " |")
    for ref in refs:
        lines += ["", f"## Seed-paired differences to {ref} (mean [t 95 % CI], n seeds)", "",
                  "| config | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
        for c in configs:
            if c == ref:
                continue
            pairs = [(f"{prefix}{c}_s{s}", f"{prefix}{ref}_s{s}") for s in SEEDS]
            pairs = [(a, b) for a, b in pairs if a in scores and b in scores]
            if len(pairs) < 2:
                continue
            cells = []
            for k in KEYS:
                t = ds.paired_t([scores[a][k] - scores[b][k] for a, b in pairs])
                cells.append(f"{t['mean']:+.2f} [{t['t_ci'][0]:+.2f}, {t['t_ci'][1]:+.2f}] n={t['n']}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PREFIX)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    nights = night_ids()
    ironly = ironly_boxes()
    cache = OUT / f"scores_{a.prefix.strip('_')}.json"
    scores = json.loads(cache.read_text()) if cache.exists() else {}
    for p in sorted(RUNS.glob(f"{a.prefix}*/ws/out/*/preds_{SPLIT}.json")):
        n, mt = p.parent.name, p.stat().st_mtime
        if scores.get(n, {}).get("_mtime") != mt:
            scores[n] = dict(score(p, nights, ironly), _mtime=mt)
            cache.write_text(json.dumps(scores))
            print(n, {k: round(v, 2) for k, v in scores[n].items() if k[0] != "_"}, flush=True)
    n_types = {t: sum(x.count(t) for _, x in ironly.values()) for t in bt.TYPES}
    dst = OUT / f"tables_{a.prefix.strip('_')}.md"
    dst.write_text("\n".join(table_lines(scores, n_types, a.prefix)) + "\n", encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests.**
Run: `.venv/Scripts/python -m pytest -q tests`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/screen_tables.py tests/test_screen_tables.py
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "screen_tables: dyn_cl_ runs, class-agnostic AP, references B and P

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Kaggle omega package and spec correction

**Files:**
- Modify: `scripts/pack_kaggle.py` (`pack_omega`)
- Modify: `tests/test_pack_kaggle.py`
- Modify: `docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md` (section 6, single-class bullet)

**Interfaces:**
- Produces: `pack_omega(proc, out, slug, user, variants=("P", "I", "V", "F"))`. It copies `ironly_w/<v>` for each variant and `stats/clean_<v>.json`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_pack_kaggle.py`:

```python
def test_pack_omega_only_clean_variants(tmp_path):
    import pack_kaggle as pk

    proc = tmp_path / "proc"
    for v in ("P", "I", "V", "F", "T"):
        d = proc / "ironly_w" / v / "train"
        d.mkdir(parents=True)
        (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3\n")
    (proc / "stats").mkdir()
    for v in ("P", "I", "V", "F"):
        (proc / "stats" / f"clean_{v}.json").write_text("{}")
    (proc / "stats" / "omega_T.json").write_text("{}")
    d = pk.pack_omega(proc, tmp_path / "out", "dynirs-omega", "someone")
    assert sorted(p.name for p in (d / "ironly_w").iterdir()) == ["F", "I", "P", "V"]
    assert sorted(p.name for p in (d / "stats").iterdir()) == [f"clean_{v}.json" for v in ("F", "I", "P", "V")]
```

The file already imports `pack_kaggle as pk` at the top; drop the local `import pack_kaggle as pk` line if you prefer. The existing `test_pack_omega_and_code` packs only a variant `T` and would break under the new default, so change its first lines to use variant `P`:

```python
    proc = tmp_path / "proc"
    (proc / "ironly_w" / "P" / "train").mkdir(parents=True)
    (proc / "ironly_w" / "P" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3
")
    (proc / "stats").mkdir()
    (proc / "stats" / "clean_P.json").write_text("{}")
    out = pk.pack_omega(proc, tmp_path / "out", "dynirs-omega", user="me", variants=("P",))
    assert (out / "ironly_w" / "P" / "train" / "a.txt").exists()
    assert (out / "stats" / "clean_P.json").exists()
```

- [ ] **Step 2: Run it and check that it fails.**
Run: `.venv/Scripts/python -m pytest -q tests/test_pack_kaggle.py`
Expected: FAIL, because `T` is copied and so is `omega_T.json`.

- [ ] **Step 3: Implement.** In `scripts/pack_kaggle.py`:

```python
CLEAN_VARIANTS = ("P", "I", "V", "F")


def pack_omega(proc, out, slug, user, variants=CLEAN_VARIANTS):
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
```

In the spec's section 6, replace the bullet that begins with `` `lowlight.IRPairDataset.update_labels_info` sets the class `` with:

```
  - The injected IR-only rows are already class 0 (clean_labels.py writes class 0), so the loader needs
    no change; Ultralytics' single_cls maps the RGB labels to 0.
```

- [ ] **Step 4: Run the tests.**
Run: `.venv/Scripts/python -m pytest -q tests`
Expected: all pass.

- [ ] **Step 5: Commit and push.**

```bash
git add scripts/pack_kaggle.py tests/test_pack_kaggle.py docs/superpowers/specs/2026-11-25-clean-ir-labels-design.md
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "pack_kaggle: omega package holds only the clean variants; spec: loader needs no single-class change

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 6: Build the labels on real data and check them (after the teacher has trained)

Prerequisite: `runs/remote/dyn_teacher_rgb1c_f{0,1}/ws/out/dyn_teacher_rgb1c_f{0,1}/preds_train.json` exist. The user starts the teacher queue after Task 1.

**Files:** none changed. The outputs are data (git-ignored): `data/dronevehicle/proc/ironly_w/{P,I,V,F}`, `proc/stats/clean_*.json` and `runs/analysis/clean/F_*.jpg`.

- [ ] **Step 1: Check the teachers.** Confirm that each teacher's validation mAP50:95 is above 0.5 (the 5-class teachers reached about 0.55), and that the predictions are single class.

```bash
for f in 0 1; do o=runs/remote/dyn_teacher_rgb1c_f$f/ws/out/dyn_teacher_rgb1c_f$f; awk -F, 'NR>1{print $1,$9}' $o/results.csv | tail -1; .venv/Scripts/python -c "import json;p=json.load(open('$o/preds_train.json'));print(len(p), sorted({x['category_id'] for x in p}))"; done
```

Expected: mAP above 0.5, and categories `[0]` for both folds.

- [ ] **Step 2: Build the labels.**
Run: `.venv/Scripts/python src/clean_labels.py build --proc data/dronevehicle/proc`
Expected (one JSON line per variant):
- `n_boxes` 34061 for every variant;
- P: `n_taught` 34061;
- I: `n_ignored` 34061;
- F: `n_images_residual` 319 (images with IR-only boxes among the 1,093 over 4 px), `n_ignored_residual` 1621, `n_moved_gt_2px` 6025;
- V: `n_taught` about 3,000–4,000 (the 5-class teacher gave 3,626);
- F: `n_taught` at least V's.

If F's numbers are far from these, stop and report.

- [ ] **Step 3: Check the loader on real data** for the ignore-carrying variants.
Run: `for v in V F; do DV_DATA=data/dronevehicle .venv/Scripts/python scripts/check_wsoft_loader.py --variant $v --batches 20 --device cpu 2>&1 | grep -E "^ok|Error"; done`
Expected: two `ok:` lines, each with both weighted boxes > 0 and ignore boxes > 0.

- [ ] **Step 4: Make the sanity sheets and look at them.**
Run: `.venv/Scripts/python src/clean_labels.py sheet --proc data/dronevehicle/proc`
Then open:
- `runs/analysis/clean/F_taught.jpg`: yellow boxes should sit on vehicles;
- `runs/analysis/clean/F_ignored_teacher.jpg`: dark, displaced or empty boxes;
- `runs/analysis/clean/F_ignored_residual.jpg`: images with uneven offsets.

Report what you see.

- [ ] **Step 5: Hand over for screening.** Give the user the commands:
  - repack the omega package (`python scripts/pack_kaggle.py omega --user <kaggle user>`) and the code package (`python scripts/pack_kaggle.py code --user <kaggle user>`), then upload both;
  - on Kaggle, run **only** `QUEUE = 'configs/queues/dyn/t4_clean_s0.json'` first (5 runs, seed 0);
  - score seed 0 and review it with the user. One seed is directional only (proxy noise is about 1.3 AP), so
    the table shows means without CIs. Run `t4_clean_s1` and `t4_clean_s2`, and then the full scale, only
    after the user decides to continue;
  - after the runs come back, score them with `.venv/Scripts/python src/screen_tables.py`.
