# Dynamic ω and Repeat Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Screen three new student configurations (CB, D, DB) that re-score the IR-only boxes during training and/or repeat the images holding learnable IR-only boxes, on the local GPU.

**Architecture:**
- A new pure-numpy module `src/dyn_omega.py` holds the update rule and the repeat rule.
- `src/job_main.py` calls it at the start of each training chunk:
  - the chunks are the stages;
  - it predicts with the run's `last.pt` on its IR-only train images, writes the stage's weight files under the run's own output directory, and repeats the selected image ids in the run's train list.
- Queues, the label build (per-box teacher score) and the score table are extended.

**Tech Stack:** Python 3.11, numpy, Ultralytics 8.4.173, pytest; `.venv/Scripts/python` on Windows.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-27-dynamic-omega-balance-design.md`.
- Fixed constants, not tuned on results: S_HI = 0.5, S_LO = 0.05, REPEAT_OMEGA = 0.5, R = 2, stages after epochs 5 and 10 (screening).
- Starting labels: variant C (`proc/ironly_w/C/train`). Boxes with ω < 0 (ignore regions) never change.
- Screening protocol unchanged: YOLO26n, `single_cls`, 20 % train subset (proxy seed 0), 15 epochs, 704 px, batch 32, seeds 0–2, val.
- Job names `dyn_cl_<code>_s<seed>`, codes CB, D, DB; queues `configs/queues/dyn/t4_dyn_s{0,1,2}.json`.
- Local GPU only, no Kaggle.
- Commit with `git -c user.email=184421321+eociV@users.noreply.github.com commit ...`. End each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run tests with `.venv/Scripts/python -m pytest -q tests` from `D:\YSL\DynIRS`.

---

### Task 1: `src/dyn_omega.py` — update rule, repeat rule, weight-dir I/O

**Files:**
- Create: `src/dyn_omega.py`
- Modify: `scripts/run_queue.py:28` (SRC_FILES)
- Test: `tests/test_dyn_omega.py`

**Interfaces:**
- Produces:
  - `read_dir(d) -> {id: (n, 6) float array}`
  - `write_dir(d, rows) -> None`
  - `box_scores(rows, preds) -> {id: (n,) array}`, where preds is `{id: (m, 5) xyxy px + score}`
  - `load_conf(path, rows) -> {id: (n,) array}`
  - `rescore(rows, s, c) -> (new_rows, stats dict)`
  - `repeat_ids(rows, thr=REPEAT_OMEGA) -> sorted list of ids`
  - constants `S_HI`, `S_LO`, `REPEAT_OMEGA`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dyn_omega.py
import json

import numpy as np
import pytest

import dyn_omega as do


def _row(x, y, w):
    return [0, x, y, 0.1, 0.1, w]


def test_rescore_rules():
    rows = {"a": np.array([_row(0.2, 0.2, 0.3),    # s 0.8 -> raised to 0.8
                           _row(0.4, 0.4, 0.6),    # s 0.55 < omega 0.6 -> stays 0.6
                           _row(0.6, 0.6, 0.3),    # s 0.2 (middle) -> kept
                           _row(0.8, 0.8, 0.3),    # s 0.01, c 0.01 -> ignore
                           _row(0.8, 0.2, 0.3),    # s 0.01, c 0.2 -> kept
                           _row(0.2, 0.8, -1.0)])}  # frozen ignore region, even with s 0.9
    s = {"a": np.array([0.8, 0.55, 0.2, 0.01, 0.01, 0.9])}
    c = {"a": np.array([0.0, 0.0, 0.0, 0.01, 0.2, 0.0])}
    new, st = do.rescore(rows, s, c)
    assert new["a"][:, 5].tolist() == pytest.approx([0.8, 0.6, 0.3, -1.0, 0.3, -1.0])
    assert rows["a"][0, 5] == 0.3  # input not modified
    assert (st["raised"], st["kept"], st["ignored"], st["frozen"]) == (1, 3, 1, 1)
    assert st["omega_taught_mean"] == pytest.approx(np.mean([0.8, 0.6, 0.3, 0.3]))


def test_rescore_leaves_unscored_images_unchanged():
    rows = {"a": np.array([_row(0.5, 0.5, 0.3)]), "b": np.array([_row(0.5, 0.5, 0.3)])}
    new, _ = do.rescore(rows, {"a": np.array([0.0])}, {"a": np.array([0.0])})
    assert new["a"][0, 5] == -1.0 and new["b"][0, 5] == 0.3


def test_box_scores_iou_match():
    rows = {"a": np.array([_row(0.5, 0.5, 0.3), _row(0.1, 0.1, 0.3)])}
    # box 0 in pixels (640x512): x 288-352, y 230.4-281.6
    preds = {"a": np.array([[288, 230.4, 352, 281.6, 0.7], [290, 232, 350, 280, 0.9], [0, 0, 5, 5, 0.99]])}
    s = do.box_scores(rows, preds)
    assert s["a"].tolist() == pytest.approx([0.9, 0.0])
    assert do.box_scores(rows, {})["a"].tolist() == [0.0, 0.0]


def test_repeat_ids():
    rows = {"a": np.array([_row(0.5, 0.5, 0.3)]), "b": np.array([_row(0.5, 0.5, 0.3), _row(0.2, 0.2, 0.5)]),
            "c": np.zeros((0, 6)), "d": np.array([_row(0.5, 0.5, -1.0)])}
    assert do.repeat_ids(rows) == ["b"]


def test_read_write_roundtrip(tmp_path):
    rows = {"a": np.array([_row(0.5, 0.25, 0.3), _row(0.1, 0.2, -1.0)]), "b": np.zeros((0, 6))}
    do.write_dir(tmp_path / "w", rows)
    back = do.read_dir(tmp_path / "w")
    assert sorted(back) == ["a", "b"] and back["b"].shape == (0, 6)
    assert back["a"] == pytest.approx(rows["a"])


def test_load_conf_checks_lengths(tmp_path):
    rows = {"a": np.array([_row(0.5, 0.5, 0.3)]), "b": np.array([_row(0.5, 0.5, 0.3)])}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": [0.4]}))
    c = do.load_conf(p, rows)
    assert c["a"].tolist() == [0.4] and c["b"].tolist() == [0.0]
    p.write_text(json.dumps({"a": [0.4, 0.1]}))
    with pytest.raises(ValueError):
        do.load_conf(p, rows)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest -q tests/test_dyn_omega.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'dyn_omega'`.

- [ ] **Step 3: Write the implementation**

```python
# src/dyn_omega.py
"""Stage-wise dynamic omega and repeat sampling of IR-only boxes
(docs/superpowers/specs/2026-11-27-dynamic-omega-balance-design.md).

Weight files: <dir>/<image id>.txt, rows "cls cx cy w h omega" (normalised boxes; omega -1 = ignore region).
At a stage boundary the student scores each box (s = best score of a detection with IoU >= 0.5):
  s >= S_HI                    -> omega = max(omega, s)
  S_LO <= s < S_HI             -> omega unchanged
  s < S_LO and teacher c < S_LO -> ignore region (omega -1)
  s < S_LO and c >= S_LO        -> omega unchanged
Ignore regions (omega < 0) never change. Images not scored are copied unchanged.
Repeat sampling: an image with a box of omega >= REPEAT_OMEGA is listed R times per epoch (job_main).
"""
import json
from pathlib import Path

import numpy as np

from dv_pairs import iou_matrix

W, H = 640, 512
MATCH_IOU = 0.5
S_HI, S_LO = 0.5, 0.05
REPEAT_OMEGA = 0.5


def read_dir(d):
    out = {}
    for f in sorted(Path(d).glob("*.txt")):
        a = np.loadtxt(f, ndmin=2, dtype=np.float64) if f.stat().st_size else np.zeros((0, 6))
        out[f.stem] = a.reshape(-1, 6)
    return out


def write_dir(d, rows):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for k, a in rows.items():
        (d / f"{k}.txt").write_text("".join(
            f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]:.4f}\n" for r in a))


def _xyxy(a):
    a = np.asarray(a, float).reshape(-1, 4)
    x, y, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def box_scores(rows, preds):
    """{id: (n,) best detection score with IoU >= MATCH_IOU per box}; preds {id: (m, 5) xyxy px + score}."""
    out = {}
    for k, a in rows.items():
        p = preds.get(k)
        if p is None or not len(p) or not len(a):
            out[k] = np.zeros(len(a))
            continue
        p = np.asarray(p, float).reshape(-1, 5)
        m = iou_matrix(_xyxy(a[:, 1:5]), p[:, :4])
        out[k] = np.where(m >= MATCH_IOU, p[None, :, 4], 0.0).max(1)
    return out


def load_conf(path, rows):
    """{id: (n,) teacher score per row}; json {id: [score per written row]}; missing ids -> zeros."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for k, a in rows.items():
        v = np.asarray(raw.get(k, [0.0] * len(a)), float)
        if len(v) != len(a):
            raise ValueError(f"{k}: {len(a)} weight rows but {len(v)} teacher scores")
        out[k] = v
    return out


def rescore(rows, s, c):
    """(new rows, stats) after one stage; s and c: {id: (n,)} for the scored images only."""
    new = {}
    n = {"raised": 0, "kept": 0, "ignored": 0, "frozen": 0}
    for k, a in rows.items():
        a = np.array(a, float, copy=True)
        if k not in s:
            new[k] = a
            continue
        w, sk, ck = a[:, 5].copy(), np.asarray(s[k], float), np.asarray(c[k], float)
        frozen = w < 0
        hi = ~frozen & (sk >= S_HI)
        ign = ~frozen & (sk < S_LO) & (ck < S_LO)
        a[hi, 5] = np.maximum(w[hi], sk[hi])
        a[ign, 5] = -1.0
        n["raised"] += int((hi & (sk > w)).sum())
        n["ignored"] += int(ign.sum())
        n["frozen"] += int(frozen.sum())
        n["kept"] += int((~frozen & ~ign).sum() - (hi & (sk > w)).sum())
        new[k] = a
    taught = np.concatenate([a[a[:, 5] > 0, 5] for a in new.values()] or [np.zeros(0)])
    stats = dict(n, n_images_scored=len(s), omega_taught_mean=float(taught.mean()) if len(taught) else 0.0,
                 n_taught=int(len(taught)), s_hi=S_HI, s_lo=S_LO)
    return new, stats


def repeat_ids(rows, thr=REPEAT_OMEGA):
    return sorted(k for k, a in rows.items() if len(a) and (a[:, 5] >= thr).any())
```

- [ ] **Step 4: Add the module to the job copy list**

In `scripts/run_queue.py`, change the SRC_FILES line to:

```python
SRC_FILES = ["lowlight.py", "dv_pairs.py", "dv_prep.py", "job_main.py", "zdce.py", "evaluate.py", "dfine_job.py",
             "dyn_omega.py"]
```

Keep the existing order of the list, and append `"dyn_omega.py"` at the end. The job runs from `ws/src`, and only the files listed here are copied there.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest -q tests/test_dyn_omega.py tests/test_run_queue.py tests/test_portability.py`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dyn_omega.py tests/test_dyn_omega.py scripts/run_queue.py
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "dyn_omega: stage-wise omega update rule and repeat rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Per-box teacher score of variant C

**Files:**
- Modify: `src/clean_labels.py` (`build_variant`, `main`)
- Test: `tests/test_clean_labels.py`

**Interfaces:**
- Consumes: `build_variant(variant, boxes, pairs, preds, rgb=None)`, which exists.
- Produces:
  - `build_variant(..., conf_out=None)`: when `conf_out` is a dict, it gets `{id: (n,) teacher score per row}` (0 where not computed).
  - `written_mask(rows_k, w_k) -> bool array`: the rows `omega.write_weights` writes.
  - `main` writes `proc/stats/clean_C_conf.json` = `{id: [c of each written row]}`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_clean_labels.py`)

```python
def test_variant_c_conf_out_aligned_with_written_rows():
    boxes = {"a": np.array([_box(0.5, 0.5), _box(0.2, 0.2), _box(0.8, 0.8)])}
    pairs = {"a": {"offsets": [[6.4, 0]] * 3}}
    rgb = {"a": om.xywhn_to_xyxy(np.array([[0.49, 0.5, 0.1, 0.1]]))}
    preds = {"a": np.array([_pred_on(_box(0.19, 0.2), 0.9)])}
    conf = {}
    rows, w, _ = cl.build_variant("C", boxes, pairs, preds, rgb, conf_out=conf)
    assert conf["a"].tolist() == pytest.approx([0.0, 0.9, 0.0])
    keep = cl.written_mask(rows["a"], w["a"])
    assert keep.tolist() == [False, True, True]  # box 0 dropped as an RGB duplicate
    assert conf["a"][keep].tolist() == pytest.approx([0.9, 0.0])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest -q tests/test_clean_labels.py -k conf_out`
Expected: FAIL with `TypeError: build_variant() got an unexpected keyword argument 'conf_out'`.

- [ ] **Step 3: Implement**

In `src/clean_labels.py`:

1. Change the signature to `def build_variant(variant, boxes, pairs, preds, rgb=None, conf_out=None):`.
2. Inside the per-image loop, set `cv = np.zeros(len(b))` right after `b[:, 0] = 0`.
3. Change the C branch to:

```python
        elif variant == "C":
            c = _conf(b, preds.get(k))
            cv = c
            x = np.where(c >= TAU, c, P_OMEGA)
```

4. Before `rows[k], w[k] = b, x`, add:

```python
        if conf_out is not None:
            conf_out[k] = np.asarray(cv, float)
```

Then add the module-level function below `rgb_boxes`:

```python
def written_mask(rows_k, w_k):
    """Rows that omega.write_weights writes (omega != 0 and a positive box size)."""
    rows_k = np.asarray(rows_k, float).reshape(-1, 5)
    return (np.asarray(w_k) != 0) & (rows_k[:, 3] > 0) & (rows_k[:, 4] > 0)
```

In `main`, replace `rows, w, st = build_variant(v, boxes, pairs, preds, rgb)` with:

```python
        conf = {}
        rows, w, st = build_variant(v, boxes, pairs, preds, rgb, conf_out=conf)
```

After the `(proc / "stats" / f"clean_{v}.json").write_text(...)` line, add:

```python
        if v == "C":
            cj = {k: [round(float(x), 4) for x in conf[k][written_mask(rows[k], w[k])]] for k in rows}
            (proc / "stats" / "clean_C_conf.json").write_text(json.dumps(cj))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest -q tests/test_clean_labels.py`
Expected: all PASS.

- [ ] **Step 5: Rebuild C on real data and check that the weight files did not change**

```bash
cat runs/remote/dyn_cl_C_s0/ws/out/dyn_cl_C_s0/omega_hash.txt
PYTHONIOENCODING=utf-8 .venv/Scripts/python src/clean_labels.py build --proc data/dronevehicle/proc --variants C --teacher "dyn_teacher_rgb26l_f{}"
.venv/Scripts/python -c "import sys; sys.path.insert(0,'src'); import job_main as jm; print(jm.hash_omega_dir('data/dronevehicle/proc/ironly_w/C/train'))"
```

Expected: the two hashes are identical. The build prints the same stats as before: `n_taught` 31914, `n_ignored` 1342. The file `data/dronevehicle/proc/stats/clean_C_conf.json` now exists.

Then check the alignment with the files:

```bash
.venv/Scripts/python -c "
import sys, json; sys.path.insert(0,'src'); import dyn_omega as do
rows = do.read_dir('data/dronevehicle/proc/ironly_w/C/train')
c = do.load_conf('data/dronevehicle/proc/stats/clean_C_conf.json', rows)
print(len(rows), sum(len(v) for v in c.values()))"
```

Expected: no ValueError. The second number is 33256 (31914 taught + 1342 ignored).

- [ ] **Step 6: Commit**

```bash
git add src/clean_labels.py tests/test_clean_labels.py
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "clean_labels: write the per-box teacher score of variant C (clean_C_conf.json)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Stage-wise re-scoring and repeat sampling in `job_main`

**Files:**
- Modify: `src/job_main.py` (`_data_yaml`, `train`, new `_epochs_done`, `_predict_ids`, `dyn_prepare`)
- Test: `tests/test_job_main.py`

**Interfaces:**
- Consumes: `dyn_omega.read_dir`, `write_dir`, `box_scores`, `load_conf`, `rescore` and `repeat_ids` (Task 1); `proc/stats/clean_C_conf.json` (Task 2).
- Produces:
  - The job key `dyn_omega`: `{"stages": [5, 10], "update": bool, "repeat": int, "conf": "clean_C_conf.json"}`.
  - `dyn_prepare(job, cfg, predict_fn=_predict_ids) -> (cfg, repeat or None)`, where repeat is `{"ids": [...], "k": int}`.
  - The in-memory job key `_repeat`, read by `_data_yaml`.
  - Per run:
    - `out/<name>/omega_dyn/stage<k>/` holds the weight files, `stats.json` and `.done`;
    - `out/<name>/train_images.jsonl` has one line per chunk: `{"from", "to", "n_images"}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_job_main.py`)

```python
import json

import numpy as np


def _proxy_setup(tmp_path, monkeypatch, n=10):
    data, ws = tmp_path / "data", tmp_path / "ws"
    img = data / "proc" / "rgb" / "train" / "images"
    img.mkdir(parents=True)
    for i in range(n):
        (img / f"i{i}.jpg").write_bytes(b"")
    monkeypatch.setattr(jm, "DATA", data)
    monkeypatch.setattr(jm, "WS", ws)
    return data, ws


def test_data_yaml_repeats_ids(tmp_path, monkeypatch):
    data, ws = _proxy_setup(tmp_path, monkeypatch)
    job = {"name": "j", "proxy_fraction": 0.5, "_repeat": {"ids": ["i1", "i3", "i9"], "k": 2}}
    jm._data_yaml("rgb_proxy", job)
    ids = json.loads((ws / "out" / "j" / "proxy_ids.json").read_text())
    lst = (data / "proc" / "rgb_train_j.txt").read_text().split()
    extra = [i for i in ids if i in ("i1", "i3", "i9")]
    assert len(lst) == len(ids) + len(extra)  # proxy_ids.json keeps the subset without repeats


def _weights(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    for k, ws_ in rows.items():
        (d / f"{k}.txt").write_text("".join(f"0 {x} {y} 0.1 0.1 {w}\n" for x, y, w in ws_))


def test_dyn_prepare_stage1_rescores_and_repeats(tmp_path, monkeypatch):
    data, ws = _proxy_setup(tmp_path, monkeypatch)
    base = data / "proc" / "ironly_w" / "C" / "train"
    _weights(base, {"i0": [(0.5, 0.5, 0.3)], "i1": [(0.5, 0.5, 0.3)], "i2": [(0.5, 0.5, 0.3)]})
    (data / "proc" / "stats").mkdir(parents=True)
    (data / "proc" / "stats" / "c.json").write_text(json.dumps({"i0": [0.0], "i1": [0.0], "i2": [0.0]}))
    out = ws / "out" / "j"
    out.mkdir(parents=True)
    (out / "proxy_ids.json").write_text(json.dumps(["i0", "i1"]))  # i2 is outside the run's subset
    monkeypatch.setattr(jm, "_epochs_done", lambda last: 5)
    seen = {}

    def fake_predict(last, ids, imgsz):
        seen["ids"] = ids
        return {"i0": np.array([[288, 230.4, 352, 281.6, 0.9]])}  # i0 seen, i1 not

    job = {"name": "j", "train": {"imgsz": 704},
           "dyn_omega": {"stages": [5, 10], "update": True, "repeat": 2, "conf": "c.json"}}
    cfg, rep = jm.dyn_prepare(job, {"ir_soft_w_dir": str(base)}, predict_fn=fake_predict)
    st1 = out / "omega_dyn" / "stage1"
    assert seen["ids"] == ["i0", "i1"] and Path(cfg["ir_soft_w_dir"]) == st1 and (st1 / ".done").exists()
    assert (st1 / "i0.txt").read_text().split()[-1] == "0.9000"
    assert (st1 / "i1.txt").read_text().split()[-1] == "-1.0000"
    assert (st1 / "i2.txt").read_text().split()[-1] == "0.3000"
    assert rep == {"ids": ["i0"], "k": 2}
    # a second call (the same chunk restarted) reuses the stage without predicting again
    seen.clear()
    cfg2, _ = jm.dyn_prepare(job, {"ir_soft_w_dir": str(base)}, predict_fn=fake_predict)
    assert not seen and cfg2 == cfg


def test_dyn_prepare_repeat_only_and_stage0(tmp_path, monkeypatch):
    data, ws = _proxy_setup(tmp_path, monkeypatch)
    base = data / "proc" / "ironly_w" / "C" / "train"
    _weights(base, {"i0": [(0.5, 0.5, 0.7)], "i1": [(0.5, 0.5, 0.3)]})
    monkeypatch.setattr(jm, "_epochs_done", lambda last: 5)
    job = {"name": "j", "train": {}, "dyn_omega": {"stages": [5, 10], "update": False, "repeat": 2}}
    cfg, rep = jm.dyn_prepare(job, {"ir_soft_w_dir": str(base)}, predict_fn=None)
    assert Path(cfg["ir_soft_w_dir"]) == base and rep == {"ids": ["i0"], "k": 2}
    monkeypatch.setattr(jm, "_epochs_done", lambda last: 0)
    job["dyn_omega"] = {"stages": [5, 10], "update": True, "repeat": 1, "conf": "c.json"}
    cfg, rep = jm.dyn_prepare(job, {"ir_soft_w_dir": str(base)}, predict_fn=None)
    assert Path(cfg["ir_soft_w_dir"]) == base and rep is None
    assert jm.dyn_prepare({"name": "j"}, {"a": 1}) == ({"a": 1}, None)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest -q tests/test_job_main.py -k "repeat or dyn_prepare"`
Expected: FAIL. The proxy list has no repeats, and `dyn_prepare` / `_epochs_done` do not exist.

- [ ] **Step 3: Implement the repeat in `_data_yaml`**

In `src/job_main.py`, `_data_yaml`, right after the existing `night_repeat` block (`ids = ids + [i for i in ids if i in night] * (rep["k"] - 1)`), add:

```python
        dyn_rep = job.get("_repeat")  # {"ids": [...], "k": R}: repeat sampling (dyn_omega)
        if dyn_rep:
            keep = set(dyn_rep["ids"])
            ids = ids + [i for i in ids if i in keep] * (dyn_rep["k"] - 1)
```

- [ ] **Step 4: Implement `_epochs_done`, `_predict_ids` and `dyn_prepare`** (place them above `def train`)

```python
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
```

`dyn_prepare` uses `json` and `Path`. `job_main.py` already imports both (check the top of the file). If one is missing, add the import.

- [ ] **Step 5: Wire it into `train`**

In `train(job)`, replace:

```python
    cfg = resolve_config(job.get("config", {}), DATA, WS)
    ll.CONFIG.update(cfg)
```

with:

```python
    cfg = resolve_config(job.get("config", {}), DATA, WS)
    cfg, rep = dyn_prepare(job, cfg)
    job = dict(job, _repeat=rep)
    ll.CONFIG.update(cfg)
```

Then, right after `over["data"] = _data_yaml(over["data"], job)`, add:

```python
    if job.get("dyn_omega"):  # images per epoch of this chunk (repeat sampling lengthens epochs)
        lst = DATA / "proc" / f"rgb_train_{job['name']}.txt"
        n = len(lst.read_text().split()) if lst.exists() else None
        frm = _epochs_done(WS / "out" / job["name"] / "weights" / "last.pt")
        to = cfg.get("stop_after_epoch") or job["train"]["epochs"]
        (WS / "out" / job["name"]).mkdir(parents=True, exist_ok=True)
        with open(WS / "out" / job["name"] / "train_images.jsonl", "a") as f:
            f.write(json.dumps({"from": frm, "to": to, "n_images": n}) + "\n")
```

- [ ] **Step 6: Run the tests**

Run: `.venv/Scripts/python -m pytest -q tests`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/job_main.py tests/test_job_main.py
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "job_main: dyn_omega jobs re-score IR-only boxes at chunk boundaries and repeat learnable images

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Queues, score table and loader check

**Files:**
- Modify: `scripts/make_dyn_queues.py`, `src/screen_tables.py`, `scripts/check_wsoft_loader.py`
- Test: `tests/test_queues.py`, `tests/test_screen_tables.py`

**Interfaces:**
- Consumes: the job key `dyn_omega` and the file `train_images.jsonl` (Task 3).
- Produces:
  - queues `t4_dyn_s{0,1,2}` with jobs `dyn_cl_{CB,D,DB}_s<seed>`;
  - `screen_tables.image_epochs(out_dir) -> int`;
  - the KEYS column `img_epochs_k`;
  - `check_wsoft_loader.py --dir <weight dir>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_queues.py`:

```python
def test_dyn_queues():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = q[f"t4_dyn_s{s}"]
        assert [j["name"] for j in jobs] == [f"dyn_cl_{c}_s{s}" for c in ("CB", "D", "DB")]
        by = {j["name"].split("_")[2]: j for j in jobs}
        for c, j in by.items():
            assert j["chunks"] == [5, 10, None] and j["proxy_fraction"] == 0.2 and j["train"]["seed"] == s
            assert j["train"]["epochs"] == 15 and j["train"]["imgsz"] == 704 and j["train"]["single_cls"]
            assert j["config"]["ir_soft_w_dir"] == "auto:C" and j["config"]["ir_ignore"] is True
            d = j["dyn_omega"]
            assert d["stages"] == [5, 10] and d["conf"] == "clean_C_conf.json"
        assert (by["CB"]["dyn_omega"]["update"], by["CB"]["dyn_omega"]["repeat"]) == (False, 2)
        assert (by["D"]["dyn_omega"]["update"], by["D"]["dyn_omega"]["repeat"]) == (True, 1)
        assert (by["DB"]["dyn_omega"]["update"], by["DB"]["dyn_omega"]["repeat"]) == (True, 2)
```

In `test_queue_names`, add `+ [f"t4_dyn_s{s}" for s in (0, 1, 2)]` to the expected list.

Append to `tests/test_screen_tables.py`:

```python
def test_image_epochs(tmp_path):
    import json

    import screen_tables as stt

    a = tmp_path / "a"
    a.mkdir()
    (a / "proxy_ids.json").write_text(json.dumps(["x"] * 100))
    (a / "results.csv").write_text("epoch\n" + "1\n" * 15)
    assert stt.image_epochs(a) == 1500
    b = tmp_path / "b"
    b.mkdir()
    (b / "train_images.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [{"from": 0, "to": 5, "n_images": 100},
                                               {"from": 5, "to": 10, "n_images": 120},
                                               {"from": 10, "to": 15, "n_images": 130}]))
    assert stt.image_epochs(b) == 1750
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest -q tests/test_queues.py tests/test_screen_tables.py`
Expected: FAIL (`KeyError: 't4_dyn_s0'`, `AttributeError: image_epochs`).

- [ ] **Step 3: Implement the queues**

In `scripts/make_dyn_queues.py`, add three entries to `CONFIGS`:

```python
    "CB": dict(PAIRED, ir_soft_w_dir="auto:C", ir_ignore=True),  # C + repeat sampling
    "D": dict(PAIRED, ir_soft_w_dir="auto:C", ir_ignore=True),   # C + stage-wise omega
    "DB": dict(PAIRED, ir_soft_w_dir="auto:C", ir_ignore=True),  # both
```

Below `SCREEN_C`, add:

```python
DYN = {"CB": {"update": False, "repeat": 2}, "D": {"update": True, "repeat": 1}, "DB": {"update": True, "repeat": 2}}
DYN_STAGES = [5, 10]  # re-scoring after epochs 5 and 10 of the 15-epoch screening
```

In `build()`, inside the seed loop after the `t4_cleanC_s{s}` entry, add:

```python
        q[f"t4_dyn_s{s}"] = [
            student(c, s, f"dyn_cl_{c}_s{s}", "t4", SCREEN_RUN["epochs"], SCREEN_RUN["imgsz"], "rgb_proxy", "val",
                    extra={"proxy_fraction": SCREEN_RUN["proxy_fraction"], "chunks": DYN_STAGES + [None],
                           "dyn_omega": dict(DYN[c], stages=DYN_STAGES, conf="clean_C_conf.json")})
            for c in DYN]
```

Extend the module docstring's queue list with one line:
`t4_dyn_s<seed> CB, D, DB: C + repeat sampling / stage-wise omega / both (src/dyn_omega.py)`.

- [ ] **Step 4: Implement the score table**

In `src/screen_tables.py`:

```python
CONFIGS = ("B", "P", "I", "V", "F", "C", "CB", "D", "DB")
REFS = ("B", "P", "C")
```

Append `"img_epochs_k"` to the end of `KEYS`. Add above `main`:

```python
def image_epochs(out):
    """Training images seen over the run (repeat sampling counted): train_images.jsonl, else subset x epochs."""
    out = Path(out)
    f = out / "train_images.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        last = {}
        for r in rows:  # a restarted chunk appends again: keep the last line per start epoch
            last[r["from"]] = r
        return int(sum(r["n_images"] * (r["to"] - r["from"]) for r in last.values()))
    n = len(json.loads((out / "proxy_ids.json").read_text()))
    epochs = len((out / "results.csv").read_text().splitlines()) - 1
    return n * epochs
```

In `main`, after the scoring loop and before `n_types = ...`, add:

```python
    for n in scores:  # recomputed every time: cheap, and old cache entries lack it
        scores[n]["img_epochs_k"] = image_epochs(RUNS / n / "ws" / "out" / n) / 1000
```

`screen_tables.py` must import `json` and `Path`. Check the top of the file and add them if missing.

- [ ] **Step 5: Implement `--dir` in the loader check**

In `scripts/check_wsoft_loader.py`, add `ap.add_argument("--dir", help="weight directory to check instead of auto:<variant>")`. Then set `w = a.dir or f"auto:{a.variant}"`, and use `w` in place of `f"auto:{a.variant}"` in both `resolve_config` calls (lines 31 and 37).

- [ ] **Step 6: Regenerate the queues and run all tests**

```bash
.venv/Scripts/python scripts/make_dyn_queues.py
.venv/Scripts/python -m pytest -q tests
```

Expected:
- the generator prints `t4_dyn_s0 3`, `t4_dyn_s1 3` and `t4_dyn_s2 3` among the queues;
- all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/make_dyn_queues.py src/screen_tables.py scripts/check_wsoft_loader.py tests/test_queues.py tests/test_screen_tables.py configs/queues/dyn
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "queues t4_dyn_s* (CB, D, DB); score table counts image-epochs; loader check takes a weight dir

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Local smoke run, then the screening queues

**Files:**
- Create (scratch, not committed): `<scratchpad>/smoke_dyn.json`

- [ ] **Step 1: Write a smoke queue**

The smoke job is DB with a 2 % subset, 3 epochs, stages after epochs 1 and 2, and no prediction:

```bash
.venv/Scripts/python - <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
import make_dyn_queues as mq
j = mq.build()["t4_dyn_s0"][2]
j = dict(j, name="smoke_dyn_DB", proxy_fraction=0.02, chunks=[1, 2, None], predict=None)
j["train"] = dict(j["train"], epochs=3)
j["dyn_omega"] = dict(j["dyn_omega"], stages=[1, 2])
open(r"C:\Users\admin\AppData\Local\Temp\claude\092d43c82f33d74b\05f727ff-7363-4b94-8d61-29975f2a5488\scratchpad\smoke_dyn.json", "w").write(json.dumps([j], indent=1))
EOF
```

- [ ] **Step 2: Run it**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_queue.py "C:\Users\admin\AppData\Local\Temp\claude\092d43c82f33d74b\05f727ff-7363-4b94-8d61-29975f2a5488\scratchpad\smoke_dyn.json"
```

Expected: three chunks with `rc 0`. Then check:

```bash
O=runs/remote/smoke_dyn_DB/ws/out/smoke_dyn_DB
ls $O/omega_dyn; cat $O/omega_dyn/stage1/stats.json $O/omega_dyn/stage2/stats.json; cat $O/train_images.jsonl
grep -h "dyn omega stage\|Scanning" runs/remote/smoke_dyn_DB/chunk_*.log | head
```

Expected:
- `stage1` and `stage2` exist, each with `.done`.
- The stats have `n_images_scored` > 0. The counts are plausible: after 1 epoch most boxes are "kept", a few "raised" or "ignored".
- `train_images.jsonl` has 3 lines. `n_images` is larger than the subset size whenever a repeat set is non-empty.
- **Ultralytics duplicate check:** the train "Scanning ... N images" count equals `n_images` of that chunk, including the repeats.
  - If N is smaller (the duplicates were collapsed), stop. Report this, and do not run the queues. Repeat sampling then needs a dataset-level change, which is a spec change.

- [ ] **Step 3: Loader check on the stage-2 weights**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/check_wsoft_loader.py --dir runs/remote/smoke_dyn_DB/ws/out/smoke_dyn_DB/omega_dyn/stage2 --batches 20
```

Expected: `ok: <n> weighted boxes, <m> ignore boxes in 20 batches`, with n > 0 and m > 0.

- [ ] **Step 4: Remove the smoke run, then run the screening (about 2 h, unmeasured)**

```bash
rm -rf runs/remote/smoke_dyn_DB data/dronevehicle/proc/rgb_train_smoke_dyn_DB.txt data/dronevehicle/rgb_proxy_smoke_dyn_DB.yaml
for s in 0 1 2; do PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_queue.py configs/queues/dyn/t4_dyn_s$s.json || break; done
PYTHONIOENCODING=utf-8 .venv/Scripts/python src/screen_tables.py
```

Expected:
- 9 runs `dyn_cl_{CB,D,DB}_s{0,1,2}`, each with `preds_val.json`;
- `runs/analysis/screen/tables_dyn_cl.md` lists B, P, I, V, F, C, CB, D and DB, with differences to B, P and C.

- [ ] **Step 5: Record the results**

Append the CB/D/DB rows and a summary to `docs/results/2026-11-25-screening-clean.md`:
- the RQ5–RQ7 answers from the "differences to C" table;
- the `img_epochs_k` of each config.

Then commit and push:

```bash
git add docs/results/2026-11-25-screening-clean.md
git -c user.email=184421321+eociV@users.noreply.github.com commit -m "results: screening of CB, D, DB (dynamic omega, repeat sampling), 3 seeds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
