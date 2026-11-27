from pathlib import Path

import pytest

import job_main as jm


def test_resolve_auto_paths(tmp_path):
    w = tmp_path / "proc" / "ironly_w" / "TL" / "train"
    w.mkdir(parents=True)
    (w / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3\n")
    cfg = jm.resolve_config({"ir_only_dir": "auto", "ir_soft_w_dir": "auto:TL"}, tmp_path, tmp_path)
    assert Path(cfg["ir_only_dir"]) == tmp_path / "proc" / "ironly" / "train"
    assert Path(cfg["ir_soft_w_dir"]) == w


def test_resolve_missing_weight_dir_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        jm.resolve_config({"ir_soft_w_dir": "auto:T"}, tmp_path, tmp_path)


def test_mix_data_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(jm, "DATA", tmp_path)
    y = Path(jm._data_yaml("mix", {"name": "j"})).read_text()
    assert "train: [rgb/train/images, ir/train/images]" in y
    assert "val: rgb/val/images" in y and "test: rgb/test/images" in y


def test_resolve_rejects_both_soft_modes(tmp_path):
    w = tmp_path / "proc" / "ironly_w" / "T" / "train"
    w.mkdir(parents=True)
    (w / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3\n")
    with pytest.raises(ValueError):
        jm.resolve_config({"ir_soft": 0.3, "ir_soft_dir": "auto", "ir_soft_w_dir": "auto:T"}, tmp_path, tmp_path)


def test_hash_omega_dir(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3\n")
    (tmp_path / "b.txt").write_text("1 0.5 0.5 0.1 0.1 0.2\n")
    (tmp_path / "ignored.json").write_text("{}")
    h = jm.hash_omega_dir(tmp_path)
    assert h == jm.hash_omega_dir(tmp_path) and len(h) == 64
    (tmp_path / "b.txt").write_text("1 0.5 0.5 0.1 0.1 0.25\n")
    assert jm.hash_omega_dir(tmp_path) != h


def test_write_atomic_replaces_without_leftovers(tmp_path):
    dst = tmp_path / "preds.json"
    dst.write_text("old")
    jm.write_atomic(dst, "[1, 2]")
    assert dst.read_text() == "[1, 2]" and list(tmp_path.iterdir()) == [dst]


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
