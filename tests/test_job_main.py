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
