import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_session as ks  # noqa: E402


def test_find_prev_runs_none_or_many(tmp_path):
    with pytest.raises(SystemExit) as e:
        ks.find_prev_runs(tmp_path)
    assert "runs/remote" in str(e.value)
    (tmp_path / "a" / "runs" / "remote").mkdir(parents=True)
    (tmp_path / "b" / "x" / "runs" / "remote").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        ks.find_prev_runs(tmp_path)
    assert str(tmp_path / "a") in str(e.value) and str(tmp_path / "b") in str(e.value)


def test_find_prev_runs_single(tmp_path):
    (tmp_path / "a" / "runs" / "remote").mkdir(parents=True)
    assert ks.find_prev_runs(tmp_path) == tmp_path / "a" / "runs" / "remote"


def test_describe_runs(tmp_path):
    for name, ep in (("fin", -1), ("mid", 9)):
        w = tmp_path / name / "ws" / "out" / name / "weights"
        w.mkdir(parents=True)
        torch.save({"epoch": ep}, w / "last.pt")
    (tmp_path / "none").mkdir()
    d = dict(ks.describe_runs(tmp_path))
    assert d == {"fin": "finished", "mid": "epoch 10 done", "none": "no last.pt"}
