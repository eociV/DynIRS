import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_auto as ka  # noqa: E402


def test_kernel_slug():
    assert ka.kernel_slug("configs/queues/dyn/t4_v8n_s0_c.json", 2) == "dynirs-t4-v8n-s0-c-r2"


def test_queue_rel_inside_and_outside(tmp_path):
    q = ROOT / "configs" / "queues" / "dyn" / "teacher.json"
    assert ka.queue_rel(q) == "configs/queues/dyn/teacher.json"
    with pytest.raises(SystemExit):
        ka.queue_rel(tmp_path / "q.json")


def test_queue_needs():
    c = {"config": {"ir_soft": 0.3}, "train": {"data": "rgb"}}
    t = {"config": {"ir_soft_w_dir": "auto:T"}, "train": {"data": "rgb"}}
    m = {"config": {}, "train": {"data": "mix"}}
    assert ka.queue_needs([c]) == (False, ["rgb"])
    assert ka.queue_needs([c, t]) == (True, ["rgb"])
    assert ka.queue_needs([m]) == (False, ["rgb", "ir"])
    real = json.loads((ROOT / "configs/queues/dyn/t4_v8n_s0_ttl.json").read_text())
    assert ka.queue_needs(real)[0] is True


def test_kernel_metadata():
    m = ka.kernel_metadata("me", "team", "dynirs-x-r2", omega=True, prev="dynirs-x-r1")
    assert m["id"] == "me/dynirs-x-r2" and m["title"] == "dynirs-x-r2"
    assert m["dataset_sources"] == ["team/dynirs-code", "team/dynirs-dv", "team/dynirs-omega"]
    assert m["kernel_sources"] == ["me/dynirs-x-r1"]
    assert m["enable_gpu"] and m["enable_internet"] and m["machine_shape"] == "NvidiaTeslaT4"
    assert ka.kernel_metadata("me", "me", "s", omega=False)["kernel_sources"] == []


def test_render_runner_is_valid_python():
    src = ka.render_runner("dynirs-x-r1", "configs/queues/dyn/smoke.json", ["a"], 11.0, ["rgb", "ir"])
    ast.parse(src)
    assert "QUEUE, ONLY, BUDGET_H, MODS = 'configs/queues/dyn/smoke.json', ['a'], 11.0, ['rgb', 'ir']" in src
    assert "'dynirs-x-r1'" in src


def test_runner_finds_inputs(tmp_path, monkeypatch):
    """Execute the input discovery part of the runner against a fake /kaggle/input."""
    inp = tmp_path / "input"
    (inp / "dynirs-code" / "scripts").mkdir(parents=True)
    (inp / "dynirs-code" / "scripts" / "kaggle_session.py").write_text("")
    (inp / "dynirs-dv" / "proc").mkdir(parents=True)
    (inp / "dynirs-dv" / "proc" / ".ready_train").write_text("")
    prev = inp / "dynirs-x-r1" / "DynIRS"
    (prev / "scripts").mkdir(parents=True)
    (prev / "scripts" / "kaggle_session.py").write_text("")
    (prev / "runs" / "remote").mkdir(parents=True)
    src = ka.render_runner("s", "q.json", [], 11.0, ["rgb"])
    src = src.replace('Path("/kaggle/input")', f"Path({str(inp)!r})").split("if code is None")[0]
    ns = {}
    exec(compile(src, "run.py", "exec"), ns)
    assert ns["code"] == inp / "dynirs-code"
    assert ns["data"] == inp / "dynirs-dv"
    assert ns["prev"] == inp / "dynirs-x-r1"
    assert ns["omega"] is None


@pytest.mark.parametrize("text,want", [
    ('me/k has status "KernelWorkerStatus.COMPLETE"', "complete"),
    ('me/k has status "complete"', "complete"),
    ('me/k has status "KernelWorkerStatus.RUNNING"', "running"),
    ('me/k has status "error"\nFailure message: "x"', "error"),
    ('me/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "cancelled"),
    ('me/k has status "queued"', "queued"),
    ("404 not found", "unknown"),
])
def test_parse_status(text, want):
    assert ka.parse_status(text) == want


def test_pending(tmp_path):
    specs = [{"name": "a", "kind": "train", "predict": {"split": "test"}},
             {"name": "b", "kind": "train", "predict": {"split": "test"}}]
    out = tmp_path / "a" / "ws" / "out" / "a"
    out.mkdir(parents=True)
    (out / "preds_test.json").write_text("[]")
    assert ka.pending(specs, tmp_path) == ["b"]
    assert ka.pending(specs, tmp_path, only=["a"]) == []
