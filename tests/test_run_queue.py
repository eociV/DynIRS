import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_queue as rq  # noqa: E402


def _spec(predict=True, chunks=(5, None), data_root=None):
    s = {"name": "j", "kind": "train", "chunks": list(chunks)}
    if predict:
        s["predict"] = {"weights": "out/j/weights/best.pt", "split": "test"}
        if data_root:
            s["predict"]["data_root"] = data_root
    return s


def _out(tmp):
    d = tmp / "j" / "ws" / "out" / "j"
    (d / "weights").mkdir(parents=True, exist_ok=True)
    return d


def _last(out, epoch):
    torch.save({"epoch": epoch}, out / "weights" / "last.pt")


def test_predict_output_name():
    assert rq.predict_output(_spec()) == "preds_test.json"
    assert rq.predict_output(_spec(data_root="rgb")) == "preds_test.json"
    assert rq.predict_output(_spec(data_root="ir")) == "preds_test_ir.json"
    assert rq.predict_output(_spec(predict=False)) is None


def test_fresh_runs_all_chunks(tmp_path):
    assert [(j["kind"], s) for j, s in rq.plan(_spec(), tmp_path)] == [("train", 5), ("train", None)]


def test_chunk_skipped_when_reached(tmp_path):
    _last(_out(tmp_path), 4)  # 5 epochs done
    assert [s for _, s in rq.plan(_spec(), tmp_path)] == [None]
    _last(_out(tmp_path), 2)
    assert [s for _, s in rq.plan(_spec(), tmp_path)] == [5, None]


def test_finished_without_prediction_runs_predict_only(tmp_path):
    _last(_out(tmp_path), -1)
    steps = rq.plan(_spec(), tmp_path)
    assert [(j["kind"], s) for j, s in steps] == [("predict", None)]
    assert steps[0][0]["predict"] == _spec()["predict"]


def test_done_when_prediction_exists(tmp_path):
    out = _out(tmp_path)
    _last(out, -1)
    (out / "preds_test.json").write_text("[]")
    assert rq.plan(_spec(), tmp_path) == []


def test_prediction_with_tag(tmp_path):
    out = _out(tmp_path)
    _last(out, -1)
    (out / "preds_test.json").write_text("[]")
    assert [j["kind"] for j, _ in rq.plan(_spec(data_root="ir"), tmp_path)] == ["predict"]


def test_no_predict_job_done_with_best_and_finished_last(tmp_path):
    out = _out(tmp_path)
    _last(out, -1)
    (out / "weights" / "best.pt").write_bytes(b"x")
    assert rq.plan(_spec(predict=False), tmp_path) == []


def test_non_train_job_runs_once(tmp_path):
    s = {"name": "j", "kind": "latency"}
    assert [(j["kind"], st) for j, st in rq.plan(s, tmp_path)] == [("latency", None)]


def test_truncated_prediction_is_not_done(tmp_path):
    out = _out(tmp_path)
    _last(out, -1)
    (out / "preds_test.json").write_text('[{"image_id": "a", "sc')
    assert [j["kind"] for j, _ in rq.plan(_spec(), tmp_path)] == ["predict"]
    (out / "preds_test.json").write_text("[]")
    assert rq.plan(_spec(), tmp_path) == []


def test_corrupt_last_pt_reruns_and_is_moved_aside(tmp_path, capsys):
    out = _out(tmp_path)
    (out / "weights" / "last.pt").write_bytes(b"garbage")
    steps = rq.plan(_spec(), tmp_path)
    assert [s for _, s in steps] == [5, None]
    assert not (out / "weights" / "last.pt").exists()
    assert (out / "weights" / "last.pt.corrupt").exists()
    assert "last.pt" in capsys.readouterr().out


def test_note_current_writes_lane_file(tmp_path, monkeypatch):
    f = tmp_path / ".lane_0_current"
    monkeypatch.setenv("DYN_LANE_FILE", str(f))
    rq.note_current("job_a", 20)
    assert f.read_text() == "job_a chunk 20"
    monkeypatch.delenv("DYN_LANE_FILE")
    rq.note_current("job_b", None)  # no env var: no-op
    assert f.read_text() == "job_a chunk 20"


def test_import_error_while_loading_checkpoint_keeps_the_file(tmp_path, monkeypatch):
    import pytest

    out = tmp_path / "out"
    (out / "weights").mkdir(parents=True)
    torch.save({"epoch": 3}, out / "weights" / "last.pt")

    def boom(*a, **k):
        raise ModuleNotFoundError("ultralytics")

    monkeypatch.setattr(torch, "load", boom)
    with pytest.raises(ModuleNotFoundError):
        rq.epochs_done(out)
    assert (out / "weights" / "last.pt").exists()


def test_predict_outputs_list(tmp_path):
    spec = _spec()
    spec["predict"] = [dict(spec["predict"], split="train"), dict(spec["predict"], split="val")]
    assert rq.predict_outputs(spec) == ["preds_train.json", "preds_val.json"]
    out = tmp_path / spec["name"] / "ws" / "out" / spec["name"]
    out.mkdir(parents=True)
    (out / "preds_train.json").write_text("[]")
    assert rq.plan(spec, runs=tmp_path)  # preds_val.json missing: still to run
    (out / "preds_val.json").write_text("[]")
    assert rq.plan(spec, runs=tmp_path) == []
