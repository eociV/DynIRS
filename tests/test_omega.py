import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import omega as om


def test_xywhn_to_xyxy():
    out = om.xywhn_to_xyxy(np.array([[0.5, 0.5, 0.25, 0.5]]))
    assert np.allclose(out, [[240, 128, 400, 384]])


def test_teacher_conf_takes_best_matching_score_class_agnostic():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], float)
    preds = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 30, 30], [300, 300, 310, 310]], float)
    scores = np.array([0.4, 0.9, 0.99, 0.8])
    c = om.teacher_conf(boxes, preds, scores)
    assert np.allclose(c, [0.9, 0.0])  # IoU([0,0,30,30]) = 0.11 < 0.5 is ignored


def test_teacher_conf_no_predictions():
    assert np.allclose(om.teacher_conf(np.array([[0, 0, 5, 5]], float), np.zeros((0, 4)), np.zeros(0)), [0.0])


def test_box_luminance():
    img = np.zeros((512, 640, 3), np.uint8)
    img[:, 320:] = 255
    v = om.box_luminance(img, np.array([[0, 0, 100, 100], [400, 0, 500, 100], [270, 0, 370, 10]], float))
    assert np.allclose(v, [0.0, 1.0, 0.5], atol=1e-3)


def test_box_luminance_degenerate_box_uses_one_pixel():
    img = np.full((512, 640, 3), 51, np.uint8)
    assert np.allclose(om.box_luminance(img, np.array([[10.2, 10.2, 10.4, 10.4]])), [0.2], atol=1e-3)


def test_raw_weights():
    c, v = np.array([0.8, 0.5]), np.array([0.25, 1.0])
    assert np.allclose(om.raw_weights(c, v, "T"), [0.8, 0.5])
    assert np.allclose(om.raw_weights(c, v, "S"), [0.8, 0.5])
    assert np.allclose(om.raw_weights(c, v, "TL"), [0.6, 0.0])
    with pytest.raises(ValueError):
        om.raw_weights(c, v, "X")


def test_mean_match_hits_target_within_bounds():
    rng = np.random.default_rng(0)
    raw = rng.beta(0.5, 2.0, 5000)
    raw[:500] = 0.0
    w, s = om.mean_match(raw)
    assert abs(w.mean() - 0.3) <= 1e-4
    assert w.min() >= 0.05 - 1e-12 and w.max() <= 1.0 + 1e-12
    assert s > 0


def test_mean_match_infeasible_raises():
    with pytest.raises(ValueError):
        om.mean_match(np.zeros(10))


def test_shuffle_is_a_seeded_permutation():
    w = np.arange(100) / 100
    a, b = om.shuffle(w, 0), om.shuffle(w, 0)
    assert np.array_equal(a, b) and not np.array_equal(a, w)
    assert np.array_equal(np.sort(a), w)


def _toy_proc(tmp_path):
    proc = tmp_path / "proc"
    d = proc / "ironly_cls" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("1 0.25 0.25 0.1 0.1\n3 0.75 0.75 0.1 0.1\n")
    (d / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (d / "c.txt").write_text("")
    for mod in ("rgb", "ir"):
        (proc / mod / "train" / "images").mkdir(parents=True)
    img = np.zeros((512, 640, 3), np.uint8)
    img[:, 320:] = 255  # box 2 of a.txt is bright, box 1 is dark
    for k in ("a", "b", "c"):
        cv2.imwrite(str(proc / "rgb" / "train" / "images" / f"{k}.jpg"), img[..., ::-1])
        cv2.imwrite(str(proc / "ir" / "train" / "images" / f"{k}.jpg"), img[..., 0])
    (proc / "stats").mkdir()
    return proc


def test_load_ironly(tmp_path):
    b = om.load_ironly(_toy_proc(tmp_path))
    assert sorted(b) == ["a", "b", "c"] and b["a"].shape == (2, 5) and b["c"].shape == (0, 5)


def test_compute_signals_with_fake_teacher(tmp_path):
    proc = _toy_proc(tmp_path)
    boxes = om.load_ironly(proc)

    def fake(paths):
        out = []
        for p in paths:
            k = Path(p).stem
            xy = om.xywhn_to_xyxy(boxes[k][:, 1:5])
            out.append((xy, np.full(len(xy), 0.8 if k == "a" else 0.6)))
        return out

    sig = om.compute_signals(proc, fake)
    assert sorted(sig) == ["a", "b", "c"]
    assert np.allclose([x[0] for x in sig["a"]], [0.8, 0.8])
    va = [x[1] for x in sig["a"]]
    assert va[0] < 0.05 and va[1] > 0.95
    assert sig["c"] == []


def test_build_and_write_weights(tmp_path):
    proc = _toy_proc(tmp_path)
    sig = {"a": [[0.9, 0.1], [0.2, 0.9]], "b": [[0.5, 0.5]], "c": []}
    for v in ("T", "TL", "S"):
        w, stats = om.build_weights(sig, v)
        flat = np.concatenate([w[k] for k in sorted(w)])
        assert abs(flat.mean() - 0.3) <= 1e-3 and flat.min() >= 0.05 and flat.max() <= 1.0
        assert stats["variant"] == v and stats["n_boxes"] == 3
        out = om.write_weights(proc, v, om.load_ironly(proc), w)
        lines = (out / "a.txt").read_text().splitlines()
        assert len(lines) == 2 and len(lines[0].split()) == 6 and lines[0].startswith("1 0.25")
        assert (out / "c.txt").read_text() == ""
    wt, _ = om.build_weights(sig, "T")
    wtl, _ = om.build_weights(sig, "TL")
    assert wt["a"][0] > wt["a"][1]          # T follows teacher confidence
    assert wtl["a"][0] > wtl["b"][0]        # TL favours the darker box


def test_cli_weights(tmp_path):
    proc = _toy_proc(tmp_path)
    (proc / "stats" / "omega_signals.json").write_text(json.dumps({"a": [[0.9, 0.1], [0.2, 0.9]], "b": [[0.5, 0.5]], "c": []}))
    om.main(["weights", "--proc", str(proc), "--variant", "TL"])
    st = json.loads((proc / "stats" / "omega_TL.json").read_text())
    assert abs(st["mean"] - 0.3) <= 1e-3
    assert (proc / "ironly_w" / "TL" / "train" / "b.txt").exists()
