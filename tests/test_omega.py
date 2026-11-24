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


def test_mean_match_raises_when_tolerance_not_met():
    with pytest.raises(ValueError):
        om.mean_match(np.linspace(0.1, 1, 50), tol=-1.0)


def test_build_weights_rejects_bad_mean(monkeypatch):
    monkeypatch.setattr(om, "mean_match", lambda raw: (np.full(len(raw), 0.5), 1.0))
    with pytest.raises(ValueError):
        om.build_weights({"a": [[0.9, 0.1], [0.2, 0.9]]}, "T")


def test_build_weights_spread_stats():
    sig = {"a": [[0.95, 0.1], [0.2, 0.9], [0.5, 0.5], [0.99, 0.3]]}
    _, st = om.build_weights(sig, "T")
    for k in ("std", "iqr", "frac_near_target", "c_std", "c_frac_ge_0_9"):
        assert k in st
    assert st["c_frac_ge_0_9"] == 0.5 and st["std"] > 0 and st["iqr"] >= 0
    assert 0 <= st["frac_near_target"] <= 1


def test_rank_weights_uniform_mean_and_ties():
    w = om.rank_weights([3.0, 1.0, 2.0, 5.0, 4.0])
    assert np.allclose(w, [0.30, 0.05, 0.175, 0.55, 0.425])
    assert abs(w.mean() - om.TARGET) < 1e-12
    t = om.rank_weights([1.0, 1.0, 2.0])  # tied pair shares the mean of ranks 0 and 1
    assert np.allclose(t, [0.175, 0.175, 0.55])
    assert np.allclose(om.rank_weights([7.0]), [0.3])


def test_rank_variants_share_one_distribution():
    rng = np.random.default_rng(1)
    n = 500
    sig = {f"im{i}": rng.random((5, 2)).tolist() for i in range(n // 5)}
    areas = {k: rng.integers(100, 5000, 5).astype(float).tolist() for k in sig}
    out = {}
    for v in ("RT", "RD", "RB", "RA", "RS"):
        w, st = om.build_weights(sig, v, areas=areas)
        flat = np.concatenate([w[k] for k in sorted(w)])
        assert abs(flat.mean() - om.TARGET) < 1e-3 and flat.min() >= 0.05 - 1e-9 and flat.max() <= 0.55 + 1e-9
        assert st["scale"] is None
        out[v] = flat
    assert np.allclose(np.sort(out["RT"]), np.sort(out["RS"]), atol=1e-4)
    c = np.array([x[0] for k in sorted(sig) for x in sig[k]])
    v = np.array([x[1] for k in sorted(sig) for x in sig[k]])
    assert np.all(np.diff(out["RT"][np.argsort(c)]) >= -1e-9)  # higher c, higher weight
    assert np.all(np.diff(out["RB"][np.argsort(v)]) >= -1e-9)
    assert np.allclose(out["RD"] + out["RB"], 0.6, atol=2e-4)  # darkness is the reversed brightness order


def test_rank_area_needs_areas():
    with pytest.raises(ValueError):
        om.build_weights({"a": [[0.5, 0.5], [0.2, 0.1]]}, "RA")


def test_box_areas():
    b = {"x": np.array([[0, 0.5, 0.5, 0.1, 0.2]])}
    assert np.allclose(om.box_areas(b)["x"], [0.1 * 640 * 0.2 * 512])


def test_image_shifts_median_and_min_pairs():
    pairs = {"a": {"offsets": [[4, -2], [5, -2], [30, 0]]}, "b": {"offsets": [[9, 9], [9, 9]]}}
    assert om.image_shifts(pairs) == {"a": (5.0, -2.0)}


def test_shift_boxes_moves_into_rgb_frame_and_clips():
    rows = np.array([[1, 0.5, 0.5, 0.1, 0.1], [0, 0.01, 0.5, 0.02, 0.1]])
    out = om.shift_boxes(rows, (6.4, -5.12))  # IR is 6.4 px right, 5.12 px up of RGB
    assert np.allclose(out[0], [1, 0.49, 0.51, 0.1, 0.1])
    x1 = (out[1, 1] - out[1, 3] / 2) * om.W
    assert x1 == pytest.approx(0) and out[1, 3] < 0.02
    assert np.array_equal(om.shift_boxes(rows, (0.0, 0.0)), rows)


def test_reg_variants_write_ignore_rows(tmp_path):
    boxes = {"a": np.array([[0, 0.5, 0.5, 0.1, 0.1]]), "b": np.zeros((0, 5))}
    pairs = {"a": {"offsets": [[6.4, 0]] * 3}, "b": {"offsets": []}}
    nb, w, st = om.build_reg_weights(boxes, pairs, "SI")
    assert st["n_moved_gt_2px"] == 1 and np.allclose(nb["a"][0, 1], 0.49)
    out = om.write_weights(tmp_path, "SI", nb, w)
    assert (out / "a.txt").read_text().split()[-1] == "-1.0000"
    nb, w, st = om.build_reg_weights(boxes, pairs, "IG")
    assert np.allclose(nb["a"][0, 1], 0.5) and st["n_in_shift_images"] == 0
