import json

import numpy as np
import pytest

import boxtypes as bt
import omega as om


def test_box_contrast_flat_and_textured():
    g = np.full((100, 100), 50.0)
    assert bt.box_contrast(g, [[20, 20, 40, 40]])[0] == pytest.approx(0.0)
    g[20:40, 20:40] = 90.0  # uniform patch 40 levels brighter than its ring
    assert bt.box_contrast(g, [[20, 20, 40, 40]])[0] == pytest.approx(40.0)
    g[20:40, 20:30] = 70.0  # half 70, half 90 inside: std 10, mean 80 -> 10 + 30
    assert bt.box_contrast(g, [[20, 20, 40, 40]])[0] == pytest.approx(40.0)


def test_box_contrast_box_covering_image_has_no_ring():
    g = np.full((10, 10), 5.0)
    assert bt.box_contrast(g, [[0, 0, 10, 10]])[0] == pytest.approx(0.0)


def test_classify_thresholds():
    assert bt.classify([0, 11.9, 12, 24.9, 25, 100], [0] * 6) == [
        "invisible", "invisible", "faint", "faint", "visible", "visible"]
    assert bt.classify([100, 0], [0.1, 0.099]) == ["dup", "invisible"]


def test_rgb_only_leaves_matched_boxes_out():
    rgb = np.array([[0, 0, 0, 10, 10], [0, 50, 50, 60, 60]], float)
    ir = np.array([[0, 0, 0, 10, 10]], float)
    assert np.allclose(bt.rgb_only(rgb, ir), [[50, 50, 60, 60]])


def test_compute_types(tmp_path):
    import cv2

    p = tmp_path / "proc"
    for d in ("ironly_cls/val", "rgb/val/labels", "ir/val/labels", "rgb/val/images"):
        (p / d).mkdir(parents=True)
    img = np.full((512, 640), 30, np.uint8)
    img[100:140, 100:180] = 200  # a bright car-sized patch -> visible
    cv2.imwrite(str(p / "rgb/val/images/a.jpg"), img)
    # IR-only boxes: one on the patch, one on flat ground, one shifted from an unmatched RGB box
    (p / "ironly_cls/val/a.txt").write_text(
        "0 0.21875 0.234375 0.125 0.078125\n0 0.7 0.7 0.05 0.05\n0 0.535 0.5 0.05 0.05\n")
    (p / "rgb/val/labels/a.txt").write_text("0 0.5 0.5 0.05 0.05\n")
    (p / "ir/val/labels/a.txt").write_text("0 0.21875 0.234375 0.125 0.078125\n0 0.7 0.7 0.05 0.05\n0 0.535 0.5 0.05 0.05\n")
    (p / "ironly_cls/val/b.txt").write_text("")
    ty = bt.compute_types(p, "val")
    assert [t for _, _, t in ty["a"]] == ["visible", "invisible", "dup"] and ty["b"] == []
    assert bt.counts(ty) == {"dup": 1, "invisible": 1, "faint": 0, "visible": 1}


def test_type_variants_and_dropped_boxes_not_written(tmp_path):
    types = {"a": [[30, 0, "visible"], [1, 0, "invisible"], [5, 0.2, "dup"]], "b": [[15, 0, "faint"]]}
    exp = {"ND": [0.3, 0.3, 0.0, 0.3], "FX": [1.0, 0.3, 0.0, 0.3], "NI": [1.0, 0.0, 0.0, 0.3]}
    for v, e in exp.items():
        w, st = om.build_type_weights(types, v)
        assert np.allclose(np.concatenate([w["a"], w["b"]]), e)
        assert st["n_dropped"] == e.count(0.0) and st["n_by_type"]["dup"] == 1
    w, _ = om.build_type_weights(types, "NI")
    boxes = {"a": np.array([[0, .1, .1, .1, .1], [1, .5, .5, .1, .1], [2, .8, .8, .1, .1]]),
             "b": np.array([[3, .3, .3, .1, .1]])}
    d = om.write_weights(tmp_path, "NI", boxes, w)
    assert (d / "a.txt").read_text().splitlines() == ["0 0.100000 0.100000 0.100000 0.100000 1.0000"]
    assert (d / "b.txt").read_text().splitlines() == ["3 0.300000 0.300000 0.100000 0.100000 0.3000"]


def test_cli_type_weights(tmp_path):
    p = tmp_path
    (p / "stats").mkdir()
    (p / "ironly_cls" / "train").mkdir(parents=True)
    (p / "ironly_cls" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    (p / "stats" / "boxtypes_train.json").write_text(json.dumps({"a": [[30, 0, "visible"], [1, 0, "dup"]]}))
    om.main(["weights", "--proc", str(p), "--variant", "FX"])
    assert (p / "ironly_w" / "FX" / "train" / "a.txt").read_text() == "0 0.500000 0.500000 0.100000 0.100000 1.0000\n"
    assert json.loads((p / "stats" / "omega_FX.json").read_text())["n_dropped"] == 1
