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


def test_variant_f_drops_boxes_that_land_on_an_rgb_label():
    # IR-only box 6.4 px right of an RGB-labelled vehicle: after registration it is the same vehicle
    boxes = {"a": np.array([_box(0.5, 0.5), _box(0.2, 0.2)])}
    pairs = {"a": {"offsets": [[6.4, 0]] * 3}}
    rgb = {"a": om.xywhn_to_xyxy(np.array([[0.49, 0.5, 0.1, 0.1]]))}
    preds = {"a": np.array([_pred_on(_box(0.49, 0.5), 0.9), _pred_on(_box(0.19, 0.2), 0.9)])}
    rows, w, st = cl.build_variant("F", boxes, pairs, preds, rgb)
    assert w["a"].tolist() == pytest.approx([0.0, 0.9])  # 0: dropped (not written), the RGB label covers it
    assert st["n_dropped_rgb_dup"] == 1 and st["n_taught"] == 1
    rows, w, st = cl.build_variant("V", boxes, pairs, preds, rgb)  # unmoved, still on the RGB label (IoU 0.82)
    assert st["n_dropped_rgb_dup"] == 1
    rows, w, st = cl.build_variant("P", boxes, pairs, preds, rgb)  # P keeps the original recipe
    assert w["a"].tolist() == [0.3, 0.3] and st["n_dropped_rgb_dup"] == 0
