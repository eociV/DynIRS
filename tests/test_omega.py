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
