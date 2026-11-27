import json

import numpy as np
import pytest

import dyn_omega as do


def _row(x, y, w):
    return [0, x, y, 0.1, 0.1, w]


def test_rescore_rules():
    rows = {"a": np.array([_row(0.2, 0.2, 0.3),    # s 0.8 -> raised to 0.8
                           _row(0.4, 0.4, 0.6),    # s 0.55 < omega 0.6 -> stays 0.6
                           _row(0.6, 0.6, 0.3),    # s 0.2 (middle) -> kept
                           _row(0.8, 0.8, 0.3),    # s 0.01, c 0.01 -> ignore
                           _row(0.8, 0.2, 0.3),    # s 0.01, c 0.2 -> kept
                           _row(0.2, 0.8, -1.0)])}  # frozen ignore region, even with s 0.9
    s = {"a": np.array([0.8, 0.55, 0.2, 0.01, 0.01, 0.9])}
    c = {"a": np.array([0.0, 0.0, 0.0, 0.01, 0.2, 0.0])}
    new, st = do.rescore(rows, s, c)
    assert new["a"][:, 5].tolist() == pytest.approx([0.8, 0.6, 0.3, -1.0, 0.3, -1.0])
    assert rows["a"][0, 5] == 0.3  # input not modified
    assert (st["raised"], st["kept"], st["ignored"], st["frozen"]) == (1, 3, 1, 1)
    assert st["omega_taught_mean"] == pytest.approx(np.mean([0.8, 0.6, 0.3, 0.3]))


def test_rescore_leaves_unscored_images_unchanged():
    rows = {"a": np.array([_row(0.5, 0.5, 0.3)]), "b": np.array([_row(0.5, 0.5, 0.3)])}
    new, _ = do.rescore(rows, {"a": np.array([0.0])}, {"a": np.array([0.0])})
    assert new["a"][0, 5] == -1.0 and new["b"][0, 5] == 0.3


def test_box_scores_iou_match():
    rows = {"a": np.array([_row(0.5, 0.5, 0.3), _row(0.1, 0.1, 0.3)])}
    # box 0 in pixels (640x512): x 288-352, y 230.4-281.6
    preds = {"a": np.array([[288, 230.4, 352, 281.6, 0.7], [290, 232, 350, 280, 0.9], [0, 0, 5, 5, 0.99]])}
    s = do.box_scores(rows, preds)
    assert s["a"].tolist() == pytest.approx([0.9, 0.0])
    assert do.box_scores(rows, {})["a"].tolist() == [0.0, 0.0]


def test_repeat_ids():
    rows = {"a": np.array([_row(0.5, 0.5, 0.3)]), "b": np.array([_row(0.5, 0.5, 0.3), _row(0.2, 0.2, 0.5)]),
            "c": np.zeros((0, 6)), "d": np.array([_row(0.5, 0.5, -1.0)])}
    assert do.repeat_ids(rows) == ["b"]


def test_read_write_roundtrip(tmp_path):
    rows = {"a": np.array([_row(0.5, 0.25, 0.3), _row(0.1, 0.2, -1.0)]), "b": np.zeros((0, 6))}
    do.write_dir(tmp_path / "w", rows)
    back = do.read_dir(tmp_path / "w")
    assert sorted(back) == ["a", "b"] and back["b"].shape == (0, 6)
    assert back["a"] == pytest.approx(rows["a"])


def test_load_conf_checks_lengths(tmp_path):
    rows = {"a": np.array([_row(0.5, 0.5, 0.3)]), "b": np.array([_row(0.5, 0.5, 0.3)])}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": [0.4]}))
    c = do.load_conf(p, rows)
    assert c["a"].tolist() == [0.4] and c["b"].tolist() == [0.0]
    p.write_text(json.dumps({"a": [0.4, 0.1]}))
    with pytest.raises(ValueError):
        do.load_conf(p, rows)
