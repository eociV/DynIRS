import sys
from pathlib import Path

import cv2
import numpy as np

import lowlight as ll

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_lanes  # noqa: E402
import warm_cache  # noqa: E402


def test_load_ir_false_uses_ultralytics_loader(monkeypatch):
    called = {}

    def fake(self, i, rect_mode=True, resize_short=False):
        called["i"] = i
        return "img", (1, 1), (1, 1)

    monkeypatch.setattr(ll.YOLODataset, "load_image", fake)
    ds = ll.IRPairDataset.__new__(ll.IRPairDataset)
    ds.cfg = {"load_ir": False}
    assert ds.load_image(3)[0] == "img" and called["i"] == 3


def test_warm_cache_matches_cv2_and_is_idempotent(tmp_path):
    img = (np.arange(64 * 80 * 3) % 251).astype(np.uint8).reshape(64, 80, 3)
    f = tmp_path / "a.jpg"
    cv2.imwrite(str(f), img)
    assert warm_cache.cache_one(str(f)) == 1
    assert np.array_equal(np.load(tmp_path / "a.npy"), cv2.imread(str(f)))
    assert warm_cache.cache_one(str(f)) == 0
    assert not list(tmp_path.glob("*.tmp*"))


def test_split_lanes_round_robin():
    assert run_lanes.split_lanes(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
    assert run_lanes.split_lanes(["a"], 2) == [["a"], []]
