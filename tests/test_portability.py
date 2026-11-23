from pathlib import Path

import numpy as np

import lowlight as ll


def test_ir_path_posix_and_windows():
    assert Path(ll.ir_path("/d/proc/rgb/train/images/1.jpg")) == Path("/d/proc/ir/train/images/1.jpg")
    assert Path(ll.ir_path(r"D:\x\proc\rgb\train\images\1.jpg")) == Path(r"D:\x\proc\ir\train\images\1.jpg")


def test_ir_path_replaces_last_rgb_component_only():
    assert Path(ll.ir_path("/rgb/proc/rgb/val/images/7.jpg")) == Path("/rgb/proc/ir/val/images/7.jpg")


def _label(tmp_path):
    return {"im_file": str(tmp_path / "rgb" / "train" / "images" / "a.jpg"),
            "cls": np.zeros((1, 1), np.float32), "bboxes": np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
            "segments": [], "keypoints": None, "bbox_format": "xywh", "normalized": True}


def test_dataset_uses_config_snapshot_not_module_global(tmp_path, monkeypatch):
    soft = tmp_path / "soft"
    soft.mkdir()
    (soft / "a.txt").write_text("2 0.2 0.2 0.05 0.05\n")
    ds = ll.IRPairDataset.__new__(ll.IRPairDataset)
    ds.use_obb = False
    ds.cfg = {"ir_soft": 0.3, "ir_soft_dir": str(soft)}
    monkeypatch.setitem(ll.CONFIG, "ir_soft", None)  # what a spawned worker sees
    out = ds.update_labels_info(_label(tmp_path))
    assert out["cls"].ravel().tolist() == [0, 2 + ll.SOFT_OFFSET]
