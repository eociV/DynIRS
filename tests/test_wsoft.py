import numpy as np
import pytest
import torch

import lowlight as ll


def legacy_split(cls, ir_soft, pseudo):
    """The original preprocess_batch logic, kept as the reference."""
    soft = cls >= ll.SOFT_OFFSET
    ps = cls >= ll.PSEUDO_OFFSET
    w = torch.where(ps, float(pseudo or 1.0), torch.where(soft, float(ir_soft or 1.0), 1.0)).to(torch.float32)
    off = torch.where(ps, ll.PSEUDO_OFFSET, ll.SOFT_OFFSET).to(cls.dtype)
    return torch.where(soft, cls - off, cls), w


def test_encode_decode_roundtrip():
    cls = np.array([0, 1, 2, 3, 4, 0, 4])
    w = np.array([0.0, 0.05, 0.3, 0.4567, 0.999, 1.0, 0.3004])
    enc = torch.from_numpy(ll.encode_wsoft(cls, w))
    assert enc.dtype == torch.float32
    assert bool((enc >= ll.WSOFT_OFFSET).all())
    c, q = ll.decode_wsoft(enc)
    assert c.tolist() == cls.tolist()
    assert np.allclose(q.numpy(), np.round(w * 1000) / 1000, atol=1e-6)


def test_split_matches_legacy_without_wsoft():
    cls = torch.tensor([0., 3., 1 + 10., 4 + 10., 2 + 20.])
    for ir_soft, pseudo in [(0.3, None), (None, 0.5), (0.3, 0.5), (None, None)]:
        a = ll.split_box_weights(cls, ir_soft, pseudo)
        b = legacy_split(cls, ir_soft, pseudo)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_constant_wsoft_equals_constant_soft():
    plain = torch.tensor([0., 1 + 10., 3 + 10.])
    enc = torch.tensor([0., *ll.encode_wsoft([1, 3], [0.3, 0.3]).tolist()])
    a = ll.split_box_weights(plain, 0.3, None)
    b = ll.split_box_weights(enc, None, None)
    assert torch.equal(a[0], b[0])
    assert torch.allclose(a[1], b[1], atol=1e-7)


def test_update_labels_info_reads_weight_files(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "a.txt").write_text("2 0.2 0.2 0.05 0.05 0.731\n0 0.6 0.6 0.1 0.1 0.05\n")
    ds = ll.IRPairDataset.__new__(ll.IRPairDataset)
    ds.use_obb = False
    ds.cfg = {"ir_soft_w_dir": str(d)}
    label = {"im_file": str(tmp_path / "rgb" / "train" / "images" / "a.jpg"),
             "cls": np.zeros((1, 1), np.float32), "bboxes": np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
             "segments": [], "keypoints": None, "bbox_format": "xywh", "normalized": True}
    out = ds.update_labels_info(label)
    c, w = ll.split_box_weights(torch.from_numpy(out["cls"].ravel()), None, None)
    assert c.tolist() == [0, 2, 0]
    assert np.allclose(w.numpy(), [1.0, 0.731, 0.05], atol=1e-6)
    assert len(out["instances"]) == 3


@pytest.mark.parametrize("cfg", ["yolov8n.yaml", "yolov10n.yaml", "yolo11n.yaml", "yolo12n.yaml", "yolo26n.yaml"])
def test_loss_uses_per_box_weights_on_every_architecture(cfg):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    m = DetectionModel(cfg, nc=5, verbose=False)
    m.args = get_cfg()
    m.train()
    img = torch.rand(2, 3, 128, 128)
    base = {"img": img, "batch_idx": torch.tensor([0., 0., 1.]), "cls": torch.tensor([[0.], [1.], [2.]]),
            "bboxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.3, 0.3], [0.5, 0.5, 0.4, 0.4]])}

    def loss(box_w):
        m.criterion = None
        return m.loss(dict(base, box_w=torch.tensor(box_w)))[0].sum().item()

    ones = loss([1., 1., 1.])
    const = loss([1., 0.3, 0.3])
    varied = loss([1., 0.9, 0.1])
    assert np.isfinite([ones, const, varied]).all()
    assert abs(ones - const) > 1e-6 and abs(const - varied) > 1e-6
