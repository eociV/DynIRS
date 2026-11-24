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


def test_update_labels_info_negative_weight_is_ignore(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "a.txt").write_text("2 0.2 0.2 0.05 0.05 -1.0000\n0 0.6 0.6 0.1 0.1 0.3\n")
    ds = ll.IRPairDataset.__new__(ll.IRPairDataset)
    ds.use_obb = False
    ds.cfg = {"ir_soft_w_dir": str(d)}
    label = {"im_file": str(tmp_path / "rgb" / "train" / "images" / "a.jpg"),
             "cls": np.zeros((1, 1), np.float32), "bboxes": np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
             "segments": [], "keypoints": None, "bbox_format": "xywh", "normalized": True}
    out = ds.update_labels_info(label)
    raw = out["cls"].ravel()
    assert raw[1] == ll.IGNORE_CLS
    c, w = ll.split_box_weights(torch.from_numpy(raw), None, None)
    assert c.tolist() == [0, ll.IGNORE_CLS, 0] and np.allclose(w.numpy()[[0, 2]], [1.0, 0.3], atol=1e-6)


def test_ignore_anchor_mask():
    pts = torch.tensor([[10., 10.], [50., 50.], [90., 90.]])
    ign = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]])  # normalized xywh on a 100x100 image
    m = ll.ignore_anchor_mask(ign, torch.tensor([1., 0.]), pts, 2, torch.tensor([100., 100.]))
    assert m.tolist() == [[True, False, False], [False, True, False]]
    assert not ll.ignore_anchor_mask(ign[:0], torch.zeros(0), pts, 2, torch.tensor([100., 100.])).any()


def test_ignore_regions_remove_background_loss(monkeypatch):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    m = DetectionModel("yolo26n.yaml", nc=5, verbose=False)
    m.args = get_cfg()
    m.train()
    base = {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.], [2.]]),
            "bboxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]]), "box_w": torch.ones(2)}

    def loss(ign, on):
        monkeypatch.setitem(ll.CONFIG, "ir_ignore", on)
        m.criterion = None
        b = dict(base, ign_bboxes=ign, ign_idx=torch.zeros(len(ign)))
        return m.loss(b)[0].sum().item()

    none = loss(torch.zeros(0, 4), True)
    region = torch.tensor([[0.75, 0.75, 0.4, 0.4]])  # background area of image 0
    assert loss(region, False) == pytest.approx(none, rel=1e-6)  # flag off: unchanged
    with_ign = loss(region, True)
    assert with_ign < none  # background BCE inside the region removed
    assert loss(torch.tensor([[0.75, 0.75, 0.4, 0.4]]) * 0 + torch.tensor([[2., 2., 0.01, 0.01]]), True) == \
        pytest.approx(none, rel=1e-6)  # a region outside the image changes nothing
