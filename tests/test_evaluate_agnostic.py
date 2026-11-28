import json

import numpy as np

import evaluate as ev


def test_agnostic_ignores_class(tmp_path):
    gt = {"width": 640, "height": 512, "classes": ["car", "truck"],
          "images": {"a": [[0, 10, 10, 60, 40]], "b": [[1, 100, 100, 180, 150]]}}
    p = tmp_path / "boxes.json"
    p.write_text(json.dumps(gt))
    preds = [{"image_id": "a", "category_id": 1, "bbox": [10, 10, 60, 40], "score": 0.9},  # wrong class
             {"image_id": "b", "category_id": 1, "bbox": [100, 100, 180, 150], "score": 0.8}]
    m5 = ev.Matched(p, preds)
    m1 = ev.Matched(p, preds, agnostic=True)
    assert m1.classes == ["vehicle"]
    assert m1.ap(np.ones(len(m1.ids))) == 1.0
    assert m5.ap(np.ones(len(m5.ids))) < 1.0


def test_ignore_regions(tmp_path):
    gt = {"width": 640, "height": 512, "classes": ["car"],
          "images": {"a": [[0, 10, 10, 60, 40]], "b": []}}
    p = tmp_path / "boxes.json"
    p.write_text(json.dumps(gt))
    preds = [{"image_id": "a", "category_id": 0, "bbox": [10, 10, 60, 40], "score": 0.5},
             # high-scoring detection on an unlabelled (IR-only) vehicle: an FP without the ignore region
             {"image_id": "b", "category_id": 0, "bbox": [200, 200, 240, 230], "score": 0.9}]
    plain = ev.Matched(p, preds, agnostic=True)
    ign = ev.Matched(p, preds, agnostic=True, ignore={"b": [[198, 198, 242, 232]]})
    w = np.ones(len(plain.ids))
    assert plain.ap(w) < 1.0
    assert abs(ign.ap(w) - 1.0) < 1e-9
    # an ignore region never adds ground truth: missing it costs nothing
    miss = ev.Matched(p, preds[:1], agnostic=True, ignore={"b": [[201, 201, 239, 229]]})
    assert abs(miss.ap(w) - 1.0) < 1e-9
    # a detection only partly inside the region is still an FP at strict IoU thresholds (COCO crowd rule)
    tight = ev.Matched(p, preds, agnostic=True, ignore={"b": [[201, 201, 239, 229]]})
    assert 0.0 < 1.0 - tight.ap(w) < 0.2
