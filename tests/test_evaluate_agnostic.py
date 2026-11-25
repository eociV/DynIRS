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
