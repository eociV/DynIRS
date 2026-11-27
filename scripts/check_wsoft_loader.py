"""Check on real data that per-box weights survive the training loader (mosaic, perspective, flips).

  python scripts/check_wsoft_loader.py --variant T [--batches 50]

Builds the paired training loader exactly as LLTrainer does, decodes every batch and checks that each
decoded weight is one of the quantised values present in the weight files, and that classes are 0-4.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="T")
    ap.add_argument("--dir", help="weight directory to check instead of auto:<variant>")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--device", default="", help='e.g. "cpu" while the GPU is busy')
    a = ap.parse_args(argv)
    wdir = a.dir or f"auto:{a.variant}"
    data = Path(os.environ.get("DV_DATA", ROOT / "data" / "dronevehicle"))
    import job_main as jm
    import lowlight as ll

    allowed = set()
    d = jm.resolve_config(dict(ir_soft_w_dir=wdir), data, ROOT)["ir_soft_w_dir"]
    for f in Path(d).glob("*.txt"):
        if f.stat().st_size:
            allowed |= set(np.round(np.loadtxt(f, ndmin=2)[:, 5] * 1000).astype(int).tolist())
    ignore = any(x < 0 for x in allowed)  # omega -1: ignore regions
    allowed = {x for x in allowed if x >= 0}
    ll.CONFIG.update(jm.resolve_config(dict(paired=True, ir_soft_w_dir=wdir, ir_ignore=ignore),
                                       data, ROOT))
    tr = ll.LLTrainer(overrides=dict(model="yolo26n.pt", data=str(data / "rgb.yaml"), imgsz=640, batch=16,
                                     workers=0, epochs=1, plots=False, device=a.device, single_cls=True))  # 0: the check needs no speed, and Windows worker shared memory can fail
    tr._setup_train()
    seen = n_ign = 0
    for i, batch in enumerate(tr.train_loader):
        if i >= a.batches:
            break
        cls, w = ll.split_box_weights(batch["cls"].view(-1), None, None)
        ign = cls == ll.IGNORE_CLS
        assert ignore or not ign.any(), "ignore boxes in a variant without omega -1"
        assert bool(((cls[~ign] >= 0) & (cls[~ign] <= 4)).all()), cls.unique()
        q = set(torch.round(w[~ign & (w < 1)] * 1000).long().tolist())
        assert q <= allowed, q - allowed
        seen += int((~ign & (w < 1)).sum())
        n_ign += int(ign.sum())
        if i == 0:  # one loss through the trainer's preprocessing (ignore regions moved to ign_bboxes)
            b = tr.preprocess_batch(batch)
            assert len(b["ign_bboxes"]) == int(ign.sum()) and not bool((b["cls"] == ll.IGNORE_CLS).any())
            tr.model.train()
            loss = tr.model.loss(b)[0].sum().item()
            assert np.isfinite(loss), loss
    print(f"ok: {seen} weighted boxes, {n_ign} ignore boxes in {a.batches} batches")
    assert seen > 0 or n_ign > 0


if __name__ == "__main__":
    main()
