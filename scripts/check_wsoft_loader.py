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
    ap.add_argument("--batches", type=int, default=50)
    a = ap.parse_args(argv)
    data = Path(os.environ.get("DV_DATA", ROOT / "data" / "dronevehicle"))
    import job_main as jm
    import lowlight as ll

    ll.CONFIG.update(jm.resolve_config(dict(paired=True, ir_soft_w_dir=f"auto:{a.variant}"), data, ROOT))
    allowed = set()
    for f in Path(ll.CONFIG["ir_soft_w_dir"]).glob("*.txt"):
        if f.stat().st_size:
            allowed |= set(np.round(np.loadtxt(f, ndmin=2)[:, 5] * 1000).astype(int).tolist())
    tr = ll.LLTrainer(overrides=dict(model="yolo26n.pt", data=str(data / "rgb.yaml"), imgsz=640, batch=16,
                                     workers=4, epochs=1, plots=False))
    tr._setup_train()
    seen = 0
    for i, batch in enumerate(tr.train_loader):
        if i >= a.batches:
            break
        cls, w = ll.split_box_weights(batch["cls"].view(-1), None, None)
        assert bool(((cls >= 0) & (cls <= 4)).all()), cls.unique()
        q = set(torch.round(w[w < 1] * 1000).long().tolist())
        assert q <= allowed, q - allowed
        seen += int((w < 1).sum())
    print(f"ok: {seen} weighted boxes in {a.batches} batches")
    assert seen > 0


if __name__ == "__main__":
    main()
