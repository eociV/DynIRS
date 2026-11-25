"""Split the IR train images into k folds.

Unit = a block of `block` consecutive ids (sorted), so near-duplicate neighbouring frames never sit on
both sides of a split. Blocks are shuffled with `seed`, then dealt greedily (largest IR-only box count
first) to the fold with the fewest IR-only boxes so far, ties broken by fewest images. The result is
deterministic for a given (ids, box counts, k, block, seed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_blocks(ids, block):
    ids = sorted(ids)
    return [ids[i:i + block] for i in range(0, len(ids), block)]


def assign(blocks, box_count, k, seed=0):
    """blocks: list of id lists; box_count: id -> number of IR-only boxes. Returns k sorted id lists."""
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(blocks) < k:
        raise ValueError(f"{len(blocks)} blocks cannot fill {k} folds")
    order = np.random.default_rng(seed).permutation(len(blocks))
    load = [sum(box_count.get(i, 0) for i in b) for b in blocks]
    order = sorted(order, key=lambda j: -load[j])  # stable: equal loads keep the shuffled order
    folds, boxes, images = [[] for _ in range(k)], [0] * k, [0] * k
    for j in order:
        f = min(range(k), key=lambda x: (boxes[x], images[x], x))
        folds[f] += blocks[j]
        boxes[f] += load[j]
        images[f] += len(blocks[j])
    return [sorted(f) for f in folds]


def make_folds(ids, box_count, k=5, block=50, seed=0):
    return assign(make_blocks(ids, block), box_count, k, seed)


def fold_stats(folds, box_count):
    return [{"fold": f, "images": len(ids), "images_with_ironly": sum(box_count.get(i, 0) > 0 for i in ids),
             "ironly_boxes": sum(box_count.get(i, 0) for i in ids)} for f, ids in enumerate(folds)]


def ironly_box_counts(proc):
    """id -> number of IR-only boxes in proc/ironly_cls/train (the boxes that get a weight)."""
    out = {}
    for f in sorted((Path(proc) / "ironly_cls" / "train").glob("*.txt")):
        out[f.stem] = sum(1 for line in f.read_text().splitlines() if line.strip())
    return out


def train_ids(proc):
    return sorted(p.stem for p in (Path(proc) / "ir" / "train" / "images").glob("*.jpg"))


# ------------------------------------------------------------------------------ fold file

def save(path, folds, cfg, stats):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"k": cfg.k, "block": cfg.block, "seed": cfg.seed, "stats": stats,
                                      "folds": folds}, indent=None, separators=(",", ":")))


def load(path):
    d = json.loads(Path(path).read_text())
    if len(d["folds"]) != d["k"]:
        raise ValueError(f"{path}: {len(d['folds'])} folds, expected {d['k']}")
    return d


def held_out(folds, fold):
    """Ids the teacher of `fold` scores (never trains on)."""
    return list(folds[fold])


def training_ids(folds, fold):
    """Ids the teacher of `fold` trains on: every other fold."""
    return sorted(i for f, ids in enumerate(folds) if f != fold for i in ids)
