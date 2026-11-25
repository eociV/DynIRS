import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pack_kaggle as pk  # noqa: E402


def test_pack_omega_and_code(tmp_path):
    proc = tmp_path / "proc"
    (proc / "ironly_w" / "P" / "train").mkdir(parents=True)
    (proc / "ironly_w" / "P" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3\n")
    (proc / "stats").mkdir()
    (proc / "stats" / "clean_P.json").write_text("{}")
    out = pk.pack_omega(proc, tmp_path / "out", "dynirs-omega", user="me", variants=("P",))
    assert (out / "ironly_w" / "P" / "train" / "a.txt").exists()
    assert (out / "stats" / "clean_P.json").exists()
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["id"] == "me/dynirs-omega"
    root = tmp_path / "repo"
    for d in ("src", "scripts", "configs/queues/dyn", "splits"):
        (root / d).mkdir(parents=True)
    (root / "src" / "a.py").write_text("x")
    (root / "requirements.txt").write_text("y")
    code = pk.pack_code(root, tmp_path / "out", "dynirs-code", user="me")
    assert (code / "src" / "a.py").exists() and (code / "requirements.txt").exists()


def test_pack_code_writes_commit(tmp_path):
    root = tmp_path / "repo"
    for d in ("src", "scripts", "configs", "splits"):
        (root / d).mkdir(parents=True)
    (root / "requirements.txt").write_text("y")
    code = pk.pack_code(root, tmp_path / "out", "dynirs-code", user="me")
    assert (code / "COMMIT").read_text().strip() == "unknown"  # tmp repo is not a git checkout
    assert pk.git_commit(pk.ROOT) not in ("", "unknown")


def test_pack_omega_only_clean_variants(tmp_path):
    proc = tmp_path / "proc"
    for v in ("P", "I", "V", "F", "C", "T"):
        d = proc / "ironly_w" / v / "train"
        d.mkdir(parents=True)
        (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.3\n")
    (proc / "stats").mkdir()
    for v in ("P", "I", "V", "F", "C"):
        (proc / "stats" / f"clean_{v}.json").write_text("{}")
    (proc / "stats" / "omega_T.json").write_text("{}")
    d = pk.pack_omega(proc, tmp_path / "out", "dynirs-omega", "someone")
    assert sorted(p.name for p in (d / "ironly_w").iterdir()) == ["C", "F", "I", "P", "V"]
    assert sorted(p.name for p in (d / "stats").iterdir()) == [f"clean_{v}.json" for v in ("C", "F", "I", "P", "V")]
