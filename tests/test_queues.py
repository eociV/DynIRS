import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dyn_queues as mq  # noqa: E402


def test_queue_names():
    q = mq.build()
    assert sorted(q) == sorted(["teacher_rgb1c"] + [f"t4_clean_s{s}" for s in (0, 1, 2)]
                               + [f"t4_cleanC_s{s}" for s in (0, 1, 2)]
                               + [f"full_clean_s{s}" for s in (0, 1, 2)])


def test_every_job_is_single_class():
    for jobs in mq.build().values():
        for j in jobs:
            assert j["train"]["single_cls"] is True


def test_rgb_teacher_rgb1c():
    jobs = mq.build()["teacher_rgb1c"]
    assert [j["name"] for j in jobs] == ["dyn_teacher_rgb1c_f0", "dyn_teacher_rgb1c_f1"]
    for f, j in enumerate(jobs):
        t = j["train"]
        assert (t["model"], t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (
            "yolo26l.pt", 30, 640, 16, 0, "rgb_proxy")
        assert j["fold"] == [f, 2] and j["proxy_seed"] == 0 and j["config"] == {}
        assert [p["split"] for p in j["predict"]] == ["train", "val"]


def test_screen_queues():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = q[f"t4_clean_s{s}"]
        assert [j["name"] for j in jobs] == [f"dyn_cl_{c}_s{s}" for c in ("B", "P", "I", "V", "F")]
        for j in jobs:
            t = j["train"]
            assert (t["model"], t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (
                "yolo26n.pt", 15, 704, 32, s, "rgb_proxy")
            assert j["proxy_fraction"] == 0.2 and "proxy_seed" not in j
            assert j["predict"]["split"] == "val" and j["predict"]["imgsz"] == 704
            assert j["splits"] == ["train", "val"] and t["workers"] == 2
        by = {j["name"].split("_")[2]: j["config"] for j in jobs}
        assert by["B"] == {}
        assert by["P"]["ir_soft_w_dir"] == "auto:P" and not by["P"].get("ir_ignore")
        for c in ("I", "V", "F"):
            assert by[c]["ir_soft_w_dir"] == f"auto:{c}" and by[c]["ir_ignore"] is True
            assert by[c]["paired"] is True and by[c]["load_ir"] is False


def test_full_queues():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = q[f"full_clean_s{s}"]
        assert [j["name"] for j in jobs] == [f"dyn_full_{c}_s{s}" for c in mq.FULL]
        for j in jobs:
            t = j["train"]
            assert (t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (60, 704, 32, s, "rgb")
            assert j["predict"]["split"] == "test" and "test" in j["splits"] and t["workers"] == 4
            assert "proxy_fraction" not in j


def test_main_removes_stale_queue_files(tmp_path, monkeypatch):
    monkeypatch.setattr(mq, "OUT", tmp_path)
    (tmp_path / "t4_v8n_s0_c.json").write_text("[]")
    mq.main([])
    names = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert "t4_v8n_s0_c" not in names and names == sorted(mq.build())


def test_fold_of_partition(tmp_path):
    import job_main as jm

    d = tmp_path / "proc" / "rgb" / "train" / "images"
    d.mkdir(parents=True)
    for i in range(11):
        (d / f"{i:05d}.jpg").write_bytes(b"")
    f = jm.fold_of(2, data=tmp_path)
    assert len(f) == 11 and sorted(set(f.values())) == [0, 1]
    assert abs(list(f.values()).count(0) - list(f.values()).count(1)) <= 1
    assert f == jm.fold_of(2, 0, tmp_path)


def test_clean_p_queues():
    q = mq.build()
    for s in (0, 1, 2):
        (j,) = q[f"t4_cleanC_s{s}"]
        assert j["name"] == f"dyn_cl_C_s{s}" and j["proxy_fraction"] == 0.2 and j["train"]["seed"] == s
        assert j["config"]["ir_soft_w_dir"] == "auto:C" and j["config"]["ir_ignore"] is True
