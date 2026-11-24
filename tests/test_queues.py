import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dyn_queues as mq  # noqa: E402


def _all(q):
    return [j for jobs in q.values() for j in jobs]


def test_matrix_counts():
    q = mq.build()
    train = [j for j in _all(q) if j["kind"] == "train" and not j["name"].startswith(("dyn_bench", "dyn_scr", "dyn_teacher_rgb"))]
    names = {j["name"] for j in train}
    assert "dyn_teacher_26l" in names
    students = names - {"dyn_teacher_26l"} - {n for n in names if n.startswith("dyn_smoke")}
    assert sorted(n for n in students if n.startswith("dyn_26n_")) == sorted(
        f"dyn_26n_{c}_s{s}" for c in ("B", "M", "C", "T", "TL", "S") for s in (0, 1, 2))
    for m in ("v8n", "11n"):
        assert sorted(n for n in students if n.startswith(f"dyn_{m}_")) == sorted(
            f"dyn_{m}_{c}_s{s}" for c in ("C", "T", "TL") for s in (0, 1, 2))
    assert len(students) == 36
    assert not any(n.startswith(("dyn_v10n", "dyn_12n")) for n in names)


def test_job_configs():
    jobs = {j["name"]: j for j in _all(mq.build())}
    b, c, t, s = (jobs[f"dyn_26n_{x}_s1"] for x in ("B", "C", "T", "S"))
    assert b["config"] == {} and b["train"]["data"] == "rgb"
    mx = jobs["dyn_26n_M_s1"]
    assert mx["config"] == {} and mx["train"]["data"] == "mix"
    assert {k: v for k, v in mx["train"].items() if k != "data"} == {k: v for k, v in b["train"].items() if k != "data"}
    assert c["config"]["ir_soft"] == 0.3 and c["config"]["paired"] is True and c["config"]["load_ir"] is False
    assert t["config"]["load_ir"] is False
    assert t["config"]["ir_soft_w_dir"] == "auto:T" and "ir_soft" not in t["config"]
    assert s["config"]["ir_soft_w_dir"] == "auto:S"
    for j in (b, c, t, s):
        assert j["train"]["epochs"] == 40 and j["train"]["imgsz"] == 640 and j["train"]["seed"] == 1
        assert j["train"]["deterministic"] is False and j["train"]["cache"] == "disk"
        assert j["train"].get("distill_model") is None
        assert j["predict"]["split"] == "test" and "test" in j["splits"]
    assert jobs["dyn_v8n_TL_s0"]["train"]["model"] == "yolov8n.pt"
    assert mq.HW["local"]["workers"] == 4
    tch = jobs["dyn_teacher_26l"]
    assert tch["train"]["model"] == "yolo26l.pt" and tch["train"]["data"] == "ir" and tch["train"]["batch"] == 16


def test_t4_c_queues_need_no_teacher():
    q = mq.build()
    for m in ("v8n", "11n"):
        for s in (0, 1, 2):
            assert [j["name"] for j in q[f"t4_{m}_s{s}_c"]] == [f"dyn_{m}_C_s{s}"]
            assert [j["name"] for j in q[f"t4_{m}_s{s}_ttl"]] == [f"dyn_{m}_T_s{s}", f"dyn_{m}_TL_s{s}"]
            assert all(j["train"]["seed"] == s for j in q[f"t4_{m}_s{s}_ttl"])


def test_smoke_tl_covers_all_models_and_no_smoke_t():
    q = mq.build()
    assert "smoke_t" not in q
    assert [j["name"] for j in q["smoke_tl"]] == [f"dyn_smoke_{m}_TL" for m in ("26n", "v8n", "11n")]
    for j in q["smoke_tl"]:
        assert j["config"]["ir_soft_w_dir"] == "auto:TL" and j["train"]["data"] == "rgb_proxy"
        assert j["train"]["epochs"] == 1 and j["proxy_fraction"] == 0.012
    assert all(j["config"]["ir_soft"] == 0.3 for j in q["smoke"])


def test_screen_queues():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = q[f"t4_screen_s{s}"]
        assert [j["name"] for j in jobs] == [f"dyn_scr_{c}_s{s}" for c in ("B", "C", "ND", "FX", "NI")]
        for j in jobs:
            t = j["train"]
            assert (t["epochs"], t["imgsz"], t["batch"], t["seed"], t["data"]) == (15, 704, 32, s, "rgb_proxy")
            assert j["proxy_fraction"] == 0.2 and "proxy_seed" not in j  # one fixed subset for every run
            assert j["predict"]["split"] == "val" and j["predict"]["imgsz"] == 704
        by = {j["name"].split("_")[2]: j for j in jobs}
        assert by["B"]["config"] == {} and by["C"]["config"]["ir_soft"] == 0.3
        for v in ("ND", "FX", "NI"):
            assert by[v]["config"]["ir_soft_w_dir"] == f"auto:{v}" and by[v]["config"]["load_ir"] is False


def test_rgb_teacher_folds():
    q = mq.build()
    jobs = q["teacher_rgb"]
    assert [j["fold"] for j in jobs] == [[0, 2], [1, 2]]
    assert all(j["train"]["data"] == "rgb_proxy" for j in jobs)
    assert all({p["split"] for p in j["predict"]} == {"train", "val"} for j in jobs)


def test_fold_of_partition(tmp_path, monkeypatch):
    import job_main as jm

    d = tmp_path / "proc" / "rgb" / "train" / "images"
    d.mkdir(parents=True)
    for i in range(11):
        (d / f"{i:05d}.jpg").write_bytes(b"")
    monkeypatch.setattr(jm, "DATA", tmp_path)
    f = jm.fold_of(2)
    assert len(f) == 11 and sorted(set(f.values())) == [0, 1]
    assert abs(list(f.values()).count(0) - list(f.values()).count(1)) <= 1
    assert f == jm.fold_of(2)


def test_screen2_queue():
    q = mq.build()
    for s in (0, 1, 2):
        jobs = {j["name"]: j for j in q[f"t4_screen2_s{s}"]}
        assert sorted(jobs) == sorted(f"dyn_scr_{c}_s{s}" for c in ("B", "C", "SH", "IG", "SI"))
        assert jobs[f"dyn_scr_IG_s{s}"]["config"]["ir_ignore"] is True
        assert jobs[f"dyn_scr_SI_s{s}"]["config"]["ir_soft_w_dir"] == "auto:SI"
        assert not jobs[f"dyn_scr_SH_s{s}"]["config"].get("ir_ignore")
        assert jobs[f"dyn_scr_C_s{s}"] == next(j for j in q[f"t4_screen_s{s}"] if j["name"] == f"dyn_scr_C_s{s}")
