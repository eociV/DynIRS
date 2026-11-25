import ast
import os
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dyncv import folds as F  # noqa: E402
from dyncv import merge as M  # noqa: E402
from dyncv import orchestrator as O  # noqa: E402
from dyncv import progress as P  # noqa: E402
from dyncv import queues as Q  # noqa: E402
from dyncv import score as S  # noqa: E402
from dyncv import session as SE  # noqa: E402
from dyncv.backends import kaggle as K  # noqa: E402
from dyncv.config import CVConfig  # noqa: E402

IDS = [f"{i:05d}" for i in range(1, 1001)]


# ------------------------------------------------------------------------------ folds

def test_folds_disjoint_complete_deterministic_and_blocked():
    counts = {i: (int(i) % 7) for i in IDS}
    a = F.make_folds(IDS, counts, k=5, block=50, seed=0)
    assert a == F.make_folds(IDS, counts, k=5, block=50, seed=0)
    assert a != F.make_folds(IDS, counts, k=5, block=50, seed=1)
    flat = [i for f in a for i in f]
    assert sorted(flat) == IDS and len(flat) == len(set(flat))
    blocks = F.make_blocks(IDS, 50)
    for b in blocks:  # a block never straddles two folds
        assert sum(set(b) <= set(f) for f in a) == 1
    loads = [s["ironly_boxes"] for s in F.fold_stats(a, counts)]
    assert max(loads) - min(loads) <= max(sum(counts[i] for i in b) for b in blocks)


def test_folds_train_and_held_out_are_complements():
    a = F.make_folds(IDS, {}, k=4, block=10)
    for f in range(4):
        assert not set(F.training_ids(a, f)) & set(F.held_out(a, f))
        assert sorted(F.training_ids(a, f) + F.held_out(a, f)) == IDS


def test_folds_rejects_too_few_blocks():
    with pytest.raises(ValueError):
        F.make_folds(IDS[:20], {}, k=5, block=10)


# ------------------------------------------------------------------------------ queues

def test_teacher_job_keeps_the_in_sample_recipe():
    import make_dyn_queues

    cfg = CVConfig()
    folds = F.make_folds(IDS, {}, k=5, block=50)
    jobs = Q.build(cfg, folds)
    ref = make_dyn_queues.teacher(cfg.epochs, name="x")
    for f, j in enumerate(jobs):
        assert j["name"] == f"cv5_teacher_26l_f{f}"
        assert j["chunks"] == ref["chunks"] and j["config"] == ref["config"] and j["kind"] == "train"
        diff = {k for k in ref["train"] if ref["train"][k] != j["train"][k]}
        assert diff <= {"data", "workers"} and j["train"]["data"] == "ir_proxy"
        assert set(j["proxy_ids"]) == set(IDS) - set(folds[f])
        assert j["cv"]["fold"] == f and j["predict"]["data_root"] == "ir"


def test_default_groups_pair_folds_per_session():
    assert Q.default_groups(5) == [[0, 1], [2, 3], [4]]


def test_generated_queues_match_fold_file():
    cfg = CVConfig()
    if not cfg.folds_file.exists():
        pytest.skip("fold file not generated")
    d = F.load(cfg.folds_file)
    for j in Q.load(cfg.queue_dir / "cv5_all.json"):
        assert not set(j["proxy_ids"]) & set(d["folds"][j["cv"]["fold"]])


# ------------------------------------------------------------------------------ score / merge

@pytest.fixture
def proc(tmp_path):
    p = tmp_path / "proc"
    for d in ("ironly_cls/train", "ir/train/images", "rgb/train/images"):
        (p / d).mkdir(parents=True)
    rgb = np.zeros((512, 640, 3), np.uint8)
    rgb[:, 320:] = 255
    for k, lines in {"00001": ["0 0.25 0.5 0.1 0.1", "1 0.75 0.5 0.1 0.1"], "00002": ["0 0.25 0.5 0.1 0.1"],
                     "00003": []}.items():
        (p / "ironly_cls" / "train" / f"{k}.txt").write_text("".join(x + "\n" for x in lines))
        cv2.imwrite(str(p / "ir" / "train" / "images" / f"{k}.jpg"), rgb)
        cv2.imwrite(str(p / "rgb" / "train" / "images" / f"{k}.jpg"), rgb)
    return p


def fake_predict(paths):
    # one prediction exactly on the left box (x 128..192, y 230..282) with score 0.8
    return [(np.array([[128.0, 230.4, 192.0, 281.6]]), np.array([0.8])) for _ in paths]


def test_fold_signals_only_scores_the_given_ids(proc):
    sig = S.fold_signals(proc, ["00001", "00003"], fake_predict, log=lambda m: None)
    assert set(sig) == {"00001"}  # 00003 has no IR-only box, 00002 is in another fold
    (c0, v0), (c1, v1) = sig["00001"]
    assert c0 == pytest.approx(0.8) and c1 == 0.0
    assert v0 == pytest.approx(0.0, abs=1e-3) and v1 == pytest.approx(1.0, abs=1e-3)
    st = S.signal_stats(sig)
    assert st["n_boxes"] == 2 and st["frac_c_zero"] == 0.5


def test_merge_requires_full_disjoint_coverage(proc):
    import omega

    boxes = omega.load_ironly(proc)
    a = {"00001": [[0.8, 0.0], [0.0, 1.0]]}
    b = {"00002": [[0.5, 0.0]]}
    m = M.merge_signals([a, b], boxes)
    assert m == {"00001": a["00001"], "00002": b["00002"], "00003": []}
    with pytest.raises(ValueError, match="in no fold"):
        M.merge_signals([a], boxes)
    with pytest.raises(ValueError, match="scored in folds"):
        M.merge_signals([a, b, a], boxes)
    with pytest.raises(ValueError, match="boxes but"):
        M.merge_signals([{"00001": [[0.8, 0.0]]}, b], boxes)


def test_compare_reports_inflation():
    cv = {"a": [[0.2, 0], [0.4, 0]], "b": [[0.6, 0]]}
    ins = {"a": [[0.9, 0], [0.95, 0]], "b": [[0.99, 0]]}
    r = M.compare(cv, ins)
    assert r["n_boxes"] == 3 and r["mean_diff_cv_minus_insample"] < 0
    assert r["insample"]["c_frac_ge_0_9"] == 1.0 and r["cv"]["c_frac_ge_0_9"] == 0.0


def _job_dir(runs, name):
    out = S.job_out(runs, name)
    out.mkdir(parents=True)
    return out


def test_score_job_rejects_leaky_job(tmp_path, proc, monkeypatch):
    cfg = CVConfig(k=2, block=1)
    folds = [["00001", "00003"], ["00002"]]
    ff = tmp_path / "folds.json"
    F.save(ff, folds, cfg, [])
    monkeypatch.setattr(S, "ROOT", tmp_path)
    spec = {"name": "j", "proxy_ids": ["00001", "00002"], "train": {"imgsz": 640},
            "cv": {"k": 2, "fold": 0, "block": 1, "seed": 0, "folds_file": "folds.json"}}
    with pytest.raises(ValueError, match="trained on images of its held-out fold"):
        S.score_job(spec, tmp_path / "runs", proc)


# ------------------------------------------------------------------------------ progress

CSV = "epoch,time,train/box_loss,metrics/mAP50-95(B)\n1,60,1.5,0.1\n2,120,1.2,0.2\n"


def test_parse_results_csv_skips_partial_last_line():
    rows = P.parse_results_csv(CSV + "3,180,1.")
    assert [r["epoch"] for r in rows] == [1, 2] and rows[1]["metrics/mAP50-95(B)"] == 0.2


def test_metric_lines_roundtrip_raw_and_kaggle_json():
    lines = [P.format_line("job", r) for r in P.parse_results_csv(CSV)]
    raw = "noise\n" + "\n".join(lines) + "\nmore"
    assert P.parse_log(raw) == {"job": P.parse_results_csv(CSV)}
    kag = json.dumps([{"stream_name": "stdout", "time": 1.0, "data": l + "\n"} for l in lines])
    assert P.parse_log(kag) == P.parse_log(raw)


def test_tailer_emits_each_epoch_once(tmp_path):
    out = tmp_path / "job" / "ws" / "out" / "job"
    out.mkdir(parents=True)
    (out / "results.csv").write_text(CSV)
    got = []
    t = P.Tailer(tmp_path, emit=got.append)
    assert t.poll() == 2 and t.poll() == 0
    (out / "results.csv").write_text(CSV + "3,180,1.0,0.3\n")
    assert t.poll() == 1 and P.parse_log("\n".join(got))["job"][-1]["epoch"] == 3


# ------------------------------------------------------------------------------ kaggle backend

def test_kaggle_names_are_separate_from_dynirs():
    assert K.kernel_slug("configs/queues/cv5/cv5_f01.json", 3) == "dyncv-cv5-f01-r3"
    m = K.kernel_metadata("me", "me", "team", "dyncv-x-r2", prev="dyncv-x-r1")
    assert m["dataset_sources"] == ["me/dynirs-cv-code", "team/dynirs-dv"]
    assert m["kernel_sources"] == ["me/dyncv-x-r1"] and m["is_private"]


def test_default_data_comes_from_prep_kernel():
    m = K.kernel_metadata("me", "me", None, "dyncv-x-r2", prev="dyncv-x-r1")
    assert m["dataset_sources"] == ["me/dynirs-cv-code"]
    assert m["kernel_sources"] == ["me/dyncv-prep", "me/dyncv-x-r1"]
    p = K.prep_metadata("me", "me")
    assert p["id"] == "me/dyncv-prep" and not p["enable_gpu"] and p["enable_internet"]
    src = K.render_prep({"train": "abc"}, "configs/cv/folds_k5_b50_s0.json")
    ast.parse(src)
    assert '{"train": "abc"}' in src and "dyncv.prep" in src


def _labels(proc, eol):
    for d in ("rgb/train/labels", "ir/train/labels", "ironly/train"):
        (proc / d).mkdir(parents=True, exist_ok=True)
        (proc / d / "00001.txt").write_bytes(f"0 0.5 0.5 0.1 0.1{eol}1 0.2 0.2 0.1 0.1{eol}".encode())
    (proc / ".ready_train").write_text("raw-hash")


def test_label_hash_ignores_line_endings_and_is_cached(tmp_path):
    from dyncv import prep as PR

    _labels(tmp_path / "win", "\r\n")
    _labels(tmp_path / "lin", "\n")
    assert PR.label_hash(tmp_path / "win", "train") == PR.label_hash(tmp_path / "lin", "train")
    cache = tmp_path / "cache.json"
    h = K.local_hashes(tmp_path / "win", cache)
    assert h == {"train": PR.label_hash(tmp_path / "lin", "train")}
    (tmp_path / "win" / "ir" / "train" / "labels" / "00001.txt").write_text("changed")
    assert K.local_hashes(tmp_path / "win", cache) == h  # cache hit: .ready_train unchanged


def test_prep_check_and_tar_roundtrip(tmp_path):
    from dyncv import prep as PR

    proc = tmp_path / "dv" / "proc"
    (proc / "ir" / "train" / "images").mkdir(parents=True)
    for i in ("00001", "00002"):
        (proc / "ir" / "train" / "images" / f"{i}.jpg").write_bytes(b"j")
        (proc / "ir" / "train" / "images" / f"{i}.npy").write_bytes(b"cache")
    ok = PR.check(proc, {"train": "h"}, {"train": "h"}, ["00002", "00001"])
    assert ok["ok"] and ok["train_images"] == 2
    bad = PR.check(proc, {"train": "x"}, {"train": "h"}, ["00001"])
    assert not bad["ok"] and len(bad["problems"]) == 2
    tar = PR.write_tar(proc, tmp_path / PR.TAR)
    got = PR.extract(tar, tmp_path / "out")
    assert sorted(p.name for p in (got / "ir" / "train" / "images").iterdir()) == ["00001.jpg", "00002.jpg"]


def test_stage_data_unpacks_tar_once(tmp_path, monkeypatch):
    from dyncv import prep as PR

    proc = tmp_path / "src" / "proc"
    proc.mkdir(parents=True)
    (proc / "x.txt").write_text("1")
    PR.write_tar(proc, tmp_path / PR.TAR)
    monkeypatch.setattr(SE, "DV", tmp_path / "dv")
    SE.stage_data(tmp_path)
    assert (tmp_path / "dv" / "proc" / "x.txt").read_text() == "1"
    (tmp_path / "dv" / "proc" / "x.txt").write_text("2")
    SE.stage_data(tmp_path)  # already there: untouched
    assert (tmp_path / "dv" / "proc" / "x.txt").read_text() == "2"


def test_runner_is_valid_python():
    src = K.render_runner("dyncv-x-r1", "configs/queues/cv5/cv5_f01.json", ["a"], 11.0)
    ast.parse(src)
    assert "dyncv.session" in src and "'configs/queues/cv5/cv5_f01.json'" in src


def test_kaggle_submit_resumes_from_last_complete_round(tmp_path):
    calls = []
    b = K.KaggleBackend(user="me", state_dir=tmp_path, run_cli=lambda *a, **k: (calls.append(a), (0, "ok"))[1])
    q = ROOT / "configs" / "queues" / "cv5" / "cv5_f01.json"
    if not q.exists():
        pytest.skip("queues not generated")
    s1 = b.submit(q)
    b.mark(s1, "complete")
    s2 = b.submit(q)
    b.mark(s2, "error")
    s3 = b.submit(q)
    meta = json.loads((tmp_path / "kernels" / s3 / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["me/dyncv-prep", f"me/{s1}"]
    assert b.active(q) == s3


def test_session_command_caches_ir_only():
    class A:
        code_in, queue, data_in, work, budget_h, lanes, gpus = "/c", "q.json", "/d", "/w", 11.0, 2, "0,1"
        prev_out, only = "/p", ["j"]

    cmd = SE.session_cmd(A)
    assert cmd[cmd.index("--mods") + 1:cmd.index("--mods") + 2] == ["ir"]
    assert cmd[-2:] == ["--only", "j"] and "--prev-out" in cmd


# ------------------------------------------------------------------------------ orchestrator

class FakeBackend:
    name = "fake"

    def __init__(self, logs="", statuses=(), no_output=(), on_fetch=None):
        self.n, self._logs, self.marked = 0, logs, {}
        self.statuses, self.no_output, self.on_fetch = list(statuses), set(no_output), on_fetch
        self.fetched = []

    def active(self, queue):
        return None

    def unfetched(self, queue):
        return None

    def submit(self, queue, only, budget_h):
        self.n += 1
        return f"s{self.n}"

    def status(self, sid):
        return self.statuses[self.n - 1] if self.statuses else "complete"

    def logs(self, sid):
        return self._logs

    def fetch(self, sid, dst):
        if sid in self.no_output:
            raise SystemExit("no runs/remote")
        self.fetched.append(sid)
        if self.on_fetch:
            self.on_fetch(sid)
        return dst

    def mark(self, sid, st=None, **info):
        self.marked.setdefault(sid, {}).update(info, **({"status": st} if st else {}))


class FakeTracker:
    def __init__(self):
        self.rows = {}

    def run_id(self, name, spec=None, tags=None):
        return name

    def log_epochs(self, name, rows):
        self.rows.setdefault(name, []).extend(rows)
        return len(rows)


class Clock:
    def strftime(self, f):
        return "00:00:00"

    def sleep(self, s):
        pass


def test_watch_pushes_live_metrics_of_queue_jobs_only():
    lines = "\n".join([P.format_line("j", {"epoch": 1, "x": 1.0}), P.format_line("other", {"epoch": 1})])
    t = FakeTracker()
    assert O.watch(FakeBackend(lines), "s1", [{"name": "j"}], t, clock=Clock()) == "complete"
    assert t.rows == {"j": [{"epoch": 1, "x": 1.0}]}


def _queue(tmp_path, names):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"name": n, "kind": "train", "cv": {}} for n in names]))
    return q


def test_run_survives_a_failed_session_with_output(tmp_path, monkeypatch):
    # s1 errors but its output (last.pt) is fetched; s2 completes and scores -> done in 2 sessions
    runs = tmp_path / "runs"
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [] if (S.job_out(r, s["name"]) / "done").exists() else [1])

    def on_fetch(sid):
        if sid == "s2":
            out = S.job_out(runs, "j")
            out.mkdir(parents=True, exist_ok=True)
            (out / "done").write_text("")
            (out / "cv_signals.json").write_text("{}")

    b = FakeBackend(statuses=["error", "complete"], on_fetch=on_fetch)
    O.run(b, _queue(tmp_path, ["j"]), runs=runs, poll=0)
    assert b.fetched == ["s1", "s2"] and b.marked["s1"] == {"status": "error", "fetched": True}


def test_run_stops_after_repeated_failures_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [1])
    b = FakeBackend(statuses=["error", "error"], no_output={"s1", "s2"})
    with pytest.raises(SystemExit, match="2 failed sessions"):
        O.run(b, _queue(tmp_path, ["j"]), runs=tmp_path / "runs", poll=0)
    assert b.marked["s1"]["no_output"] and b.n == 2


def test_run_does_nothing_when_queue_is_done(tmp_path, monkeypatch):
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [])
    runs = tmp_path / "runs"
    _job_dir(runs, "j")
    (S.job_out(runs, "j") / "cv_signals.json").write_text("{}")
    b = FakeBackend()
    assert O.run(b, _queue(tmp_path, ["j"]), runs=runs) == [] and b.n == 0


def test_kaggle_resumes_from_failed_session_that_was_fetched(tmp_path):
    q = ROOT / "configs" / "queues" / "cv5" / "cv5_f01.json"
    if not q.exists():
        pytest.skip("queues not generated")
    b = K.KaggleBackend(user="me", state_dir=tmp_path, run_cli=lambda *a, **k: (0, "ok"))
    s1 = b.submit(q)
    b.mark(s1, "error")
    assert b.unfetched(q) == s1
    b.mark(s1, fetched=True)
    assert b.unfetched(q) is None
    s2 = b.submit(q)
    assert json.loads((tmp_path / "kernels" / s2 / "kernel-metadata.json").read_text())["kernel_sources"] == ["me/dyncv-prep", f"me/{s1}"]


def test_pending_counts_unscored_cv_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(O.run_queue, "plan", lambda s, runs: [])
    specs = [{"name": "a", "cv": {}}, {"name": "b", "cv": {}}, {"name": "c"}]
    _job_dir(tmp_path, "a")
    (S.job_out(tmp_path, "a") / "cv_signals.json").write_text("{}")
    assert O.pending(specs, tmp_path) == ["b"]


# ------------------------------------------------------------------------------ tracking

def test_tracker_is_idempotent(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from dyncv import tracking as T

    monkeypatch.setattr(T.C, "mlflow_artifacts", lambda: (tmp_path / "art").as_uri())
    tr = T.Tracker(f"sqlite:///{(tmp_path / 'm.db').as_posix()}", "test")
    spec = {"name": "job", "train": {"epochs": 3, "model": "yolo26l.pt"}, "proxy_ids": ["1", "2"], "cv": {"fold": 0}}
    out = _job_dir(tmp_path / "runs", "job")
    (out / "results.csv").write_text(CSV)
    (out / "weights").mkdir()
    (out / "weights" / "best.pt").write_bytes(b"x")
    assert tr.sync_job(spec, tmp_path / "runs") == 2
    assert tr.sync_job(spec, tmp_path / "runs") == 0
    assert T.Tracker(f"sqlite:///{(tmp_path / 'm.db').as_posix()}", "test").logged_epochs("job") == {1, 2}
    run = tr.client.get_run(tr.run_id("job"))
    assert run.data.params["n_train_images"] == "2" and run.data.tags["dyncv.state"] == "epoch 2/3"
    assert "metrics/mAP50-95_B" in run.data.metrics
    assert {a.path for a in tr.client.list_artifacts(run.info.run_id, "weights")} == {"weights/best.pt"}


# ------------------------------------------------------------------------------ tick / schedule

class TickBackend(FakeBackend):
    def __init__(self, active=None, status="running", fails=0, **kw):
        super().__init__(statuses=[status] * 9, **kw)
        self._active, self._fails = active, fails

    def active(self, queue):
        return self._active

    def failures(self, queue):
        return self._fails

    def status(self, sid):
        return self.statuses[0]


def test_tick_running_session_only_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [1])
    b = TickBackend(active="s9", status="running")
    msg = O.tick(b, _queue(tmp_path, ["j"]), runs=tmp_path / "runs")
    assert msg == "q: s9 running" and b.n == 0 and b.fetched == []


def test_tick_ended_session_is_fetched_then_next_pushed(tmp_path, monkeypatch):
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [1])
    b = TickBackend(active="s9", status="complete")
    msg = O.tick(b, _queue(tmp_path, ["j"]), runs=tmp_path / "runs")
    assert b.fetched == ["s9"] and b.marked["s9"]["status"] == "complete"
    assert msg.startswith("q: pushed s1")


def test_tick_done_and_stopped(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [])
    _job_dir(runs, "j")
    (S.job_out(runs, "j") / "cv_signals.json").write_text("{}")
    assert O.tick(TickBackend(), _queue(tmp_path, ["j"]), runs=runs) == "q: done"
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [1])
    b = TickBackend(fails=2)
    assert "STOPPED" in O.tick(b, _queue(tmp_path, ["j"]), runs=runs) and b.n == 0


def test_kaggle_failures_count_trailing_errors(tmp_path):
    q = ROOT / "configs" / "queues" / "cv5" / "cv5_f01.json"
    if not q.exists():
        pytest.skip("queues not generated")
    b = K.KaggleBackend(user="me", state_dir=tmp_path, run_cli=lambda *a, **k: (0, "ok"))
    for st in ("error", "complete", "error", "cancelled"):
        b.mark(b.submit(q), st)
    assert b.failures(q) == 2


def test_lock_is_exclusive_and_released(tmp_path):
    from dyncv.schedule import lock

    f = tmp_path / "t.lock"
    with lock(f) as a:
        assert a
        with lock(f) as b:
            assert not b
    assert not f.exists()
    f.write_text("old")
    os.utime(f, (0, 0))  # stale lock from a dead tick
    with lock(f, stale_s=60) as c:
        assert c


def test_tick_command_runs_from_repo(tmp_path):
    from dyncv.schedule import tick_command

    s = tick_command([ROOT / "configs/queues/cv5/cv5_f4.json"])
    assert f'cd /d "{ROOT}"' in s and "-m dyncv tick" in s and "cv5_f4.json" in s and "tick.log" in s


# ------------------------------------------------------------------------------ live (HF)

class FakeHf:
    def __init__(self):
        self.files = {}

    def upload_file(self, path_or_fileobj, path_in_repo, **kw):
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()


def test_pusher_sends_csv_each_change_and_ckpt_every_k(tmp_path):
    from dyncv import live as L

    out = _job_dir(tmp_path, "job")
    (out / "weights").mkdir()
    rows = "".join(f"{e},60,1.0,0.5\n" for e in range(1, 11))
    (out / "results.csv").write_text(CSV.splitlines()[0] + "\n" + rows)
    last = out / "weights" / "last.pt"
    last.write_bytes(b"ckpt10")
    import time as _t

    old = _t.time() - 120
    os.utime(out / "results.csv", (old, old))
    os.utime(last, (old + 10, old + 10))  # written after the row, and settled (> 30 s ago)
    api = FakeHf()
    p = L.Pusher(tmp_path, "u/r", "tok", ckpt_every=10, api=api)
    assert p.poll() == 2
    assert api.files["runs/job/weights/last.pt"] == b"ckpt10" and "runs/job/results.csv" in api.files
    assert p.poll() == 0  # nothing changed


def test_reader_and_tick_log_live_epochs(tmp_path, monkeypatch):
    from dyncv import live as L

    class Api:
        def hf_hub_download(self, repo, path, repo_type, local_dir, force_download):
            f = Path(local_dir) / "r.csv"
            f.write_text(CSV)
            return str(f)

    rows = L.Reader("u/r", api=Api()).rows("j")
    assert [r["epoch"] for r in rows] == [1, 2]
    monkeypatch.setattr(O.run_queue, "plan", lambda s, r: [1])
    t = FakeTracker()
    t.set_tags = lambda *a, **k: None
    msg = O.tick(TickBackend(active="s1"), _queue(tmp_path, ["j"]), t, runs=tmp_path / "runs",
                 reader=L.Reader("u/r", api=Api()))
    assert msg == "q: s1 running; live j epoch 2 (+2)" and len(t.rows["j"]) == 2


def test_live_metadata_mounts_secrets_and_runner_passes_it():
    m = K.kernel_metadata("me", "me", None, "dyncv-x-r1", live=True)
    assert "me/dyncv-secrets" in m["dataset_sources"]
    assert "me/dyncv-secrets" not in K.kernel_metadata("me", "me", None, "dyncv-x-r1")["dataset_sources"]
    src = K.render_runner("dyncv-x-r1", "configs/queues/cv5/cv5_f01.json", None, 11.0)
    ast.parse(src)
    assert '"--live"' in src


def test_secrets_setup_leaves_no_token_file(tmp_path, monkeypatch):
    from dyncv import live as L

    monkeypatch.setattr(L, "STATE_DIR", tmp_path)
    monkeypatch.setattr(L, "LOCAL_CFG", tmp_path / "live.json")
    seen = []

    def cli(*a, check=True):
        if a[:2] == ("datasets", "create"):
            seen.append(json.loads((Path(a[3]) / "live.json").read_text()))
        return (1, "") if a[:2] == ("datasets", "status") else (0, "ok")

    L.setup("me", "hf_secret", "u/r", cli, out_dir=tmp_path / "pkg")
    assert seen == [{"repo": "u/r", "token": "hf_secret"}]
    assert not (tmp_path / "pkg").exists()
    assert json.loads((tmp_path / "live.json").read_text()) == {"repo": "u/r"}
