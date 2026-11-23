import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_lanes  # noqa: E402


def test_over_budget():
    assert not run_lanes.over_budget(0, 100, 11.0)
    assert not run_lanes.over_budget(100, 100 + 11 * 3600 - 1, 11.0)
    assert run_lanes.over_budget(100, 100 + 11 * 3600, 11.0)
    assert not run_lanes.over_budget(0, 10**9, None)
    assert not run_lanes.over_budget(0, 10**9, 0)  # 0 / None = no budget


def test_resolve_deadline():
    assert run_lanes.resolve_deadline(None, None, 100.0) is None
    assert run_lanes.resolve_deadline(None, 1.0, 100.0) == 100.0 + 3600
    assert run_lanes.resolve_deadline(555.0, 1.0, 100.0) == 555.0  # explicit deadline wins
    assert run_lanes.past_deadline(None, 1e12) is False
    assert run_lanes.past_deadline(10.0, 10.0) and not run_lanes.past_deadline(10.0, 9.9)


def test_kill_tree_kills_grandchildren(tmp_path):
    import subprocess
    import time

    import psutil

    pidfile = tmp_path / "gc.pid"
    child = (
        "import subprocess, sys, time\n"
        f"g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pidfile}', 'w').write(str(g.pid))\n"
        "time.sleep(60)\n"
    )
    p = subprocess.Popen([sys.executable, "-c", child], **run_lanes.lane_popen_kwargs())
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    gc = int(pidfile.read_text())
    assert psutil.pid_exists(gc)
    run_lanes.kill_tree(p)
    for _ in range(50):
        if not psutil.pid_exists(gc) and p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is not None
    assert not psutil.pid_exists(gc)


def test_current_job_file(tmp_path):
    f = tmp_path / ".lane_0_current"
    assert run_lanes.read_current(f) == "unknown"
    f.write_text("job_a chunk 20")
    assert run_lanes.read_current(f) == "job_a chunk 20"
