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
