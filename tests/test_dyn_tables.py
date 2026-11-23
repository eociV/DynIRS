import dyn_tables as dt


def test_needs_rescore():
    assert dt.needs_rescore(None, 5.0)
    assert dt.needs_rescore({"a": 1.0}, 5.0)  # entry from before mtimes were recorded
    assert dt.needs_rescore({"_mtime": 4.0}, 5.0)
    assert not dt.needs_rescore({"_mtime": 5.0}, 5.0)
