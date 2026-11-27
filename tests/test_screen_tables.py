import screen_tables as st


def test_table_lines_means_and_paired_diffs():
    keys = st.KEYS
    scores = {f"dyn_cl_{c}_s{s}": {k: base + s * 0.1 for k in keys}
              for c, base in (("B", 10.0), ("P", 11.0)) for s in (0, 1, 2)}
    lines = st.table_lines(scores, {"dup": 1, "invisible": 2, "faint": 3, "visible": 4})
    text = "\n".join(lines)
    assert "| B | 3 | 10.10 ± 0.10" in text
    assert "| P | 3 | 11.10 ± 0.10" in text
    assert "## Seed-paired differences to B" in text and "| P | +1.00 [" in text
    assert "| I |" not in text  # no runs of I


def test_image_epochs(tmp_path):
    import json

    import screen_tables as stt

    a = tmp_path / "a"
    a.mkdir()
    (a / "proxy_ids.json").write_text(json.dumps(["x"] * 100))
    (a / "results.csv").write_text("epoch\n" + "1\n" * 15)
    assert stt.image_epochs(a) == 1500
    b = tmp_path / "b"
    b.mkdir()
    (b / "train_images.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [{"from": 0, "to": 5, "n_images": 100},
                                               {"from": 5, "to": 10, "n_images": 120},
                                               {"from": 10, "to": 15, "n_images": 130}]))
    assert stt.image_epochs(b) == 1750
