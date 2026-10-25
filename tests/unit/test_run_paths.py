"""RunPaths + hằng số run directory (evaluate/runs.py) — xem plan5.md Phần C."""

from slidegen_rag.evaluate import runs


def test_for_new_run_creates_directory_and_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)

    rp = runs.RunPaths.for_new_run("chu_de_test")

    assert rp.root.exists()
    assert rp.root == tmp_path / "chu_de_test" / "runs" / rp.run_id
    assert rp.answer == rp.root / "answer.txt"
    assert rp.chunks == rp.root / "chunks.txt"
    assert rp.summary == rp.root / "summary.json"
    assert rp.scores == rp.root / "scores.csv"
    assert rp.run_config == rp.root / "run_config.json"


def test_for_new_run_dedupes_same_second_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    import datetime as dt

    now = dt.datetime(2026, 9, 16, 20, 45, 0)

    rp1 = runs.RunPaths.for_new_run("chu_de_test", now=now)
    rp2 = runs.RunPaths.for_new_run("chu_de_test", now=now)
    rp3 = runs.RunPaths.for_new_run("chu_de_test", now=now)

    assert rp1.run_id == "20260916_204500"
    assert rp2.run_id == "20260916_204500_2"
    assert rp3.run_id == "20260916_204500_3"
    assert rp1.root != rp2.root != rp3.root


def test_list_run_ids_sorted_newest_first_and_skips_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    d = runs.runs_dir("chu_de_test")
    d.mkdir(parents=True)
    for run_id in ["20260101_000000", "20260301_000000", "20260201_000000"]:
        (d / run_id).mkdir()
        (d / run_id / "summary.json").write_text("{}", encoding="utf-8")
    # Run dở dang (không có summary.json) không được liệt kê.
    (d / "20260401_000000").mkdir()

    assert runs.list_run_ids("chu_de_test") == [
        "20260301_000000",
        "20260201_000000",
        "20260101_000000",
    ]


def test_resolve_latest_reads_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = runs.RunPaths.for_new_run("chu_de_test")
    rp.summary.write_text("{}", encoding="utf-8")
    runs.write_latest_pointer("chu_de_test", rp.run_id)

    assert runs.resolve_latest("chu_de_test") == rp.run_id


def test_resolve_latest_falls_back_to_max_when_pointer_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    d = runs.runs_dir("chu_de_test")
    d.mkdir(parents=True)
    for run_id in ["20260101_000000", "20260301_000000"]:
        (d / run_id).mkdir()
        (d / run_id / "summary.json").write_text("{}", encoding="utf-8")

    assert runs.resolve_latest("chu_de_test") == "20260301_000000"


def test_resolve_latest_falls_back_when_pointer_targets_missing_run(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    d = runs.runs_dir("chu_de_test")
    d.mkdir(parents=True)
    (d / "20260101_000000").mkdir()
    (d / "20260101_000000" / "summary.json").write_text("{}", encoding="utf-8")
    (d / "LATEST.txt").write_text("20269999_000000\n", encoding="utf-8")  # run "ma"

    assert runs.resolve_latest("chu_de_test") == "20260101_000000"


def test_write_latest_pointer_is_atomic_no_leftover_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    runs.runs_dir("chu_de_test").mkdir(parents=True)

    runs.write_latest_pointer("chu_de_test", "20260916_204500")

    pointer = runs.runs_dir("chu_de_test") / "LATEST.txt"
    assert pointer.read_text(encoding="utf-8").strip() == "20260916_204500"
    assert not (runs.runs_dir("chu_de_test") / "LATEST.txt.tmp").exists()


def test_for_run_does_not_require_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = runs.RunPaths.for_run("chu_de_test", "20260101_000000")
    assert not rp.root.exists()
    assert rp.run_id == "20260101_000000"


def _make_run(tmp_path, topic, run_id, *, chunks_content="x"):
    d = runs.runs_dir(topic)
    d.mkdir(parents=True, exist_ok=True)
    rp = runs.RunPaths.for_run(topic, run_id)
    rp.root.mkdir(parents=True, exist_ok=True)
    rp.summary.write_text("{}", encoding="utf-8")
    rp.chunks.write_text(chunks_content, encoding="utf-8")
    return rp


def test_resolve_source_run_prefers_explicit_over_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    _make_run(tmp_path, "t", "20260101_000000")
    _make_run(tmp_path, "t", "20260201_000000")
    runs.write_latest_pointer("t", "20260201_000000")

    run_id, reason = runs.resolve_source_run("t", explicit_run_id="20260101_000000")
    assert run_id == "20260101_000000"
    assert "chỉ định" in reason


def test_resolve_source_run_falls_back_to_predicate_when_latest_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    _make_run(tmp_path, "t", "20260101_000000", chunks_content="has-chunks")
    _make_run(tmp_path, "t", "20260201_000000", chunks_content="")  # latest, empty
    runs.write_latest_pointer("t", "20260201_000000")

    def predicate(rp):
        return bool(rp.chunks.read_text(encoding="utf-8").strip())

    run_id, reason = runs.resolve_source_run("t", predicate=predicate)
    assert run_id == "20260101_000000"
    assert "LATEST.txt" in reason


def test_resolve_source_run_reports_reason_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    _make_run(tmp_path, "t", "20260101_000000", chunks_content="")

    run_id, reason = runs.resolve_source_run(
        "t", predicate=lambda rp: bool(rp.chunks.read_text(encoding="utf-8").strip())
    )
    assert run_id is None
    assert "20260101_000000" in reason


def test_resolve_source_run_no_runs_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    run_id, reason = runs.resolve_source_run("t_empty")
    assert run_id is None
    assert "chưa có run nào" in reason
