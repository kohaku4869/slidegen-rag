"""run_config.json schema + config_fingerprint (evaluate/run_config.py)."""

from slidegen_rag.evaluate import run_config as rc
from slidegen_rag.evaluate.runs import RunPaths


def _base_kwargs(run_paths, **overrides):
    kwargs = dict(
        run_paths=run_paths,
        topic="chu_de_test",
        started_at="2026-09-16T20:45:00",
        finished_at="2026-09-16T20:50:00",
        mode="full",
        generate_provider="gemini",
        lowcost=False,
        max_turns=5,
        web_search_enabled=False,
        max_retrieve=5,
        gemini_model="gemini-3.5-flash-lite",
        local_base_url="http://localhost:8000",
        index_path=run_paths.root / "fake.pkl",
        index_source_sha1="abc123",
        index_pkl_sha1=None,
        questions_sha1="qsha",
        references_sha1="rsha",
        total_questions=30,
    )
    kwargs.update(overrides)
    return kwargs


def test_build_run_config_has_stable_fingerprint(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    cfg1 = rc.build_run_config(**_base_kwargs(rp))
    cfg2 = rc.build_run_config(**_base_kwargs(rp))

    assert cfg1["config_fingerprint"] == cfg2["config_fingerprint"]
    assert cfg1["config_fingerprint"] is not None
    assert cfg1["schema_version"] == 1


def test_fingerprint_changes_when_fingerprint_field_changes(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    cfg_gemini = rc.build_run_config(**_base_kwargs(rp, generate_provider="gemini"))
    cfg_local = rc.build_run_config(**_base_kwargs(rp, generate_provider="local"))

    assert cfg_gemini["config_fingerprint"] != cfg_local["config_fingerprint"]


def test_fingerprint_unaffected_by_non_fingerprint_fields(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    cfg1 = rc.build_run_config(**_base_kwargs(rp, finished_at="2026-09-16T21:00:00"))
    cfg2 = rc.build_run_config(**_base_kwargs(rp, finished_at="2026-09-16T22:00:00"))

    assert cfg1["config_fingerprint"] == cfg2["config_fingerprint"]


def test_migrated_run_has_null_fingerprint(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    cfg = rc.build_run_config(**_base_kwargs(rp, migrated=True))
    assert cfg["config_fingerprint"] is None
    assert cfg["migrated"] is True


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")
    cfg = rc.build_run_config(**_base_kwargs(rp))

    rc.save_run_config(rp, cfg)
    loaded = rc.load_run_config(rp)

    assert loaded == cfg


def test_load_run_config_missing_returns_none(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    assert rc.load_run_config(rp) is None
