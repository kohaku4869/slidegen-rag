"""scripts/migrate_eval_runs.py — thuần thao tác file, test bằng tmp_path
(không đụng data/test/ thật). Xem plan5.md Phần C4, bước verification 5-7."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_eval_runs.py"
_spec = importlib.util.spec_from_file_location("migrate_eval_runs", _SCRIPT_PATH)
mig = importlib.util.module_from_spec(_spec)
sys.modules["migrate_eval_runs"] = mig
_spec.loader.exec_module(mig)


def _write_topic(tmp_path, topic="pldc_v2"):
    d = tmp_path / topic
    d.mkdir()
    (d / "question.txt").write_text("1. Q1?\n\n2. Q2?\n", encoding="utf-8")
    (d / "reference.txt").write_text("-1. R1\n\n-2. R2\n", encoding="utf-8")
    return d


def _write_pair(topic_dir, ts, *, responses, summary_extra=None):
    csv_lines = ["STT,topic,question,response,reference"]
    for stt, resp in responses.items():
        csv_lines.append(f'{stt},{topic_dir.name},"Q{stt}?","{resp}","R{stt}"')
    (topic_dir / f"scores_{ts}.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8-sig"
    )
    summary = {
        "topic": topic_dir.name,
        "timestamp": ts,
        "total_score": 5.0,
        "total_questions": len(responses),
    }
    if summary_extra:
        summary.update(summary_extra)
    (topic_dir / f"summary_{ts}.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _patch_data_test_dir(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs as runs_mod
    from slidegen_rag.evaluate import settings as settings_mod

    monkeypatch.setattr(runs_mod, "DATA_TEST_DIR", tmp_path)
    monkeypatch.setattr(settings_mod, "DATA_TEST_DIR", tmp_path)
    return tmp_path


def test_migrate_moves_pairs_and_writes_run_config(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "cũ 1", 2: "cũ 2"})
    _write_pair(
        topic_dir,
        "20260911_100000",
        responses={1: "mới 1", 2: "mới 2"},
        summary_extra={"generate_provider": "gemini"},
    )
    (topic_dir / "answer.txt").write_text(
        "-1. mới 1\n\n-2. mới 2\n", encoding="utf-8"
    )
    (topic_dir / "chunks.txt").write_text("=== CÂU HỎI 1 ===\n\n", encoding="utf-8")

    migrated = mig.migrate_one_topic(topic_dir)
    assert migrated is True

    from slidegen_rag.evaluate import runs as runs_mod

    run_ids = runs_mod.list_run_ids("pldc_v2")
    assert set(run_ids) == {"20260910_100000", "20260911_100000"}

    # scores_<ts>.csv/summary_<ts>.json phẳng đã bị MOVE đi, không còn nữa.
    assert not (topic_dir / "scores_20260910_100000.csv").exists()
    assert not (topic_dir / "summary_20260911_100000.json").exists()

    rp_new = runs_mod.RunPaths.for_run("pldc_v2", "20260911_100000")
    assert rp_new.scores.exists()
    assert rp_new.summary.exists()
    cfg_new = json.loads(rp_new.run_config.read_text(encoding="utf-8"))
    assert cfg_new["migrated"] is True
    assert cfg_new["mode"] == "unknown"
    assert cfg_new["config_fingerprint"] is None
    assert cfg_new["generate_provider"] == "gemini"

    rp_old = runs_mod.RunPaths.for_run("pldc_v2", "20260910_100000")
    cfg_old = json.loads(rp_old.run_config.read_text(encoding="utf-8"))
    assert cfg_old["generate_provider"] is None  # summary cũ không có field này


def test_winner_run_gets_artifacts_copied_by_evidence(tmp_path):
    """answer.txt hiện tại khớp scores.csv của ts MỚI HƠN (không phải ts mới
    nhất theo tên) -> ts đó phải được gán artifacts, không phải ts mới nhất
    theo tên file."""
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "sai 1", 2: "sai 2"})
    _write_pair(topic_dir, "20260911_100000", responses={1: "đúng 1", 2: "đúng 2"})
    (topic_dir / "answer.txt").write_text(
        "-1. đúng 1\n\n-2. đúng 2\n", encoding="utf-8"
    )
    (topic_dir / "chunks.txt").write_text("nội dung chunk", encoding="utf-8")

    mig.migrate_one_topic(topic_dir)

    from slidegen_rag.evaluate import runs as runs_mod

    rp_winner = runs_mod.RunPaths.for_run("pldc_v2", "20260911_100000")
    cfg_winner = json.loads(rp_winner.run_config.read_text(encoding="utf-8"))
    assert cfg_winner["artifacts_verified"] is True
    assert rp_winner.answer.exists()
    assert rp_winner.chunks.exists()

    rp_loser = runs_mod.RunPaths.for_run("pldc_v2", "20260910_100000")
    cfg_loser = json.loads(rp_loser.run_config.read_text(encoding="utf-8"))
    assert cfg_loser["artifacts_verified"] is False
    assert cfg_loser["artifacts"] == "missing"
    assert not rp_loser.answer.exists()

    notes = (runs_mod.runs_dir("pldc_v2") / "MIGRATION_NOTES.md").read_text(
        encoding="utf-8"
    )
    assert "20260910_100000" in notes


def test_no_run_matches_answers_falls_back_to_newest_unverified(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "a", 2: "b"})
    _write_pair(topic_dir, "20260911_100000", responses={1: "c", 2: "d"})
    # answer.txt hiện tại khớp KHÔNG run nào (dữ liệu bị các lần chạy sau đè).
    (topic_dir / "answer.txt").write_text(
        "-1. hoàn toàn khác\n\n-2. hoàn toàn khác\n", encoding="utf-8"
    )

    mig.migrate_one_topic(topic_dir)

    from slidegen_rag.evaluate import runs as runs_mod

    rp_newest = runs_mod.RunPaths.for_run("pldc_v2", "20260911_100000")
    cfg = json.loads(rp_newest.run_config.read_text(encoding="utf-8"))
    assert cfg["artifacts_verified"] is False
    assert "artifacts" not in cfg or cfg.get("artifacts") != "missing"


def test_unpaired_ts_still_migrated_with_artifacts_complete_false(tmp_path):
    topic_dir = _write_topic(tmp_path)
    # Chỉ có scores.csv, KHÔNG có summary.json cho ts này.
    _write_pair(topic_dir, "20260910_100000", responses={1: "x"})
    (topic_dir / "summary_20260910_100000.json").unlink()

    mig.migrate_one_topic(topic_dir)

    from slidegen_rag.evaluate import runs as runs_mod

    rp = runs_mod.RunPaths.for_run("pldc_v2", "20260910_100000")
    assert rp.scores.exists()
    assert not rp.summary.exists()
    cfg = json.loads(rp.run_config.read_text(encoding="utf-8"))
    assert cfg["artifacts_complete"] is False


def test_idempotent_second_run_is_noop(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "a"})

    first = mig.migrate_one_topic(topic_dir)
    second = mig.migrate_one_topic(topic_dir)

    assert first is True
    assert second is False  # không còn gì phẳng để migrate


def test_dry_run_touches_nothing(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "a"})

    mig.migrate_one_topic(topic_dir, dry_run=True)

    assert (topic_dir / "scores_20260910_100000.csv").exists()
    assert (topic_dir / "summary_20260910_100000.json").exists()
    from slidegen_rag.evaluate import runs as runs_mod

    assert runs_mod.list_run_ids("pldc_v2") == []


def test_verify_detects_sha1_drift(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "a"})
    mig.migrate_one_topic(topic_dir)

    assert mig.verify_topic(topic_dir) is True

    from slidegen_rag.evaluate import runs as runs_mod

    rp = runs_mod.RunPaths.for_run("pldc_v2", "20260910_100000")
    rp.scores.write_text("tampered", encoding="utf-8")

    assert mig.verify_topic(topic_dir) is False


def test_rollback_restores_flat_layout(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "a"})
    before_csv = (topic_dir / "scores_20260910_100000.csv").read_text(encoding="utf-8")
    before_json = (topic_dir / "summary_20260910_100000.json").read_text(
        encoding="utf-8"
    )

    mig.migrate_one_topic(topic_dir)
    rolled = mig.rollback_topic(topic_dir)

    assert rolled is True
    assert (topic_dir / "scores_20260910_100000.csv").read_text(
        encoding="utf-8"
    ) == before_csv
    assert (topic_dir / "summary_20260910_100000.json").read_text(
        encoding="utf-8"
    ) == before_json

    from slidegen_rag.evaluate import runs as runs_mod

    assert runs_mod.list_run_ids("pldc_v2") == []


def test_rollback_refuses_when_sha1_drifted(tmp_path):
    topic_dir = _write_topic(tmp_path)
    _write_pair(topic_dir, "20260910_100000", responses={1: "a"})
    mig.migrate_one_topic(topic_dir)

    from slidegen_rag.evaluate import runs as runs_mod

    rp = runs_mod.RunPaths.for_run("pldc_v2", "20260910_100000")
    rp.scores.write_text("changed after migration", encoding="utf-8")

    with pytest.raises(mig.MigrationError):
        mig.rollback_topic(topic_dir)

    # Không move gì cả khi phát hiện lệch — file vẫn ở nguyên chỗ vừa sửa.
    assert rp.scores.read_text(encoding="utf-8") == "changed after migration"
