"""Chế độ "chỉ chạy lại generate" — xem runner.regenerate_answers_for_topic.

Sau plan5.md Phần C, chunks.txt/generate_state.json/trajectory.txt không còn
nằm phẳng trong data/test/<topic>/ mà ở một RUN NGUỒN dưới
data/test/<topic>/runs/<run_id>/ — regenerate_answers_for_topic đọc từ đó
(qua resolve_source_run) và ghi answer.txt MỚI vào run_dir riêng.
"""

import json

import pytest

from slidegen_rag.evaluate import io as ev
from slidegen_rag.evaluate import runner
from slidegen_rag.evaluate import runs
from slidegen_rag.evaluate.runs import RunPaths
from slidegen_rag.evaluate.settings import GENERATE_STATE_FILENAME


def test_generate_state_round_trip(tmp_path):
    path = tmp_path / GENERATE_STATE_FILENAME
    states = {
        2: {
            "facet_sufficiency": [{"query": "b?", "sufficient": True}],
            "document_priorities": [],
        },
        1: {
            "facet_sufficiency": [
                {"query": "a?", "sufficient": False, "missing_aspects": ["thiếu x"]}
            ],
            "document_priorities": [2, 1, 0],
        },
    }
    ev.save_generate_state(path, states)

    loaded = ev.load_generate_state(path)
    # Khoá JSON là string, load phải trả về int để khớp `num` của load_questions.
    assert loaded == states
    assert all(isinstance(k, int) for k in loaded)


def test_load_generate_state_missing_or_corrupt_returns_empty(tmp_path):
    assert ev.load_generate_state(tmp_path / "không-tồn-tại.json") == {}

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert ev.load_generate_state(empty) == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{không phải json", encoding="utf-8")
    assert ev.load_generate_state(corrupt) == {}


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    """Dựng 1 topic tối thiểu (2 câu) + 1 run NGUỒN sẵn chunks/generate_state
    trong thư mục tạm, trỏ DATA_TEST_DIR của evaluate.runs (KHÔNG phải
    evaluate.runner — sau plan5.md Phần C, path nằm ở runs.py) vào đó."""
    topic_dir = tmp_path / "chu_de_test"
    topic_dir.mkdir()
    (topic_dir / "question.txt").write_text(
        "1. Câu hỏi một?\n\n2. Câu hỏi hai?\n", encoding="utf-8"
    )
    (topic_dir / "reference.txt").write_text(
        "-1. Đáp án một\n\n-2. Đáp án hai\n", encoding="utf-8"
    )

    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)

    rp = RunPaths.for_new_run("chu_de_test")
    rp.chunks.write_text(
        "=== CÂU HỎI 1 ===\n\n[CHUNK 1]\nNội dung chunk A\n\n[CHUNK 2]\nNội dung chunk B\n\n"
        "=== CÂU HỎI 2 ===\n\n[CHUNK 1]\nNội dung chunk C\n\n",
        encoding="utf-8",
    )
    # Bắt buộc phải có, nếu không regenerate_answers_for_topic từ chối chạy.
    ev.save_generate_state(
        rp.generate_state,
        {
            1: {"facet_sufficiency": [], "document_priorities": []},
            2: {"facet_sufficiency": [], "document_priorities": []},
        },
    )
    rp.summary.write_text("{}", encoding="utf-8")
    runs.write_latest_pointer("chu_de_test", rp.run_id)
    return rp


def _stub_generate(monkeypatch, recorder):
    """Thay generate_node bằng stub ghi lại state nhận được, không gọi LLM thật."""
    from slidegen_rag.agent.nodes import generate_node as gn

    async def fake_generate_node(state):
        recorder.append(state)
        return {"generation": f"Trả lời cho: {state['query']}"}

    monkeypatch.setattr(gn, "generate_node", fake_generate_node)


def test_regenerate_reuses_chunks_and_replays_generate_state(source_run, monkeypatch):
    source_run.generate_state.write_text(
        json.dumps(
            {
                "1": {
                    "facet_sufficiency": [
                        {"query": "Câu hỏi một?", "sufficient": False}
                    ],
                    "document_priorities": [2, 1],
                },
                "2": {"facet_sufficiency": [], "document_priorities": []},
            }
        ),
        encoding="utf-8",
    )
    seen: list[dict] = []
    _stub_generate(monkeypatch, seen)

    chunks_before = source_run.chunks.read_text(encoding="utf-8")
    results, resolved_source_id = runner.regenerate_answers_for_topic("chu_de_test")

    assert resolved_source_id == source_run.run_id

    # Chunk từ chunks.txt được nạp lại nguyên vẹn làm documents cho generate.
    assert [s["documents"] for s in seen] == [
        ["Nội dung chunk A", "Nội dung chunk B"],
        ["Nội dung chunk C"],
    ]
    # facet_sufficiency/document_priorities được phát lại đúng như lần chạy đầy đủ.
    assert seen[0]["facet_sufficiency"] == [
        {"query": "Câu hỏi một?", "sufficient": False}
    ]
    assert seen[0]["document_priorities"] == [2, 1]

    assert [r["response"] for r in results] == [
        "Trả lời cho: Câu hỏi một?",
        "Trả lời cho: Câu hỏi hai?",
    ]
    assert [r["reference"] for r in results] == ["Đáp án một", "Đáp án hai"]

    # answer.txt MỚI nằm trong run MỚI (không phải run nguồn); chunks.txt của
    # run nguồn giữ nguyên (chunk là đầu vào cố định).
    new_run_dir = next(
        p
        for p in runs.runs_dir("chu_de_test").iterdir()
        if p.name != source_run.run_id
    )
    assert ev.load_answers(new_run_dir / "answer.txt") == {
        1: "Trả lời cho: Câu hỏi một?",
        2: "Trả lời cho: Câu hỏi hai?",
    }
    assert source_run.chunks.read_text(encoding="utf-8") == chunks_before
    # Run mới vẫn tự chứa — chunks.txt được copy sang, không chỉ answer.txt.
    assert (new_run_dir / "chunks.txt").read_text(encoding="utf-8") == chunks_before


def test_regenerate_without_any_source_run_refuses_to_run(tmp_path, monkeypatch):
    """Chưa có run nguồn nào có generate_state.json + chunks.txt thì DỪNG,
    không chạy ra điểm gây hiểu nhầm."""
    topic_dir = tmp_path / "chu_de_trong"
    topic_dir.mkdir()
    (topic_dir / "question.txt").write_text("1. Câu hỏi một?\n", encoding="utf-8")
    (topic_dir / "reference.txt").write_text("-1. Đáp án một\n", encoding="utf-8")
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)

    seen: list[dict] = []
    _stub_generate(monkeypatch, seen)

    with pytest.raises(FileNotFoundError, match=GENERATE_STATE_FILENAME):
        runner.regenerate_answers_for_topic("chu_de_trong")

    assert seen == []  # dừng TRƯỚC khi tốn bất kỳ lượt gọi LLM nào


def test_regenerate_requires_existing_chunks(source_run, monkeypatch):
    # File rỗng -> load_chunks() trả {} -> predicate resolve_source_run loại
    # ngay run này (any({}.values()) == False).
    source_run.chunks.write_text("", encoding="utf-8")
    _stub_generate(monkeypatch, [])

    with pytest.raises(FileNotFoundError, match="không run nào thỏa điều kiện"):
        runner.regenerate_answers_for_topic("chu_de_test")


def test_regenerate_survives_one_failing_question(source_run, monkeypatch):
    """1 câu lỗi không được làm hỏng cả lượt chạy — giống đường chạy đầy đủ."""
    from slidegen_rag.agent.nodes import generate_node as gn

    async def flaky_generate_node(state):
        if "hai" in state["query"]:
            raise RuntimeError("LLM sập")
        return {"generation": "OK"}

    monkeypatch.setattr(gn, "generate_node", flaky_generate_node)

    results, _source_id = runner.regenerate_answers_for_topic("chu_de_test")

    assert results[0]["response"] == "OK"
    assert results[1]["response"].startswith("[LỖI GENERATE]")


def test_regenerate_honors_explicit_source_run_id(tmp_path, monkeypatch):
    """Chỉ định source_run_id tường minh phải thắng LATEST.txt."""
    topic_dir = tmp_path / "chu_de_test"
    topic_dir.mkdir()
    (topic_dir / "question.txt").write_text("1. Câu hỏi một?\n", encoding="utf-8")
    (topic_dir / "reference.txt").write_text("-1. Đáp án một\n", encoding="utf-8")
    monkeypatch.setattr(runs, "DATA_TEST_DIR", tmp_path)

    def _make_source(run_id_suffix, chunk_text):
        import datetime as dt

        rp = RunPaths.for_new_run(
            "chu_de_test", now=dt.datetime(2026, 1, 1, 0, 0, run_id_suffix)
        )
        rp.chunks.write_text(
            f"=== CÂU HỎI 1 ===\n\n[CHUNK 1]\n{chunk_text}\n\n", encoding="utf-8"
        )
        ev.save_generate_state(
            rp.generate_state, {1: {"facet_sufficiency": [], "document_priorities": []}}
        )
        rp.summary.write_text("{}", encoding="utf-8")
        return rp

    old_run = _make_source(1, "chunk cũ")
    new_run = _make_source(2, "chunk mới")
    runs.write_latest_pointer("chu_de_test", new_run.run_id)

    seen: list[dict] = []
    _stub_generate(monkeypatch, seen)

    _results, resolved_id = runner.regenerate_answers_for_topic(
        "chu_de_test", source_run_id=old_run.run_id
    )

    assert resolved_id == old_run.run_id
    assert seen[0]["documents"] == ["chunk cũ"]
