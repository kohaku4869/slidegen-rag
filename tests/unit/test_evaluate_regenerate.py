"""Chế độ "chỉ chạy lại generate" của evaluate.py — xem
evaluate.regenerate_answers_for_topic."""

import json

import pytest

from slidegen_rag.evaluate import evaluate as ev


def test_generate_state_round_trip(tmp_path):
    path = tmp_path / ev.GENERATE_STATE_FILENAME
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
def topic_dir(tmp_path, monkeypatch):
    """Dựng 1 topic tối thiểu (2 câu) trong thư mục tạm và trỏ DATA_TEST_DIR vào đó."""
    topic = tmp_path / "chu_de_test"
    topic.mkdir()
    (topic / "question.txt").write_text(
        "1. Câu hỏi một?\n\n2. Câu hỏi hai?\n", encoding="utf-8"
    )
    (topic / "reference.txt").write_text(
        "-1. Đáp án một\n\n-2. Đáp án hai\n", encoding="utf-8"
    )
    (topic / "chunks.txt").write_text(
        "=== CÂU HỎI 1 ===\n\n[CHUNK 1]\nNội dung chunk A\n\n[CHUNK 2]\nNội dung chunk B\n\n"
        "=== CÂU HỎI 2 ===\n\n[CHUNK 1]\nNội dung chunk C\n\n",
        encoding="utf-8",
    )
    # Bắt buộc phải có, nếu không regenerate_answers_for_topic từ chối chạy.
    ev.save_generate_state(
        topic / ev.GENERATE_STATE_FILENAME,
        {
            1: {"facet_sufficiency": [], "document_priorities": []},
            2: {"facet_sufficiency": [], "document_priorities": []},
        },
    )
    monkeypatch.setattr(ev, "DATA_TEST_DIR", tmp_path)
    return topic


def _stub_generate(monkeypatch, recorder):
    """Thay generate_node bằng stub ghi lại state nhận được, không gọi LLM thật."""
    from slidegen_rag.agent.nodes import generate_node as gn

    async def fake_generate_node(state):
        recorder.append(state)
        return {"generation": f"Trả lời cho: {state['query']}"}

    monkeypatch.setattr(gn, "generate_node", fake_generate_node)


def test_regenerate_reuses_chunks_and_replays_generate_state(topic_dir, monkeypatch):
    (topic_dir / ev.GENERATE_STATE_FILENAME).write_text(
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

    chunks_before = (topic_dir / "chunks.txt").read_text(encoding="utf-8")
    results = ev.regenerate_answers_for_topic("chu_de_test")

    # Chunk từ chunks.txt được nạp lại nguyên vẹn làm documents cho generate.
    assert [s["documents"] for s in seen] == [
        ["Nội dung chunk A", "Nội dung chunk B"],
        ["Nội dung chunk C"],
    ]
    # facet_sufficiency/document_priorities được phát lại đúng như lần chạy đầy đủ
    # — thiếu chúng thì không tái hiện được hành vi rào đón đang cần đo.
    assert seen[0]["facet_sufficiency"] == [
        {"query": "Câu hỏi một?", "sufficient": False}
    ]
    assert seen[0]["document_priorities"] == [2, 1]

    assert [r["response"] for r in results] == [
        "Trả lời cho: Câu hỏi một?",
        "Trả lời cho: Câu hỏi hai?",
    ]
    assert [r["reference"] for r in results] == ["Đáp án một", "Đáp án hai"]

    # answer.txt bị ghi đè, chunks.txt giữ nguyên (chunk là đầu vào cố định).
    assert ev.load_answers(topic_dir / "answer.txt") == {
        1: "Trả lời cho: Câu hỏi một?",
        2: "Trả lời cho: Câu hỏi hai?",
    }
    assert (topic_dir / "chunks.txt").read_text(encoding="utf-8") == chunks_before


def test_regenerate_without_generate_state_refuses_to_run(topic_dir, monkeypatch):
    """Thiếu generate_state.json thì DỪNG, không chạy ra điểm gây hiểu nhầm.

    Không có file này thì mọi câu đều nhận note "ngữ cảnh ĐẦY ĐỦ" — điểm trông
    như so sánh được với run đầy đủ nhưng thực chất đo điều kiện khác hẳn.
    """
    (topic_dir / ev.GENERATE_STATE_FILENAME).unlink()
    seen: list[dict] = []
    _stub_generate(monkeypatch, seen)

    with pytest.raises(FileNotFoundError, match=ev.GENERATE_STATE_FILENAME):
        ev.regenerate_answers_for_topic("chu_de_test")

    assert seen == []  # dừng TRƯỚC khi tốn bất kỳ lượt gọi LLM nào


def test_regenerate_requires_existing_chunks(topic_dir, monkeypatch):
    (topic_dir / "chunks.txt").write_text("", encoding="utf-8")
    _stub_generate(monkeypatch, [])

    with pytest.raises(FileNotFoundError, match="chunks.txt"):
        ev.regenerate_answers_for_topic("chu_de_test")


def test_regenerate_survives_one_failing_question(topic_dir, monkeypatch):
    """1 câu lỗi không được làm hỏng cả lượt chạy — giống đường chạy đầy đủ."""
    from slidegen_rag.agent.nodes import generate_node as gn

    async def flaky_generate_node(state):
        if "hai" in state["query"]:
            raise RuntimeError("LLM sập")
        return {"generation": "OK"}

    monkeypatch.setattr(gn, "generate_node", flaky_generate_node)

    results = ev.regenerate_answers_for_topic("chu_de_test")

    assert results[0]["response"] == "OK"
    assert results[1]["response"].startswith("[LỖI GENERATE]")
