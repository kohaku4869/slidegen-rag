"""Unit tests cho `ConversationManager` (agent/graph/conversation.py) sau khi
thêm persistence tuỳ chọn (plan4.md B3). Dùng `InMemorySessionRepository`
làm fake — không cần Postgres. Bao phủ đúng 5 điều plan4.md liệt kê ở mục
Verification > Chiến lược test: `repo=None` giữ nguyên hành vi cũ, resume,
watermark không gộp lại 2 lần, lọc cited node theo sha1, compaction vẫn đúng
1 LLM call."""

from types import SimpleNamespace

import pytest

from slidegen_rag.agent.graph import conversation as conv_module
from slidegen_rag.agent.graph.conversation import ConversationManager
from slidegen_rag.memory.dto import DocumentRef
from slidegen_rag.memory.memory_store import InMemorySessionRepository


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _fake_llm(content: str):
    calls = {"n": 0}

    async def ainvoke(messages):
        calls["n"] += 1
        return _FakeResponse(content)

    return SimpleNamespace(ainvoke=ainvoke), calls


# ── repo=None: hành vi in-memory hôm nay KHÔNG đổi (guard hồi quy) ─────────


def test_repo_none_record_turn_unchanged():
    conv = ConversationManager(max_turns=5)
    conv.record_turn("câu hỏi 1", "câu trả lời 1", [10, 20, None])
    assert conv.chat_history[0]["role"] == "user"
    assert conv.chat_history[0]["content"] == "câu hỏi 1"
    assert conv.chat_history[1]["content"] == "câu trả lời 1"
    assert conv.cited_node_ids_last_turn == [10, 20]


async def test_repo_none_arecord_turn_behaves_like_record_turn():
    conv = ConversationManager(max_turns=5)
    await conv.arecord_turn("q", "standalone q", "a", [1, 2])
    assert len(conv.chat_history) == 2
    assert conv.cited_node_ids_last_turn == [1, 2]


async def test_repo_none_maybe_compact_still_summarizes(monkeypatch):
    conv = ConversationManager(max_turns=1)
    llm, calls = _fake_llm("tóm tắt")
    monkeypatch.setattr(conv_module, "_compact_llm", llm)
    for i in range(3):
        conv.record_turn(f"q{i}", f"a{i}", [])
    await conv.maybe_compact()
    assert calls["n"] == 1
    assert conv.history_summary == "tóm tắt"
    assert len(conv.chat_history) == 2  # chỉ giữ lại lượt gần nhất (max_turns=1)


# ── arecord_turn: persist qua repo, gán _seq ────────────────────────────────


async def test_arecord_turn_persists_and_sets_seq():
    repo = InMemorySessionRepository()
    state = await repo.create_session(bound_doc_id="pldc_v2")
    doc_ref = DocumentRef(doc_id="pldc_v2", source_sha1="sha-abc")
    conv = ConversationManager(repo=repo, session_id=state.id, doc_ref=doc_ref)

    await conv.arecord_turn("q1", "standalone q1", "a1", [5])

    assert conv.chat_history[0]["_seq"] == 1
    assert conv.chat_history[1]["_seq"] == 2
    loaded = await repo.load_session(state.id)
    assert [e.content for e in loaded.entries] == ["q1", "a1"]
    assert loaded.last_cited[0].node_index == 5
    assert loaded.last_cited[0].doc_id == "pldc_v2"
    assert loaded.last_cited[0].source_sha1 == "sha-abc"


async def test_arecord_turn_db_failure_marks_degraded_but_keeps_chat_going():
    class _BrokenRepo:
        async def append_turn(self, *a, **k):
            raise RuntimeError("DB rớt")

    conv = ConversationManager(repo=_BrokenRepo(), session_id="fake-id")
    await conv.arecord_turn("q1", "q1", "a1", [])
    assert conv._degraded is True
    assert len(conv.chat_history) == 2  # câu trả lời không bị mất

    # Lượt kế tiếp không thử ghi DB lại (không raise lần 2)
    await conv.arecord_turn("q2", "q2", "a2", [])
    assert len(conv.chat_history) == 4


# ── resume(): nạp lại state, watermark, lọc citation theo doc_ref ──────────


async def test_resume_restores_history_and_summary():
    repo = InMemorySessionRepository()
    state = await repo.create_session(bound_doc_id="pldc_v2", max_turns=3)
    doc_ref = DocumentRef(doc_id="pldc_v2", source_sha1="sha-abc")
    for i in range(2):
        await repo.append_turn(
            state.id,
            raw_query=f"q{i}",
            standalone_query=f"q{i}",
            answer=f"a{i}",
            cited=[],
        )
    await repo.save_compaction(state.id, summary="tóm tắt cũ", compacted_through_seq=2)

    conv = await ConversationManager.resume(repo, state.id, doc_ref)

    assert conv.history_summary == "tóm tắt cũ"
    # seq 1,2 đã bị nén -> chỉ còn lượt seq 3,4
    assert [e["_seq"] for e in conv.chat_history] == [3, 4]
    assert conv.chat_history[0]["content"] == "q1"


async def test_resume_unknown_session_raises():
    repo = InMemorySessionRepository()
    with pytest.raises(ValueError):
        await ConversationManager.resume(repo, "khong-ton-tai", DocumentRef("d", "s"))


async def test_resume_filters_cited_nodes_when_doc_rebuilt():
    """Rủi ro quan trọng nhất của plan4.md B1.1: index build lại (sha1 đổi)
    -> citation cũ PHẢI bị bỏ, không được trỏ nhầm node."""
    repo = InMemorySessionRepository()
    state = await repo.create_session(bound_doc_id="pldc_v2")
    old_ref = DocumentRef(doc_id="pldc_v2", source_sha1="sha-OLD")
    await repo.append_turn(
        state.id,
        raw_query="q",
        standalone_query="q",
        answer="a",
        cited=[
            conv_module.CitedNodeRef(
                doc_id="pldc_v2", source_sha1="sha-OLD", node_index=42
            )
        ],
    )

    # Resume nhưng index hiện tại đã build lại -> sha1 mới
    new_ref = DocumentRef(doc_id="pldc_v2", source_sha1="sha-NEW")
    conv = await ConversationManager.resume(repo, state.id, new_ref)
    assert conv.cited_node_ids_last_turn == []

    # Cùng sha1 (chưa rebuild) -> vẫn giữ được citation
    conv_same = await ConversationManager.resume(repo, state.id, old_ref)
    assert conv_same.cited_node_ids_last_turn == [42]


# ── maybe_compact() với repo: watermark ghi đúng 1 lần, đúng 1 LLM call ────


async def test_maybe_compact_persists_watermark_exactly_once(monkeypatch):
    repo = InMemorySessionRepository()
    state = await repo.create_session(bound_doc_id="pldc_v2", max_turns=1)
    doc_ref = DocumentRef(doc_id="pldc_v2", source_sha1="sha-abc")
    conv = ConversationManager(
        max_turns=1, repo=repo, session_id=state.id, doc_ref=doc_ref
    )

    llm, calls = _fake_llm("tóm tắt mới")
    monkeypatch.setattr(conv_module, "_compact_llm", llm)

    for i in range(3):
        await conv.arecord_turn(f"q{i}", f"q{i}", f"a{i}", [])
    await conv.maybe_compact()

    assert calls["n"] == 1
    loaded = await repo.load_session(state.id)
    assert loaded.history_summary == "tóm tắt mới"
    assert loaded.compacted_through_seq == 4  # giữ lại lượt cuối (seq 5,6)

    # Gọi lại maybe_compact khi chưa đủ ngưỡng lần nữa -> không nén thêm,
    # không gọi LLM thêm lần nào.
    await conv.maybe_compact()
    assert calls["n"] == 1
