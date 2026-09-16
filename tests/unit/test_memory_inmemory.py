"""Unit tests for `InMemorySessionRepository` — implementation thật thứ hai
của `SessionRepository` (Protocol), không phải mock. Không cần Postgres, xem
plan4.md phần Verification > Chiến lược test."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from slidegen_rag.memory.dto import CitedNodeRef
from slidegen_rag.memory.memory_store import InMemorySessionRepository
from slidegen_rag.memory.repository import SessionRepository


def _set_last_active(
    repo: InMemorySessionRepository, session_id, when: datetime
) -> None:
    """`last_active_at` được gán bằng `datetime.now()` — quá gần nhau giữa 2
    `create_session()` liên tiếp trong test để phân định thứ tự đáng tin cậy,
    nên set thẳng field nội bộ thay vì `time.sleep()`."""
    repo._sessions[session_id].last_active_at = when


def test_conforms_to_protocol():
    assert isinstance(InMemorySessionRepository(), SessionRepository)


async def test_create_session_defaults():
    repo = InMemorySessionRepository()
    state = await repo.create_session(bound_doc_id="pldc_v2", max_turns=7)
    assert state.bound_doc_id == "pldc_v2"
    assert state.max_turns == 7
    assert state.next_seq == 1
    assert state.compacted_through_seq == 0
    assert state.entries == []
    assert state.last_cited == []


async def test_append_turn_allocates_sequential_seq():
    repo = InMemorySessionRepository()
    state = await repo.create_session()

    user_seq, assistant_seq = await repo.append_turn(
        state.id, raw_query="q1", standalone_query="q1", answer="a1", cited=[]
    )
    assert (user_seq, assistant_seq) == (1, 2)

    user_seq2, assistant_seq2 = await repo.append_turn(
        state.id, raw_query="q2", standalone_query="q2", answer="a2", cited=[]
    )
    assert (user_seq2, assistant_seq2) == (3, 4)

    loaded = await repo.load_session(state.id)
    assert [e.seq for e in loaded.entries] == [1, 2, 3, 4]
    assert loaded.next_seq == 5


async def test_append_turn_stores_citations_for_last_assistant_message_only():
    repo = InMemorySessionRepository()
    state = await repo.create_session()
    ref1 = CitedNodeRef(doc_id="d", source_sha1="s", node_index=3)
    ref2 = CitedNodeRef(doc_id="d", source_sha1="s", node_index=7)

    await repo.append_turn(
        state.id, raw_query="q1", standalone_query="q1", answer="a1", cited=[ref1]
    )
    loaded = await repo.load_session(state.id)
    assert loaded.last_cited == [ref1]

    await repo.append_turn(
        state.id, raw_query="q2", standalone_query="q2", answer="a2", cited=[ref2]
    )
    loaded = await repo.load_session(state.id)
    assert loaded.last_cited == [ref2]  # chỉ lượt gần nhất, không cộng dồn


async def test_save_compaction_sets_watermark_and_hides_old_entries():
    repo = InMemorySessionRepository()
    state = await repo.create_session()
    for i in range(3):
        await repo.append_turn(
            state.id,
            raw_query=f"q{i}",
            standalone_query=f"q{i}",
            answer=f"a{i}",
            cited=[],
        )
    # 3 lượt -> seq 1..6. Nén tới seq=4 (giữ lại lượt thứ 3: seq 5,6).
    await repo.save_compaction(
        state.id, summary="tóm tắt 2 lượt đầu", compacted_through_seq=4
    )

    loaded = await repo.load_session(state.id)
    assert loaded.compacted_through_seq == 4
    assert loaded.history_summary == "tóm tắt 2 lượt đầu"
    assert [e.seq for e in loaded.entries] == [5, 6]


async def test_save_compaction_is_idempotent_never_moves_watermark_backward():
    repo = InMemorySessionRepository()
    state = await repo.create_session()
    for i in range(3):
        await repo.append_turn(
            state.id,
            raw_query=f"q{i}",
            standalone_query=f"q{i}",
            answer=f"a{i}",
            cited=[],
        )

    await repo.save_compaction(state.id, summary="v1", compacted_through_seq=4)
    await repo.save_compaction(state.id, summary="stale retry", compacted_through_seq=2)

    loaded = await repo.load_session(state.id)
    assert loaded.compacted_through_seq == 4
    assert loaded.history_summary == "v1"


async def test_latest_session_filters_by_bound_doc_id_and_active_status():
    repo = InMemorySessionRepository()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    s_a = await repo.create_session(bound_doc_id="docA")
    _set_last_active(repo, s_a.id, base)
    await repo.create_session(bound_doc_id="docB")
    s_a2 = await repo.create_session(bound_doc_id="docA")
    _set_last_active(repo, s_a2.id, base + timedelta(minutes=1))

    latest = await repo.latest_session(bound_doc_id="docA")
    assert latest.id == s_a2.id

    await repo.archive_session(s_a2.id)
    latest_after_archive = await repo.latest_session(bound_doc_id="docA")
    assert latest_after_archive.id == s_a.id


async def test_latest_session_none_when_nothing_matches():
    repo = InMemorySessionRepository()
    assert await repo.latest_session(bound_doc_id="unknown") is None


async def test_load_session_missing_returns_none():
    repo = InMemorySessionRepository()
    assert await repo.load_session(uuid4()) is None


async def test_list_sessions_most_recent_first():
    repo = InMemorySessionRepository()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    s1 = await repo.create_session(bound_doc_id="docA")
    _set_last_active(repo, s1.id, base)
    s2 = await repo.create_session(bound_doc_id="docA")
    _set_last_active(repo, s2.id, base + timedelta(minutes=1))
    rows = await repo.list_sessions(bound_doc_id="docA")
    assert [r.id for r in rows] == [s2.id, s1.id]
