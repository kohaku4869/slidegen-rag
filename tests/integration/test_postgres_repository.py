"""Integration tests cho `PostgresSessionRepository` — hành vi CHỈ đúng
được với Postgres thật (cấp `seq` đồng thời dưới lock, cascade delete), khác
với `tests/unit/test_memory_inmemory.py` vốn chỉ cần
`InMemorySessionRepository`. Xem plan4.md, mục Verification > Chiến lược
test.

Chạy: `TEST_DATABASE_URL=postgresql+asyncpg://... uv run pytest -m integration`
DB test đã migrate: `TEST_DATABASE_URL=... uv run alembic upgrade head`.
Skip hoàn toàn (không fail) nếu thiếu `TEST_DATABASE_URL` — CI trắng không
cần Postgres, xem AGENTS.md > Tests.
"""

import asyncio
import os

import pytest
from sqlalchemy import text

from slidegen_rag.memory.dto import CitedNodeRef
from slidegen_rag.memory.engine import get_sessionmaker
from slidegen_rag.memory.repository import PostgresSessionRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL, reason="cần TEST_DATABASE_URL (xem docstring module)"
    ),
]


@pytest.fixture
async def repo():
    sessionmaker = get_sessionmaker(TEST_DATABASE_URL)
    r = PostgresSessionRepository(sessionmaker)
    yield r
    async with sessionmaker() as db:
        await db.execute(text("TRUNCATE sessions CASCADE"))
        await db.commit()


async def test_concurrent_append_turn_never_duplicates_seq(repo):
    """`SELECT next_seq ... FOR UPDATE` (repository.py) phải tuần tự hoá cấp
    phát seq dưới ghi đồng thời thật — InMemorySessionRepository chỉ mô
    phỏng bằng asyncio.Lock trong 1 tiến trình, không chứng minh được lock ở
    mức DB."""
    state = await repo.create_session(bound_doc_id="concurrency-test")

    async def one_turn(i: int):
        return await repo.append_turn(
            state.id,
            raw_query=f"q{i}",
            standalone_query=f"q{i}",
            answer=f"a{i}",
            cited=[],
        )

    results = await asyncio.gather(*(one_turn(i) for i in range(8)))
    all_seqs = sorted(seq for pair in results for seq in pair)
    assert all_seqs == list(range(1, 17))  # 8 lượt x 2 seq, không trùng, không hụt

    loaded = await repo.load_session(state.id)
    assert loaded.next_seq == 17
    assert len(loaded.entries) == 16


async def test_save_compaction_idempotent_against_concurrent_stale_write(repo):
    state = await repo.create_session(bound_doc_id="compaction-test")
    for i in range(3):
        await repo.append_turn(
            state.id,
            raw_query=f"q{i}",
            standalone_query=f"q{i}",
            answer=f"a{i}",
            cited=[],
        )

    await repo.save_compaction(state.id, summary="v moi nhat", compacted_through_seq=4)
    # Một request cũ (giả lập race) cố ghi đè watermark thấp hơn -> bị bỏ qua.
    await repo.save_compaction(
        state.id, summary="v cu, chua ai thay", compacted_through_seq=2
    )

    loaded = await repo.load_session(state.id)
    assert loaded.compacted_through_seq == 4
    assert loaded.history_summary == "v moi nhat"


async def test_cascade_delete_session_removes_messages_and_citations(repo):
    state = await repo.create_session(bound_doc_id="cascade-test")
    await repo.append_turn(
        state.id,
        raw_query="q",
        standalone_query="q",
        answer="a",
        cited=[CitedNodeRef(doc_id="d", source_sha1="s" * 40, node_index=1)],
    )

    sessionmaker = get_sessionmaker(TEST_DATABASE_URL)
    async with sessionmaker() as db:
        await db.execute(text("DELETE FROM sessions WHERE id = :id"), {"id": state.id})
        await db.commit()

        remaining_messages = (
            await db.execute(
                text("SELECT count(*) FROM messages WHERE session_id = :id"),
                {"id": state.id},
            )
        ).scalar_one()
        remaining_citations = (
            await db.execute(text("SELECT count(*) FROM message_citations"))
        ).scalar_one()

    assert remaining_messages == 0
    assert remaining_citations == 0


async def test_message_citations_survive_char40_roundtrip(repo):
    """source_sha1 là CHAR(40) — Postgres đệm space bên phải khi đọc lại,
    PostgresSessionRepository phải rstrip() để không lệch với sha1 hex thật
    (luôn đúng 40 ký tự, không có khoảng trắng)."""
    sha1 = "a" * 40
    state = await repo.create_session(bound_doc_id="char40-test")
    await repo.append_turn(
        state.id,
        raw_query="q",
        standalone_query="q",
        answer="a",
        cited=[CitedNodeRef(doc_id="d", source_sha1=sha1, node_index=9)],
    )
    loaded = await repo.load_session(state.id)
    assert loaded.last_cited[0].source_sha1 == sha1
