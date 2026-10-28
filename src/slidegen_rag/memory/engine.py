"""
engine.py — Async SQLAlchemy engine cho `MemoryConfig.DATABASE_URL`.

`ping()` gọi ở CLI startup (main.py) để fail loudly nếu `DATABASE_URL` được
khai nhưng không kết nối được — memory KHÔNG theo quy ước fail-soft của
observability trong repo này (xem plan4.md B4): mất một trace LangSmith
không mất thứ người dùng yêu cầu, nhưng chat 20 lượt rồi phát hiện chẳng gì
được lưu thì tệ hơn crash ngay lúc khởi động.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from slidegen_rag.config import MemoryConfig

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine(database_url: str | None = None) -> AsyncEngine:
    global _engine
    url = database_url or MemoryConfig.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL chưa được cấu hình — không thể tạo engine.")
    if _engine is None or database_url is not None:
        engine = create_async_engine(
            url,
            pool_size=MemoryConfig.POOL_SIZE,
            pool_pre_ping=True,  # bỏ connection đã bị Postgres đóng thay vì văng "connection is closed"
            connect_args={"timeout": MemoryConfig.CONNECT_TIMEOUT_S},
        )
        if database_url is not None:
            return (
                engine  # engine tạm cho test (TEST_DATABASE_URL), không cache toàn cục
            )
        _engine = engine
    return _engine


def get_sessionmaker(
    database_url: str | None = None,
) -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if database_url is not None:
        return async_sessionmaker(get_engine(database_url), expire_on_commit=False)
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def ping(database_url: str | None = None) -> None:
    """Xác nhận kết nối được tới DATABASE_URL. Lỗi ở đây phải văng lên cho
    caller xử lý fail loudly (sys.exit), không nuốt exception."""
    engine = get_engine(database_url)
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
