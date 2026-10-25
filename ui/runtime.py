"""
runtime.py — `AgentRuntime`: một event loop asyncio bất tử chạy trong thread
nền, dùng chung cho toàn bộ process Streamlit (xem plan6.md A1).

Vì sao cần: `memory/engine.py` giữ `_engine`/`_sessionmaker` ở module-global
và không bao giờ invalidate — engine asyncpg bị buộc vào loop đã tạo ra nó
(xem `main.py::_async_entry`). Streamlit rerun toàn bộ script mỗi lần bấm
nút; `asyncio.run(...)` mỗi rerun sẽ tạo loop MỚI rồi đóng nó khi xong, nên
lần rerun sau dùng lại engine cũ qua loop đã đóng -> treo hoặc
`RuntimeError: Event loop is closed`. Mọi coroutine chạy trên CÙNG MỘT loop
sống suốt đời process giải quyết vấn đề này — landmine biến thành tính năng.

CẤM dùng `asyncio.run(...)` trong script thread của Streamlit. CẤM import
`slidegen_rag.evaluate.runner` từ package `ui/` (nó gọi
`LLMModel.reset_usage()` và có `asyncio.run` riêng trong vòng lặp — xung đột
với runtime loop này).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


class AgentRuntime:
    """Một event loop chạy `run_forever()` trong thread daemon riêng. Mọi
    coroutine của agent/memory phải đi qua `submit()`/`run()` — không bao giờ
    gọi `asyncio.run()` hay `.result()` của coroutine trực tiếp từ script
    thread của Streamlit."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, name="agent-runtime-loop", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        self.agent_lock: asyncio.Lock = self.run(self._make_lock())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    async def _make_lock(self) -> asyncio.Lock:
        # Phải tạo TRÊN loop này (asyncio.Lock() gắn với loop hiện hành lúc
        # await đầu tiên) — tạo ở thread Streamlit sẽ gắn nhầm loop.
        return asyncio.Lock()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def submit(self, coro: Coroutine[object, object, T]) -> concurrent.futures.Future:
        """Đưa coroutine vào runtime loop, trả về ngay một
        `concurrent.futures.Future` — không block thread gọi."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run(self, coro: Coroutine[object, object, T], timeout: float | None = None) -> T:
        """Chạy coroutine trên runtime loop và block tới khi xong — dùng cho
        các thao tác ngắn (ping DB, load danh sách project/session)."""
        return self.submit(coro).result(timeout=timeout)
