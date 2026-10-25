"""Fixture dùng chung cho toàn bộ test.

Lý do tồn tại: agent giữ khá nhiều state ở cấp module — index đang dùng
(`tools._ra`), cờ bật/tắt web search, provider của generate_node, và 2 cache
LLM. Chúng được đặt bằng hàm setter và KHÔNG được khôi phục: ví dụ
`evaluate/runner.py` gọi `set_web_search_enabled(False)` cho benchmark
closed-book rồi để nguyên như vậy, nên trong cùng một tiến trình, mọi thứ chạy
sau đó đều mất web search. Trong test, kiểu rò rỉ này còn khó chịu hơn: thứ tự
chạy đổi là kết quả đổi, và lỗi hiện ra ở test KHÁC với test gây ra nó.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_agent_globals():
    """Chụp lại state cấp module trước mỗi test và trả về nguyên trạng sau đó."""
    from slidegen_rag.agent.models.llm import LLMModel
    from slidegen_rag.agent.nodes import generate_node, retrieve_node
    from slidegen_rag.agent.tools import tools

    saved = (
        tools._ra,
        tools._web_search_enabled,
        generate_node._provider,
        dict(generate_node._llm_cache),
        dict(retrieve_node._llm_cache),
        LLMModel.tracker.snapshot(),
    )

    yield

    (
        tools._ra,
        tools._web_search_enabled,
        generate_node._provider,
        gen_cache,
        ret_cache,
        tracker_snapshot,
    ) = saved
    LLMModel.tracker.reset()
    for key, value in tracker_snapshot.items():
        setattr(LLMModel.tracker, key, value)
    # Gán lại NỘI DUNG chứ không thay object: có module đã giữ tham chiếu tới
    # chính dict này từ lúc import, thay object sẽ khiến chúng trỏ vào bản cũ.
    generate_node._llm_cache.clear()
    generate_node._llm_cache.update(gen_cache)
    retrieve_node._llm_cache.clear()
    retrieve_node._llm_cache.update(ret_cache)
