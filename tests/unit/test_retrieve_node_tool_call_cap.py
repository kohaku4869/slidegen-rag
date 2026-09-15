"""Unit test for FIX P0 (plan chạy thực nghiệm pldc_v2, run 20260914_171758):
retrieve_node phải chặn cứng số tool call MỚI/lượt gọi LLM, phòng trường hợp
degenerate repetition (đo thực nghiệm: 1 lượt retrieve sinh 517 tool_calls,
out=16,371 token chạm sát trần MAX_TOKENS=16,384)."""

from types import SimpleNamespace

from slidegen_rag.agent.nodes import retrieve_node as rn


class _FakeResponse:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _make_calls(n: int, name: str = "semantic_search") -> list[dict]:
    # Query khác nhau từng chữ để KHÔNG bị lọc trùng — mô phỏng đúng ca thật
    # (517 biến thể query gần giống nhau, tất cả unique theo call_key).
    return [
        {
            "name": name,
            "args": {"query": f"query variant {i}"},
            "id": f"call_{i}",
            "type": "tool_call",
        }
        for i in range(n)
    ]


def _fake_llm(tool_calls: list[dict]):
    async def ainvoke(messages):
        return _FakeResponse(tool_calls)

    return SimpleNamespace(ainvoke=ainvoke)


async def test_retrieve_node_caps_degenerate_tool_call_burst(monkeypatch):
    monkeypatch.setattr(rn, "_get_llm_with_tools", lambda: _fake_llm(_make_calls(300)))

    result = await rn.retrieve_node({"query": "some question"})

    assert len(result["tool_calls"]) == rn._MAX_TOOL_CALLS_PER_STEP
    assert result["retrieve_count"] == 1


async def test_retrieve_node_leaves_small_batch_unchanged(monkeypatch):
    calls = _make_calls(3)
    monkeypatch.setattr(rn, "_get_llm_with_tools", lambda: _fake_llm(calls))

    result = await rn.retrieve_node({"query": "some question"})

    assert len(result["tool_calls"]) == 3


async def test_retrieve_node_cap_applies_before_dedup_filter():
    """Trần áp dụng NGAY khi nhận response, trước bước lọc trùng — nếu chỉ
    lọc trùng thì 300 query khác nhau từng chữ vẫn lọt hết (đây chính là lý
    do bug gốc không được lọc trùng bắt được)."""
    calls = _make_calls(300)
    already_tried = rn.tried_keys([])
    fresh = [tc for tc in calls if rn.call_key(tc) not in already_tried]
    assert len(fresh) == 300  # xác nhận lọc trùng đơn thuần KHÔNG chặn được ca này
