"""
Conditional edge functions cho LangGraph CRAG pipeline.
"""

from langgraph.graph import END

from agent.graph.state import AgentState

# Số lần retrieve tối đa trước khi bắt buộc generate dù context chưa đủ
MAX_RETRIEVE = 3


def decide_after_context_eval(state: AgentState) -> str:
    """
    Sau context_evaluator_node:
    - Context đủ → "generate"
    - Context chưa đủ VÀ chưa đạt MAX_RETRIEVE → "retrieve" (loop với feedback)
    - Context chưa đủ NHƯNG đã đạt MAX_RETRIEVE → "generate" (generate với best-effort)

    Returns:
        "generate" | "retrieve"
    """
    sufficient = state.get("context_sufficient", False)
    retrieve_count = state.get("retrieve_count", 0)

    if sufficient:
        return "generate"

    if retrieve_count >= MAX_RETRIEVE:
        return "generate"  # best-effort, tránh infinite loop

    # No-progress guard: vòng vừa rồi không mang về chunk MỚI nào (tool_node
    # ghi đè pending_documents mỗi vòng) — lặp lại retrieve cũng vô ích vì
    # retrieve_node đã lọc trùng (tool, query), nên thử thêm 1 vòng nữa chỉ
    # tốn token mà không đổi được gì. Chỉ áp dụng từ vòng 2 trở đi để vẫn cho
    # phép retry sau lần retrieve đầu tiên.
    if retrieve_count >= 2 and not state.get("pending_documents", []):
        return "generate"

    return "retrieve"
