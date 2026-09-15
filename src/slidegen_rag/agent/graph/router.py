"""
Conditional edge functions cho LangGraph CRAG pipeline.
"""

from langgraph.graph import END

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.utils.search_history import new_chunks_in_round
from slidegen_rag.config import AgentConfig

# Re-export để router vẫn là nơi tra cứu tự nhiên cho logic dừng vòng lặp;
# nguồn sự thật nằm ở config.AgentConfig (trajectory.py cũng đọc từ đó).
MAX_RETRIEVE = AgentConfig.MAX_RETRIEVE

# Số vòng tối thiểu trước khi bắt đầu xét no-progress guard — luôn cho phép
# thử lại ít nhất 1 lần sau vòng đầu trước khi có thể bị dừng sớm.
MIN_RETRIEVE_BEFORE_STALL_CHECK = 3


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

    # No-progress guard: cần HAI vòng LIÊN TIẾP không mang về chunk MỚI nào
    # (không chỉ vòng vừa rồi) mới coi là cạn kiệt — 1 vòng "0 chunk mới" là
    # tín hiệu yếu (có thể do LLM chọn nhầm facet/tool đã cạn ở facet đó,
    # không phải do topic thực sự hết), agent cần ít nhất 1 cơ hội đổi chiến
    # lược trước khi bị ép dừng. Chỉ bắt đầu xét từ retrieve_count >= 3 để
    # luôn cho retry sau lần đầu tiên.
    if retrieve_count >= MIN_RETRIEVE_BEFORE_STALL_CHECK:
        history = state.get("search_history", [])
        current_round_new = new_chunks_in_round(history, retrieve_count)
        previous_round_new = new_chunks_in_round(history, retrieve_count - 1)
        if current_round_new == 0 and previous_round_new == 0:
            return "generate"

    return "retrieve"
