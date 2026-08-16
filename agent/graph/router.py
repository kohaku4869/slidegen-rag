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

    return "retrieve"
