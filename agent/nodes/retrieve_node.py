import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.models.llm import LLMModel
from agent.tools.tools import keyword_search, semantic_search
from agent.utils.prompts import (
    RETRIEVE_SYSTEM_PROMPT,
    RETRIEVE_USER_PROMPT_NO_FEEDBACK,
    RETRIEVE_USER_PROMPT_WITH_FEEDBACK,
)

logger = logging.getLogger(__name__)

# LLM bound với retrieval tools — temperature=0 để output tool_calls ổn định
_retrieval_tools = [semantic_search, keyword_search]
_llm_with_tools = LLMModel(temperature=0.0).bind_tools(
    _retrieval_tools, tool_choice="any"  # bắt buộc phải gọi ít nhất 1 tool
)

_FALLBACK_TOOL_CALL = lambda query: {
    "name": "semantic_search",
    "args": {"query": query},
    "id": "fallback_0",
    "type": "tool_call",
}


async def retrieve_node(state: AgentState) -> dict:
    """
    Node quyết định retrieval strategy bằng LLM + bind_tools.

    LLM phân tích query (và feedback nếu có từ context_evaluator) để tự
    quyết định nên gọi tool nào (semantic_search / keyword_search / cả hai)
    và với args gì. Kết quả là danh sách tool_calls được lưu vào state để
    tool_node thực thi ở bước tiếp theo.

    Luồng:
        retrieve_node → [tool_node] → grade_documents_node
                ↑ (loop nếu context chưa đủ, kèm feedback)

    Args:
        state (AgentState): Cần có 'query'. 'feedback' tùy chọn (retry).

    Returns:
        dict: {"tool_calls": list[dict], "retrieve_count": int}
    """
    query = state["query"]
    feedback = state.get("feedback", "")
    retrieve_count = state.get("retrieve_count", 0)

    # Build user message — kèm feedback nếu đây là lần retry
    if feedback:
        user_content = RETRIEVE_USER_PROMPT_WITH_FEEDBACK.format(
            query=query, feedback=feedback
        )
        logger.info(
            f"[retrieve_node] Retry #{retrieve_count} with feedback: '{feedback[:80]}'"
        )
    else:
        user_content = RETRIEVE_USER_PROMPT_NO_FEEDBACK.format(query=query)
        logger.info(f"[retrieve_node] Initial retrieval for: '{query[:80]}'")

    messages = [
        SystemMessage(content=RETRIEVE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await _llm_with_tools.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", []) or []
    except Exception as e:
        logger.error(f"[retrieve_node] LLM error: {e} — using fallback tool call")
        tool_calls = []

    if not tool_calls:
        logger.warning("[retrieve_node] No tool_calls from LLM — using fallback")
        tool_calls = [_FALLBACK_TOOL_CALL(query)]

    logger.info(
        f"[retrieve_node] Generated {len(tool_calls)} tool call(s): "
        f"{[tc['name'] for tc in tool_calls]}"
    )

    return {
        "tool_calls": tool_calls,
        "retrieve_count": retrieve_count + 1,
    }
