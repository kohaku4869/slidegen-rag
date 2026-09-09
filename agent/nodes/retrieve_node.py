import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.models.llm import LLMModel
from agent.tools.tools import expand_node, keyword_search, semantic_search, web_search, web_search_enabled
from agent.utils.prompts import (
    RETRIEVE_DUPLICATE_RETRY_PROMPT,
    RETRIEVE_SYSTEM_PROMPT,
    RETRIEVE_USER_PROMPT_NO_FEEDBACK,
    RETRIEVE_USER_PROMPT_WITH_FEEDBACK,
)
from agent.utils.search_history import call_key, format_history, tried_keys

logger = logging.getLogger(__name__)

# LLM bound với retrieval tools — temperature=0 để output tool_calls ổn định.
# web_search chỉ được bind khi web_search_enabled() (mặc định = có
# TAVILY_API_KEY và evaluate.py không chủ động tắt cho benchmark closed-book).
# Cache theo trạng thái enabled để không build lại LLMModel mỗi lần gọi node,
# nhưng vẫn cho phép evaluate.py đổi cờ giữa các topic trong cùng 1 process.
_llm_cache: dict[bool, LLMModel] = {}


def _get_llm_with_tools():
    enabled = web_search_enabled()
    llm = _llm_cache.get(enabled)
    if llm is None:
        retrieval_tools = [semantic_search, keyword_search, expand_node]
        if enabled:
            retrieval_tools.append(web_search)
        else:
            logger.info("[retrieve_node] web_search bị tắt (TAVILY_API_KEY thiếu hoặc benchmark closed-book)")
        llm = LLMModel(temperature=0.0).bind_tools(
            retrieval_tools, tool_choice="any"  # bắt buộc phải gọi ít nhất 1 tool
        )
        _llm_cache[enabled] = llm
    return llm


def _fallback_tool_call(query: str, tried: set[tuple[str, str]]) -> dict | None:
    """Chọn 1 tool CHƯA thử với đúng query gốc, theo thứ tự ưu tiên. Trả về
    None nếu mọi lựa chọn đều đã thử (khi đó retrieve_node không thêm gì mới,
    router sẽ tự thoát nhờ no-progress guard)."""
    candidates = [
        ("semantic_search", {"name": "semantic_search", "args": {"query": query}}),
        ("keyword_search", {"name": "keyword_search", "args": {"query": query}}),
    ]
    if web_search_enabled():
        candidates.append(("web_search", {"name": "web_search", "args": {"query": query}}))
    for name, tc in candidates:
        key = (name, query.lower().strip())
        if key not in tried:
            return {**tc, "id": "fallback_0", "type": "tool_call"}
    return None


def _format_digest(digest: list[dict]) -> str:
    if not digest:
        return "(chưa có)"
    lines = []
    for d in digest:
        tag = " [có thể expand]" if d.get("has_children") else ""
        lines.append(f"- §N{d['id']}|L{d['layer']}§{tag}: {d['snippet']}")
    return "\n".join(lines)


async def retrieve_node(state: AgentState) -> dict:
    """
    Node quyết định retrieval strategy bằng LLM + bind_tools.

    LLM phân tích query (và feedback/lịch sử tìm kiếm nếu có) để tự quyết
    định nên phân rã câu hỏi thành các facet nào, gọi tool nào cho từng facet
    (semantic_search / keyword_search / expand_node / web_search), với args
    gì. Sau khi LLM trả lời, retrieve_node LỌC TRÙNG bằng code (không chỉ dựa
    vào prompt): bỏ mọi tool call trùng (tool, query) đã có trong
    search_history — đảm bảo cứng để vòng lặp không lặp lại chính nó dù LLM
    có bỏ qua chỉ dẫn trong prompt.

    Luồng:
        retrieve_node → [tool_node] → grade_documents_node
                ↑ (loop nếu context chưa đủ, kèm feedback)

    Args:
        state (AgentState): Cần có 'query'. 'feedback', 'search_history',
                            'missing_aspects', 'node_digest' tuỳ chọn (retry).

    Returns:
        dict: {"tool_calls": list[dict], "retrieve_count": int}
    """
    query = state["query"]
    feedback = state.get("feedback", "")
    retrieve_count = state.get("retrieve_count", 0)
    history = state.get("search_history", [])
    missing = state.get("missing_aspects", [])
    digest = state.get("node_digest", [])

    if feedback or history:
        user_content = RETRIEVE_USER_PROMPT_WITH_FEEDBACK.format(
            query=query,
            round=retrieve_count + 1,
            feedback=feedback or "(không có)",
            missing="; ".join(missing) if missing else "(không rõ)",
            digest=_format_digest(digest),
            tried=format_history(history) or "(chưa có)",
        )
        logger.info(f"[retrieve_node] Retry #{retrieve_count} with feedback: '{feedback[:80]}'")
    else:
        user_content = RETRIEVE_USER_PROMPT_NO_FEEDBACK.format(query=query)
        logger.info(f"[retrieve_node] Initial retrieval for: '{query[:80]}'")

    messages = [
        SystemMessage(content=RETRIEVE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    llm_with_tools = _get_llm_with_tools()

    try:
        response = await llm_with_tools.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", []) or []
    except Exception as e:
        logger.error(f"[retrieve_node] LLM error: {e} — using fallback tool call")
        tool_calls = []

    # Lọc trùng bằng code: bảo đảm cứng, không phụ thuộc việc LLM có tuân thủ
    # chỉ dẫn "đừng lặp lại" trong prompt hay không.
    already_tried = tried_keys(history)
    fresh, dupes = [], []
    seen_in_batch = set(already_tried)
    for tc in tool_calls:
        key = call_key(tc)
        if key in seen_in_batch:
            dupes.append(tc)
        else:
            fresh.append(tc)
            seen_in_batch.add(key)  # chặn cả trùng trong cùng 1 batch

    if dupes and not fresh:
        logger.warning(f"[retrieve_node] Toàn bộ {len(dupes)} tool call trùng — nhắc lại LLM 1 lần")
        retry_messages = messages + [
            HumanMessage(content=RETRIEVE_DUPLICATE_RETRY_PROMPT.format(
                duplicates="\n".join(
                    f"- {tc['name']}({(tc.get('args') or {}).get('query', (tc.get('args') or {}).get('node_id', ''))!r})"
                    for tc in dupes
                )
            ))
        ]
        try:
            retry_response = await llm_with_tools.ainvoke(retry_messages)
            retry_calls = getattr(retry_response, "tool_calls", []) or []
        except Exception as e:
            logger.error(f"[retrieve_node] Retry LLM error: {e}")
            retry_calls = []

        for tc in retry_calls:
            key = call_key(tc)
            if key not in seen_in_batch:
                fresh.append(tc)
                seen_in_batch.add(key)

    if not fresh:
        fb = _fallback_tool_call(query, already_tried)
        if fb is not None:
            logger.warning("[retrieve_node] Không có tool call mới hợp lệ — dùng fallback")
            fresh = [fb]
        else:
            logger.warning("[retrieve_node] Mọi lựa chọn fallback đều đã thử — không thêm tool call mới")

    logger.info(
        f"[retrieve_node] Generated {len(fresh)} tool call(s): "
        f"{[tc['name'] for tc in fresh]}"
    )

    return {
        "tool_calls": fresh,
        "retrieve_count": retrieve_count + 1,
    }
