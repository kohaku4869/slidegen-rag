import logging

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.tools.tools import (
    expand_node,
    find_in_subtree,
    keyword_search,
    semantic_search,
    web_search,
    web_search_enabled,
)
from slidegen_rag.agent.utils.prompts import (
    RETRIEVE_DUPLICATE_RETRY_PROMPT,
    RETRIEVE_SYSTEM_PROMPT,
    RETRIEVE_USER_PROMPT_NO_FEEDBACK,
    RETRIEVE_USER_PROMPT_WITH_FEEDBACK,
)
from slidegen_rag.agent.utils.search_history import call_key, format_history, tried_keys

logger = logging.getLogger(__name__)

# LLM bound với retrieval tools — temperature=0 để output tool_calls ổn định.
# web_search chỉ được bind khi web_search_enabled() (mặc định = có
# TAVILY_API_KEY và evaluate/runner.py không chủ động tắt cho benchmark closed-book).
# Cache theo trạng thái enabled để không build lại LLMModel mỗi lần gọi node,
# nhưng vẫn cho phép evaluate/runner.py đổi cờ giữa các topic trong cùng 1 process.
_llm_cache: dict[bool, LLMModel] = {}


def _get_llm_with_tools():
    enabled = web_search_enabled()
    llm = _llm_cache.get(enabled)
    if llm is None:
        retrieval_tools = [
            semantic_search,
            keyword_search,
            expand_node,
            find_in_subtree,
        ]
        if enabled:
            retrieval_tools.append(web_search)
        else:
            logger.info(
                "[retrieve_node] web_search bị tắt (TAVILY_API_KEY thiếu hoặc benchmark closed-book)"
            )
        llm = LLMModel(temperature=0.0).bind_tools(
            retrieval_tools,
            tool_choice="any",  # bắt buộc phải gọi ít nhất 1 tool
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
        candidates.append(
            ("web_search", {"name": "web_search", "args": {"query": query}})
        )
    for name, tc in candidates:
        key = (name, query.lower().strip())
        if key not in tried:
            return {**tc, "id": "fallback_0", "type": "tool_call"}
    return None


# Ngưỡng chọn tool đào sâu — đặt ở đây vì đây là heuristic hướng prompt của
# retrieve_node, không phải cấu trúc dữ liệu của node_digest.
_DIGEST_MANY_CHILDREN = 5
_DIGEST_MIN_SIBLINGS = 2

# Trần cứng số tool call MỚI mỗi lượt gọi LLM của retrieve_node — chặn
# degenerate repetition: đo được trên benchmark pldc_v2 (run 20260914_171758,
# câu 22, facet "Lý luận về nhà nước") 1 lượt retrieve sinh ra 517 tool_calls
# (biến thể query gần như trùng nhau: "lý luận nhà nước", "lý luận về nhà
# nước", "lý luận nhà nước là gì", ...), out=16,371 token — chạm sát trần
# LLMConfig.MAX_TOKENS=16,384 — tốn ~189K token Gemini và 221s chỉ cho 1
# facet/1 câu (~11% token cả run 30 câu). Lọc trùng ở dưới không cứu được vì
# các query đều khác nhau từng chữ. Cắt cứng số lượng NGAY khi nhận response,
# trước khi đưa vào bước lọc trùng — không dựa vào model tự kiềm chế.
_MAX_TOOL_CALLS_PER_STEP = 6


def _digest_hint(d: dict) -> str:
    """Gợi ý tool/direction nên dùng cho 1 entry của node_digest, theo thứ tự
    ưu tiên: nhiều con → find_in_subtree (đọc hết cả nhánh quá đắt); có con
    (ít) → expand_node(children); không con nhưng nhiều anh em → node này là
    1 MỤC trong danh sách liệt kê, anh em mới là nơi trả lời trực tiếp →
    expand_node(siblings); còn lại nếu có cha → expand_node(parent) để lấy
    bối cảnh bao quát hơn. Trả về "" khi không có gợi ý nào (node lá cô lập)
    hoặc khi describe_node không tra được (n_children is None)."""
    n_children = d.get("n_children")
    if n_children is None:
        return ""
    if n_children >= _DIGEST_MANY_CHILDREN:
        return f"find_in_subtree(root_node_id={d['id']}, patterns=[...])"
    if n_children >= 1:
        return f'expand_node(node_id={d["id"]}, direction="children")'
    if (d.get("n_siblings") or 0) >= _DIGEST_MIN_SIBLINGS:
        return f'expand_node(node_id={d["id"]}, direction="siblings")'
    if d.get("has_parent"):
        return f'expand_node(node_id={d["id"]}, direction="parent")'
    return ""


def _format_digest(digest: list[dict]) -> str:
    if not digest:
        return "(chưa có)"
    lines = []
    for d in digest:
        n_children = d.get("n_children")
        if n_children is None:
            # describe_node không tra được (tree chưa sẵn sàng/node_id lạ) —
            # giữ nguyên tag cũ, không bịa số con/anh em.
            counts = " [có thể expand]" if d.get("has_children") else ""
        else:
            counts = f" | {n_children} con, {d.get('n_siblings') or 0} anh em"
        heading = d.get("heading") or ""
        heading_part = f" {heading}" if heading else ""
        hint = _digest_hint(d)
        hint_part = f" → gợi ý: {hint}" if hint else ""
        lines.append(f"- §N{d['id']}|L{d['layer']}§{heading_part}{counts}{hint_part}")
        lines.append(f"  {d['snippet']}")
    return "\n".join(lines)


_CANDIDATE_SNIPPET_CHARS = 150


def _format_drill_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "(chưa có)"
    lines = []
    for c in candidates:
        reason = c.get("reason", "")
        if reason == "need_detail":
            hint = f'expand_node(node_id={c["node_id"]}, direction="children")'
        elif reason == "need_siblings":
            hint = f'expand_node(node_id={c["node_id"]}, direction="siblings")'
        elif reason == "need_context":
            hint = f'expand_node(node_id={c["node_id"]}, direction="parent")'
        else:
            hint = f"expand_node(node_id={c['node_id']})"
        snippet = str(c.get("content", ""))[:_CANDIDATE_SNIPPET_CHARS]
        lines.append(f"- §N{c['node_id']}§ [{reason}] → gợi ý: {hint}\n  {snippet}")
    return "\n".join(lines)


async def retrieve_node(state: AgentState) -> dict:
    """
    Node quyết định retrieval strategy bằng LLM + bind_tools.

    LLM phân tích query (và feedback/lịch sử tìm kiếm nếu có) để tự quyết
    định nên phân rã câu hỏi thành các facet nào, gọi tool nào cho từng facet
    (semantic_search / keyword_search / expand_node / find_in_subtree /
    web_search), với args gì. `drill_candidates` (nếu có, từ
    grade_documents_node hoặc seed từ lượt hỏi trước) được ưu tiên hiển thị
    TRƯỚC node_digest trong prompt — hiện thực hoá ưu tiên Drill-down trước
    Discovery mà không cần 1 node/branch quyết định riêng. Sau khi LLM trả
    lời, retrieve_node LỌC TRÙNG bằng code (không chỉ dựa vào prompt): bỏ mọi
    tool call trùng (tool, query) đã có trong search_history — đảm bảo cứng
    để vòng lặp không lặp lại chính nó dù LLM có bỏ qua chỉ dẫn trong prompt.

    Args:
        state (AgentState): Cần có 'query'. 'feedback', 'search_history',
                            'missing_aspects', 'node_digest',
                            'drill_candidates' tuỳ chọn (retry/follow-up).

    Returns:
        dict: {"tool_calls": list[dict], "retrieve_count": int}
    """
    query = state["query"]
    feedback = state.get("feedback", "")
    retrieve_count = state.get("retrieve_count", 0)
    history = state.get("search_history", [])
    missing = state.get("missing_aspects", [])
    digest = state.get("node_digest", [])
    drill_candidates = state.get("drill_candidates", [])

    if feedback or history or drill_candidates:
        user_content = RETRIEVE_USER_PROMPT_WITH_FEEDBACK.format(
            query=query,
            round=retrieve_count + 1,
            feedback=feedback or "(không có)",
            missing="; ".join(missing) if missing else "(không rõ)",
            drill_candidates=_format_drill_candidates(drill_candidates),
            digest=_format_digest(digest),
            tried=format_history(history) or "(chưa có)",
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

    llm_with_tools = _get_llm_with_tools()

    try:
        response = await llm_with_tools.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", []) or []
    except Exception as e:
        logger.error(f"[retrieve_node] LLM error: {e} — using fallback tool call")
        tool_calls = []

    if len(tool_calls) > _MAX_TOOL_CALLS_PER_STEP:
        logger.warning(
            f"[retrieve_node] LLM trả về {len(tool_calls)} tool call trong 1 lượt "
            f"(> trần {_MAX_TOOL_CALLS_PER_STEP}) — nghi degenerate repetition, "
            f"cắt còn {_MAX_TOOL_CALLS_PER_STEP} call đầu tiên"
        )
        tool_calls = tool_calls[:_MAX_TOOL_CALLS_PER_STEP]

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
        logger.warning(
            f"[retrieve_node] Toàn bộ {len(dupes)} tool call trùng — nhắc lại LLM 1 lần"
        )
        retry_messages = messages + [
            HumanMessage(
                content=RETRIEVE_DUPLICATE_RETRY_PROMPT.format(
                    duplicates="\n".join(
                        f"- {tc['name']}({(tc.get('args') or {}).get('query', (tc.get('args') or {}).get('node_id', ''))!r})"
                        for tc in dupes
                    )
                )
            )
        ]
        try:
            retry_response = await llm_with_tools.ainvoke(retry_messages)
            retry_calls = getattr(retry_response, "tool_calls", []) or []
        except Exception as e:
            logger.error(f"[retrieve_node] Retry LLM error: {e}")
            retry_calls = []

        if len(retry_calls) > _MAX_TOOL_CALLS_PER_STEP:
            logger.warning(
                f"[retrieve_node] Retry LLM trả về {len(retry_calls)} tool call "
                f"— cắt còn {_MAX_TOOL_CALLS_PER_STEP} call đầu tiên"
            )
            retry_calls = retry_calls[:_MAX_TOOL_CALLS_PER_STEP]

        for tc in retry_calls:
            key = call_key(tc)
            if key not in seen_in_batch:
                fresh.append(tc)
                seen_in_batch.add(key)

    if not fresh:
        fb = _fallback_tool_call(query, already_tried)
        if fb is not None:
            logger.warning(
                "[retrieve_node] Không có tool call mới hợp lệ — dùng fallback"
            )
            fresh = [fb]
        else:
            logger.warning(
                "[retrieve_node] Mọi lựa chọn fallback đều đã thử — không thêm tool call mới"
            )

    logger.info(
        f"[retrieve_node] Generated {len(fresh)} tool call(s): "
        f"{[tc['name'] for tc in fresh]}"
    )

    return {
        "tool_calls": fresh,
        "retrieve_count": retrieve_count + 1,
    }
