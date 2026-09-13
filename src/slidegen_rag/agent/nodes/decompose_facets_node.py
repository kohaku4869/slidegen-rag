import logging

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.utils.prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    DECOMPOSE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# Cap cứng số facet fan-out song song — hằng số RIÊNG của module này (không
# đặt ở router.py) vì đây là fan-out cap (số nhánh Send), khác hẳn
# MAX_RETRIEVE (số vòng retry NỘI BỘ trong 1 facet).
MAX_FACETS = 4

_FALLBACK_LABEL = "Toàn bộ câu hỏi"

# temperature=0.0 theo house style _eval_llm/_grader_llm — parse text thủ
# công, không dùng structured output vì codebase hiện chưa dùng pattern đó ở
# đâu cả.
_decompose_llm = LLMModel(temperature=0.0, max_tokens=512)


def _parse_facets(content: str, original_query: str) -> list[dict]:
    """
    Parse output dạng "LABEL|SUB-QUESTION" mỗi dòng.

    Enforce lại bằng code (không tin riêng LLM):
    - Parse lỗi/không dòng nào hợp lệ -> fail-closed về đúng 1 facet (câu hỏi
      gốc) -> KHÔNG BAO GIỜ trả 0 facet.
    - Đúng 1 dòng parse được (câu hỏi 1 chủ đề) -> sub-question BẮT BUỘC là
      nguyên văn câu hỏi gốc, bất kể LLM viết gì trong đó — đây là fast-path
      bắt buộc để không đổi hành vi/chi phí của câu hỏi đơn giản.
    - Cap cứng MAX_FACETS.
    """
    facets: list[dict] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        label, _, sub_question = line.partition("|")
        label = label.strip().strip("-* ").strip()
        sub_question = sub_question.strip()
        if not label or not sub_question:
            continue
        facets.append({"label": label, "query": sub_question})

    if not facets:
        return [{"label": _FALLBACK_LABEL, "query": original_query}]

    if len(facets) == 1:
        facets[0]["query"] = original_query
        return facets

    return facets[:MAX_FACETS]


async def decompose_facets_node(state: dict) -> dict:
    """
    Tách câu hỏi gốc thành các facet độc lập, mỗi facet sẽ chạy song song qua
    facet_subgraph (xem agent/graph/builder.py::fan_out_facets).

    Luồng:
        START -> decompose_facets -> [fan-out Send] -> facet_subgraph (N bản)

    Args:
        state (AgentState): Cần có 'query'.

    Returns:
        dict: {"facet_queries": list[{"label": str, "query": str}]}
    """
    query = state["query"]
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=DECOMPOSE_USER_PROMPT.format(query=query)),
    ]

    try:
        response = await _decompose_llm.ainvoke(messages)
        facets = _parse_facets(response.content, query)
    except Exception as e:
        logger.error(
            f"[decompose_facets_node] LLM error: {e} — fallback về 1 facet (câu hỏi gốc)"
        )
        facets = [{"label": _FALLBACK_LABEL, "query": query}]

    logger.info(
        f"[decompose_facets_node] {len(facets)} facet(s): {[f['label'] for f in facets]}"
    )

    return {"facet_queries": facets}
