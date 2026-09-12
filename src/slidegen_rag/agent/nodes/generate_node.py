import logging

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.utils.context_budget import CONTEXT_BUDGET_CHARS, build_context
from slidegen_rag.agent.utils.prompts import (
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# temperature=0.2 (không phải 0.5): prompt yêu cầu Strict Grounding, nhiệt độ
# cao làm tăng phương sai đúng ở bước dễ vi phạm "không suy diễn/bịa" nhất.
_generate_llm = LLMModel(temperature=0.2)


def _format_sufficiency_note(
    context_sufficient: bool, missing_aspects: list[str]
) -> str:
    """Tóm tắt nhận định của context_evaluator_node thành 1 đoạn ngắn để
    generate_node THAM KHẢO — KHÔNG phải kết luận cuối cùng để chép lại y
    nguyên. context_evaluator_node có thể sai (đọc lướt, bỏ sót thông tin
    gián tiếp), nên generate_node vẫn phải tự đọc lại context trước khi
    quyết định có rào đón hay không (xem GENERATE_SYSTEM_PROMPT mục 3) —
    tránh lặp lại lỗi cũ: model chép nguyên văn "không đề cập" từ nhận định
    sơ bộ dù chính ngữ cảnh nó đang đọc thực ra có thông tin đó."""
    if context_sufficient or not missing_aspects:
        return "Ngữ cảnh được đánh giá là ĐẦY ĐỦ để trả lời toàn bộ câu hỏi. Không cần rào đón bất kỳ phần nào."
    facets = "; ".join(missing_aspects)
    return (
        "Nhận định sơ bộ: ngữ cảnh CHƯA ĐẦY ĐỦ, các khía cạnh sau bị NGHI NGỜ là thiếu — "
        f"nhưng đây chỉ là nghi ngờ, hãy tự đọc lại ngữ cảnh bên dưới để xác minh từng khía cạnh "
        f"(chỉ dựa trên dữ kiện CÓ TRONG ngữ cảnh, không dùng kiến thức ngoài tài liệu để lấp chỗ trống) "
        f"trước khi quyết định có rào đón hay không: {facets}"
    )


async def generate_node(state: AgentState) -> dict:
    """
    Tổng hợp câu trả lời cuối cùng từ các documents đã được graded.

    Args:
        state (AgentState): Cần có 'documents' và 'query'.

    Returns:
        dict: {"generation": str}
    """
    query = state["query"]
    documents = state.get("documents", [])
    context_sufficient = state.get("context_sufficient", True)
    missing_aspects = state.get("missing_aspects", [])

    context = (
        build_context(documents, CONTEXT_BUDGET_CHARS)
        if documents
        else "No context was retrieved."
    )
    sufficiency_note = _format_sufficiency_note(context_sufficient, missing_aspects)

    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=GENERATE_USER_PROMPT.format(
                context=context, query=query, sufficiency_note=sufficiency_note
            )
        ),
    ]

    try:
        response = await _generate_llm.ainvoke(messages)
        generation = response.content.strip()
        logger.info(f"[generate_node] Generated {len(generation)} chars")
    except Exception as e:
        logger.error(f"[generate_node] Generation failed: {e}")
        generation = "Xin lỗi, không thể tổng hợp câu trả lời lúc này."

    return {"generation": generation}
