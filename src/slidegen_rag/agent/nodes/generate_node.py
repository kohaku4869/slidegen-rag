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


def _format_sufficiency_note(facet_sufficiency: list[dict]) -> str:
    """Tóm tắt nhận định của context_evaluator_node (1 lượt/facet, được
    collect_facets_node gộp thành facet_sufficiency) thành 1 đoạn ngắn để
    generate_node THAM KHẢO — KHÔNG phải kết luận cuối cùng để chép lại y
    nguyên. context_evaluator_node có thể sai (đọc lướt, bỏ sót thông tin
    gián tiếp), nên generate_node vẫn phải tự đọc lại context trước khi
    quyết định có rào đón hay không (xem GENERATE_SYSTEM_PROMPT mục 3) —
    tránh lặp lại lỗi cũ: model chép nguyên văn "không đề cập" từ nhận định
    sơ bộ dù chính ngữ cảnh nó đang đọc thực ra có thông tin đó.

    Câu hỏi đã được decompose_facets_node tách thành một hoặc nhiều khía
    cạnh (mỗi khía cạnh ứng với 1 phần câu hỏi gốc — câu hỏi 1 chủ đề luôn
    có đúng 1 khía cạnh), nên note được liệt kê THEO TỪNG khía cạnh thay vì
    1 khối phẳng như thiết kế cũ."""
    if not facet_sufficiency:
        return "Ngữ cảnh được đánh giá là ĐẦY ĐỦ để trả lời toàn bộ câu hỏi. Không cần rào đón bất kỳ phần nào."

    if all(f.get("sufficient", False) for f in facet_sufficiency):
        return (
            "Ngữ cảnh được đánh giá là ĐẦY ĐỦ để trả lời toàn bộ câu hỏi (mọi khía cạnh). "
            "Không cần rào đón bất kỳ phần nào."
        )

    lines = []
    for f in facet_sufficiency:
        label = f.get("label", "?")
        if f.get("sufficient", False):
            lines.append(f'- Khía cạnh "{label}": ĐẦY ĐỦ.')
        else:
            missing = f.get("missing_aspects") or []
            missing_str = "; ".join(missing) if missing else "(không rõ chi tiết)"
            lines.append(
                f'- Khía cạnh "{label}": CHƯA ĐẦY ĐỦ — nghi ngờ thiếu: {missing_str}.'
            )

    return (
        "Nhận định sơ bộ theo từng khía cạnh của câu hỏi (một hoặc nhiều khía cạnh, mỗi "
        "khía cạnh ứng với 1 phần câu hỏi gốc) — đây chỉ là nghi ngờ, hãy tự đọc lại ngữ "
        "cảnh bên dưới để xác minh từng khía cạnh (chỉ dựa trên dữ kiện CÓ TRONG ngữ cảnh, "
        "không dùng kiến thức ngoài tài liệu để lấp chỗ trống) trước khi quyết định có rào "
        "đón phần nào hay không:\n" + "\n".join(lines)
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
    facet_sufficiency = state.get("facet_sufficiency", [])

    # CONTEXT_BUDGET_CHARS dùng KHÔNG ĐỔI ở đây dù có N facet — KHÔNG chia
    # CONTEXT_BUDGET_CHARS / N: generate chỉ chạy đúng 1 lần bất kể N facet,
    # nên ngân sách cuối cùng cho LLM tổng hợp câu trả lời phải giữ nguyên.
    context = (
        build_context(documents, CONTEXT_BUDGET_CHARS)
        if documents
        else "No context was retrieved."
    )
    sufficiency_note = _format_sufficiency_note(facet_sufficiency)

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
