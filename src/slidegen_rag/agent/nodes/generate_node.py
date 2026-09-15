import logging

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.utils.context_budget import CONTEXT_BUDGET_CHARS, build_context
from slidegen_rag.agent.utils.prompts import (
    GENERATE_RETRY_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_PROMPT,
)
from slidegen_rag.agent.utils.text_guard import (
    collapse_pathological_repetition,
    has_pathological_repetition,
)

logger = logging.getLogger(__name__)

# max_tokens RIÊNG, cao hơn hẳn LLMConfig.MAX_TOKENS (16,384) mà node này vốn
# kế thừa: đã đo được một câu 3-facet dùng hết out=16,084/16,384 — chạm trần —
# và bị cắt ngang giữa bảng so sánh, chỉ còn 1,327 ký tự nội dung thật (điểm
# rơi từ 9.0 xuống 2.75). Phần lớn ngân sách bị reasoning token của Gemini 2.5
# Flash ăn mất trước khi kịp sinh nội dung hiển thị, nên trần phải nới cho
# riêng bước tổng hợp cuối — các node khác (output ngắn) giữ nguyên 16,384.
_MAX_GENERATE_TOKENS = 32_768

# Cờ chọn provider cho generate_node — mặc định "gemini". "lowcost" (giá trị
# "local") chuyển sang model tự host qua LLM_BASE_URL (cùng server Qwen3.5 mà
# grade_documents_node đang dùng) để cắt chi phí Gemini ở bước tốn token nhất/
# lượt gọi (xem so sánh token retrieve vs generate) — LLMModel tự fallback
# sang Gemini nếu self-host lỗi/timeout (fallback_to_gemini=True mặc định).
# Cache theo provider, KHÔNG dựng client mỗi lượt gọi node, nhưng vẫn cho
# evaluate.py đổi cờ giữa các lần chạy trong cùng 1 process — cùng pattern với
# set_web_search_enabled ở agent/tools/tools.py.
_provider = "gemini"
_llm_cache: dict[str, LLMModel] = {}


def set_generate_provider(provider: str) -> None:
    """Bật/tắt provider của RIÊNG generate_node. Gọi TRƯỚC build_graph().
    KHÔNG ảnh hưởng retrieve/decompose/context_evaluator (vẫn Gemini) hay
    grade_documents (vẫn cố định "local")."""
    global _provider
    _provider = (provider or "gemini").strip().lower()


def generate_provider() -> str:
    return _provider


def _get_generate_llm() -> LLMModel:
    llm = _llm_cache.get(_provider)
    if llm is None:
        llm = LLMModel(
            provider=_provider, temperature=0.2, max_tokens=_MAX_GENERATE_TOKENS
        )
        _llm_cache[_provider] = llm
    return llm

# finish_reason báo output bị cắt vì chạm trần token (tên khác nhau tuỳ
# provider: Gemini dùng "MAX_TOKENS", OpenAI-compatible dùng "length").
_TRUNCATED_FINISH_REASONS = {"MAX_TOKENS", "LENGTH"}


def _is_truncated(response) -> bool:
    """True nếu provider báo output bị cắt giữa chừng do chạm trần token."""
    meta = getattr(response, "response_metadata", None) or {}
    reason = str(meta.get("finish_reason") or "").upper()
    return reason in _TRUNCATED_FINISH_REASONS


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
    1 khối phẳng như thiết kế cũ.

    Hiển thị nguyên văn sub-question (facet_sufficiency[i]["query"]) thay vì
    "label" — label giờ chỉ là 1 chuỗi số thứ tự + cắt ngắn do code sinh ra
    (xem decompose_facets_node._make_label), không còn ý nghĩa để đọc; fallback
    về "label" khi thiếu "query" (fixture cũ/facet fail-closed)."""
    if not facet_sufficiency:
        return "Ngữ cảnh được đánh giá là ĐẦY ĐỦ để trả lời toàn bộ câu hỏi. Không cần rào đón bất kỳ phần nào."

    if all(f.get("sufficient", False) for f in facet_sufficiency):
        return (
            "Ngữ cảnh được đánh giá là ĐẦY ĐỦ để trả lời toàn bộ câu hỏi (mọi khía cạnh). "
            "Không cần rào đón bất kỳ phần nào."
        )

    lines = []
    for f in facet_sufficiency:
        shown = f.get("query") or f.get("label") or "?"
        if f.get("sufficient", False):
            lines.append(f'- Khía cạnh "{shown}": ĐẦY ĐỦ.')
        else:
            missing = f.get("missing_aspects") or []
            missing_str = "; ".join(missing) if missing else "(không rõ chi tiết)"
            lines.append(
                f'- Khía cạnh "{shown}": CHƯA ĐẦY ĐỦ — nghi ngờ thiếu: {missing_str}.'
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
    document_priorities = state.get("document_priorities", [])
    facet_sufficiency = state.get("facet_sufficiency", [])

    # CONTEXT_BUDGET_CHARS dùng KHÔNG ĐỔI ở đây dù có N facet — KHÔNG chia
    # CONTEXT_BUDGET_CHARS / N: generate chỉ chạy đúng 1 lần bất kể N facet,
    # nên ngân sách cuối cùng cho LLM tổng hợp câu trả lời phải giữ nguyên.
    # document_priorities (từ grade_documents_node) cho build_context ưu
    # tiên giữ chunk CHẤT LƯỢNG CAO khi vượt ngân sách, thay vì chỉ giữ chunk
    # MỚI NHẤT — xem docstring context_budget.build_context.
    context = (
        build_context(documents, CONTEXT_BUDGET_CHARS, priorities=document_priorities)
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

    llm = _get_generate_llm()
    try:
        response = await llm.ainvoke(messages)
        generation = response.content.strip()

        # Hai chế độ hỏng đã gặp thật, đều KHÔNG raise exception nên trước đây
        # lọt thẳng ra ngoài: (1) lặp ký tự vô hạn, (2) bị cắt do chạm trần
        # token. Cả hai đều khiến judge chấm rất thấp dù retrieval hoàn toàn ổn.
        degenerate = has_pathological_repetition(generation)
        truncated = _is_truncated(response)

        if degenerate or truncated:
            why = "lặp bất thường" if degenerate else "bị cắt do chạm trần token"
            logger.warning(
                f"[generate_node] Output hỏng ({why}, {len(generation)} ký tự) — sinh lại 1 lần"
            )
            try:
                retry_response = await llm.ainvoke(
                    messages + [HumanMessage(content=GENERATE_RETRY_PROMPT)]
                )
                retry_generation = retry_response.content.strip()
                # Chỉ nhận bản mới nếu nó thực sự sạch hơn — nếu vẫn hỏng thì
                # giữ bản đầu (fail-safe: không bao giờ tệ hơn hiện trạng).
                if retry_generation and not has_pathological_repetition(
                    retry_generation
                ):
                    generation = retry_generation
                    logger.info(
                        f"[generate_node] Sinh lại OK ({len(generation)} ký tự)"
                    )
                else:
                    logger.warning("[generate_node] Sinh lại vẫn hỏng — giữ bản đầu")
            except Exception as e:
                logger.error(f"[generate_node] Sinh lại lỗi: {e} — giữ bản đầu")

            # Dù nhận bản nào cũng phải rút gọn đoạn lặp trước khi trả ra: text
            # 266k ký tự toàn ký tự lặp không có giá trị cho ai, và còn làm
            # phình answer.txt / prompt của LLM judge ở bước sau.
            generation = collapse_pathological_repetition(generation)

        logger.info(f"[generate_node] Generated {len(generation)} chars")
    except Exception as e:
        logger.error(f"[generate_node] Generation failed: {e}")
        generation = "Xin lỗi, không thể tổng hợp câu trả lời lúc này."

    return {"generation": generation}
