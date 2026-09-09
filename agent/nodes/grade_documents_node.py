import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.models.llm import LLMModel
from agent.nodes.tool_node import _merge_documents
from agent.utils.prompts import GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT

logger = logging.getLogger(__name__)

# temperature=0 để output ổn định; dùng provider="local" (LLM_BASE_URL), tự động fallback sang Gemini.
# max_tokens=24 (trước là 16): output giờ là 1 trong 3 từ (direct/related/no)
# thay vì yes/no, cần thêm vài token dự phòng.
_grader_llm = LLMModel(provider="local", temperature=0.0, max_tokens=24, fallback_to_gemini=True)

# Nếu KHÔNG có chunk "direct" nào, giữ tối đa bấy nhiêu chunk "related" để
# generate_node vẫn có gì đó nói "tài liệu không đề cập trực tiếp X, nhưng có Y".
_MAX_RELATED_FALLBACK = 10

# Sàn cứng cuối cùng: nếu grader loại sạch (kể cả related) mà pending vẫn có
# chunk, giữ lại top-K theo đúng thứ tự retrieval trả về (đã được rank bởi
# similarity/BM25) — thà đưa nhầm 1 chunk lân cận vào generate còn hơn để
# agent trả lời "không có thông tin nào" trong khi rõ ràng có tài liệu liên
# quan bị grader từ chối oan (câu hỏi ghép nhiều vế dễ bị vậy).
_HARD_FLOOR = 5


async def _grade_one(document: str, query: str, sem: asyncio.Semaphore) -> str:
    """Trả về 'direct' | 'related' | 'no'."""
    messages = [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=GRADER_USER_PROMPT.format(document=document, query=query)),
    ]
    async with sem:
        try:
            response = await _grader_llm.ainvoke(messages)
            answer = response.content.strip().lower()
            if "direct" in answer:
                return "direct"
            if "related" in answer:
                return "related"
            return "no"
        except Exception as e:
            logger.warning(f"[grade_documents_node] Grading error: {e} — giữ lại document (related)")
            return "related"  # fail-safe: giữ lại thay vì loại khi grader lỗi


async def grade_documents_node(state: AgentState) -> dict:
    """
    Chấm điểm CHỈ các chunk MỚI trong pending_documents (chunk cũ đã đậu ở
    vòng trước nằm trong `documents` và KHÔNG bị chấm lại — tool_node đã lọc
    trùng bằng seen_fingerprints nên pending_documents chỉ chứa chunk chưa
    từng thấy).

    Phân loại 3 mức: 'direct' (trả lời trực tiếp câu hỏi), 'related' (cùng
    lĩnh vực/lân cận, không trả lời trực tiếp nhưng hữu ích để agent nói rõ
    giới hạn tài liệu), 'no' (không liên quan). Luôn ưu tiên giữ 'direct'; nếu
    round này không có 'direct' nào, promote một phần 'related' để tránh trả
    về context rỗng cho các câu hỏi mà tài liệu chỉ đề cập lân cận (không có
    câu trả lời chính xác nhưng KHÔNG có nghĩa là không có gì để nói).

    Luồng:
        tool_node → grade_documents_node → context_evaluator_node

    Args:
        state (AgentState): Cần có 'pending_documents', 'documents', 'query'.

    Returns:
        dict: {"documents": list[str]} — survivor tích luỹ (documents cũ + phần
              chunk mới vừa đậu).
    """
    query = state["query"]
    documents = state.get("documents", [])
    pending = state.get("pending_documents", [])

    if not pending:
        logger.info(f"[grade_documents_node] Không có chunk mới | giữ {len(documents)} document(s) cũ")
        return {"documents": documents}

    logger.info(f"[grade_documents_node] Grading {len(pending)} chunk mới (đã có {len(documents)} survivor)...")

    sem = asyncio.Semaphore(16)
    grades = await asyncio.gather(*[_grade_one(doc, query, sem) for doc in pending])

    direct = [doc for doc, g in zip(pending, grades) if g == "direct"]
    related = [doc for doc, g in zip(pending, grades) if g == "related"]

    if direct:
        kept = direct
    elif related:
        # Không có chunk trả lời trực tiếp — giữ related để agent có thể nói
        # "tài liệu không đề cập trực tiếp X, nhưng có ghi nhận Y" (đúng văn
        # phong của gold reference) thay vì bỏ trắng.
        kept = related[:_MAX_RELATED_FALLBACK]
        logger.info(f"[grade_documents_node] Không có 'direct' — promote {len(kept)} 'related' chunk")
    else:
        kept = []

    if not kept and pending:
        # Sàn cứng: grader loại sạch nhưng rõ ràng có tài liệu được retrieve
        # về (retrieval không rỗng) — một grader nhị phân/3-mức sai trên câu
        # hỏi ghép nhiều vế không nên khiến agent trả lời "không có gì".
        kept = pending[:_HARD_FLOOR]
        logger.warning(
            f"[grade_documents_node] Grader loại sạch {len(pending)} chunk — "
            f"áp sàn cứng, giữ {len(kept)} chunk đầu theo thứ tự retrieval"
        )

    merged = _merge_documents(documents, kept)
    logger.info(
        f"[grade_documents_node] direct={len(direct)} related={len(related)} "
        f"| giữ {len(kept)}/{len(pending)} chunk mới | tổng survivor: {len(merged)}"
    )

    return {"documents": merged}
