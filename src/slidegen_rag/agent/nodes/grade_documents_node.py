import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.nodes.tool_node import _merge_documents_with_ids
from slidegen_rag.agent.tools.tools import describe_node
from slidegen_rag.agent.utils.prompts import GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT

logger = logging.getLogger(__name__)

# temperature=0 để output ổn định; dùng provider="local" (LLM_BASE_URL), tự động fallback sang Gemini.
# max_tokens=24: output là 1 trong 3 từ (relevant/irrelevant/relevant_incomplete).
_grader_llm = LLMModel(provider="local", temperature=0.0, max_tokens=24, fallback_to_gemini=True)

# Sàn cứng cuối cùng: nếu grader loại sạch (kể cả relevant_incomplete) mà
# pending vẫn có chunk, giữ lại top-K theo đúng thứ tự retrieval trả về (đã
# được rank bởi similarity/BM25) — thà đưa nhầm 1 chunk lân cận vào generate
# còn hơn để agent trả lời "không có thông tin nào" trong khi rõ ràng có tài
# liệu liên quan bị grader từ chối oan (câu hỏi ghép nhiều vế dễ bị vậy).
_HARD_FLOOR = 5

# Giới hạn số candidate tích luỹ trong drill_candidates — tránh phình prompt
# của retrieve_node qua nhiều vòng nếu tài liệu có rất nhiều chỗ "incomplete".
_MAX_DRILL_CANDIDATES = 10


async def _grade_one(document: str, query: str, sem: asyncio.Semaphore) -> str:
    """Trả về 'relevant' | 'irrelevant' | 'relevant_incomplete'."""
    messages = [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=GRADER_USER_PROMPT.format(document=document, query=query)),
    ]
    async with sem:
        try:
            response = await _grader_llm.ainvoke(messages)
            answer = response.content.strip().lower()
            if "relevant_incomplete" in answer or "incomplete" in answer:
                return "relevant_incomplete"
            if "irrelevant" in answer:
                return "irrelevant"
            if "relevant" in answer:
                return "relevant"
            return "irrelevant"
        except Exception as e:
            logger.warning(f"[grade_documents_node] Grading error: {e} — giữ lại document (relevant_incomplete)")
            return "relevant_incomplete"  # fail-safe: giữ lại thay vì loại khi grader lỗi


def _prune_drilled(drill_candidates: list[dict], tool_calls: list[dict]) -> tuple[list[dict], set]:
    """Bỏ khỏi drill_candidates các node vừa bị `expand_node`/`find_in_subtree`
    nhắm tới trong vòng này (đọc từ state["tool_calls"], vẫn còn nguyên ở
    thời điểm grade_documents_node chạy) — tránh gợi ý lặp lại node đã đào,
    dù nội dung mới có đậu hay không (đã trả lời rồi thì thôi).

    Returns:
        (drill_candidates đã prune, set node_id vừa bị đào trong vòng này)
    """
    drilled_ids: set = set()
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        if name == "expand_node" and args.get("node_id") is not None:
            drilled_ids.add(args["node_id"])
        elif name == "find_in_subtree" and args.get("root_node_id") is not None:
            drilled_ids.add(args["root_node_id"])

    if not drilled_ids:
        return drill_candidates, drilled_ids
    return [c for c in drill_candidates if c["node_id"] not in drilled_ids], drilled_ids


def _build_drill_candidates(
    existing: list[dict],
    incomplete_docs: list[str],
    incomplete_ids: list,
    already_drilled: set,
) -> list[dict]:
    """Thêm entry mới vào drill_candidates cho các chunk vừa được grade
    'relevant_incomplete', kèm hint need_detail/need_context suy ra từ
    tree topology (describe_node) — không cần hỏi lại LLM lần 2."""
    candidates = list(existing)
    known_ids = {c["node_id"] for c in candidates}

    for doc, node_id in zip(incomplete_docs, incomplete_ids):
        if len(candidates) >= _MAX_DRILL_CANDIDATES:
            break
        if node_id is None or node_id in known_ids or node_id in already_drilled:
            continue
        info = describe_node(node_id)
        if info is None:
            continue
        if info["has_children"]:
            reason = "need_detail"
        elif info["has_parent"]:
            reason = "need_context"
        else:
            continue  # node lá, không cha — không còn gì để đào sâu thêm

        candidates.append({"node_id": node_id, "content": doc, "reason": reason})
        known_ids.add(node_id)

    return candidates


async def grade_documents_node(state: AgentState) -> dict:
    """
    Chấm điểm CHỈ các chunk MỚI trong pending_documents (chunk cũ đã đậu ở
    vòng trước nằm trong `documents` và KHÔNG bị chấm lại — tool_node đã lọc
    trùng bằng seen_fingerprints nên pending_documents chỉ chứa chunk chưa
    từng thấy).

    Phân loại 3 mức: 'relevant' (trả lời trực tiếp HOẶC đã đủ đầy đủ để hữu
    ích ngay), 'relevant_incomplete' (đúng chủ đề nhưng chỉ là 1 mảnh/tóm tắt
    — nhiều khả năng cây có node con chi tiết hơn hoặc node cha bao quát hơn
    đáng đào tiếp), 'irrelevant' (không liên quan).

    'relevant' VÀ 'relevant_incomplete' đều được giữ nguyên trong `documents`
    (generate_node chỉ đọc field này) — 'relevant_incomplete' KHÔNG bị giữ
    riêng trong drill_candidates để tránh trường hợp hết lượt retry mà chưa
    kịp đào thì nội dung biến mất khỏi câu trả lời. drill_candidates chỉ là
    metadata TRỎ THÊM tới cùng node đó để retrieve_node ưu tiên đào sâu.

    Luồng:
        tool_node → grade_documents_node → context_evaluator_node

    Args:
        state (AgentState): Cần có 'pending_documents', 'pending_meta',
            'documents', 'document_node_ids', 'drill_candidates',
            'tool_calls', 'query'.

    Returns:
        dict: {"documents", "document_node_ids", "drill_candidates"}.
    """
    query = state["query"]
    documents = state.get("documents", [])
    document_node_ids = state.get("document_node_ids", [])
    pending = state.get("pending_documents", [])
    pending_meta = state.get("pending_meta", [])
    tool_calls = state.get("tool_calls", [])

    drill_candidates, drilled_ids = _prune_drilled(list(state.get("drill_candidates", [])), tool_calls)

    if not pending:
        logger.info(f"[grade_documents_node] Không có chunk mới | giữ {len(documents)} document(s) cũ")
        return {
            "documents": documents,
            "document_node_ids": document_node_ids,
            "drill_candidates": drill_candidates,
        }

    if len(pending_meta) != len(pending):
        logger.warning("[grade_documents_node] pending_meta lệch kích thước với pending_documents — bỏ qua node_id")
        pending_meta = [{"node_id": None, "layer": None}] * len(pending)

    logger.info(f"[grade_documents_node] Grading {len(pending)} chunk mới (đã có {len(documents)} survivor)...")

    sem = asyncio.Semaphore(16)
    grades = await asyncio.gather(*[_grade_one(doc, query, sem) for doc in pending])

    relevant_docs, relevant_ids = [], []
    incomplete_docs, incomplete_ids = [], []
    irrelevant_count = 0
    for doc, meta, g in zip(pending, pending_meta, grades):
        if g == "relevant":
            relevant_docs.append(doc)
            relevant_ids.append(meta.get("node_id"))
        elif g == "relevant_incomplete":
            incomplete_docs.append(doc)
            incomplete_ids.append(meta.get("node_id"))
        else:
            irrelevant_count += 1

    kept_docs = relevant_docs + incomplete_docs
    kept_ids = relevant_ids + incomplete_ids

    if not kept_docs and pending:
        # Sàn cứng: grader loại sạch nhưng rõ ràng có tài liệu được retrieve
        # về (retrieval không rỗng) — một grader 3-mức sai trên câu hỏi ghép
        # nhiều vế không nên khiến agent trả lời "không có gì".
        kept_docs = pending[:_HARD_FLOOR]
        kept_ids = [m.get("node_id") for m in pending_meta[:_HARD_FLOOR]]
        logger.warning(
            f"[grade_documents_node] Grader loại sạch {len(pending)} chunk — "
            f"áp sàn cứng, giữ {len(kept_docs)} chunk đầu theo thứ tự retrieval"
        )

    merged_docs, merged_ids = _merge_documents_with_ids(documents, document_node_ids, kept_docs, kept_ids)
    drill_candidates = _build_drill_candidates(drill_candidates, incomplete_docs, incomplete_ids, drilled_ids)

    logger.info(
        f"[grade_documents_node] relevant={len(relevant_docs)} "
        f"relevant_incomplete={len(incomplete_docs)} irrelevant={irrelevant_count} "
        f"| giữ {len(kept_docs)}/{len(pending)} chunk mới | tổng survivor: {len(merged_docs)} "
        f"| drill_candidates: {len(drill_candidates)}"
    )

    return {
        "documents": merged_docs,
        "document_node_ids": merged_ids,
        "drill_candidates": drill_candidates,
    }
