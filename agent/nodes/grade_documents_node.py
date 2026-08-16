import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.models.llm import LLMModel
from agent.utils.prompts import GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT

logger = logging.getLogger(__name__)

# temperature=0 để output yes/no ổn định; max_tokens=512 để đủ chỗ cho thinking tokens của Gemini
_grader_llm = LLMModel(temperature=0.0, max_tokens=512)


async def _grade_one(document: str, query: str) -> bool:
    """Trả về True nếu document liên quan đến query."""
    messages = [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=GRADER_USER_PROMPT.format(document=document, query=query)),
    ]
    try:
        response = await _grader_llm.ainvoke(messages)
        answer = response.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        logger.warning(f"[grade_documents_node] Grading error: {e} — giữ lại document")
        return True  # fail-safe


async def grade_documents_node(state: AgentState) -> dict:
    """
    Lọc các document chunk không liên quan đến query.

    Chấm điểm từng chunk song song bằng LLM grader, loại bỏ các chunk
    có điểm 'no', và cập nhật state["documents"] với chỉ các chunk liên quan.

    Luồng:
        tool_node → grade_documents_node → context_evaluator_node

    Args:
        state (AgentState): Cần có 'documents' và 'query'.

    Returns:
        dict: {"documents": list[str]} — chỉ giữ lại các chunk relevant.
    """
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("[grade_documents_node] No documents to grade")
        return {"documents": []}

    logger.info(f"[grade_documents_node] Grading {len(documents)} chunks...")

    grade_results = await asyncio.gather(
        *[_grade_one(doc, query) for doc in documents]
    )

    relevant = [doc for doc, ok in zip(documents, grade_results) if ok]
    removed = len(documents) - len(relevant)

    logger.info(
        f"[grade_documents_node] Kept {len(relevant)}/{len(documents)} "
        f"| Removed {removed} irrelevant chunks"
    )

    return {"documents": relevant}
