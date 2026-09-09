import logging

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.utils.prompts import CONTEXT_EVAL_SYSTEM_PROMPT, CONTEXT_EVAL_USER_PROMPT
from slidegen_rag.agent.utils.search_history import format_history

logger = logging.getLogger(__name__)

# temperature=0 cho output ổn định; max_tokens=1024 đủ chỗ cho thinking + feedback của Gemini
_eval_llm = LLMModel(temperature=0.0, max_tokens=1024)

_MAX_CONTEXT_CHARS = 8_000


def _parse_eval_response(content: str) -> tuple[bool, str, list[str]]:
    """
    Parse phản hồi của LLM theo format:
        SUFFICIENT: yes/no
        MISSING: facet1; facet2
        FEEDBACK: <text, có thể nhiều dòng>

    Returns:
        (is_sufficient: bool, feedback: str, missing_aspects: list[str])
    """
    sufficient = False
    missing: list[str] = []
    feedback_lines: list[str] = []
    mode: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        up = line.upper()
        if up.startswith("SUFFICIENT:"):
            val = line.split(":", 1)[1].strip().lower()
            sufficient = "yes" in val
            mode = None
        elif up.startswith("MISSING:"):
            val = line.split(":", 1)[1].strip()
            missing = [m.strip() for m in val.split(";") if m.strip()]
            mode = None
        elif up.startswith("FEEDBACK:"):
            feedback_lines = [line.split(":", 1)[1].strip()]
            mode = "feedback"
        elif mode == "feedback" and line:
            # Feedback nhiều dòng — trước đây _parse_eval_response chỉ đọc
            # dòng đầu nên phần gợi ý chi tiết bị mất nếu LLM xuống dòng.
            feedback_lines.append(line)

    return sufficient, " ".join(feedback_lines).strip(), missing


async def context_evaluator_node(state: AgentState) -> dict:
    """
    Đánh giá xem context hiện có đã đủ để trả lời query chưa.

    - Nếu ĐỦ → set context_sufficient = True → router chuyển sang generate.
    - Nếu CHƯA ĐỦ → set context_sufficient = False + ghi feedback/missing_aspects
      → router quay lại retrieve_node để thử retrieval strategy khác.

    Luồng:
        grade_documents_node → context_evaluator_node
            ├─ sufficient     → generate_node
            └─ not sufficient → retrieve_node  (kèm feedback)

    Args:
        state (AgentState): Cần có 'query' và 'documents'.

    Returns:
        dict: {"context_sufficient": bool, "feedback": str, "missing_aspects": list[str]}
    """
    query = state["query"]
    documents = state.get("documents", [])
    history = state.get("search_history", [])
    tried = format_history(history) or "(chưa có)"

    if not documents:
        logger.warning("[context_evaluator_node] No documents in state — marking insufficient")
        return {
            "context_sufficient": False,
            "missing_aspects": state.get("missing_aspects", []),
            "feedback": (
                "No chunk survived retrieval/grading. These searches were already tried "
                f"and must NOT be repeated:\n{tried}\n"
                "The document tree may use different terminology than the question — "
                "split the question into facets and search each with the domain/legal term "
                "instead of the plain wording. If two rounds still find nothing relevant, "
                "use web_search."
            ),
        }

    context = "\n\n".join(documents)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"

    messages = [
        SystemMessage(content=CONTEXT_EVAL_SYSTEM_PROMPT),
        HumanMessage(content=CONTEXT_EVAL_USER_PROMPT.format(
            query=query, context=context, tried=tried
        )),
    ]

    try:
        response = await _eval_llm.ainvoke(messages)
        is_sufficient, feedback, missing = _parse_eval_response(response.content)
    except Exception as e:
        logger.error(f"[context_evaluator_node] Eval failed: {e} — assuming sufficient")
        is_sufficient, feedback, missing = True, "", []

    logger.info(
        f"[context_evaluator_node] Sufficient: {is_sufficient} "
        f"| Missing: {missing} | Feedback: '{feedback[:80]}'"
    )

    return {
        "context_sufficient": is_sufficient,
        "feedback": feedback,
        "missing_aspects": missing,
    }
