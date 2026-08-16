import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.models.llm import LLMModel
from agent.utils.prompts import CONTEXT_EVAL_SYSTEM_PROMPT, CONTEXT_EVAL_USER_PROMPT

logger = logging.getLogger(__name__)

# temperature=0 cho output ổn định; max_tokens=1024 đủ chỗ cho thinking + feedback của Gemini
_eval_llm = LLMModel(temperature=0.0, max_tokens=1024)

_MAX_CONTEXT_CHARS = 8_000


def _parse_eval_response(content: str) -> tuple[bool, str]:
    """
    Parse phản hồi của LLM theo format:
        SUFFICIENT: yes/no
        FEEDBACK: <text>

    Returns:
        (is_sufficient: bool, feedback: str)
    """
    sufficient = False
    feedback = ""

    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("SUFFICIENT:"):
            val = line.split(":", 1)[1].strip().lower()
            sufficient = "yes" in val
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return sufficient, feedback


async def context_evaluator_node(state: AgentState) -> dict:
    """
    Đánh giá xem context hiện có đã đủ để trả lời query chưa.

    - Nếu ĐỦ → set context_sufficient = True → router chuyển sang generate.
    - Nếu CHƯA ĐỦ → set context_sufficient = False + ghi feedback → router
      quay lại retrieve_node để thử retrieval strategy khác.

    Luồng:
        grade_documents_node → context_evaluator_node
            ├─ sufficient     → generate_node
            └─ not sufficient → retrieve_node  (kèm feedback)

    Args:
        state (AgentState): Cần có 'query' và 'documents'.

    Returns:
        dict: {"context_sufficient": bool, "feedback": str}
    """
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning(
            "[context_evaluator_node] No documents in state — marking insufficient"
        )
        return {
            "context_sufficient": False,
            "feedback": (
                "No relevant documents were retrieved. "
                "Try broader search terms or use different keywords."
            ),
        }

    context = "\n\n".join(documents)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"

    messages = [
        SystemMessage(content=CONTEXT_EVAL_SYSTEM_PROMPT),
        HumanMessage(content=CONTEXT_EVAL_USER_PROMPT.format(
            query=query, context=context
        )),
    ]

    try:
        response = await _eval_llm.ainvoke(messages)
        is_sufficient, feedback = _parse_eval_response(response.content)
    except Exception as e:
        logger.error(f"[context_evaluator_node] Eval failed: {e} — assuming sufficient")
        is_sufficient, feedback = True, ""

    logger.info(
        f"[context_evaluator_node] Sufficient: {is_sufficient} "
        f"| Feedback: '{feedback[:80]}'"
    )

    return {
        "context_sufficient": is_sufficient,
        "feedback": feedback,
    }
