import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.models.llm import LLMModel
from agent.utils.prompts import GENERATE_SYSTEM_PROMPT, GENERATE_USER_PROMPT

logger = logging.getLogger(__name__)

_generate_llm = LLMModel(temperature=0.5)
_MAX_CONTEXT_CHARS = 24_000


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

    if documents:
        context = "\n\n".join(documents)
        if len(context) > _MAX_CONTEXT_CHARS:
            context = context[:_MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"
    else:
        context = "No context was retrieved."

    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=GENERATE_USER_PROMPT.format(context=context, query=query)),
    ]

    try:
        response = await _generate_llm.ainvoke(messages)
        generation = response.content.strip()
        logger.info(f"[generate_node] Generated {len(generation)} chars")
    except Exception as e:
        logger.error(f"[generate_node] Generation failed: {e}")
        generation = "Xin lỗi, không thể tổng hợp câu trả lời lúc này."

    return {"generation": generation}
