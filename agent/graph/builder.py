"""
LangGraph builder cho Corrective RAG (CRAG) Agent — redesigned flow.

Pipeline:
    START
      → retrieve_node         (LLM quyết định tool_calls)
      → tool_node             (thực thi tool_calls, lấy raw docs)
      → grade_documents_node  (LLM lọc docs liên quan)
      → context_evaluator_node
            ├─ sufficient     → generate_node → END
            └─ not sufficient → retrieve_node  (kèm feedback, tối đa MAX_RETRIEVE lần)

Sử dụng:
    from agent.graph.builder import build_graph
    from langchain_core.messages import HumanMessage

    graph = build_graph()
    result = await graph.ainvoke({"query": "câu hỏi của bạn"})
    print(result["generation"])
"""

import logging

from langgraph.graph import END, StateGraph

from agent.graph.router import decide_after_context_eval
from agent.graph.state import AgentState
from agent.nodes.context_evaluator_node import context_evaluator_node
from agent.nodes.generate_node import generate_node
from agent.nodes.grade_documents_node import grade_documents_node
from agent.nodes.retrieve_node import retrieve_node
from agent.nodes.tool_node import tool_node
from agent.observability.tracing import init_langsmith

logger = logging.getLogger(__name__)


def build_graph():
    """
    Lắp ráp và compile CRAG LangGraph.

    Returns:
        CompiledGraph: Graph sẵn sàng để invoke / ainvoke / stream.
    """
    # Bật LangSmith tracing (nếu .env cấu hình) — điểm khởi tạo duy nhất, tự
    # phủ mọi entry point (main.py, evaluate.py). Fail-soft: không API key
    # thì tự tắt, không raise.
    init_langsmith()

    workflow = StateGraph(AgentState)

    # ── Đăng ký nodes ─────────────────────────────────────────────────────
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("context_evaluator", context_evaluator_node)
    workflow.add_node("generate", generate_node)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("retrieve")

    # ── Edges ─────────────────────────────────────────────────────────────
    # retrieve → tool: LLM đã quyết định tool_calls, tool_node thực thi
    workflow.add_edge("retrieve", "tool")

    # tool → grade: raw docs được lọc bằng LLM grader
    workflow.add_edge("tool", "grade_documents")

    # grade → evaluator: docs đã lọc được đánh giá xem đủ chưa
    workflow.add_edge("grade_documents", "context_evaluator")

    # evaluator → generate hoặc quay lại retrieve (kèm feedback)
    workflow.add_conditional_edges(
        "context_evaluator",
        decide_after_context_eval,
        {
            "generate": "generate",
            "retrieve": "retrieve",
        },
    )

    # generate → END
    workflow.add_edge("generate", END)

    graph = workflow.compile()
    logger.info("[build_graph] CRAG graph compiled successfully")
    return graph
