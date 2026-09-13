"""
LangGraph builder cho Multi-Facet Parallel Retrieval CRAG Agent.

Pipeline:
    START
      → decompose_facets      (LLM tách câu hỏi gốc thành N facet độc lập)
      → [fan-out Send]        (N bản facet_subgraph chạy SONG SONG, mỗi bản
                                giữ documents/search_history/retrieve_count/
                                context_sufficient RIÊNG — xem
                                agent/graph/facet_state.py)
      → facet_subgraph × N    (mỗi bản: retrieve→tool→grade_documents→
                                context_evaluator, retry nội bộ như cũ, thoát
                                khi đủ HOẶC hết MAX_RETRIEVE riêng của facet đó)
      → collect_facets        (gộp + dedup documents xuyên-facet, tính
                                facet_sufficiency — CHẠY ĐÚNG 1 LẦN)
      → generate               (đọc CONTEXT_BUDGET_CHARS KHÔNG ĐỔI trên pool
                                đã gộp — CHẠY ĐÚNG 1 LẦN bất kể N facet)
      → END

Sử dụng:
    from slidegen_rag.agent.graph.builder import build_graph
    from langchain_core.messages import HumanMessage

    graph = build_graph()
    result = await graph.ainvoke({"query": "câu hỏi của bạn"})
    print(result["generation"])
"""

import logging

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from slidegen_rag.agent.graph.facet_state import FacetState, make_initial_facet_state
from slidegen_rag.agent.graph.facet_subgraph import facet_branch_node
from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.nodes.collect_facets_node import collect_facets_node
from slidegen_rag.agent.nodes.decompose_facets_node import decompose_facets_node
from slidegen_rag.agent.nodes.generate_node import generate_node
from slidegen_rag.agent.observability.tracing import init_langsmith

logger = logging.getLogger(__name__)


def fan_out_facets(state: AgentState) -> list[Send]:
    """
    Conditional edge sau decompose_facets: phát ra 1 Send("facet_subgraph", ...)
    cho mỗi facet trong state["facet_queries"] — LangGraph chạy các Send này
    SONG SONG, mỗi bản với 1 FacetState độc lập (không chia sẻ documents/
    search_history/retrieve_count với nhau).

    `seed_drill_candidates` (nếu có, từ ConversationManager khi câu hỏi là
    follow-up) được copy NGUYÊN VẸN vào MỌI facet — không chia theo facet vì
    không có cách xác định đáng tin cậy node cũ thuộc facet mới nào; trùng
    lặp vô hại vì grade_documents_node._prune_drilled tự loại khi đã đào.
    """
    facet_queries = state.get("facet_queries", [])
    seed_drill_candidates = state.get("seed_drill_candidates", [])
    return [
        Send(
            "facet_subgraph",
            make_initial_facet_state(
                fq.get("label") or fq.get("query", "")[:40],
                fq["query"],
                seed_drill_candidates=seed_drill_candidates,
            ),
        )
        for fq in facet_queries
    ]


def build_graph():
    """
    Lắp ráp và compile Multi-Facet Parallel Retrieval CRAG LangGraph.

    Returns:
        CompiledGraph: Graph sẵn sàng để invoke / ainvoke / stream.
    """
    # Bật LangSmith tracing (nếu .env cấu hình) — điểm khởi tạo duy nhất, tự
    # phủ mọi entry point (main.py, evaluate.py). Fail-soft: không API key
    # thì tự tắt, không raise.
    init_langsmith()

    workflow = StateGraph(AgentState)

    # ── Đăng ký nodes ─────────────────────────────────────────────────────
    workflow.add_node("decompose_facets", decompose_facets_node)
    workflow.add_node("facet_subgraph", facet_branch_node, input_schema=FacetState)
    workflow.add_node("collect_facets", collect_facets_node)
    workflow.add_node("generate", generate_node)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("decompose_facets")

    # ── Edges ─────────────────────────────────────────────────────────────
    # decompose_facets → fan-out Send theo facet (chạy song song)
    workflow.add_conditional_edges("decompose_facets", fan_out_facets)

    # mỗi bản facet_subgraph (đã thoát nội bộ ở END của chính nó) → collect_facets
    workflow.add_edge("facet_subgraph", "collect_facets")

    # collect_facets → generate (1 lần, trên pool documents đã gộp)
    workflow.add_edge("collect_facets", "generate")

    # generate → END
    workflow.add_edge("generate", END)

    graph = workflow.compile()
    logger.info(
        "[build_graph] Multi-Facet Parallel Retrieval CRAG graph compiled successfully"
    )
    return graph
