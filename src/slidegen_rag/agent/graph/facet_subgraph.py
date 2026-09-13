"""
Subgraph 1 facet — retrieve→tool→grade_documents→context_evaluator, y hệt
pipeline CRAG cũ (agent/graph/builder.py trước khi tách facet), chạy trên
FacetState riêng của từng nhánh Send (xem agent/graph/builder.py::fan_out_facets).

Tái sử dụng nguyên vẹn 4 node cũ và `router.decide_after_context_eval` không
đổi 1 dòng nào — chỉ khác cách map kết quả: "generate" -> END (thay vì sang
node `generate` thật, vì generate chỉ chạy đúng 1 lần ở graph cha sau khi đã
gộp mọi facet, xem agent/nodes/collect_facets_node.py).
"""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from slidegen_rag.agent.graph.facet_state import FacetState
from slidegen_rag.agent.graph.router import decide_after_context_eval
from slidegen_rag.agent.nodes.context_evaluator_node import context_evaluator_node
from slidegen_rag.agent.nodes.grade_documents_node import grade_documents_node
from slidegen_rag.agent.nodes.retrieve_node import retrieve_node
from slidegen_rag.agent.nodes.tool_node import tool_node

logger = logging.getLogger(__name__)


def build_facet_subgraph():
    """Compile subgraph 1 facet — dùng bởi module này (singleton) và bởi test
    (build độc lập để mock 4 node bên trong).

    2 điểm CẦN THIẾT khi đăng ký lại 4 node cũ (viết ra cho `AgentState` gốc,
    trước khi tách facet) vào graph này (state schema giờ là `FacetState`):

    1. `input_schema=FacetState` PHẢI truyền tường minh cho cả 4 `add_node()`.
       4 node cũ vẫn còn annotate `state: AgentState` (không đổi, đúng như
       plan yêu cầu tái sử dụng nguyên vẹn) — nhưng `AgentState` giờ mang ý
       nghĩa HOÀN TOÀN khác (state graph CHA: query/facet_queries/
       facet_results/facet_sufficiency/documents/document_node_ids/
       generation/seed_drill_candidates), không còn field nào của node cũ
       cần (tool_calls/retrieve_count/feedback/search_history/...). Đã xác
       minh bằng thực nghiệm: nếu KHÔNG truyền `input_schema`, LangGraph tự
       suy ra input schema của node từ type hint tham số đầu tiên của hàm
       (`get_type_hints` trên function — xem `langgraph/graph/state.py`,
       `add_node()`), tức là suy nhầm ra `AgentState` (annotation cũ) thay vì
       `FacetState` (state schema THẬT của graph này) — hệ quả: state bị lọc
       sai schema, `tool_calls`/`retrieve_count`/`feedback`/... KHÔNG được
       đọc/ghi đúng, khiến retrieve_node tưởng luôn là vòng đầu tiên
       (retrieve_count reset), context_evaluator luôn thấy "chưa đủ" (do đọc
       nhầm field), facet không bao giờ hội tụ.
    2. `decide_after_context_eval` được bọc trong 1 closure MỚI (`_route`)
       tạo lại ở MỖI lần gọi hàm này thay vì đăng ký thẳng — vì cùng lý do ở
       trên: hàm này CŨNG annotate `state: AgentState` (cũ), nên nếu đăng ký
       trực tiếp làm routing callable, LangGraph suy schema tương tự và lọc
       sai state truyền vào nó (đọc `context_sufficient`/`retrieve_count` từ
       schema sai → luôn thấy default `False`/`0` → không bao giờ thoát vòng
       lặp). `_route` không có type hint nên LangGraph KHÔNG suy được gì,
       dùng đúng state schema của graph (FacetState). Logic routing thực sự
       KHÔNG đổi 1 dòng nào — `_route` chỉ forward thẳng sang
       `decide_after_context_eval`.
    """
    workflow = StateGraph(FacetState)

    workflow.add_node("retrieve", retrieve_node, input_schema=FacetState)
    workflow.add_node("tool", tool_node, input_schema=FacetState)
    workflow.add_node("grade_documents", grade_documents_node, input_schema=FacetState)
    workflow.add_node("context_evaluator", context_evaluator_node, input_schema=FacetState)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "tool")
    workflow.add_edge("tool", "grade_documents")
    workflow.add_edge("grade_documents", "context_evaluator")

    def _route(state: FacetState) -> str:
        return decide_after_context_eval(state)

    workflow.add_conditional_edges(
        "context_evaluator",
        _route,
        {
            "generate": END,
            "retrieve": "retrieve",
        },
    )

    return workflow.compile()


# Compiled 1 lần ở import-time (không phụ thuộc RA/config nào tại thời điểm
# compile — set_retrieval_augmentation() được gọi sau, trước khi graph thực
# sự invoke) — facet_branch_node dùng lại instance này cho mọi nhánh Send.
_compiled_facet_subgraph = build_facet_subgraph()


async def facet_branch_node(state: FacetState, config: RunnableConfig) -> dict:
    """
    Node thực sự được add vào graph cha (agent/graph/builder.py), nhận 1
    FacetState (từ Send) và chạy trọn vẹn facet_subgraph cho tới khi thoát ở
    END. `config` PHẢI được forward vào `.ainvoke()` — nếu không,
    TrajectoryCallbackHandler (gắn qua config.callbacks ở
    agent/observability/trajectory.py) sẽ không thấy bất kỳ tool/LLM call nào
    xảy ra bên trong facet, và LangGraph cũng không thể gắn đúng namespace
    lồng nhau cho `astream(..., subgraphs=True)`.

    Returns:
        dict: {"facet_results": [final FacetState]} — field CHỈ có reducer
            trên AgentState (Annotated[list[dict], operator.add]), để N nhánh
            song song ghi đồng thời không đè lên nhau.
    """
    final_state = await _compiled_facet_subgraph.ainvoke(state, config)
    return {"facet_results": [final_state]}
