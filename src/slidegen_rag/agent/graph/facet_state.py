from typing import Optional, TypedDict


class FacetState(TypedDict):
    """
    State cho 1 nhánh facet chạy song song (Send fan-out từ AgentState —
    xem agent/graph/builder.py::fan_out_facets).

    `query` mang câu hỏi CON của facet này (không phải câu hỏi gốc của user)
    để 4 node cũ (retrieve_node, tool_node, grade_documents_node,
    context_evaluator_node) dùng lại nguyên vẹn không cần sửa signature — với
    các node đó, FacetState và AgentState cũ trông giống hệt nhau.

    `facet_label`: tên khía cạnh (do decompose_facets_node đặt, hoặc câu hỏi
    gốc rút gọn khi fail-closed về 1 facet) — dùng bởi collect_facets_node để
    build facet_sufficiency, và bởi TrajectoryRecorder để gắn nhãn facet vào
    từng NodeStep.

    Các field còn lại single-writer và KHÔNG cần reducer: mỗi Send tạo một
    bản FacetState riêng nên các nhánh song song không ghi đè lẫn nhau (quy
    tắc reducer đầy đủ: xem docstring AgentState ở agent/graph/state.py).
    """

    query: str
    facet_label: str
    tool_calls: list[dict]
    documents: list[str]
    pending_documents: list[str]
    seen_fingerprints: list[str]
    search_history: list[dict]
    missing_aspects: list[str]
    node_digest: list[dict]
    pending_meta: list[dict]
    feedback: str
    context_sufficient: bool
    retrieve_count: int
    document_node_ids: list[Optional[int]]
    document_priorities: list[int]
    drill_candidates: list[dict]


def make_initial_facet_state(
    label: str, query: str, seed_drill_candidates: Optional[list[dict]] = None
) -> FacetState:
    """
    Factory tạo FacetState khởi tạo cho 1 nhánh Send.

    Args:
        label: facet_label — nhãn khía cạnh (từ decompose_facets_node).
        query: câu hỏi con của facet này.
        seed_drill_candidates: copy NGUYÊN VẸN từ
            AgentState["seed_drill_candidates"] — xem docstring field đó ở
            agent/graph/state.py để biết vì sao không chia theo facet.
    """
    return {
        "query": query,
        "facet_label": label,
        "tool_calls": [],
        "documents": [],
        "pending_documents": [],
        "seen_fingerprints": [],
        "search_history": [],
        "missing_aspects": [],
        "node_digest": [],
        "pending_meta": [],
        "feedback": "",
        "context_sufficient": False,
        "retrieve_count": 0,
        "document_node_ids": [],
        "document_priorities": [],
        "drill_candidates": list(seed_drill_candidates)
        if seed_drill_candidates
        else [],
    }
