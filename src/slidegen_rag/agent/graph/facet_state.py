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

    Các field còn lại giữ Y HỆT ý nghĩa/single-writer như AgentState gốc
    trước khi tách facet (xem lịch sử agent/graph/state.py) — chỉ khác là giờ
    có N bản độc lập chạy song song (mỗi Send một bản riêng) thay vì 1 bản
    duy nhất, nên KHÔNG cần reducer: các nhánh không ghi đè lẫn nhau.
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
            AgentState["seed_drill_candidates"] (xem builder.py::fan_out_facets)
            — không chia theo facet vì không có cách xác định đáng tin cậy
            node cũ (trích dẫn ở lượt hội thoại trước) thuộc facet mới nào;
            trùng lặp vô hại vì grade_documents_node._prune_drilled tự loại
            khi đã đào.
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
        "drill_candidates": list(seed_drill_candidates)
        if seed_drill_candidates
        else [],
    }
