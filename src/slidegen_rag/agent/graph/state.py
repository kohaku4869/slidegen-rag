import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict):
    """
    State cho graph CHA của Multi-Facet Parallel Retrieval CRAG Agent.

    Kiến trúc (xem agent/graph/builder.py):
        START -> decompose_facets -> [fan-out Send theo facet] -> facet_subgraph
               (N bản song song, mỗi bản: retrieve->tool->grade_documents->
               context_evaluator, state riêng ở agent/graph/facet_state.py::FacetState)
               -> collect_facets (gộp + dedup) -> generate (1 lần) -> END

    Field cấp CHA (field ephemeral-per-facet như tool_calls/documents/
    search_history/context_sufficient/retrieve_count/... đã CHUYỂN HẲN sang
    FacetState — xem agent/graph/facet_state.py):

        query:                  Câu hỏi gốc của user.
        facet_queries:          [{"label": str, "query": str}] — do
                                 decompose_facets_node ghi 1 lần; nguồn cho
                                 fan_out_facets tạo N Send("facet_subgraph", ...).
        facet_results:          List các FacetState CUỐI CÙNG (1 phần tử/facet),
                                 ghi bởi facet_branch_node. Field DUY NHẤT trên
                                 AgentState cần Annotated reducer (operator.add)
                                 — vì N nhánh Send ghi ĐỒNG THỜI vào field này,
                                 khác mọi field khác (chỉ 1 writer, không cần
                                 reducer, giống lý do AgentState gốc trước khi
                                 tách facet không cần reducer).
        facet_sufficiency:      [{"label", "sufficient", "missing_aspects",
                                 "retrieve_count"}] mỗi facet — do
                                 collect_facets_node build từ facet_results,
                                 thay thế hẳn cặp context_sufficient/
                                 missing_aspects toàn cục cũ. Single-writer:
                                 collect_facets_node.
        documents:              Pool document đã gộp + dedup xuyên-facet, do
                                 collect_facets_node build từ facet_results
                                 (dùng _merge_documents_with_ids của
                                 tool_node.py). Single-writer: collect_facets_node.
        document_node_ids:      node_id (hoặc None) cho từng phần tử trong
                                 `documents`, CÙNG THỨ TỰ (1:1) — dùng bởi
                                 ConversationManager để seed drill_candidates
                                 cho lượt hỏi tiếp theo. Single-writer:
                                 collect_facets_node.
        generation:             Câu trả lời tổng hợp từ generate_node.
                                 Single-writer: generate_node.
        seed_drill_candidates:  Seed ban đầu (từ ConversationManager khi câu
                                 hỏi là follow-up của lượt trước — xem
                                 agent/graph/conversation.py) — copy NGUYÊN VẸN
                                 vào drill_candidates của MỌI facet khi
                                 fan-out (không chia theo facet, xem
                                 fan_out_facets). Ghi 1 lần từ ngoài graph qua
                                 make_initial_state(); graph không ghi lại.

    Lưu ý thiết kế: graph này CÓ fan-out thật sự (N bản facet_subgraph chạy
    song song qua Send) — khác thiết kế cũ (tuần tự, không fan-out, không cần
    reducer). Vì mỗi field cấp cha ở trên (trừ facet_results) chỉ có ĐÚNG MỘT
    writer (không phải N nhánh song song ghi), chúng không cần reducer.
    `facet_results` là field DUY NHẤT nhiều nhánh cùng ghi nên PHẢI có
    reducer — nếu sau này thêm field nào khác cũng được N nhánh Send ghi, đó
    cũng phải là Annotated[..., operator.add] (hoặc reducer tương ứng), nếu
    không update sẽ bị mất âm thầm (chỉ giữ lại 1 trong N bản ghi đè lên nhau).
    """

    query: str
    facet_queries: list[dict]
    facet_results: Annotated[list[dict], operator.add]
    facet_sufficiency: list[dict]
    documents: list[str]
    document_node_ids: list[Optional[int]]
    generation: str
    seed_drill_candidates: list[dict]
