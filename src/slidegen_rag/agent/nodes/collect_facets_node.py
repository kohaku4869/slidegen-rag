import logging

from slidegen_rag.agent.nodes.tool_node import _merge_documents_with_ids

logger = logging.getLogger(__name__)


async def collect_facets_node(state: dict) -> dict:
    """
    Gộp kết quả của mọi facet (state["facet_results"], list of final
    FacetState — 1 phần tử/facet, ghi bởi facet_branch_node qua reducer
    operator.add) thành 1 pool `documents`/`document_node_ids` DUY NHẤT cho
    generate_node, và build `facet_sufficiency` (thay thế hẳn cặp
    context_sufficient/missing_aspects toàn cục cũ).

    Dùng đúng `_merge_documents_with_ids` đã có sẵn ở tool_node.py (dedup
    theo string y hệt logic dedup xuyên-vòng hiện tại, giờ dùng xuyên-facet —
    chunk trùng giữa 2 facet là chuyện bình thường, cần dedup).

    Luồng:
        [facet_subgraph] (N bản song song) -> collect_facets -> generate

    Args:
        state (AgentState): Cần có 'facet_results'.

    Returns:
        dict: {"documents", "document_node_ids", "document_priorities",
            "facet_sufficiency"}
    """
    facet_results = state.get("facet_results", [])

    documents: list[str] = []
    document_node_ids: list = []
    document_priorities: list[int] = []
    facet_sufficiency: list[dict] = []

    for facet_state in facet_results:
        label = facet_state.get("facet_label", "?")
        facet_docs = facet_state.get("documents", [])
        facet_ids = facet_state.get("document_node_ids", [])
        facet_priorities = facet_state.get("document_priorities", [])
        if len(facet_priorities) != len(facet_docs):
            # Facet thiếu/lệch document_priorities (state cũ hoặc fixture
            # test không set) — fallback về ưu tiên thấp nhất thay vì lệch
            # index với facet_docs.
            facet_priorities = [0] * len(facet_docs)

        # Snapshot TRƯỚC khi merge — dùng làm existing_docs cho CẢ 2 lượt
        # merge bên dưới, để 2 lượt dedup y hệt nhau (xem docstring
        # _merge_documents_with_ids) và document_priorities không bị lệch
        # index so với documents/document_node_ids.
        prev_documents = documents
        documents, document_node_ids = _merge_documents_with_ids(
            prev_documents, document_node_ids, facet_docs, facet_ids
        )
        _, document_priorities = _merge_documents_with_ids(
            prev_documents, document_priorities, facet_docs, facet_priorities
        )
        facet_sufficiency.append(
            {
                "label": label,
                "query": facet_state.get("query", ""),
                "sufficient": bool(facet_state.get("context_sufficient", False)),
                "missing_aspects": facet_state.get("missing_aspects", []),
                "retrieve_count": facet_state.get("retrieve_count", 0),
            }
        )

    logger.info(
        f"[collect_facets_node] {len(facet_results)} facet(s) gộp lại -> "
        f"{len(documents)} document(s) sau dedup xuyên-facet"
    )

    return {
        "documents": documents,
        "document_node_ids": document_node_ids,
        "document_priorities": document_priorities,
        "facet_sufficiency": facet_sufficiency,
    }
