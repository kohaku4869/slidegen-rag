import logging
import os
from typing import Literal, Optional

import dotenv
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from tavily import TavilyClient

from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation
from slidegen_rag.tree.structure.tree_structure import Tree
from slidegen_rag.tree.utils import (
    dedup_parent_child,
    distances_from_embeddings,
    get_embeddings,
)

dotenv.load_dotenv()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# Cờ bật/tắt web_search ở tầng retrieve_node.bind_tools — mặc định BẬT nếu có
# TAVILY_API_KEY. evaluate/runner.py chủ động TẮT khi chạy benchmark PLDC (đóng
# tài liệu = closed-book) vì reference của bộ câu hỏi này được viết trên giả
# định agent CHỈ dùng data/src/pldc.md; nếu agent tra được web thì câu trả
# lời sẽ mâu thuẫn với reference dù có thể "đúng" hơn trong thực tế.
# Khi dùng thật (main.py / production) giữ mặc định BẬT.
_web_search_enabled = tavily_client is not None


def set_web_search_enabled(flag: bool) -> None:
    """Bật/tắt việc bind web_search cho retrieve_node. Gọi trước khi build_graph()."""
    global _web_search_enabled
    _web_search_enabled = bool(flag) and tavily_client is not None


def web_search_enabled() -> bool:
    return _web_search_enabled


# Tiền tố các thông báo "lỗi mềm" — khi search thất bại, các tool dưới đây
# KHÔNG raise exception mà trả về 1 chuỗi lỗi như kết quả bình thường (để
# LLM/pipeline không crash). agent/nodes/tool_node.py dùng để lọc chunk lỗi
# khỏi kết quả; agent/observability/trajectory.py dùng để đánh dấu
# ToolCallRecord.soft_failed trong trajectory.txt. Sửa tool nào ở đây thì
# phải cập nhật tiền tố tương ứng trong tuple này.
SOFT_FAILURE_PREFIXES = (
    "Semantic search failed",
    "Keyword search failed",
    "Error searching",
    "Error expanding node",
    "No relevant results found",
    "No web search results found",
    "No tree available",
    "Tavily API key is not configured",
    "Node not found",
    "Node has no parent",
    "Node has no siblings",
    "Node has no children",  # expand_node(direction="children") trên node lá
    "No parent content found for node",
    "No siblings content found for node",
    "No matches found in subtree",
    "No children to search",
)


# Marker gắn trước mỗi node trong context trả về, để tool_node tách được
# node nào ứng với đoạn text nào (phục vụ node_digest cho expand_node) mà
# KHÔNG cần đổi kiểu trả về của tool (LangChain tool vẫn phải trả str).
# LLM không bao giờ thấy marker này — nó bị tool_node bóc ra (xem
# agent/nodes/tool_node.py:NODE_MARKER_RE) trước khi đưa vào
# state["documents"]/pending_documents hay vào prompt sinh câu trả lời.
def _node_marker(index: int, layer: int) -> str:
    return f"§N{index}|L{layer}§ "


# Module-level RA instance — must be set before using semantic_search / keyword_search
_ra: Optional[RetrievalAugmentation] = None


def set_retrieval_augmentation(ra: RetrievalAugmentation):
    """
    Injects the RetrievalAugmentation instance used by the search tools.
    Must be called once before invoking semantic_search or keyword_search.

    Args:
        ra (RetrievalAugmentation): The initialized RA instance containing the built tree.
    """
    global _ra
    _ra = ra


def clear_bm25_cache(tree: Optional[Tree] = None) -> None:
    """Xóa cache BM25. Nếu tree=None, xóa cache của tree hiện tại trong _ra."""
    target_tree = tree if tree is not None else (_ra.tree if _ra is not None else None)
    if target_tree is not None and hasattr(target_tree, "_bm25_cache"):
        delattr(target_tree, "_bm25_cache")


def _get_ra() -> RetrievalAugmentation:
    if _ra is None:
        raise RuntimeError(
            "RetrievalAugmentation instance not set. "
            "Call set_retrieval_augmentation(ra) before using search tools."
        )
    return _ra


def _flatten(text: str) -> str:
    """Làm phẳng text của 1 node giống hệt tree/utils.py:get_text() — gộp các
    dòng nội bộ thành 1 dòng bằng khoảng trắng. Bắt buộc dùng trước khi join
    nhiều node bằng "\\n\\n", nếu không các node summary (chứa \\n\\n nội bộ)
    sẽ bị agent/nodes/tool_node.py băm vụn thành nhiều chunk rời rạc."""
    return " ".join(str(text or "").splitlines())


def _get_bm25(ra: RetrievalAugmentation) -> tuple[Optional[BM25Okapi], Optional[list]]:
    """
    Lấy hoặc khởi tạo BM25 index và danh sách node từ Tree.
    Cache được lưu trực tiếp trên instance của Tree (tree._bm25_cache) thay vì
    dict toàn cục theo id(tree) để tránh va chạm địa chỉ ô nhớ khi CPython
    tái sử dụng id cho tree mới, đồng thời tự động giải phóng khi tree bị GC (tránh memory leak).
    """
    tree = ra.tree
    if tree is None:
        return None, None
    cached = getattr(tree, "_bm25_cache", None)
    if cached is not None:
        return cached
    all_nodes = [n for n in tree.all_nodes.values() if n.text and n.text.strip()]
    tokenized_corpus = [node.text.lower().split() for node in all_nodes]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    cached = (bm25, all_nodes)
    tree._bm25_cache = cached
    return cached


def describe_node(node_id: int) -> Optional[dict]:
    """
    Metadata gọn của 1 node trong cây hiện tại (không đọc text) — dùng bởi
    grade_documents_node để quyết định hint đào sâu (need_detail/need_context)
    cho drill_candidates, và bởi ConversationManager để seed drill_candidates
    từ node đã trích dẫn ở lượt hỏi trước.

    Returns:
        dict | None: {"has_children", "has_parent", "heading_path", "layer",
            "n_children", "n_siblings"}, None nếu chưa có tree hoặc node_id
            không tồn tại.
    """
    if _ra is None or _ra.tree is None:
        return None
    tree = _ra.tree
    node = tree.all_nodes.get(node_id)
    if node is None:
        return None
    parent_of_map = getattr(tree, "parent_of", None) or {}
    parent_id = (
        node.parent
        if getattr(node, "parent", None) is not None
        else parent_of_map.get(node_id)
    )
    node_to_layer = _ra.retriever.tree_node_index_to_layer
    children = getattr(node, "children", None) or ()
    parent = tree.all_nodes.get(parent_id) if parent_id is not None else None
    sibling_count = (
        max(len(getattr(parent, "children", None) or ()) - 1, 0) if parent else 0
    )
    return {
        "has_children": bool(children),
        "has_parent": parent_id is not None,
        "heading_path": getattr(node, "heading_path", ()) or (),
        "layer": node_to_layer.get(node_id, getattr(node, "layer", -1)),
        "n_children": len(children),
        "n_siblings": sibling_count,
    }


def _collect_subtree(tree: Tree, root_node_id: int, limit: int = 500) -> list:
    """BFS các node con/cháu của root_node_id (không gồm chính root), cap số
    node quét để tránh subtree khổng lồ làm find_in_subtree quá chậm."""
    root = tree.all_nodes.get(root_node_id)
    if root is None:
        return []

    collected = []
    seen = {root_node_id}
    queue = list(getattr(root, "children", None) or [])
    while queue and len(collected) < limit:
        idx = queue.pop(0)
        if idx in seen:
            continue
        seen.add(idx)
        node = tree.all_nodes.get(idx)
        if node is None:
            continue
        collected.append(node)
        queue.extend(getattr(node, "children", None) or [])
    return collected


@tool
def semantic_search(query: str, top_k: int = 3) -> str:
    """
    Performs semantic search over the document tree using vector embeddings.
    Searches across ALL layers of the tree (collapsed_tree=True) and returns
    the most semantically relevant context chunks for the given query.

    Use this tool when the query requires understanding of meaning, concepts,
    or when the exact keywords are unknown. Keep the query SHORT and focused
    on a single facet/aspect — do not paste the entire user question.

    Args:
        query (str): A short, focused natural language query for ONE facet.
        top_k (int): Maximum number of nodes to retrieve. Defaults to 3 — use
            a small value per facet and issue multiple calls for multi-facet
            questions instead of one call with a large top_k.

    Returns:
        str: Retrieved context text from the most relevant nodes in the tree.
    """
    ra = _get_ra()

    try:
        context, layer_info = ra.retrieve(
            question=query,
            collapse_tree=True,
            top_k=top_k,
            max_tokens=4000,
            return_layer_information=True,
        )
        # context đã được get_text() làm phẳng và join bằng "\n\n" theo đúng
        # thứ tự layer_info -> gắn marker node vào từng đoạn tương ứng.
        pieces = [p for p in context.split("\n\n") if p.strip()]
        marked = [
            _node_marker(li["node_index"], li["layer_number"]) + piece
            for li, piece in zip(layer_info, pieces)
        ]
        result = "\n\n".join(marked)
        logging.info(
            f"[semantic_search] Query: {query[:60]}... | top_k={top_k} | nodes={len(marked)}"
        )
        return result
    except Exception as e:
        logging.error(f"[semantic_search] Error: {e}")
        return f"Semantic search failed: {e}"


@tool
def keyword_search(query: str, top_k: int = 3) -> str:
    """
    Performs keyword-based search over the document tree using BM25 scoring.
    Searches all nodes in the tree (all layers) and returns the most relevant
    nodes based on exact term frequency matching.

    Use this tool when searching for specific terms, names, numbers, or
    when exact keyword matching is important. Keep the query SHORT — a
    handful of key terms for ONE facet, not the entire user question.

    Args:
        query (str): The keyword query to search for (short, one facet).
        top_k (int): Maximum number of nodes to retrieve. Defaults to 3 — use
            a small value per facet and issue multiple calls for multi-facet
            questions instead of one call with a large top_k.

    Returns:
        str: Retrieved context text from the top BM25-matched nodes.
    """
    ra = _get_ra()

    try:
        tree = ra.tree
        if tree is None:
            return "No tree available. Call add_documents first."

        bm25, all_nodes = _get_bm25(ra)
        if bm25 is None or not all_nodes:
            return "No tree available. Call add_documents first."

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        top_nodes = [all_nodes[i] for i in ranked_indices if scores[i] > 0]

        # Cùng heuristic dedup cha/con như semantic_search — tránh trả về
        # một node summary VÀ chính các con của nó cùng lúc.
        top_nodes = dedup_parent_child(top_nodes)

        if not top_nodes:
            return "No relevant results found for the given keywords."

        node_to_layer = ra.retriever.tree_node_index_to_layer
        marked = [
            _node_marker(node.index, node_to_layer.get(node.index, -1))
            + _flatten(node.text)
            for node in top_nodes
        ]
        context = "\n\n".join(marked)
        logging.info(
            f"[keyword_search] Query: {query[:60]}... | Top nodes: {len(top_nodes)}"
        )
        return context

    except Exception as e:
        logging.error(f"[keyword_search] Error: {e}")
        return f"Keyword search failed: {e}"


@tool
def expand_node(
    node_id: int, direction: Literal["children", "parent", "siblings"] = "children"
) -> str:
    """
    Expands a node returned by a previous search call (identified by the
    `§N<id>|L<layer>§` marker prefix of a chunk) to drill down for more
    detail or zoom out for broader context, WITHOUT re-searching.

    Use `direction="children"` (default) when a result is a high-level
    summary that looks relevant but lacks the specific detail needed — reads
    the node's more detailed child content instead of guessing a new query.
    Use `direction="parent"` when a result is too narrow/specific and you
    need the broader context it sits under. Use `direction="siblings"` to
    see neighboring sections at the same level (e.g. other conditions/cases
    listed alongside this one).

    Args:
        node_id (int): The node index from a `§N<id>|...§` marker.
        direction (str): "children" (default), "parent", or "siblings".

    Returns:
        str: Concatenated text of the target nodes (or the node's own text
            if `direction="children"` and it is a leaf with no children).
    """
    ra = _get_ra()

    try:
        tree = ra.tree
        if tree is None:
            return "No tree available. Call add_documents first."

        node = tree.all_nodes.get(node_id)
        if node is None:
            return f"Node not found: {node_id}"

        if direction == "children":
            target_indices = sorted(getattr(node, "children", None) or [])
            if not target_indices:
                return f"Node has no children (leaf node): {node_id}"
        else:
            parent_of_map = getattr(tree, "parent_of", None) or {}
            parent_id = (
                node.parent
                if getattr(node, "parent", None) is not None
                else parent_of_map.get(node_id)
            )
            if direction == "parent":
                if parent_id is None:
                    return f"Node has no parent (root node): {node_id}"
                target_indices = [parent_id]
            else:  # siblings
                if parent_id is None:
                    return f"Node has no siblings (root node): {node_id}"
                parent = tree.all_nodes.get(parent_id)
                target_indices = sorted(
                    (getattr(parent, "children", None) or set()) - {node_id}
                )
                if not target_indices:
                    return f"Node has no siblings: {node_id}"

        node_to_layer = ra.retriever.tree_node_index_to_layer
        marked = []
        for idx in target_indices:
            target = tree.all_nodes.get(idx)
            if target is None or not target.text or not target.text.strip():
                continue
            marked.append(
                _node_marker(idx, node_to_layer.get(idx, -1)) + _flatten(target.text)
            )

        if not marked:
            if direction == "children":
                return f"Node has no children (leaf node): {node_id}"
            return f"No {direction} content found for node {node_id}."

        logging.info(
            f"[expand_node] node_id={node_id} | direction={direction} | results={len(marked)}"
        )
        return "\n\n".join(marked)

    except Exception as e:
        logging.error(f"[expand_node] Error: {e}")
        return f"Error expanding node: {e}"


@tool
def find_in_subtree(
    root_node_id: int,
    patterns: list[str],
    mode: Literal["keyword", "semantic"] = "keyword",
    max_matches_per_pattern: int = 2,
) -> str:
    """
    Searches ONLY inside the subtree rooted at `root_node_id` (a node from a
    previous search or `expand_node` result) for specific keywords/phrases,
    to pinpoint exact details without reading the entire branch.

    Use this after `expand_node` still returns a broad section and you are
    looking for one or two specific terms/facts within it — cheaper and more
    precise than expanding the whole branch and reading everything.

    Args:
        root_node_id (int): Node index whose descendants define the search
            scope (from a `§N<id>|...§` marker).
        patterns (list[str]): Keywords/phrases to look for, one search per
            pattern.
        mode (str): "keyword" (BM25, default) or "semantic" (embedding
            similarity) — use "semantic" when the exact wording is unknown.
        max_matches_per_pattern (int): Max nodes returned per pattern.
            Defaults to 2.

    Returns:
        str: Concatenated text of the matched nodes, deduplicated.
    """
    ra = _get_ra()

    try:
        tree = ra.tree
        if tree is None:
            return "No tree available. Call add_documents first."

        if tree.all_nodes.get(root_node_id) is None:
            return f"Node not found: {root_node_id}"

        subtree_nodes = _collect_subtree(tree, root_node_id)
        subtree_nodes = [n for n in subtree_nodes if n.text and n.text.strip()]
        if not subtree_nodes:
            return f"No children to search (leaf node): {root_node_id}"

        if not patterns:
            return "No matches found in subtree: no patterns provided."

        node_to_layer = ra.retriever.tree_node_index_to_layer
        seen_ids: set = set()
        ordered_matches = []

        if mode == "semantic":
            embeddings = get_embeddings(
                subtree_nodes, ra.retriever.context_embedding_model
            )
            for pattern in patterns:
                query_embedding = ra.retriever.create_embedding(pattern)
                distances = distances_from_embeddings(query_embedding, embeddings)
                ranked = sorted(range(len(distances)), key=lambda i: distances[i])
                count = 0
                for i in ranked:
                    node = subtree_nodes[i]
                    if node.index in seen_ids:
                        continue
                    seen_ids.add(node.index)
                    ordered_matches.append(node)
                    count += 1
                    if count >= max_matches_per_pattern:
                        break
        else:  # keyword
            tokenized = [n.text.lower().split() for n in subtree_nodes]
            token_sets = [set(t) for t in tokenized]
            bm25 = BM25Okapi(tokenized) if tokenized else None
            for pattern in patterns:
                if bm25 is None:
                    continue
                query_tokens = set(pattern.lower().split())
                scores = bm25.get_scores(list(query_tokens))
                ranked = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )
                count = 0
                for i in ranked:
                    # rank_bm25's classic IDF goes to 0 (or negative) once a
                    # term appears in >=50% of the corpus — trivial on the
                    # small corpora find_in_subtree scopes to, which would
                    # silently zero out every score. Fall back to plain token
                    # overlap so "the term is actually in this node" still
                    # counts as a match even when BM25's score can't
                    # discriminate; ranking (order) still comes from BM25.
                    if scores[i] <= 0 and not (query_tokens & token_sets[i]):
                        continue
                    node = subtree_nodes[i]
                    if node.index in seen_ids:
                        continue
                    seen_ids.add(node.index)
                    ordered_matches.append(node)
                    count += 1
                    if count >= max_matches_per_pattern:
                        break

        if not ordered_matches:
            return f"No matches found in subtree of node {root_node_id}."

        marked = [
            _node_marker(n.index, node_to_layer.get(n.index, -1)) + _flatten(n.text)
            for n in ordered_matches
        ]
        logging.info(
            f"[find_in_subtree] root={root_node_id} | patterns={patterns} | "
            f"mode={mode} | matches={len(marked)}"
        )
        return "\n\n".join(marked)

    except Exception as e:
        logging.error(f"[find_in_subtree] Error: {e}")
        return f"Error searching subtree: {e}"


@tool
def web_search(
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    top_k: int = 10,
) -> str:
    """
    Searches the internet for up-to-date or external information using Tavily.

    Use this tool when the document tree does not contain the answer — the
    two rounds over the document tree failed, the topic is clearly outside
    the uploaded documents, or the query needs current/external facts. Try
    semantic_search and keyword_search first; switch to this tool as soon as
    the document tree has clearly failed. Keep search_depth at "basic" unless
    the topic is niche, then use "advanced".

    Args:
        query (str): The search query to look up on the web.
        search_depth (str): "basic" or "advanced". Defaults to "basic".
        top_k (int): Maximum number of results to retrieve. Defaults to 10.

    Returns:
        str: Search results from Tavily including relevant web content.
    """
    if tavily_client is None:
        return "Tavily API key is not configured (TAVILY_API_KEY is missing). Web search is unavailable."

    # Phòng thủ: Gemini đôi khi emit giá trị ngoài enum (vd. "deep",
    # "comprehensive") dù schema đã khai Literal — Tavily sẽ reject và phí
    # mất một vòng retrieve nếu không chặn ở đây.
    if search_depth not in ("basic", "advanced"):
        search_depth = "basic"

    try:
        response = tavily_client.search(
            query=query, search_depth=search_depth, max_results=min(max(top_k, 1), 10)
        )

        results = response.get("results", [])
        if not results:
            return "No web search results found for the query."
        # Format kết quả sạch sẽ cho Agent dễ đọc
        formatted_results = []
        for idx, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            url = res.get("url", "")
            content = res.get("content", "")
            formatted_results.append(f"[{idx}] {title}\nURL: {url}\nContent: {content}")
        logging.info(
            f"[web_search] Query: {query[:60]}... | Found {len(results)} results"
        )
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        logging.error(f"[web_search] Error: {e}")
        return f"Error searching the web: {e}"


tools = [semantic_search, keyword_search, web_search, expand_node, find_in_subtree]
TOOL_MAP = {tool.name: tool for tool in tools}
