import logging
import os
from typing import Literal, Optional

import dotenv
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from tavily import TavilyClient

from tree.RetrievalAugmentation import RetrievalAugmentation
from tree.utils import dedup_parent_child

dotenv.load_dotenv()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# Cờ bật/tắt web_search ở tầng retrieve_node.bind_tools — mặc định BẬT nếu có
# TAVILY_API_KEY. evaluate.py chủ động TẮT khi chạy benchmark PLDC (đóng
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

# Cache BM25Okapi theo id(tree) — tránh dựng lại index trên TOÀN BỘ node mỗi
# lần gọi keyword_search (tree pldc_raptor có hàng trăm node, số lần gọi
# keyword_search/câu hỏi sẽ tăng lên khi retrieve_node phân rã nhiều facet).
_bm25_cache: dict[int, tuple[BM25Okapi, list]] = {}


def set_retrieval_augmentation(ra: RetrievalAugmentation):
    """
    Injects the RetrievalAugmentation instance used by the search tools.
    Must be called once before invoking semantic_search or keyword_search.

    Args:
        ra (RetrievalAugmentation): The initialized RA instance containing the built tree.
    """
    global _ra
    _ra = ra
    _bm25_cache.clear()


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


def _get_bm25(ra: RetrievalAugmentation):
    tree = ra.tree
    if tree is None:
        return None, None
    key = id(tree)
    cached = _bm25_cache.get(key)
    if cached is not None:
        return cached
    all_nodes = [n for n in tree.all_nodes.values() if n.text and n.text.strip()]
    tokenized_corpus = [node.text.lower().split() for node in all_nodes]
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_cache[key] = (bm25, all_nodes)
    return bm25, all_nodes


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
        logging.info(f"[semantic_search] Query: {query[:60]}... | top_k={top_k} | nodes={len(marked)}")
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
        if bm25 is None:
            return "No tree available. Call add_documents first."

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_nodes = [all_nodes[i] for i in ranked_indices if scores[i] > 0]

        # Cùng heuristic dedup cha/con như semantic_search — tránh trả về
        # một node summary VÀ chính các con của nó cùng lúc.
        top_nodes = dedup_parent_child(top_nodes)

        if not top_nodes:
            return "No relevant results found for the given keywords."

        node_to_layer = ra.retriever.tree_node_index_to_layer
        marked = [
            _node_marker(node.index, node_to_layer.get(node.index, -1)) + _flatten(node.text)
            for node in top_nodes
        ]
        context = "\n\n".join(marked)
        logging.info(f"[keyword_search] Query: {query[:60]}... | Top nodes: {len(top_nodes)}")
        return context

    except Exception as e:
        logging.error(f"[keyword_search] Error: {e}")
        return f"Keyword search failed: {e}"


@tool
def expand_node(node_id: int) -> str:
    """
    Expands a node returned by a previous semantic_search / keyword_search call
    (identified by the `§N<id>|L<layer>§` marker prefix of a chunk) into its
    child nodes, to drill down for more detail.

    Use this when a search result is a high-level summary that looks relevant
    but does not contain the specific detail needed — expand it to read the
    underlying, more detailed content instead of re-searching with a new query.

    Args:
        node_id (int): The node index from a `§N<id>|...§` marker.

    Returns:
        str: Concatenated text of the node's child nodes (or the node's own
            text if it is a leaf with no children).
    """
    ra = _get_ra()

    try:
        tree = ra.tree
        if tree is None:
            return "No tree available. Call add_documents first."

        node = tree.all_nodes.get(node_id)
        if node is None:
            return f"Node not found: {node_id}"

        children_indices = sorted(getattr(node, "children", None) or [])
        if not children_indices:
            return f"Node has no children (leaf node): {_flatten(node.text)}"

        node_to_layer = ra.retriever.tree_node_index_to_layer
        marked = []
        for idx in children_indices:
            child = tree.all_nodes.get(idx)
            if child is None or not child.text or not child.text.strip():
                continue
            marked.append(_node_marker(idx, node_to_layer.get(idx, -1)) + _flatten(child.text))

        if not marked:
            return f"Node has no children (leaf node): {_flatten(node.text)}"

        logging.info(f"[expand_node] node_id={node_id} | children={len(marked)}")
        return "\n\n".join(marked)

    except Exception as e:
        logging.error(f"[expand_node] Error: {e}")
        return f"Error expanding node: {e}"


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
            query=query,
            search_depth=search_depth,
            max_results=min(max(top_k, 1), 10)
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
            formatted_results.append(
                f"[{idx}] {title}\nURL: {url}\nContent: {content}"
            )
        logging.info(f"[web_search] Query: {query[:60]}... | Found {len(results)} results")
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        logging.error(f"[web_search] Error: {e}")
        return f"Error searching the web: {e}"


tools = [semantic_search, keyword_search, web_search, expand_node]
TOOL_MAP = {tool.name: tool for tool in tools}
