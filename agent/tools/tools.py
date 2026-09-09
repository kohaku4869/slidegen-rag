import logging
import os
from typing import Optional

import dotenv
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from tavily import TavilyClient

from tree.RetrievalAugmentation import RetrievalAugmentation

dotenv.load_dotenv()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

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
    "No relevant results found",
    "No web search results found",
    "No tree available",
    "Tavily API key is not configured",
)

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


def _get_ra() -> RetrievalAugmentation:
    if _ra is None:
        raise RuntimeError(
            "RetrievalAugmentation instance not set. "
            "Call set_retrieval_augmentation(ra) before using search tools."
        )
    return _ra


@tool
def semantic_search(query: str, top_k: int = 10) -> str:
    """
    Performs semantic search over the document tree using vector embeddings.
    Searches across ALL layers of the tree (collapsed_tree=True) and returns
    the most semantically relevant context chunks for the given query.

    Use this tool when the query requires understanding of meaning, concepts,
    or when the exact keywords are unknown.

    Args:
        query (str): The natural language query to search for.
        top_k (int): Maximum number of nodes to retrieve. Defaults to 10.

    Returns:
        str: Retrieved context text from the most relevant nodes in the tree.
    """
    ra = _get_ra()

    try:
        result = ra.retrieve(
            question=query,
            collapse_tree=True,
            top_k=top_k,
            max_tokens=4000,
            return_layer_information=False,
        )
        logging.info(f"[semantic_search] Query: {query[:60]}... | top_k={top_k} | Result length: {len(result)}")
        return result
    except Exception as e:
        logging.error(f"[semantic_search] Error: {e}")
        return f"Semantic search failed: {e}"


@tool
def keyword_search(query: str, top_k: int = 10) -> str:
    """
    Performs keyword-based search over the document tree using BM25 scoring.
    Searches all nodes in the tree (all layers) and returns the most relevant
    nodes based on exact term frequency matching.

    Use this tool when searching for specific terms, names, numbers, or
    when exact keyword matching is important. 

    Args:
        query (str): The keyword query to search for.
        top_k (int): Maximum number of nodes to retrieve. Defaults to 10.

    Returns:
        str: Retrieved context text from the top BM25-matched nodes.
    """
    ra = _get_ra()

    try:
        tree = ra.tree
        if tree is None:
            return "No tree available. Call add_documents first."

        # Collect all nodes from all layers
        all_nodes = list(tree.all_nodes.values())

        # Tokenize node texts (simple whitespace tokenize)
        tokenized_corpus = [node.text.lower().split() for node in all_nodes]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Sort nodes by BM25 score descending, take top_k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        top_nodes = [all_nodes[i] for i in ranked_indices if scores[i] > 0]

        if not top_nodes:
            return "No relevant results found for the given keywords."

        context = "\n\n".join(node.text for node in top_nodes)
        logging.info(
            f"[keyword_search] Query: {query[:60]}... | Top nodes: {len(top_nodes)}"
        )
        return context

    except Exception as e:
        logging.error(f"[keyword_search] Error: {e}")
        return f"Keyword search failed: {e}"


@tool
def web_search(query: str, search_depth: str = "basic",top_k: int = 10) -> str:
    """
    Searches the internet for up-to-date or external information using Tavily.

    Use this tool when the answer cannot be found in the document tree, or when
    the query requires current events, real-time data, or information not present
    in the uploaded documents. Try keyword_search and semantic_search first.

    Args:
        query (str): The search query to look up on the web.
        top_k (int): Maximum number of results to retrieve. Defaults to 10.
        search_depth (str): The depth of the search. "basic" or "advanced". Defaults to "basic".

    Returns:
        str: Search results from Tavily including relevant web content.
    """
    if tavily_client is None:
        return "Tavily API key is not configured (TAVILY_API_KEY is missing). Web search is unavailable."

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


tools = [semantic_search, keyword_search, web_search]
TOOL_MAP = {tool.name: tool for tool in tools}


