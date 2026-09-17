"""tree_cache.py — Bản không-Streamlit của `ui/data.py::load_tree_cached`,
`_tree_builder_retriever_config`, `_build_merged_ra_cached`,
`_load_merge_parts`, `get_merged_ra`.

Logic giữ nguyên; chỉ thay `@st.cache_resource` bằng cache thủ công dựa trên
`OrderedDict` (LRU đơn giản, cap kích thước) được lưu ở `app.state.tree_cache`/
`app.state.merged_ra_cache` (xem `api/deps.py`), và thay `run_async(...)`
bằng `await` trực tiếp — route handler của FastAPI đã chạy trong coroutine.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

from slidegen_rag.config import RetrieverConfig, TreeConfig
from slidegen_rag.memory.catalog import DocumentRepository
from slidegen_rag.memory.dto import DocumentIndexRow, DocumentRow
from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation
from ui.merge import merge_trees

_TREE_CACHE_MAX_ENTRIES = 8
_RA_CACHE_MAX_ENTRIES = 2


def _lru_get_or_set(cache: OrderedDict, key, build, max_entries: int):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = build()
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)
    return value


def load_tree(tree_cache: OrderedDict, path: str):
    """`mtime_ns` trong cache key: build lại (re-upload/re-index) đổi mtime
    -> cache tự invalidate mà không cần xoá thủ công."""
    mtime_ns = Path(path).stat().st_mtime_ns
    key = (path, mtime_ns)

    def _build():
        from slidegen_rag.tree import legacy_pickle  # noqa: F401 — cài shim unpickle

        with open(path, "rb") as f:
            return pickle.load(f)

    return _lru_get_or_set(tree_cache, key, _build, _TREE_CACHE_MAX_ENTRIES)


def _tree_builder_retriever_config():
    from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
    from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
    from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentationConfig
    from slidegen_rag.tree.retriever.tree_retriever import TreeRetrieverConfig

    embedding_model = EmbeddingModel(
        model_name=TreeConfig.EMBEDDING_MODEL_NAME,
        base_url=TreeConfig.EMBEDDING_BASE_URL or None,
    )
    tb_config = TreeBuilderConfig(
        summarization_length=TreeConfig.SUMMARIZATION_LENGTH,
        embedding_models={TreeConfig.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=TreeConfig.CLUSTER_EMBEDDING_MODEL,
    )
    tr_config = TreeRetrieverConfig(
        threshold=RetrieverConfig.THRESHOLD,
        top_k=RetrieverConfig.TOP_K,
        selection_mode=RetrieverConfig.SELECTION_MODE,
        context_embedding_model=RetrieverConfig.CONTEXT_EMBEDDING_MODEL,
        embedding_model=embedding_model,
        num_layers=RetrieverConfig.NUM_LAYERS,
        start_layer=RetrieverConfig.START_LAYER,
    )
    return RetrievalAugmentationConfig(
        tree_builder_config=tb_config, tree_retriever_config=tr_config
    )


def build_merged_ra(
    ra_cache: OrderedDict,
    tree_cache: OrderedDict,
    parts_key: tuple[tuple[int, str, str, str], ...],
) -> RetrievalAugmentation:
    """`parts_key`: tuple đã sort của `(ref_no, doc_key, title, artifact_path)`."""

    def _build():
        parts = []
        for ref_no, doc_key, title, artifact_path in parts_key:
            tree = load_tree(tree_cache, artifact_path)
            parts.append((ref_no, doc_key, title, tree))
        merged_tree = merge_trees(parts)
        return RetrievalAugmentation(
            config=_tree_builder_retriever_config(), tree=merged_tree
        )

    return _lru_get_or_set(ra_cache, parts_key, _build, _RA_CACHE_MAX_ENTRIES)


async def load_merge_parts(
    document_repo: DocumentRepository, documents: Sequence[DocumentRow]
) -> list[tuple[int, str, str, str]]:
    import asyncio

    indexes: list[DocumentIndexRow | None] = await asyncio.gather(
        *(document_repo.get_active_index(d.id) for d in documents)
    )
    parts = []
    for doc, idx in zip(documents, indexes):
        if idx is None or not idx.artifact_path:
            continue  # tài liệu chưa có bản index nào ready — bỏ qua, không chặn cả lượt chat
        parts.append((doc.ref_no, str(doc.id), doc.title, idx.artifact_path))
    return parts


async def get_merged_ra(
    document_repo: DocumentRepository,
    documents: Sequence[DocumentRow],
    *,
    tree_cache: OrderedDict,
    ra_cache: OrderedDict,
) -> RetrievalAugmentation | None:
    """Điểm vào DUY NHẤT để có `RetrievalAugmentation` cho một lượt chat —
    "chọn 1 tài liệu chỉ là gộp N=1"."""
    parts_meta = await load_merge_parts(document_repo, documents)
    if not parts_meta:
        return None
    return build_merged_ra(ra_cache, tree_cache, tuple(sorted(parts_meta)))
