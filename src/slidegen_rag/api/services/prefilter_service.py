"""prefilter_service.py — Bản không-Streamlit của
`ui/data.py::get_embedding_model_cached`, `prefilter_documents_by_query`.

Model embedding cache trên `app.state.embedding_model` (lazy-init, xem
`api/app.py::lifespan`) thay vì `@st.cache_resource`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from slidegen_rag.config import TreeConfig
from slidegen_rag.memory.dto import DocumentRow


def get_embedding_model(state):
    """`state`: `request.app.state`. Lazy-init một lần cho cả process."""
    if state.embedding_model is None:
        from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel

        state.embedding_model = EmbeddingModel(
            model_name=TreeConfig.EMBEDDING_MODEL_NAME,
            base_url=TreeConfig.EMBEDDING_BASE_URL or None,
        )
    return state.embedding_model


def prefilter_documents_by_query(
    state, query: str, documents: Sequence[DocumentRow], top_k: int = 3
) -> list[DocumentRow]:
    """>3 tài liệu đang chọn -> cosine trên `doc_summary` cấp tài liệu để thu
    hẹp TRƯỚC KHI gộp cây. Không thêm LLM call — chỉ N+1 embedding call rẻ."""
    if len(documents) <= 3:
        return list(documents)
    model = get_embedding_model(state)
    query_vec = np.asarray(model.create_embedding(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    scored = []
    for doc in documents:
        summary = doc.doc_summary or doc.title
        vec = np.asarray(model.create_embedding(summary), dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            vec = vec / vec_norm
        scored.append((float(query_vec @ vec), doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
