"""
DocStore — read-only access layer the harness's 4 tools are built on top of:
node lookup, source line spans, and dense/sparse retrieval over a schema-v2
Tree. Requires `tree.schema_version >= 2` (line spans, heading_path, depth,
parent — see tree/structure/tree_structure.py) since `find`/`open` need real
line spans to operate; a v1 tree (or a tree with no source_text at all)
cannot back this harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi

from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel
from slidegen_rag.tree.structure.tree_structure import Node, Tree
from slidegen_rag.tree.utils import distances_from_embeddings


class DocStoreError(Exception):
    """Raised when a DocStore is asked to do something its Tree can't support
    (missing schema v2 metadata, unknown node id, ...)."""


@dataclass(frozen=True)
class ScoredNode:
    node: Node
    score: float


class DocStore:
    """Wraps one schema-v2 Tree. One DocStore per document/run — the BM25
    index is cached on the instance (not module-global), so building two
    DocStores over two different Trees never collide or go stale, unlike
    the old `agent/tools/tools.py:_bm25_cache` keyed on `id(tree)`."""

    def __init__(
        self,
        tree: Tree,
        embedding_model: BaseEmbeddingModel,
        embedding_model_key: str = "SBERT",
    ) -> None:
        if getattr(tree, "schema_version", 1) < 2:
            raise DocStoreError(
                "Tree.schema_version < 2 — this index predates line-span metadata "
                "and cannot back find/open. Run scripts/migrate_index_v2.py or "
                "re-ingest."
            )
        if not tree.source_text:
            raise DocStoreError("Tree.source_text is empty — no source to open/find over.")

        self.tree = tree
        self.embedding_model = embedding_model
        self.embedding_model_key = embedding_model_key
        self._source_lines: list[str] = tree.source_text.split("\n")
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_nodes: Optional[list[Node]] = None

    @classmethod
    def from_ra(cls, ra) -> "DocStore":
        """Build a DocStore from an already-loaded RetrievalAugmentation."""
        return cls(
            tree=ra.tree,
            embedding_model=ra.retriever.embedding_model,
            embedding_model_key=ra.retriever.context_embedding_model,
        )

    # ── node access ──────────────────────────────────────────────────────
    def node(self, node_id: int) -> Node:
        try:
            return self.tree.all_nodes[node_id]
        except KeyError:
            raise DocStoreError(f"No node with id {node_id}") from None

    def children(self, node_id: int) -> list[Node]:
        return [self.tree.all_nodes[c] for c in sorted(self.node(node_id).children) if c in self.tree.all_nodes]

    def parent(self, node_id: int) -> Optional[Node]:
        p = self.node(node_id).parent
        return self.tree.all_nodes.get(p) if p is not None else None

    # ── source text access ───────────────────────────────────────────────
    @property
    def source_lines(self) -> list[str]:
        return self._source_lines

    @property
    def line_count(self) -> int:
        return len(self._source_lines)

    def span_text(self, start: int, end: int) -> list[str]:
        """Lines [start, end), clamped to the document bounds."""
        start = max(0, min(start, self.line_count))
        end = max(start, min(end, self.line_count))
        return self._source_lines[start:end]

    # ── retrieval ────────────────────────────────────────────────────────
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        nodes = [n for n in self.tree.all_nodes.values() if n.text and n.text.strip()]
        tokenized = [n.text.lower().split() for n in nodes]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._bm25_nodes = nodes

    def bm25(self, query: str, k: int) -> list[ScoredNode]:
        self._ensure_bm25()
        if self._bm25 is None or not self._bm25_nodes:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredNode(self._bm25_nodes[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def dense(self, query: str, k: int) -> list[ScoredNode]:
        nodes = [
            n for n in self.tree.all_nodes.values()
            if n.text and n.text.strip() and self.embedding_model_key in n.embeddings
        ]
        if not nodes:
            return []
        query_embedding = self.embedding_model.create_embedding(query)
        embeddings = [n.embeddings[self.embedding_model_key] for n in nodes]
        distances = distances_from_embeddings(query_embedding, embeddings)
        ranked = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
        return [ScoredNode(nodes[i], 1.0 - distances[i]) for i in ranked]
