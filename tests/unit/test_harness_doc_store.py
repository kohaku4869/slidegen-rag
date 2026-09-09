"""
Unit tests for DocStore, ReferenceRegistry and RRF fusion — the harness's
foundation layer. Uses a small synthetic tree built via TreeBuilder with fake
models (same pattern as test_markdown_spans.py), no LLM/network involved.
"""

import pytest
import tiktoken

from slidegen_rag.agent.harness.doc_store import DocStore, DocStoreError
from slidegen_rag.agent.harness.fusion import rrf_fuse
from slidegen_rag.agent.harness.refs import ReferenceRegistry
from slidegen_rag.tree.builder.tree_builder import TreeBuilder, TreeBuilderConfig
from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import BaseSummarizationModel
from slidegen_rag.tree.structure.tree_structure import Tree


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Deterministic hash-of-text -> fixed-dim vector, so cosine similarity
    is stable and testable without any real model."""

    def create_embedding(self, text):
        h = hash(text)
        return [float((h >> (8 * i)) % 97) for i in range(8)]

    def create_embeddings(self, texts):
        return [self.create_embedding(t) for t in texts]


class FakeSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return f"tóm tắt: {context[:40]}"

    def batch_summarize(self, contexts, max_tokens=150):
        return [self.summarize(c) for c in contexts]


SAMPLE_MD = """\
# CHƯƠNG 1: LUẬT DÂN SỰ
## 1.1 Năng lực hành vi
Người từ đủ 18 tuổi có năng lực hành vi dân sự đầy đủ.
Người dưới 6 tuổi không có năng lực hành vi dân sự.

## 1.2 Hợp đồng
Hợp đồng dân sự là sự thỏa thuận giữa các bên.

# CHƯƠNG 2: LUẬT HÌNH SỰ
## 2.1 Tội phạm
Tội phạm là hành vi nguy hiểm cho xã hội.
"""


@pytest.fixture()
def tree() -> Tree:
    config = TreeBuilderConfig(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        summarization_model=FakeSummarizationModel(),
        embedding_models={"SBERT": FakeEmbeddingModel()},
    )
    return TreeBuilder(config).build_from_text(SAMPLE_MD, doc_id="sample", source_path="sample.md")


@pytest.fixture()
def store(tree) -> DocStore:
    return DocStore(tree, embedding_model=FakeEmbeddingModel(), embedding_model_key="SBERT")


def test_doc_store_rejects_v1_tree():
    v1_tree = Tree(all_nodes={}, root_nodes={}, leaf_nodes={}, num_layers=1, layer_to_nodes={})
    # v1_tree.schema_version defaults to 1 (class-level fallback) since not passed
    v1_tree.schema_version = 1
    with pytest.raises(DocStoreError):
        DocStore(v1_tree, embedding_model=FakeEmbeddingModel())


def test_doc_store_rejects_empty_source_text(tree):
    tree.source_text = ""
    with pytest.raises(DocStoreError):
        DocStore(tree, embedding_model=FakeEmbeddingModel())


def test_node_children_parent_access(store, tree):
    nang_luc = next(n for n in tree.leaf_nodes.values() if n.title == "Năng lực hành vi")
    parent = store.parent(nang_luc.index)
    assert parent is not None
    assert nang_luc.index in {c.index for c in store.children(parent.index)}


def test_span_text_clamped_to_bounds(store):
    # way out of range on both ends should clamp, never raise or go negative
    assert store.span_text(-100, 3) == store.source_lines[0:3]
    assert store.span_text(store.line_count - 1, store.line_count + 1000) == store.source_lines[-1:]


def test_bm25_finds_exact_keyword(store, tree):
    hits = store.bm25("năng lực hành vi", k=5)
    assert hits, "expected at least one BM25 hit"
    titles = {h.node.title for h in hits}
    assert "Năng lực hành vi" in titles


def test_bm25_no_match_returns_empty(store):
    # every word here is absent from SAMPLE_MD (unlike a naive "totally
    # unrelated" query built from common Vietnamese function words, which
    # can accidentally share tokens like "không" with real corpus content).
    assert store.bm25("qwerty asdfgh zxcvbn plugh xyzzy", k=5) == []


def test_dense_returns_scored_nodes(store):
    hits = store.dense("năng lực hành vi dân sự", k=3)
    assert len(hits) <= 3
    assert all(hasattr(h, "score") for h in hits)


def test_rrf_fuse_merges_and_ranks_by_agreement(store, tree):
    dense_hits = store.dense("năng lực hành vi", k=5)
    bm25_hits = store.bm25("năng lực hành vi", k=5)
    fused = rrf_fuse([dense_hits, bm25_hits], k=60)
    assert fused
    # a node present in BOTH rankings should outrank one present in only one,
    # all else equal — check the top result is a real, sensible node
    assert fused[0].node.title in {"Năng lực hành vi", "CHƯƠNG 1: LUẬT DÂN SỰ"}


def test_rrf_fuse_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_reference_registry_dedup_and_stable_ids(tree):
    registry = ReferenceRegistry()
    node = next(iter(tree.leaf_nodes.values()))

    ref1, is_new1 = registry.register(node, turn=1)
    ref2, is_new2 = registry.register(node, turn=2)  # same node, later turn

    assert is_new1 is True
    assert is_new2 is False
    assert ref1 == ref2  # stable across turns
    assert ref1.startswith("turn1search")  # id reflects the FIRST turn seen

    entry = registry.resolve(ref1)
    assert entry.node_id == node.index
    assert entry.first_seen_turn == 1


def test_reference_registry_counter_is_global_not_per_turn(tree):
    registry = ReferenceRegistry()
    nodes = list(tree.leaf_nodes.values())[:3]
    ids = [registry.register(n, turn=1)[0] for n in nodes]
    more_ids = [registry.register(n, turn=5)[0] for n in list(tree.leaf_nodes.values())[3:5]]
    all_ids = ids + more_ids
    counters = [int(rid.split("search")[1]) for rid in all_ids]
    assert counters == sorted(counters)  # strictly increasing, never resets
    assert len(set(counters)) == len(counters)  # all unique
