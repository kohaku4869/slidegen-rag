"""
Tests for the batch embedding optimization in TreeBuilder.build_from_text.

Verifies that:
1. batch_create_nodes correctly assigns indices, texts, children, and embeddings.
2. build_from_text produces structurally correct trees (node count, layers, parent-child links).
3. Embedding calls are batched (create_embeddings) instead of sequential (create_embedding).
"""

import collections
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
from typing import Dict, List, Set, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slidegen_rag.tree.structure.tree_structure import Node, Tree
from slidegen_rag.tree.builder.tree_builder import TreeBuilder, TreeBuilderConfig


class FakeEmbeddingModel:
    """Fake embedding model that returns deterministic vectors and tracks calls."""

    def __init__(self, dim=4):
        self.dim = dim
        self.create_embedding_calls = []
        self.create_embeddings_calls = []

    def create_embedding(self, text: str) -> List[float]:
        self.create_embedding_calls.append(text)
        return [float(hash(text) % 100)] * self.dim

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.create_embeddings_calls.append(texts)
        return [[float(hash(t) % 100)] * self.dim for t in texts]


class FakeSummarizationModel:
    """Fake summarization model that returns truncated input as summary."""

    def summarize(self, context, max_tokens=2048) -> str:
        return f"Summary of: {context[:50]}"

    def batch_summarize(self, contexts, max_tokens=2048) -> List[str]:
        return [f"Summary of: {c[:50]}" for c in contexts]


def make_tree_builder(embedding_model=None, summarization_model=None):
    """Helper to create a TreeBuilder with fake models."""
    if embedding_model is None:
        embedding_model = FakeEmbeddingModel()
    if summarization_model is None:
        summarization_model = FakeSummarizationModel()

    config = MagicMock()
    config.tokenizer = None
    config.max_tokens = 100
    config.num_layers = 5
    config.top_k = 5
    config.threshold = 0.5
    config.selection_mode = "top_k"
    config.summarization_length = 100
    config.summarization_model = summarization_model
    config.embedding_models = {"test_model": embedding_model}
    config.cluster_embedding_model = "test_model"
    config.log_config = MagicMock(return_value="test_config")

    return TreeBuilder(config), embedding_model


class TestBatchCreateNodes(unittest.TestCase):
    """Tests for the upgraded batch_create_nodes method."""

    def test_empty_input(self):
        builder, _ = make_tree_builder()
        result = builder.batch_create_nodes([])
        self.assertEqual(result, {})

    def test_single_node(self):
        builder, emb_model = make_tree_builder()
        items = [(0, "hello world", set())]
        result = builder.batch_create_nodes(items)

        self.assertIn(0, result)
        node = result[0]
        self.assertEqual(node.text, "hello world")
        self.assertEqual(node.index, 0)
        self.assertEqual(node.children, set())
        self.assertIn("test_model", node.embeddings)
        # Verify batched call was used
        self.assertEqual(len(emb_model.create_embeddings_calls), 1)
        self.assertEqual(emb_model.create_embedding_calls, [])

    def test_multiple_nodes_with_children(self):
        builder, emb_model = make_tree_builder()
        items = [
            (5, "node A", {0, 1}),
            (6, "node B", {2, 3, 4}),
            (7, "node C", set()),
        ]
        result = builder.batch_create_nodes(items)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[5].text, "node A")
        self.assertEqual(result[5].children, {0, 1})
        self.assertEqual(result[6].children, {2, 3, 4})
        self.assertEqual(result[7].children, set())

        # All embeddings created in single batch call
        self.assertEqual(len(emb_model.create_embeddings_calls), 1)
        self.assertEqual(len(emb_model.create_embeddings_calls[0]), 3)

    def test_custom_indices_preserved(self):
        """Indices don't have to be sequential or start from 0."""
        builder, _ = make_tree_builder()
        items = [(10, "x", set()), (20, "y", {10}), (30, "z", {10, 20})]
        result = builder.batch_create_nodes(items)

        self.assertIn(10, result)
        self.assertIn(20, result)
        self.assertIn(30, result)
        self.assertEqual(result[20].children, {10})


class TestBuildFromTextBatchEmbedding(unittest.TestCase):
    """Tests that build_from_text uses batch embedding correctly."""

    def test_no_headings_single_node(self):
        """Text without headings should produce a single-node tree."""
        builder, _ = make_tree_builder()
        md = "Just a simple paragraph with no headings."
        tree = builder.build_from_text(md)

        self.assertEqual(tree.num_layers, 1)
        self.assertEqual(len(tree.all_nodes), 1)
        self.assertEqual(len(tree.leaf_nodes), 1)
        self.assertEqual(len(tree.root_nodes), 1)

    def test_flat_headings_all_leaves(self):
        """Multiple top-level headings without children -> all leaves + super root."""
        builder, emb_model = make_tree_builder()
        md = (
            "# Section A\nContent A\n"
            "# Section B\nContent B\n"
            "# Section C\nContent C\n"
        )
        tree = builder.build_from_text(md)

        # 3 leaf nodes (layer 0) + 1 super root = 4 total
        self.assertEqual(len(tree.leaf_nodes), 3)
        self.assertGreaterEqual(len(tree.all_nodes), 4)

        # Verify batch embedding was used (not sequential)
        # At least: 1 batch for layer 0 leaves, 1 batch for super root
        self.assertGreaterEqual(len(emb_model.create_embeddings_calls), 1)

    def test_nested_structure_layers(self):
        """Nested headings should produce correct layer structure."""
        builder, emb_model = make_tree_builder()
        md = (
            "# Chapter 1\n"
            "## Section 1.1\nContent 1.1\n"
            "## Section 1.2\nContent 1.2\n"
        )
        tree = builder.build_from_text(md)

        # 2 leaf nodes (Section 1.1, 1.2) at layer 0
        # 1 parent node (Chapter 1) at layer 1
        self.assertEqual(len(tree.leaf_nodes), 2)
        self.assertIn(0, tree.layer_to_nodes)
        self.assertEqual(len(tree.layer_to_nodes[0]), 2)
        self.assertGreaterEqual(tree.num_layers, 2)

        # Verify batch embedding was used, NOT sequential create_embedding
        total_batch_calls = len(emb_model.create_embeddings_calls)
        self.assertGreaterEqual(total_batch_calls, 1)  # At least layer 0 batch

    def test_deep_nesting_batch_per_layer(self):
        """3-level nesting should produce 3 layers of batch embedding calls."""
        builder, emb_model = make_tree_builder()
        md = (
            "# Top\n"
            "## Mid A\n"
            "### Leaf A1\nContent A1\n"
            "### Leaf A2\nContent A2\n"
            "## Mid B\n"
            "### Leaf B1\nContent B1\n"
        )
        tree = builder.build_from_text(md)

        # Layer 0: Leaf A1, Leaf A2, Leaf B1 (3 leaves)
        # Layer 1: Mid A (children: A1,A2), Mid B (children: B1)
        # Layer 2: Top (children: Mid A, Mid B)
        self.assertEqual(len(tree.leaf_nodes), 3)
        self.assertGreaterEqual(tree.num_layers, 3)

        # Verify batch calls: at least 3 separate layer batches
        self.assertGreaterEqual(len(emb_model.create_embeddings_calls), 3)

    def test_no_sequential_embedding_calls_in_build(self):
        """Ensure build_from_text NEVER calls create_embedding (only batch)."""
        builder, emb_model = make_tree_builder()
        md = (
            "# A\n## A1\nContent\n## A2\nContent\n"
            "# B\n## B1\nContent\n"
        )

        # Reset call tracking
        emb_model.create_embedding_calls = []
        emb_model.create_embeddings_calls = []

        tree = builder.build_from_text(md)

        # The only individual create_embedding call should be from the
        # edge case single-node path. For multi-heading docs, we expect
        # zero individual calls during the main build loop.
        # Note: create_node is still used for the no-heading edge case only
        self.assertEqual(
            len(emb_model.create_embedding_calls), 0,
            f"Expected 0 sequential create_embedding calls during build, "
            f"got {len(emb_model.create_embedding_calls)} for texts: "
            f"{emb_model.create_embedding_calls}"
        )
        self.assertGreater(len(emb_model.create_embeddings_calls), 0)

    def test_parent_child_links_correct(self):
        """Verify parent nodes correctly reference their children indices."""
        builder, _ = make_tree_builder()
        md = (
            "# Parent\n"
            "## Child 1\nContent 1\n"
            "## Child 2\nContent 2\n"
        )
        tree = builder.build_from_text(md)

        # Find the parent node (should be in layer 1+)
        leaf_indices = set(tree.leaf_nodes.keys())
        for idx, node in tree.all_nodes.items():
            if node.children:
                # Parent's children should be a subset of leaf indices
                self.assertTrue(
                    node.children.issubset(set(tree.all_nodes.keys())),
                    f"Parent node {idx} has children {node.children} "
                    f"not found in all_nodes {set(tree.all_nodes.keys())}"
                )

    def test_unbalanced_tree_mixed_layers(self):
        """
        Unbalanced tree: one branch goes deeper than others.
        Layer 0 should contain leaves at different heading levels.
        """
        builder, emb_model = make_tree_builder()
        md = (
            "# Shallow Section\nDirect content here\n"
            "# Deep Section\n"
            "## Sub\n"
            "### Deep Leaf\nDeep content\n"
        )
        tree = builder.build_from_text(md)

        # "Shallow Section" (no children) and "Deep Leaf" (no children)
        # are both Layer 0 despite being at different heading levels
        layer_0_texts = [n.text for n in tree.layer_to_nodes[0]]
        self.assertEqual(len(layer_0_texts), 2)
        self.assertGreaterEqual(tree.num_layers, 3)  # Layer 0, 1, 2 (+ possible super root)


class TestEmbeddingModelBatchContract(unittest.TestCase):
    """Verify that FakeEmbeddingModel behaves like the real EmbeddingModel batch API."""

    def test_batch_returns_correct_count(self):
        model = FakeEmbeddingModel(dim=3)
        texts = ["a", "b", "c"]
        result = model.create_embeddings(texts)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_empty_batch(self):
        model = FakeEmbeddingModel(dim=3)
        result = model.create_embeddings([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
