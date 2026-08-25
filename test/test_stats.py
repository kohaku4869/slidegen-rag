import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tree.stats import BuildStats, LayerStat, format_ingest_report, format_multi_file_summary
from tree.builder.tree_builder import TreeBuilder
from tree.structure.tree_structure import Tree


class FakeEmbeddingModel:
    def __init__(self, dim=4):
        self.dim = dim

    def create_embedding(self, text: str):
        return [1.0] * self.dim

    def create_embeddings(self, texts):
        return [[1.0] * self.dim for _ in texts]


class FakeSummarizationModel:
    def summarize(self, context, max_tokens=2048) -> str:
        return f"Summary of: {context[:30]}"

    def batch_summarize(self, contexts, max_tokens=2048):
        return [f"Summary of: {c[:30]}" for c in contexts]


def make_test_builder():
    config = MagicMock()
    config.tokenizer = None
    config.max_tokens = 100
    config.num_layers = 5
    config.top_k = 5
    config.threshold = 0.5
    config.selection_mode = "top_k"
    config.summarization_length = 100
    config.summarization_model = FakeSummarizationModel()
    config.embedding_models = {"test_model": FakeEmbeddingModel()}
    config.cluster_embedding_model = "test_model"
    config.log_config = MagicMock(return_value="test_config")

    return TreeBuilder(config)


class TestStatsModule(unittest.TestCase):
    def test_layer_stat_creation(self):
        stat = LayerStat(
            layer=0,
            name="Leaf Nodes",
            num_nodes=10,
            num_llm_calls=0,
            summarize_time=0.0,
            embedding_time=0.5,
        )
        self.assertEqual(stat.total_time, 0.5)
        d = stat.to_dict()
        self.assertEqual(d["layer"], 0)
        self.assertEqual(d["num_nodes"], 10)
        self.assertEqual(d["num_llm_calls"], 0)
        self.assertEqual(d["summarize_time"], 0.0)
        self.assertEqual(d["embedding_time"], 0.5)
        self.assertEqual(d["total_time"], 0.5)

    def test_build_stats_creation_and_dict(self):
        l0 = LayerStat(0, "Leaf Nodes", 10, 0, 0.0, 0.5)
        l1 = LayerStat(1, "Layer 1 Summaries", 2, 2, 3.2, 0.1)
        stats = BuildStats(
            filename="test.md",
            src_path="data/src/test.md",
            out_path="data/index/test.pkl",
            total_time=4.0,
            parse_time=0.05,
            total_summarize_time=3.2,
            total_embedding_time=0.6,
            save_time=0.1,
            file_size_bytes=1024,
            total_nodes=12,
            total_llm_calls=2,
            num_layers=2,
            layers=[l0, l1],
        )
        d = stats.to_dict()
        self.assertEqual(d["filename"], "test.md")
        self.assertEqual(d["total_nodes"], 12)
        self.assertEqual(d["total_llm_calls"], 2)
        self.assertEqual(len(d["layers"]), 2)

    def test_format_ingest_report_rendering(self):
        l0 = LayerStat(0, "Leaf Nodes", 20, 0, 0.0, 0.8)
        l1 = LayerStat(1, "Layer 1 Summaries", 4, 4, 8.5, 0.2)
        stats = BuildStats(
            filename="pldc.md",
            src_path="data/src/pldc.md",
            out_path="data/index/pldc.pkl",
            total_time=10.0,
            parse_time=0.1,
            total_summarize_time=8.5,
            total_embedding_time=1.0,
            save_time=0.2,
            file_size_bytes=50000,
            total_nodes=24,
            total_llm_calls=4,
            num_layers=2,
            layers=[l0, l1],
        )
        report = format_ingest_report(stats)
        self.assertIn("INGESTION PERFORMANCE & METRICS REPORT", report)
        self.assertIn("Leaf Nodes", report)
        self.assertIn("Layer 1 Summaries", report)
        self.assertIn("LLM Summarization Phase", report)
        self.assertIn("Vector Embedding Phase", report)
        self.assertIn("BOTTLENECK ANALYSIS", report)

    def test_format_multi_file_summary(self):
        s1 = BuildStats(filename="doc1.md", total_nodes=10, total_llm_calls=2, total_summarize_time=2.0, total_embedding_time=0.5, total_time=3.0)
        s2 = BuildStats(filename="doc2.md", total_nodes=20, total_llm_calls=4, total_summarize_time=4.0, total_embedding_time=1.0, total_time=5.5)
        summary = format_multi_file_summary([s1, s2])
        self.assertIn("BATCH INGESTION MULTI-FILE SUMMARY", summary)
        self.assertIn("doc1.md", summary)
        self.assertIn("doc2.md", summary)
        self.assertIn("TOTAL ( 2 files)", summary)


class TestTreeBuilderBuildStats(unittest.TestCase):
    def test_build_from_text_stats_attached(self):
        builder = make_test_builder()
        md = (
            "# Chapter 1\n"
            "## Section 1.1\nLeaf text 1\n"
            "## Section 1.2\nLeaf text 2\n"
            "# Chapter 2\n"
            "## Section 2.1\nLeaf text 3\n"
        )
        tree = builder.build_from_text(md)

        self.assertIsNotNone(tree.build_stats)
        self.assertIsInstance(tree.build_stats, BuildStats)
        self.assertEqual(builder.last_build_stats, tree.build_stats)

        # Check nodes count matches
        self.assertEqual(tree.build_stats.total_nodes, len(tree.all_nodes))
        self.assertEqual(tree.build_stats.num_layers, tree.num_layers)

        # Check layer stats
        layers = tree.build_stats.layers
        self.assertGreaterEqual(len(layers), 2)

        # Layer 0 has 0 LLM calls
        self.assertEqual(layers[0].layer, 0)
        self.assertEqual(layers[0].num_llm_calls, 0)
        self.assertEqual(layers[0].num_nodes, len(tree.layer_to_nodes[0]))

        # Higher layers have LLM calls
        total_higher_calls = sum(l.num_llm_calls for l in layers[1:])
        self.assertGreater(total_higher_calls, 0)
        self.assertEqual(tree.build_stats.total_llm_calls, total_higher_calls)

    def test_single_node_stats(self):
        builder = make_test_builder()
        md = "Just plain text without headers"
        tree = builder.build_from_text(md)

        self.assertIsNotNone(tree.build_stats)
        self.assertEqual(tree.build_stats.total_nodes, 1)
        self.assertEqual(tree.build_stats.total_llm_calls, 0)
        self.assertEqual(len(tree.build_stats.layers), 1)
        self.assertEqual(tree.build_stats.layers[0].num_nodes, 1)


if __name__ == "__main__":
    unittest.main()
