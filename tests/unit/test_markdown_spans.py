"""
Unit tests for TreeBuilder's schema-v2 metadata: line spans, heading path,
depth, and token counts captured by parse_markdown_to_hierarchical_json /
_apply_json_metadata / build_from_text. No LLM/embedding calls — uses fake
models exactly like benchmarks/test_batch_embedding.py.

Line numbers throughout are relative to `markdown_text.strip().split("\n")`,
which is also what build_from_text stores as Tree.source_text — so a node's
`source_lines[node.line_start:node.line_end]` should always be non-empty and
start with something recognizable from the node's own heading.
"""

import tiktoken

from slidegen_rag.tree.builder.tree_builder import TreeBuilder, TreeBuilderConfig
from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import BaseSummarizationModel


class FakeEmbeddingModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [float(hash(text) % 100)]

    def create_embeddings(self, texts):
        return [self.create_embedding(t) for t in texts]


class FakeSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return f"tóm tắt: {context[:40]}"

    def batch_summarize(self, contexts, max_tokens=150):
        return [self.summarize(c) for c in contexts]


def _builder() -> TreeBuilder:
    config = TreeBuilderConfig(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        summarization_model=FakeSummarizationModel(),
        embedding_models={"SBERT": FakeEmbeddingModel()},
    )
    return TreeBuilder(config)


SAMPLE_MD = """\
# CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG
Đoạn mở đầu chương 1.

## 1.1 Khái niệm
Nội dung mục 1.1 dòng một.
Nội dung mục 1.1 dòng hai.

## 1.2 Đặc điểm
Nội dung mục 1.2.

# CHƯƠNG 2: PHẦN THỨ HAI
Nội dung chương 2, không có mục con.
"""


def test_tree_has_schema_v2_and_source_text():
    tree = _builder().build_from_text(SAMPLE_MD, doc_id="sample", source_path="sample.md")
    assert tree.schema_version == 2
    assert tree.doc_id == "sample"
    assert tree.source_path == "sample.md"
    assert tree.source_text == SAMPLE_MD.strip()
    assert tree.source_sha1  # non-empty


def test_leaf_line_span_recovers_own_content_from_source():
    tree = _builder().build_from_text(SAMPLE_MD, doc_id="sample")
    source_lines = tree.source_text.split("\n")

    khai_niem = next(n for n in tree.leaf_nodes.values() if n.title == "Khái niệm")
    span = source_lines[khai_niem.line_start:khai_niem.line_end]
    assert any("1.1 Khái niệm" in line for line in span)
    assert any("dòng một" in line for line in span)
    assert any("dòng hai" in line for line in span)


def test_heading_path_and_depth_reflect_nesting():
    tree = _builder().build_from_text(SAMPLE_MD, doc_id="sample")
    khai_niem = next(n for n in tree.leaf_nodes.values() if n.title == "Khái niệm")
    assert khai_niem.heading_path == ("CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG", "Khái niệm")
    assert khai_niem.depth == 1

    chuong1 = next(
        n for nodes in tree.layer_to_nodes.values() for n in nodes
        if n.title == "CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG"
    )
    assert chuong1.heading_path == ("CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG",)
    assert chuong1.depth == 0
    assert chuong1.heading_level == 1  # CHƯƠNG keyword forces level 1


def test_summary_node_subtree_span_covers_all_children():
    tree = _builder().build_from_text(SAMPLE_MD, doc_id="sample")
    source_lines = tree.source_text.split("\n")

    chuong1 = next(
        n for nodes in tree.layer_to_nodes.values() for n in nodes
        if n.title == "CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG"
    )
    span_text = "\n".join(source_lines[chuong1.line_start:chuong1.line_end])
    assert "1.1 Khái niệm" in span_text
    assert "1.2 Đặc điểm" in span_text
    assert "dòng một" in span_text
    # Chapter 1's span must not bleed into chapter 2's content
    assert "không có mục con" not in span_text


def test_parent_pointer_and_kind_and_subtree_tokens():
    tree = _builder().build_from_text(SAMPLE_MD, doc_id="sample")

    khai_niem = next(n for n in tree.leaf_nodes.values() if n.title == "Khái niệm")
    chuong1 = next(
        n for nodes in tree.layer_to_nodes.values() for n in nodes
        if n.title == "CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG"
    )
    assert khai_niem.kind == "leaf"
    assert khai_niem.parent == chuong1.index
    assert tree.parent_of[khai_niem.index] == chuong1.index

    root = next(iter(tree.root_nodes.values()))
    assert root.kind == "root"

    # subtree_tokens must be at least the sum of the direct children's
    # subtree_tokens (parent's own summary text adds a bit more on top).
    child_total = sum(tree.all_nodes[c].subtree_tokens for c in chuong1.children)
    assert chuong1.subtree_tokens >= child_total
    assert chuong1.subtree_tokens == chuong1.n_tokens + child_total


def test_bullet_after_hash_treated_as_content_not_heading():
    md = """\
# CHƯƠNG 1: TIÊU ĐỀ
## 1.1 Mục con
# - đây là gạch đầu dòng bị OCR gắn nhầm dấu #
Nội dung tiếp theo.
"""
    tree = _builder().build_from_text(md, doc_id="sample")
    titles = [n.title for n in tree.all_nodes.values()]
    assert "đây là gạch đầu dòng bị OCR gắn nhầm dấu #".lstrip("- ") not in titles
    muc_con = next(n for n in tree.leaf_nodes.values() if n.title == "Mục con")
    assert "gạch đầu dòng" in muc_con.text


def test_no_heading_document_gets_single_root_node_with_full_span():
    md = "Chỉ có một đoạn văn bản, không có heading nào cả.\nDòng thứ hai."
    tree = _builder().build_from_text(md, doc_id="flat")
    assert tree.num_layers == 1
    node = next(iter(tree.all_nodes.values()))
    assert node.kind == "root"
    assert node.doc_id == "flat"
    source_lines = tree.source_text.split("\n")
    assert node.line_start == 0
    assert node.line_end == len(source_lines)
    assert node.n_tokens > 0
    assert node.subtree_tokens == node.n_tokens
