"""Unit tests cho `progress_callback` của `TreeBuilder.build_from_text` (xem
plan6.md C3). Không gọi LLM/embedding thật — dùng fake model giống hệt
`tests/unit/test_markdown_spans.py`."""

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


# Hai H1 top-level -> root_jsons có 2 phần tử -> kích hoạt nhánh super-root.
TWO_ROOTS_MD = """\
# CHƯƠNG 1: NHỮNG VẤN ĐỀ CHUNG
Đoạn mở đầu chương 1.

## 1.1 Khái niệm
Nội dung mục 1.1.

## 1.2 Đặc điểm
Nội dung mục 1.2.

# CHƯƠNG 2: PHẦN THỨ HAI
Nội dung chương 2, không có mục con.
"""

# Một H1 duy nhất -> root_jsons có 1 phần tử -> KHÔNG có nhánh super-root.
ONE_ROOT_MD = """\
# CHƯƠNG DUY NHẤT
Đoạn mở đầu.

## 1.1 Mục con
Nội dung mục con.
"""

FLAT_MD = "Chỉ có một đoạn văn bản, không có heading nào cả.\nDòng thứ hai."


def test_progress_callback_total_matches_sum_of_done_units_two_roots():
    calls = []
    _builder().build_from_text(
        TWO_ROOTS_MD, doc_id="sample", progress_callback=lambda label, done, total: calls.append((label, done, total))
    )
    assert calls, "progress_callback phải được gọi ít nhất 1 lần"
    totals = {c[2] for c in calls}
    assert len(totals) == 1, "total_units phải cố định xuyên suốt 1 lần build"
    total = totals.pop()
    assert calls[-1][1] == total, "done_units ở lần gọi cuối phải bằng total_units"
    # done_units phải tăng đơn điệu, không lặp/giảm.
    dones = [c[1] for c in calls]
    assert dones == sorted(dones)
    assert len(set(dones)) == len(dones)


def test_progress_callback_no_super_root_branch_when_single_top_level():
    calls = []
    _builder().build_from_text(
        ONE_ROOT_MD, doc_id="sample", progress_callback=lambda label, done, total: calls.append((label, done, total))
    )
    assert all("Super root" not in c[0] for c in calls)
    assert calls[-1][1] == calls[-1][2]


def test_progress_callback_flat_document_single_call():
    calls = []
    _builder().build_from_text(
        FLAT_MD, doc_id="flat", progress_callback=lambda label, done, total: calls.append((label, done, total))
    )
    assert calls == [(calls[0][0], 1, 1)]


def test_progress_callback_none_does_not_change_behavior():
    tree_with = _builder().build_from_text(TWO_ROOTS_MD, doc_id="sample", progress_callback=lambda *a: None)
    tree_without = _builder().build_from_text(TWO_ROOTS_MD, doc_id="sample")
    assert len(tree_with.all_nodes) == len(tree_without.all_nodes)
    assert tree_with.num_layers == tree_without.num_layers


def test_progress_callback_exception_propagates_and_aborts_build():
    class Cancelled(Exception):
        pass

    def cancel_after_first(label, done, total):
        raise Cancelled("huỷ job")

    try:
        _builder().build_from_text(
            TWO_ROOTS_MD, doc_id="sample", progress_callback=cancel_after_first
        )
        assert False, "phải raise Cancelled"
    except Cancelled:
        pass
