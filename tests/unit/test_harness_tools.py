"""
Unit tests for the 4 harness tools (search/find/open/summarize) and
compaction — driven purely off DocStore + ReferenceRegistry, no LLM/network.
"""

import pytest
import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from slidegen_rag.agent.harness.compaction import compact_messages
from slidegen_rag.agent.harness.config import AgenticConfig
from slidegen_rag.agent.harness.doc_store import DocStore
from slidegen_rag.agent.harness.refs import ReferenceRegistry
from slidegen_rag.agent.harness.tools import TurnBox, build_agentic_tools
from slidegen_rag.tree.builder.tree_builder import TreeBuilder, TreeBuilderConfig
from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import BaseSummarizationModel

# Deliberately verbose leaf bodies so open()'s token-budget windowing has
# something real to bite into, and long enough chapters that a summary
# node's span meaningfully differs from any one child's span.
SAMPLE_MD = """\
# CHƯƠNG 1: LUẬT DÂN SỰ
## 1.1 Năng lực hành vi
Người từ đủ 18 tuổi có năng lực hành vi dân sự đầy đủ theo quy định pháp luật.
Người dưới 6 tuổi không có năng lực hành vi dân sự và mọi giao dịch phải do
người đại diện theo pháp luật xác lập và thực hiện thay cho họ.
Người từ đủ 6 tuổi đến chưa đủ 18 tuổi khi xác lập giao dịch dân sự phải
được người đại diện theo pháp luật đồng ý, trừ giao dịch phục vụ nhu cầu
sinh hoạt hàng ngày phù hợp với lứa tuổi.

## 1.2 Hợp đồng
Hợp đồng dân sự là sự thỏa thuận giữa các bên về việc xác lập, thay đổi
hoặc chấm dứt quyền, nghĩa vụ dân sự. Hợp đồng vô hiệu khi vi phạm điều
cấm của luật hoặc trái đạo đức xã hội.

# CHƯƠNG 2: LUẬT HÌNH SỰ
## 2.1 Tội phạm
Tội phạm là hành vi nguy hiểm cho xã hội được quy định trong Bộ luật hình sự.
"""


class FakeEmbeddingModel(BaseEmbeddingModel):
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


@pytest.fixture()
def store() -> DocStore:
    config = TreeBuilderConfig(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        summarization_model=FakeSummarizationModel(),
        embedding_models={"SBERT": FakeEmbeddingModel()},
    )
    tree = TreeBuilder(config).build_from_text(SAMPLE_MD, doc_id="sample", source_path="sample.md")
    return DocStore(tree, embedding_model=FakeEmbeddingModel(), embedding_model_key="SBERT")


@pytest.fixture()
def harness(store):
    registry = ReferenceRegistry()
    turn_box = TurnBox()
    turn_box.turn = 1
    tools = build_agentic_tools(store, registry, AgenticConfig(), turn_box)
    by_name = {t.name: t for t in tools}
    return store, registry, turn_box, by_name


def test_search_registers_stable_refs_and_returns_snippets(harness):
    store, registry, turn_box, tools = harness
    result = tools["search"].invoke({"queries": ["năng lực hành vi dân sự"]})
    assert "turn1search" in result
    assert len(registry) >= 1

    # calling again with the same query should re-surface at least one
    # already-registered ref as a one-liner, not duplicate it under a new id
    result2 = tools["search"].invoke({"queries": ["năng lực hành vi dân sự"]})
    assert "đã thấy trước đó" in result2


def test_search_coerces_bare_string_query(harness):
    store, registry, turn_box, tools = harness
    # queries as a bare str (not a list) must not raise
    result = tools["search"].invoke({"queries": "hợp đồng dân sự"})
    assert result and "search: không có" not in result


def test_search_respects_multi_query_disabled(store):
    registry = ReferenceRegistry()
    turn_box = TurnBox()
    turn_box.turn = 1
    config = AgenticConfig(enable_multi_query=False)
    tools = build_agentic_tools(store, registry, config, turn_box)
    by_name = {t.name: t for t in tools}
    # two queries passed, but only the first should be used
    by_name["search"].invoke({"queries": ["năng lực hành vi", "hợp đồng"]})
    # can't directly observe which query ran, but this should not error and
    # should still return something for the first query's topic
    assert len(registry) >= 1


def test_open_on_summary_returns_children_toc(harness):
    store, registry, turn_box, tools = harness
    chuong1 = next(n for n in store.tree.all_nodes.values() if n.title == "CHƯƠNG 1: LUẬT DÂN SỰ")
    ref_id, _ = registry.register(chuong1, turn=1)

    result = tools["open"].invoke({"reference_id": ref_id})
    assert "children" in result
    assert "Năng lực hành vi" in result
    assert "Hợp đồng" in result
    # children should now be registered with their own ref ids
    child_titles = {c.title for c in store.children(chuong1.index)}
    assert child_titles == {"Năng lực hành vi", "Hợp đồng"}
    for child in store.children(chuong1.index):
        assert registry.ref_id_for_node(child.index) is not None


def test_open_on_leaf_windows_source_and_offers_continuation(harness):
    store, registry, turn_box, tools = harness
    leaf = next(n for n in store.tree.leaf_nodes.values() if n.title == "Năng lực hành vi")
    ref_id, _ = registry.register(leaf, turn=1)

    result = tools["open"].invoke({"reference_id": ref_id})
    assert "Viewing lines" in result
    assert "18 tuổi" in result
    # line numbers are actually printed
    assert f"{leaf.line_start:>5}|" in result or str(leaf.line_start) in result


def test_open_unknown_reference_id_is_soft_failure(harness):
    store, registry, turn_box, tools = harness
    result = tools["open"].invoke({"reference_id": "turn99search99"})
    assert "không xác định" in result


def test_find_returns_passage_for_known_string_and_explicit_no_match(harness):
    store, registry, turn_box, tools = harness
    leaf = next(n for n in store.tree.leaf_nodes.values() if n.title == "Năng lực hành vi")
    ref_id, _ = registry.register(leaf, turn=1)

    result = tools["find"].invoke({"reference_id": ref_id, "patterns": ["18 tuổi", "một cụm từ chắc chắn không tồn tại"]})
    assert "18 tuổi" in result
    assert "không khớp" in result


def test_find_document_wide_scan(harness):
    store, registry, turn_box, tools = harness
    result = tools["find"].invoke({"reference_id": "document", "patterns": ["Tội phạm"]})
    assert "Tội phạm" in result
    assert "không khớp" not in result


def test_find_unicode_nfc_nfd_normalization(harness):
    import unicodedata
    store, registry, turn_box, tools = harness
    # "Hợp đồng" as NFD (decomposed diacritics) must still match NFC source text
    nfd_pattern = unicodedata.normalize("NFD", "Hợp đồng")
    result = tools["find"].invoke({"reference_id": "document", "patterns": [nfd_pattern]})
    assert "không khớp" not in result


def test_summarize_tool_confirms_kept_refs(harness):
    store, registry, turn_box, tools = harness
    leaf = next(n for n in store.tree.leaf_nodes.values() if n.title == "Năng lực hành vi")
    ref_id, _ = registry.register(leaf, turn=1)

    result = tools["summarize"].invoke({"notes": "đã tìm thấy độ tuổi", "keep_reference_ids": [ref_id]})
    assert ref_id in result
    assert "Bỏ qua" not in result  # nothing unresolved


def test_summarize_tool_reports_unresolved_ref(harness):
    store, registry, turn_box, tools = harness
    result = tools["summarize"].invoke({"notes": "x", "keep_reference_ids": ["turn99search99"]})
    assert "Bỏ qua" in result


def test_navigation_tools_absent_when_disabled(store):
    registry = ReferenceRegistry()
    turn_box = TurnBox()
    config = AgenticConfig(enable_navigation=False)
    tools = build_agentic_tools(store, registry, config, turn_box)
    names = {t.name for t in tools}
    assert "open" not in names
    assert "find" not in names
    assert "search" in names


def test_summarize_tool_absent_when_disabled(store):
    registry = ReferenceRegistry()
    turn_box = TurnBox()
    config = AgenticConfig(enable_summarize=False)
    tools = build_agentic_tools(store, registry, config, turn_box)
    names = {t.name for t in tools}
    assert "summarize" not in names


# ── compaction ────────────────────────────────────────────────────────────

def test_compact_messages_removes_tool_rounds_and_keeps_full_content_for_kept_refs(store):
    registry = ReferenceRegistry()
    leaf = next(n for n in store.tree.leaf_nodes.values() if n.title == "Năng lực hành vi")
    ref_id, _ = registry.register(leaf, turn=1)

    messages = [
        HumanMessage(content="câu hỏi gốc", id="h1"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "call1"}], id="a1"),
        ToolMessage(content="kết quả tìm kiếm", tool_call_id="call1", id="t1"),
    ]

    removals, summary_message = compact_messages(messages, "đã biết độ tuổi", {ref_id}, registry, store)

    removed_ids = {r.id for r in removals}
    assert removed_ids == {"a1", "t1"}
    assert all(isinstance(r, RemoveMessage) for r in removals)
    assert ref_id in summary_message.content
    assert leaf.text in summary_message.content  # kept ref's full text is preserved
    assert "đã biết độ tuổi" in summary_message.content


def test_compact_messages_preserves_headers_for_dropped_refs_too(store):
    """Production lesson from the paper: the model must still be able to
    open() a reference it did NOT ask to keep — only its full content is
    dropped, not its header."""
    registry = ReferenceRegistry()
    leaf = next(n for n in store.tree.leaf_nodes.values() if n.title == "Hợp đồng")
    ref_id, _ = registry.register(leaf, turn=1)

    _, summary_message = compact_messages([], "notes", keep_ref_ids=set(), registry=registry, store=store)
    assert ref_id in summary_message.content
    assert leaf.title in summary_message.content
