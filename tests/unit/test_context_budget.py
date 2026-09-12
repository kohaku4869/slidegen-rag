from slidegen_rag.agent.utils.context_budget import CONTEXT_BUDGET_CHARS, build_context


def test_build_context_passthrough_when_under_budget():
    docs = ["chunk one", "chunk two"]
    assert build_context(docs, max_chars=1000) == "chunk one\n\nchunk two"


def test_build_context_empty_list_returns_empty_string():
    assert build_context([], max_chars=1000) == ""


def test_build_context_over_budget_keeps_newest_chunks_in_original_order():
    docs = ["a" * 40, "b" * 40, "c" * 40]  # oldest -> newest
    result = build_context(
        docs, max_chars=90
    )  # fits last two ("b"*40 + sep + "c"*40 = 82)
    assert result.index("b" * 40) < result.index("c" * 40)
    assert "a" * 40 not in result


def test_build_context_drops_note_mentions_correct_count():
    docs = ["a" * 40, "b" * 40, "c" * 40]
    result = build_context(docs, max_chars=90)
    assert "1 older chunk(s) omitted for length" in result


def test_build_context_single_huge_chunk_is_truncated_not_dropped():
    docs = ["short old chunk", "z" * 500]
    result = build_context(docs, max_chars=100)
    assert result != ""
    assert "short old chunk" not in result
    assert "z" * 100 in result


def test_build_context_default_uses_shared_budget_constant():
    docs = ["x"]
    assert build_context(docs) == "x"
    assert CONTEXT_BUDGET_CHARS == 32_000
