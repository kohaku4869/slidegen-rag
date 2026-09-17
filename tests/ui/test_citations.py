"""Unit tests cho `ui/components/citations.py::extract_citations` — phần
thuần, không Streamlit (xem plan6.md D1)."""

from ui.components.citations import extract_citations


def test_extract_citations_filters_none_node_ids():
    final_state = {
        "documents": ["a", "b", "c"],
        "document_node_ids": [1, None, 3],
        "document_priorities": [5, 9, 2],
    }
    result = extract_citations(final_state)
    assert {c["node_id"] for c in result} == {1, 3}


def test_extract_citations_sorted_by_priority_descending():
    final_state = {
        "documents": ["a", "b", "c"],
        "document_node_ids": [1, 2, 3],
        "document_priorities": [1, 9, 5],
    }
    result = extract_citations(final_state)
    assert [c["node_id"] for c in result] == [2, 3, 1]


def test_extract_citations_dedupes_keeping_highest_priority():
    final_state = {
        "documents": ["a", "b"],
        "document_node_ids": [7, 7],
        "document_priorities": [2, 8],
    }
    result = extract_citations(final_state)
    assert len(result) == 1
    assert result[0]["priority"] == 8


def test_extract_citations_caps_at_max_chips():
    final_state = {
        "documents": [f"d{i}" for i in range(20)],
        "document_node_ids": list(range(20)),
        "document_priorities": list(range(20)),
    }
    result = extract_citations(final_state, max_chips=8)
    assert len(result) == 8


def test_extract_citations_empty_state_returns_empty_list():
    assert extract_citations({}) == []
