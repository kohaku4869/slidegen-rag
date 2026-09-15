"""Unit tests for generate_node._format_sufficiency_note (facet-aware
version — reads facet_sufficiency: list[dict] instead of the old global
context_sufficient/missing_aspects pair)."""

from slidegen_rag.agent.nodes.generate_node import _format_sufficiency_note


def test_empty_facet_sufficiency_is_treated_as_fully_sufficient():
    note = _format_sufficiency_note([])
    assert "ĐẦY ĐỦ" in note
    assert "CÒN ĐÁNG TÌM THÊM" not in note


def test_all_facets_sufficient():
    facet_sufficiency = [
        {
            "label": "Facet A",
            "sufficient": True,
            "missing_aspects": [],
            "retrieve_count": 1,
        },
        {
            "label": "Facet B",
            "sufficient": True,
            "missing_aspects": [],
            "retrieve_count": 2,
        },
    ]
    note = _format_sufficiency_note(facet_sufficiency)
    assert "ĐẦY ĐỦ" in note
    assert "mọi khía cạnh" in note
    assert "CÒN ĐÁNG TÌM THÊM" not in note


def test_mixed_sufficiency_lists_each_facet_separately():
    facet_sufficiency = [
        {
            "label": "Facet A",
            "sufficient": True,
            "missing_aspects": [],
            "retrieve_count": 1,
        },
        {
            "label": "Facet B",
            "sufficient": False,
            "missing_aspects": ["chi tiết X"],
            "retrieve_count": 5,
        },
    ]
    note = _format_sufficiency_note(facet_sufficiency)

    assert 'Khía cạnh "Facet A": không cần tìm thêm' in note
    assert 'Khía cạnh "Facet B": CÒN ĐÁNG TÌM THÊM' in note
    assert "chi tiết X" in note


def test_all_facets_insufficient():
    facet_sufficiency = [
        {
            "label": "Facet A",
            "sufficient": False,
            "missing_aspects": ["a"],
            "retrieve_count": 5,
        },
        {
            "label": "Facet B",
            "sufficient": False,
            "missing_aspects": ["b"],
            "retrieve_count": 5,
        },
    ]
    note = _format_sufficiency_note(facet_sufficiency)

    assert 'Khía cạnh "Facet A": CÒN ĐÁNG TÌM THÊM' in note
    assert 'Khía cạnh "Facet B": CÒN ĐÁNG TÌM THÊM' in note


def test_query_shown_instead_of_label_when_both_present():
    facet_sufficiency = [
        {
            "label": "1. some truncated code-generated label…",
            "query": "Câu hỏi đầy đủ của khía cạnh này là gì?",
            "sufficient": False,
            "missing_aspects": ["a"],
            "retrieve_count": 5,
        },
    ]
    note = _format_sufficiency_note(facet_sufficiency)
    assert 'Khía cạnh "Câu hỏi đầy đủ của khía cạnh này là gì?"' in note
    assert "some truncated code-generated label" not in note


def test_falls_back_to_label_when_query_missing():
    facet_sufficiency = [
        {
            "label": "Facet A",
            "sufficient": False,
            "missing_aspects": ["a"],
            "retrieve_count": 5,
        },
    ]
    note = _format_sufficiency_note(facet_sufficiency)
    assert 'Khía cạnh "Facet A"' in note


def test_missing_aspects_empty_list_shows_placeholder():
    facet_sufficiency = [
        {
            "label": "Facet A",
            "sufficient": False,
            "missing_aspects": [],
            "retrieve_count": 5,
        },
    ]
    note = _format_sufficiency_note(facet_sufficiency)
    assert "không rõ chi tiết" in note
