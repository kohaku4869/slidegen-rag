"""Unit tests for collect_facets_node: merges facet_results into a single
deduped documents pool (string-identity dedup, first-seen id kept) and builds
facet_sufficiency with the right shape/count/content."""

import pytest

from slidegen_rag.agent.nodes.collect_facets_node import collect_facets_node


@pytest.mark.asyncio
async def test_merges_and_dedups_documents_across_facets():
    facet_results = [
        {
            "facet_label": "Facet A",
            "documents": ["doc shared", "doc only in A"],
            "document_node_ids": [1, 2],
            "context_sufficient": True,
            "missing_aspects": [],
            "retrieve_count": 1,
        },
        {
            "facet_label": "Facet B",
            "documents": ["doc shared", "doc only in B"],
            "document_node_ids": [99, 3],
            "context_sufficient": False,
            "missing_aspects": ["some detail"],
            "retrieve_count": 4,
        },
    ]

    result = await collect_facets_node({"facet_results": facet_results})

    # "doc shared" appears once, keeping the id from the FIRST facet that saw it.
    assert result["documents"] == ["doc shared", "doc only in A", "doc only in B"]
    assert result["document_node_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_facet_sufficiency_shape_and_content():
    facet_results = [
        {
            "facet_label": "Facet A",
            "documents": [],
            "document_node_ids": [],
            "context_sufficient": True,
            "missing_aspects": [],
            "retrieve_count": 2,
        },
        {
            "facet_label": "Facet B",
            "documents": [],
            "document_node_ids": [],
            "context_sufficient": False,
            "missing_aspects": ["thiếu chi tiết X"],
            "retrieve_count": 5,
        },
        {
            "facet_label": "Facet C",
            "documents": [],
            "document_node_ids": [],
            "context_sufficient": True,
            "missing_aspects": [],
            "retrieve_count": 1,
        },
    ]

    result = await collect_facets_node({"facet_results": facet_results})
    sufficiency = result["facet_sufficiency"]

    assert len(sufficiency) == 3
    assert sufficiency[0] == {
        "label": "Facet A",
        "sufficient": True,
        "missing_aspects": [],
        "retrieve_count": 2,
    }
    assert sufficiency[1] == {
        "label": "Facet B",
        "sufficient": False,
        "missing_aspects": ["thiếu chi tiết X"],
        "retrieve_count": 5,
    }
    assert sufficiency[2]["label"] == "Facet C"
    assert sufficiency[2]["sufficient"] is True


@pytest.mark.asyncio
async def test_no_facet_results_returns_empty_pool():
    result = await collect_facets_node({"facet_results": []})

    assert result["documents"] == []
    assert result["document_node_ids"] == []
    assert result["facet_sufficiency"] == []
