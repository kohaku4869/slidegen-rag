"""
Unit tests for agent.utils.search_history — the (tool, query) dedup-key logic
that retrieve_node relies on to avoid re-issuing identical tool calls. Pure
functions, no LLM/network.
"""

from slidegen_rag.agent.utils.search_history import (
    call_key,
    fingerprint,
    normalize_query,
    tried_keys,
)


def test_normalize_query_lowercases_and_collapses_whitespace():
    assert normalize_query("  Năng Lực   Hành Vi  ") == "năng lực hành vi"


def test_normalize_query_handles_none():
    assert normalize_query(None) == ""


def test_call_key_uses_normalized_query_for_search_tools():
    tc = {"name": "semantic_search", "args": {"query": "  Quan Hệ Pháp Luật  ", "top_k": 5}}
    assert call_key(tc) == ("semantic_search", "quan hệ pháp luật")


def test_call_key_ignores_top_k_so_same_query_different_top_k_collides():
    tc_a = {"name": "keyword_search", "args": {"query": "hợp đồng", "top_k": 3}}
    tc_b = {"name": "keyword_search", "args": {"query": "hợp đồng", "top_k": 10}}
    assert call_key(tc_a) == call_key(tc_b)


def test_call_key_uses_node_id_for_expand_node():
    tc = {"name": "expand_node", "args": {"node_id": 42}}
    assert call_key(tc) == ("expand_node", "42")


def test_tried_keys_normalizes_history_entries():
    history = [
        {"tool": "semantic_search", "query": "Năng lực hành vi"},
        {"tool": "keyword_search", "query": "  hợp đồng "},
    ]
    assert tried_keys(history) == {
        ("semantic_search", "năng lực hành vi"),
        ("keyword_search", "hợp đồng"),
    }


def test_fingerprint_is_deterministic_and_whitespace_insensitive():
    a = fingerprint("Điều  20   Bộ luật dân sự")
    b = fingerprint("Điều 20 Bộ luật dân sự")
    assert a == b
    assert len(a) == 16  # blake2b digest_size=8 -> 16 hex chars


def test_fingerprint_differs_for_different_content():
    assert fingerprint("nội dung A") != fingerprint("nội dung B")
