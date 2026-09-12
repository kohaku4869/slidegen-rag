"""
Unit tests for agent.utils.search_history — the (tool, query) dedup-key logic
that retrieve_node relies on to avoid re-issuing identical tool calls. Pure
functions, no LLM/network.
"""

from slidegen_rag.agent.utils.search_history import (
    call_key,
    fingerprint,
    new_chunks_in_round,
    normalize_query,
    tried_keys,
)


def test_normalize_query_lowercases_and_collapses_whitespace():
    assert normalize_query("  Năng Lực   Hành Vi  ") == "năng lực hành vi"


def test_normalize_query_handles_none():
    assert normalize_query(None) == ""


def test_call_key_uses_normalized_query_for_search_tools():
    tc = {
        "name": "semantic_search",
        "args": {"query": "  Quan Hệ Pháp Luật  ", "top_k": 5},
    }
    assert call_key(tc) == ("semantic_search", "quan hệ pháp luật")


def test_call_key_ignores_top_k_so_same_query_different_top_k_collides():
    tc_a = {"name": "keyword_search", "args": {"query": "hợp đồng", "top_k": 3}}
    tc_b = {"name": "keyword_search", "args": {"query": "hợp đồng", "top_k": 10}}
    assert call_key(tc_a) == call_key(tc_b)


def test_call_key_uses_node_id_for_expand_node():
    tc = {"name": "expand_node", "args": {"node_id": 42}}
    assert call_key(tc) == ("expand_node", "42:children")


def test_call_key_defaults_expand_node_direction_to_children():
    tc = {"name": "expand_node", "args": {"node_id": 42, "direction": "children"}}
    assert call_key(tc) == ("expand_node", "42:children")


def test_call_key_distinguishes_expand_node_directions():
    children = {"name": "expand_node", "args": {"node_id": 42, "direction": "children"}}
    parent = {"name": "expand_node", "args": {"node_id": 42, "direction": "parent"}}
    siblings = {"name": "expand_node", "args": {"node_id": 42, "direction": "siblings"}}
    keys = {call_key(children), call_key(parent), call_key(siblings)}
    assert len(keys) == 3


def test_call_key_for_find_in_subtree_uses_root_and_sorted_patterns():
    tc = {
        "name": "find_in_subtree",
        "args": {"root_node_id": 7, "patterns": ["Beta", "alpha"]},
    }
    assert call_key(tc) == ("find_in_subtree", "7:alpha,beta")


def test_call_key_find_in_subtree_same_patterns_different_order_collide():
    tc_a = {
        "name": "find_in_subtree",
        "args": {"root_node_id": 7, "patterns": ["alpha", "beta"]},
    }
    tc_b = {
        "name": "find_in_subtree",
        "args": {"root_node_id": 7, "patterns": ["beta", "alpha"]},
    }
    assert call_key(tc_a) == call_key(tc_b)


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


def test_new_chunks_in_round_sums_multiple_entries_same_round():
    history = [
        {"round": 2, "tool": "semantic_search", "query": "a", "new_chunks": 3},
        {"round": 2, "tool": "keyword_search", "query": "b", "new_chunks": 1},
        {"round": 3, "tool": "semantic_search", "query": "c", "new_chunks": 5},
    ]
    assert new_chunks_in_round(history, 2) == 4
    assert new_chunks_in_round(history, 3) == 5


def test_new_chunks_in_round_returns_zero_for_unknown_round():
    assert new_chunks_in_round([], 1) == 0
