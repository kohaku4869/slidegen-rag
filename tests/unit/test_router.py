from slidegen_rag.agent.graph.router import MAX_RETRIEVE, decide_after_context_eval


def _history(*rounds_new_chunks: int) -> list[dict]:
    """Builds one search_history entry per round index (1-based) with the
    given new_chunks count."""
    return [
        {
            "round": i + 1,
            "tool": "semantic_search",
            "query": "q",
            "new_chunks": n,
            "soft_failed": False,
        }
        for i, n in enumerate(rounds_new_chunks)
    ]


def test_sufficient_short_circuits_to_generate_regardless_of_history():
    state = {
        "context_sufficient": True,
        "retrieve_count": 1,
        "search_history": [],
        "pending_documents": [],
    }
    assert decide_after_context_eval(state) == "generate"


def test_max_retrieve_forces_generate_regardless_of_new_logic():
    state = {
        "context_sufficient": False,
        "retrieve_count": MAX_RETRIEVE,
        "search_history": _history(
            3, 2, 1, 0, 5
        ),  # last round productive, still forced
        "pending_documents": ["something"],
    }
    assert decide_after_context_eval(state) == "generate"


def test_single_stalled_round_then_productive_round_does_not_trigger_early_generate():
    # round 3 stalled (0), round 4 productive (2) -> at retrieve_count=4, previous=0 current=2
    state = {
        "context_sufficient": False,
        "retrieve_count": 4,
        "search_history": _history(2, 1, 0, 2),
        "pending_documents": ["fresh chunk"],
    }
    assert decide_after_context_eval(state) == "retrieve"


def test_two_consecutive_stalled_rounds_from_round_three_triggers_generate():
    # round 2 stalled (0), round 3 stalled (0) -> at retrieve_count=3
    state = {
        "context_sufficient": False,
        "retrieve_count": 3,
        "search_history": _history(2, 0, 0),
        "pending_documents": [],
    }
    assert decide_after_context_eval(state) == "generate"


def test_stall_guard_does_not_apply_before_retrieve_count_three():
    # two stalled rounds but retrieve_count only 2 -> guard must not fire yet
    state = {
        "context_sufficient": False,
        "retrieve_count": 2,
        "search_history": _history(0, 0),
        "pending_documents": [],
    }
    assert decide_after_context_eval(state) == "retrieve"
