"""Unit tests for TokenTracker (agent/models/llm.py): Gemini and local token
usage must be tracked SEPARATELY, and cost must be computed ONLY from the
Gemini portion — a local-model call must never inflate the estimated cost."""

from slidegen_rag.agent.models.llm import TokenTracker


def _usage(input_tokens: int, output_tokens: int, reasoning: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_token_details": {"reasoning": reasoning} if reasoning else {},
    }


def test_gemini_only_usage_tracked_and_priced():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)

    summary = tracker.get_summary("gemini-2.5-flash")

    assert summary["gemini_calls"] == 1
    assert summary["local_calls"] == 0
    assert summary["gemini_input_tokens"] == 1000
    assert summary["gemini_output_tokens"] == 500
    assert summary["gemini_total_tokens"] == 1500
    assert summary["total_cost_usd"] > 0


def test_local_only_usage_has_zero_cost():
    tracker = TokenTracker()
    tracker.add_usage(_usage(10_000, 5_000), is_local=True)

    summary = tracker.get_summary("gemini-2.5-flash")

    assert summary["gemini_calls"] == 0
    assert summary["local_calls"] == 1
    assert summary["local_input_tokens"] == 10_000
    assert summary["local_output_tokens"] == 5_000
    assert summary["local_total_tokens"] == 15_000
    # A local (self-hosted) call must contribute ZERO to the Gemini cost.
    assert summary["total_cost_usd"] == 0.0
    assert summary["total_cost_vnd"] == 0.0


def test_mixed_usage_cost_reflects_only_gemini_portion():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    tracker.add_usage(_usage(50_000, 50_000), is_local=True)  # much bigger, but free

    gemini_only_tracker = TokenTracker()
    gemini_only_tracker.add_usage(_usage(1000, 500), is_local=False)

    mixed_summary = tracker.get_summary("gemini-2.5-flash")
    gemini_only_summary = gemini_only_tracker.get_summary("gemini-2.5-flash")

    # The huge local call must not change the estimated Gemini cost at all.
    assert mixed_summary["total_cost_usd"] == gemini_only_summary["total_cost_usd"]
    # But the backward-compat combined totals DO include both providers.
    assert mixed_summary["input_tokens"] == 51_000
    assert mixed_summary["output_tokens"] == 50_500
    assert mixed_summary["calls"] == 2


def test_reasoning_tokens_tracked_for_gemini_only():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500, reasoning=300), is_local=False)
    tracker.add_usage(_usage(200, 100, reasoning=999), is_local=True)

    summary = tracker.get_summary()
    assert summary["gemini_reasoning_tokens"] == 300


def test_reset_clears_both_gemini_and_local():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    tracker.add_usage(_usage(200, 100), is_local=True)

    tracker.reset()
    summary = tracker.get_summary()

    assert summary["calls"] == 0
    assert summary["gemini_total_tokens"] == 0
    assert summary["local_total_tokens"] == 0
    assert summary["total_cost_usd"] == 0.0


def test_format_report_mentions_local_only_when_present():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    report_no_local = tracker.format_report()
    assert "Local (miễn phí)" not in report_no_local

    tracker.add_usage(_usage(200, 100), is_local=True)
    report_with_local = tracker.format_report()
    assert "Local (miễn phí)" in report_with_local
