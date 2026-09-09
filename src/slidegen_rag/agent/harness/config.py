"""
AgenticConfig — every tunable knob of the AgenticRAG-style harness, in one
place so the paper's ablation rows are literally flag flips.

Plain dataclass rather than pydantic-settings: the harness has no consumer
yet that needs env-var/TOML precedence layering (that's `settings.py` work
for later, once there's more than one config surface to unify), and a
dataclass keeps this dependency-free and trivially constructible in tests.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticConfig:
    # ── Loop bounds (Algorithm 1 in the paper) ───────────────────────────
    max_iterations: int = 15

    # ── Context management ───────────────────────────────────────────────
    # Paper uses 128K; kept lower here (see restructuring plan) because
    # current spend is ~45K tokens/query — a 128K ceiling would essentially
    # never fire, making the summarize ablation unmeasurable.
    context_threshold_tokens: int = 96_000
    context_warn_ratio: float = 0.9
    # Independent per-tool-result cap, enforced regardless of summarize —
    # this is what actually neutralizes an oversized (10K+ token) node.
    tool_result_max_tokens: int = 6_000

    # ── search ────────────────────────────────────────────────────────────
    search_max_queries: int = 5       # paper: "up to 5 reformulations per call"
    search_per_query_k: int = 8       # scaled down from the paper's 10 — see plan §A2
    search_max_results: int = 12      # merged cap after RRF fusion
    rrf_k: int = 60

    # ── open ──────────────────────────────────────────────────────────────
    # Paper's 1800-LINE default window doesn't port literally to this corpus
    # (OCR flattens whole paragraphs onto one line — 1800 lines = ~98K
    # tokens). Line numbers stay as the addressing scheme; window size is a
    # token budget instead. See plan §A1.
    open_token_budget: int = 3_500
    open_max_lines: int = 400

    # ── find ──────────────────────────────────────────────────────────────
    find_max_passages_per_pattern: int = 2
    find_context_lines: int = 2       # lines of context around a match, each side
    find_token_budget: int = 6_000    # not the paper's 11K — that's 4% of this corpus
    # Ablation: paper measured semantic find as NEGATIVE value (46.34 vs
    # 43.49 recall@1 without it) — keep it possible to flip on to reproduce
    # that result, but off by default.
    enable_semantic_find: bool = False

    # ── ablation switches ─────────────────────────────────────────────────
    enable_multi_query: bool = True   # False -> search accepts only 1 query
    enable_navigation: bool = True    # False -> drop find + open entirely
    enable_summarize: bool = True     # False -> hard-truncate at threshold, no tool
