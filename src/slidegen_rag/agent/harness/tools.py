"""
The four AgenticRAG tools (search/find/open/summarize), built as LangChain
`@tool`s by a factory closing over one DocStore + ReferenceRegistry + config
+ TurnBox per run — no module-level mutable state (contrast with
agent/tools/tools.py's `_ra`/`_bm25_cache` globals, which cannot be made
safe for concurrent runs without this kind of per-run construction).

Design decisions specific to this single-document, heading-derived corpus
are explained inline; see the restructuring plan (§B2) for the fuller
reasoning:
  - search fuses dense + BM25 per query via RRF, then fuses across queries —
    multi-query agreement is itself part of the ranking signal.
  - search does NOT apply dedup_parent_child: a parent surfacing alongside
    its child is signal here (the parent is the coarse entry point), not
    redundancy to filter out.
  - open windows over a node's *source line span*, not its (builder-
    reformatted) `.text` — for a heading-derived tree the two are the same
    physical region of the source, so this recovers real markdown formatting
    the builder's text normalization discards.
  - find is lexical-only (NFC-normalized, diacritics NOT folded — "có" and
    "cỏ" are different words) and emits an explicit no-match line; the
    paper's semantic-find ablation measured NEGATIVE value, so it isn't
    implemented at all here (config.enable_semantic_find only annotates the
    tool description, it doesn't change behavior).
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Union

import tiktoken
from langchain_core.tools import BaseTool, tool

from slidegen_rag.agent.harness.config import AgenticConfig
from slidegen_rag.agent.harness.doc_store import DocStore, ScoredNode
from slidegen_rag.agent.harness.fusion import rrf_fuse
from slidegen_rag.agent.harness.refs import ReferenceRegistry

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class TurnBox:
    """Mutable holder for "which loop iteration are we in" — tools are built
    once per run, before the graph starts, but need the CURRENT iteration
    number at call time to allocate turnNsearchM ref ids correctly. Updated
    by the agent node at the start of each iteration."""

    def __init__(self) -> None:
        self.turn: int = 0


def _n_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text)) if text else 0


def _coerce_str_list(value) -> list[str]:
    """Providers sometimes emit a bare string where a list was declared —
    coerce rather than raise (existing precedent: agent/tools/tools.py's
    web_search search_depth clamp)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


def _snippet(text: str, max_chars: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _kind_note(node) -> str:
    if node.kind == "summary":
        return f"summary of {len(node.children)} children | subtree {node.subtree_tokens} tokens"
    return f"{node.kind} | {node.n_tokens} tokens"


def build_search_tool(
    store: DocStore, registry: ReferenceRegistry, config: AgenticConfig, turn_box: TurnBox
) -> BaseTool:
    @tool("search")
    def search(queries: Union[str, list[str]]) -> str:
        """Discovers relevant sections across the entire document by combining
        dense (semantic) and BM25 (keyword) retrieval, fused via Reciprocal
        Rank Fusion. Accepts up to 5 short, focused query reformulations in
        ONE call, each targeting a different facet of the question, instead
        of pasting the whole question in one query. Results are deduplicated
        against everything already seen this conversation; each new result
        gets a stable reference id (e.g. "turn1search3") used by open/find.
        """
        queries_list = _coerce_str_list(queries)
        if not queries_list:
            return "search: không có truy vấn nào được cung cấp."
        queries_list = queries_list[:1] if not config.enable_multi_query else queries_list[: config.search_max_queries]

        per_query_rankings: list[list[ScoredNode]] = []
        for q in queries_list:
            dense = store.dense(q, config.search_per_query_k)
            bm25 = store.bm25(q, config.search_per_query_k)
            per_query_rankings.append(rrf_fuse([dense, bm25], k=config.rrf_k))

        fused = rrf_fuse(per_query_rankings, k=config.rrf_k)[: config.search_max_results]
        if not fused:
            return f"search: không tìm thấy kết quả nào cho {queries_list!r}."

        turn = turn_box.turn
        blocks: list[str] = []
        for scored in fused:
            node = scored.node
            ref_id, is_new = registry.register(node, turn)
            if not is_new:
                blocks.append(f"[{ref_id}] (đã thấy trước đó — dùng open/find trên tham chiếu này thay vì search lại)")
                continue
            path = " > ".join(node.heading_path) if node.heading_path else (node.title or "(không tiêu đề)")
            doc_name = node.doc_id or "document"
            blocks.append(
                f"[{ref_id}] {node.title or '(không tiêu đề)'}\n"
                f"  path: {path}\n"
                f"  {doc_name} (markdown) | lines {node.line_start}-{node.line_end} | {_kind_note(node)}\n"
                f"  {_snippet(node.text)}"
            )
        return "\n\n".join(blocks)

    return search


def build_open_tool(
    store: DocStore, registry: ReferenceRegistry, config: AgenticConfig, turn_box: TurnBox
) -> BaseTool:
    @tool("open")
    def open_ref(reference_id: str, line_number: int = None, num_lines: int = None) -> str:
        """Retrieves windowed, line-numbered source content for a reference
        returned by search/open/find. For a SECTION reference (a heading with
        children), calling this WITHOUT line_number returns a table of
        contents of its children (each getting its own reference id) instead
        of raw text — use this to navigate from a broad section down to a
        specific part instead of guessing a new search query; pass an
        explicit line_number to force reading the section's own raw text
        instead. For a leaf reference, returns a token-budgeted window of the
        source starting at line_number (defaults to the reference's own
        start); if the output is truncated it ends with a "Continue:" hint
        giving the line_number to resume from.
        """
        entry = registry.resolve(reference_id)
        if entry is None:
            return f"open: reference_id {reference_id!r} không xác định — hãy dùng một id từ kết quả search/open/find trước đó."

        node = store.node(entry.node_id)

        if node.kind == "summary" and line_number is None:
            children = store.children(node.index)
            if children:
                lines = [
                    f"[{entry.ref_id}] {node.title or '(không tiêu đề)'} — summary, "
                    f"lines {node.line_start}-{node.line_end}, {len(children)} children:"
                ]
                turn = turn_box.turn
                for child in children:
                    child_ref, _ = registry.register(child, turn)
                    lines.append(
                        f"  [{child_ref}] {child.title or '(không tiêu đề)'}   "
                        f"lines {child.line_start}-{child.line_end}   {_kind_note(child)}"
                    )
                return "\n".join(lines)

        span_start, span_end = node.line_start, node.line_end
        start = line_number if line_number is not None else span_start
        start = max(0, min(start, store.line_count))
        out_of_span = start < span_start or start >= span_end
        max_lines = min(num_lines, config.open_max_lines) if num_lines else config.open_max_lines

        end = start
        used_tokens = 0
        lines_out: list[str] = []
        while end < store.line_count and (end - start) < max_lines:
            line = store.source_lines[end]
            t = _n_tokens(line) + 1
            if lines_out and used_tokens + t > config.open_token_budget:
                break
            lines_out.append(line)
            used_tokens += t
            end += 1

        header = (
            f"Viewing lines [{start}-{max(start, end - 1)}] of node span [{span_start}-{span_end}] "
            f"— [{entry.ref_id}] \"{node.title}\""
        )
        if out_of_span:
            header += " (line_number nằm ngoài span mặc định của reference này)"
        header += f"\nTài liệu {node.doc_id or '(chưa rõ)'}, {store.line_count} dòng."
        if end < store.line_count and end > start:
            header += f"\nContinue: open(reference_id={entry.ref_id!r}, line_number={end})"

        if not lines_out:
            return f"{header}\n\n(không có nội dung trong khoảng dòng này)"
        body = "\n".join(f"{i + start:>5}| {l}" for i, l in enumerate(lines_out))
        return f"{header}\n\n{body}"

    return open_ref


def build_find_tool(store: DocStore, registry: ReferenceRegistry, config: AgenticConfig) -> BaseTool:
    @tool("find")
    def find(reference_id: str, patterns: Union[str, list[str]]) -> str:
        """Lexical (case-insensitive substring) search for one or more patterns
        within a single reference's source span, or the whole document if
        reference_id is the literal string "document". Use this when you know
        exactly what to look for (a figure, a term, an article number) inside
        a section you already have. Returns up to 2 passages per pattern
        (matching line plus surrounding context) and an EXPLICIT no-match
        line for any pattern not found — that is a valid, informative answer
        ("the documents don't cover this"), not a failure to work around.
        """
        patterns_list = _coerce_str_list(patterns)
        if not patterns_list:
            return "find: không có pattern nào được cung cấp."

        if reference_id == "document":
            scope_start, scope_end = 0, store.line_count
            scope_label = "[document]"
        else:
            entry = registry.resolve(reference_id)
            if entry is None:
                return (
                    f"find: reference_id {reference_id!r} không xác định — dùng một id từ "
                    f"search/open trước đó, hoặc \"document\" để quét toàn bộ tài liệu."
                )
            scope_start, scope_end = entry.line_start, entry.line_end
            scope_label = f"[{reference_id}]"

        lines = store.source_lines
        seen_hashes: set[str] = set()
        used_tokens = 0
        blocks: list[str] = []

        for pattern in patterns_list:
            needle = unicodedata.normalize("NFC", pattern).lower()
            hits = 0
            for i in range(scope_start, scope_end):
                if hits >= config.find_max_passages_per_pattern:
                    break
                haystack = unicodedata.normalize("NFC", lines[i]).lower()
                if needle not in haystack:
                    continue
                lo = max(scope_start, i - config.find_context_lines)
                hi = min(scope_end, i + config.find_context_lines + 1)
                passage_body = "\n".join(f"{ln:>5}| {lines[ln]}" for ln in range(lo, hi))
                digest = hashlib.blake2b(passage_body.encode("utf-8"), digest_size=8).hexdigest()
                if digest in seen_hashes:
                    continue
                t = _n_tokens(passage_body)
                if used_tokens + t > config.find_token_budget:
                    break
                seen_hashes.add(digest)
                used_tokens += t
                hits += 1
                blocks.append(f"{scope_label} find {pattern!r} — dòng {lo}-{hi - 1}:\n{passage_body}")
            if hits == 0:
                blocks.append(f"{scope_label} find {pattern!r} — không khớp trong dòng {scope_start}-{scope_end}.")

        return "\n\n".join(blocks)

    return find


def build_summarize_tool(registry: ReferenceRegistry) -> BaseTool:
    """The tool itself only validates and confirms — the actual message-list
    compaction is a graph-level side effect (see harness/compaction.py and
    harness/nodes.py's tools_node), since a plain @tool function can only
    return a string for its own ToolMessage, not remove OTHER messages."""

    @tool("summarize")
    def summarize(notes: str, keep_reference_ids: Union[str, list[str]] = None) -> str:
        """Consolidates reasoning so far and frees context space. `notes`
        should capture what you've learned and what's still missing.
        `keep_reference_ids` lists which references' FULL content to keep in
        context — everything else's full content is dropped, but every
        reference's id/title/line-span header stays available so you can
        still open() it again later if you change your mind.
        """
        keep = set(_coerce_str_list(keep_reference_ids))
        resolved = sorted(rid for rid in keep if registry.resolve(rid) is not None)
        unresolved = sorted(keep - set(resolved))
        lines = [
            f"summarize: đã ghi nhận notes ({len(notes or '')} ký tự).",
            f"Giữ nội dung đầy đủ cho {len(resolved)} reference: {resolved}.",
        ]
        if unresolved:
            lines.append(f"Bỏ qua {len(unresolved)} reference_id không xác định: {unresolved}.")
        return "\n".join(lines)

    return summarize


def build_agentic_tools(
    store: DocStore, registry: ReferenceRegistry, config: AgenticConfig, turn_box: TurnBox
) -> list[BaseTool]:
    tools = [build_search_tool(store, registry, config, turn_box)]
    if config.enable_navigation:
        tools.append(build_open_tool(store, registry, config, turn_box))
        tools.append(build_find_tool(store, registry, config))
    if config.enable_summarize:
        tools.append(build_summarize_tool(registry))
    return tools
