"""
The agentic loop's two active nodes (`agent`, `tools`) plus `finalize` and
the `should_continue` router — see harness/graph.py for how they wire into a
2-node LangGraph cycle. This is Algorithm 1 from the paper: `agent` is the
"call LLM with tool schemas, act on tool_calls or return" step, `tools` is
`EXECUTETOOL`, `finalize` is `FORMATANSWER`.
"""

from __future__ import annotations

import asyncio
import logging
import re

import tiktoken
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from slidegen_rag.agent.harness.compaction import compact_messages
from slidegen_rag.agent.harness.config import AgenticConfig
from slidegen_rag.agent.harness.doc_store import DocStore
from slidegen_rag.agent.harness.prompts import CONTEXT_WARN_PROMPT, FORCE_ANSWER_PROMPT
from slidegen_rag.agent.harness.refs import ReferenceRegistry
from slidegen_rag.agent.harness.state import AgenticState
from slidegen_rag.agent.harness.tools import TurnBox, _coerce_str_list, build_agentic_tools
from slidegen_rag.agent.models.llm import LLMModel

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
# Matches the ref id itself wherever it appears — NOT anchored to an exact
# "[turnMsearchN]" bracket with nothing else inside, because models
# routinely cite as "[turn1search5, dòng 2284-2287]" (extra content in the
# same brackets) despite the system prompt's literal example.
_REF_ID_RE = re.compile(r"turn\d+search\d+")


def _estimate_tokens(messages: list[AnyMessage]) -> int:
    """Cheap, deliberately-over-counting estimate (Vietnamese tokenizes worse
    under cl100k_base than under Gemini's own tokenizer — over-estimating is
    the safe direction for a threshold trigger)."""
    total = 0
    for m in messages:
        content = getattr(m, "content", "")
        total += len(_TOKENIZER.encode(content if isinstance(content, str) else str(content)))
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            total += len(_TOKENIZER.encode(str(tool_calls)))
    return total


def build_harness(store: DocStore, config: AgenticConfig):
    """Constructs one run's worth of harness state: a fresh ReferenceRegistry
    and TurnBox (never shared across runs — see tools.py's module docstring),
    the tool list, and the three node callables closing over all of it.
    Returns (agent_node, tools_node, finalize_node, registry)."""
    registry = ReferenceRegistry()
    turn_box = TurnBox()
    tools: list[BaseTool] = build_agentic_tools(store, registry, config, turn_box)
    tool_by_name = {t.name: t for t in tools}
    llm = LLMModel(provider="gemini", temperature=0.2)

    async def agent_node(state: AgenticState) -> dict:
        iteration = state.get("iteration", 0) + 1
        turn_box.turn = iteration
        messages = list(state["messages"])
        updates: dict = {"iteration": iteration}

        if iteration > config.max_iterations:
            force_msg = HumanMessage(content=FORCE_ANSWER_PROMPT)
            response = await llm.client.ainvoke(messages + [force_msg])
            _track_usage(response)
            updates["forced_completion"] = True
            return {**updates, "messages": [force_msg, response]}

        token_estimate = _estimate_tokens(messages)
        summarize_tool = tool_by_name.get("summarize")

        if config.enable_summarize and summarize_tool is not None and token_estimate >= config.context_threshold_tokens:
            bound = llm.client.bind_tools([summarize_tool], tool_choice="summarize")
            response = await bound.ainvoke(messages)
            _track_usage(response)
            return {**updates, "messages": [response]}

        extra: list[AnyMessage] = []
        warn_ratio_hit = token_estimate >= config.context_threshold_tokens * config.context_warn_ratio
        if warn_ratio_hit and not state.get("warned_context", False):
            warn_msg = SystemMessage(content=CONTEXT_WARN_PROMPT)
            messages = messages + [warn_msg]
            extra.append(warn_msg)
            updates["warned_context"] = True

        bound = llm.client.bind_tools(tools, tool_choice="auto")
        response = await bound.ainvoke(messages)
        _track_usage(response)
        return {**updates, "messages": extra + [response]}

    async def tools_node(state: AgenticState) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        async def run_one(tc: dict):
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            fn = tool_by_name.get(name)
            if fn is None:
                return tc, f"Không tìm thấy tool: {name!r}"
            try:
                out = await asyncio.to_thread(fn.invoke, args)
                return tc, str(out)
            except Exception as e:  # soft failure — visible to the model, not filtered out
                logger.warning(f"[harness.tools_node] tool {name!r} raised: {e}")
                return tc, f"Tool {name!r} thất bại: {e}"

        outcomes = await asyncio.gather(*[run_one(tc) for tc in tool_calls])

        new_messages: list[AnyMessage] = []
        for tc, output in outcomes:
            new_messages.append(ToolMessage(content=output, tool_call_id=tc.get("id", ""), name=tc.get("name", "")))
            if tc.get("name") == "summarize":
                args = tc.get("args", {}) or {}
                keep_ids = set(_coerce_str_list(args.get("keep_reference_ids")))
                removals, summary_message = compact_messages(
                    state["messages"] + new_messages, args.get("notes", ""), keep_ids, registry, store
                )
                new_messages.append(summary_message)
                new_messages.extend(removals)

        return {"messages": new_messages}

    async def finalize_node(state: AgenticState) -> dict:
        last = state["messages"][-1]
        answer = _extract_text(last.content)
        cited_ids = sorted(set(_REF_ID_RE.findall(answer)))

        resolvable = [rid for rid in cited_ids if registry.resolve(rid) is not None]
        unresolved = [rid for rid in cited_ids if registry.resolve(rid) is None]
        if resolvable:
            answer = answer.rstrip() + "\n\nNguồn tham khảo:\n" + "\n".join(
                _format_source_line(rid, registry) for rid in resolvable
            )
        if unresolved:
            logger.warning(f"[harness.finalize_node] answer cited unresolved ref ids: {unresolved}")

        documents = [store.node(e.node_id).text for e in registry.all_refs()]
        return {"generation": answer, "documents": documents, "cited_refs": resolvable}

    return agent_node, tools_node, finalize_node, registry


def should_continue(state: AgenticState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def _extract_text(content) -> str:
    """AIMessage.content is a plain str in the common case, but a
    tool-calling conversation's FINAL (non-tool-calling) Gemini response can
    come back as a list of content parts — text blocks plus opaque
    "thought signature" blobs from the function-calling turn. Naively
    str()-ing that list dumps a huge base64 blob into the answer; extract
    just the text parts instead."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        if parts:
            return "\n".join(parts)
    return str(content)


def _format_source_line(ref_id: str, registry: ReferenceRegistry) -> str:
    entry = registry.resolve(ref_id)
    return f"[{ref_id}] {entry.title or '(không tiêu đề)'} — {entry.doc_id or 'document'}, dòng {entry.line_start}-{entry.line_end}"


def _track_usage(response) -> None:
    """bind_tools() returns the raw bound BaseChatModel, bypassing
    LLMModel.ainvoke's own usage tracking — record it manually here so
    LLMModel.get_usage_report() still reflects harness runs."""
    usage = getattr(response, "usage_metadata", None)
    if usage:
        LLMModel.tracker.add_usage(usage, is_local=False)
