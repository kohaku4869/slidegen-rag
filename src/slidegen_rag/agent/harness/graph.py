"""
Two-node LangGraph loop implementing Algorithm 1 (the paper's bounded
agentic loop): `agent` decides tool_calls or a final answer; `tools` executes
them and always feeds results back to `agent`. This is deliberately kept as
LangGraph (not a bare while-loop) so `graph.astream(stream_mode=["updates",
"values"])` keeps working for observability and LangSmith tracing, exactly
as agent/graph/builder.py's CRAG graph does — a cyclic 2-node graph already
IS Algorithm 1, "the paper writes a while loop" isn't an argument to drop
LangGraph.

    START → agent → should_continue → {tools → agent | finalize → END}

Usage:
    from slidegen_rag.agent.harness.graph import build_agentic_graph
    from slidegen_rag.agent.harness.doc_store import DocStore
    from slidegen_rag.agent.harness.config import AgenticConfig

    store = DocStore.from_ra(ra)
    graph, registry = build_agentic_graph(store, AgenticConfig())
    result = await graph.ainvoke(make_initial_state("câu hỏi của bạn"))
    print(result["generation"])
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from slidegen_rag.agent.harness.config import AgenticConfig
from slidegen_rag.agent.harness.doc_store import DocStore
from slidegen_rag.agent.harness.nodes import build_harness, should_continue
from slidegen_rag.agent.harness.prompts import AGENTIC_SYSTEM_PROMPT
from slidegen_rag.agent.harness.state import AgenticState
from slidegen_rag.agent.observability.tracing import init_langsmith

logger = logging.getLogger(__name__)


def make_initial_state(query: str) -> AgenticState:
    return AgenticState(
        messages=[SystemMessage(content=AGENTIC_SYSTEM_PROMPT), HumanMessage(content=query)],
        query=query,
        iteration=0,
        forced_completion=False,
        warned_context=False,
        generation="",
        documents=[],
        cited_refs=[],
    )


def build_agentic_graph(store: DocStore, config: AgenticConfig | None = None):
    """Returns (compiled_graph, registry). `registry` is the SAME
    ReferenceRegistry the graph's tools use — callers that want to inspect
    citations/refs after a run (trajectory recording, tests) need this
    handle since it's per-run state, not reachable from the compiled graph
    object itself."""
    config = config or AgenticConfig()
    init_langsmith()

    agent_node, tools_node, finalize_node, registry = build_harness(store, config)

    workflow = StateGraph(AgenticState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "finalize": "finalize"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)

    graph = workflow.compile()
    logger.info("[build_agentic_graph] agentic harness graph compiled successfully")
    return graph, registry
