"""
AgenticState — unlike agent/graph/state.py:AgentState, this HAS a `messages`
field with the `add_messages` reducer, so ToolMessage results genuinely round
-trip back to the model. That absence (retrieve_node emitting tool calls
"blind", never seeing prior tool output) is the root cause the harness fixes;
everything else in the retrofit is amplification of this one change.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgenticState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    iteration: int
    forced_completion: bool
    warned_context: bool
    generation: str
    documents: list[str]
    cited_refs: list[str]
