"""
ReferenceRegistry — allocates and tracks the paper's `turn{m}search{n}`
reference IDs. One instance per run (per query), never module-global — the
old `agent/tools/tools.py` pattern of module-level `_ra`/`_bm25_cache` is
exactly what made concurrent runs (e.g. a future parallel slide-generation
pipeline) unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from slidegen_rag.tree.structure.tree_structure import Node


@dataclass(frozen=True)
class RefEntry:
    ref_id: str
    node_id: int
    title: str
    heading_path: tuple
    kind: str
    doc_id: str
    line_start: int
    line_end: int
    first_seen_turn: int


class ReferenceRegistry:
    """
    `register(node, turn)` allocates a new ref id the first time a node is
    seen, using ONE globally-incrementing counter (matching the paper's
    "format turn{m}search{n} using a globally incrementing counter") — `m`
    is the turn a node was FIRST seen in, `n` never resets. Re-registering an
    already-seen node returns its original ref id instead of a new one, which
    is what makes cross-turn dedup and stable citations possible.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._ref_by_node: dict[int, str] = {}
        self._entry_by_ref: dict[str, RefEntry] = {}

    def register(self, node: Node, turn: int) -> tuple[str, bool]:
        """Returns (ref_id, is_new)."""
        existing = self._ref_by_node.get(node.index)
        if existing is not None:
            return existing, False

        self._counter += 1
        ref_id = f"turn{turn}search{self._counter}"
        entry = RefEntry(
            ref_id=ref_id,
            node_id=node.index,
            title=node.title,
            heading_path=node.heading_path,
            kind=node.kind,
            doc_id=node.doc_id,
            line_start=node.line_start,
            line_end=node.line_end,
            first_seen_turn=turn,
        )
        self._ref_by_node[node.index] = ref_id
        self._entry_by_ref[ref_id] = entry
        return ref_id, True

    def resolve(self, ref_id: str) -> Optional[RefEntry]:
        return self._entry_by_ref.get(ref_id)

    def ref_id_for_node(self, node_id: int) -> Optional[str]:
        return self._ref_by_node.get(node_id)

    def all_refs(self) -> list[RefEntry]:
        """Insertion order == allocation order (Python dicts preserve it)."""
        return list(self._entry_by_ref.values())

    def __len__(self) -> int:
        return len(self._entry_by_ref)
