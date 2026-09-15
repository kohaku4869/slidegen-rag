from typing import Dict, List, Optional, Set, Tuple


class Node:
    """
    Represents a node in the hierarchical tree structure.

    v2 adds source-provenance metadata (heading path, depth, line spans, token
    counts, ...) — `describe_node` in `agent/tools/tools.py` reads it to tell
    the retrieve node which direction to expand. Declared as class-level
    defaults (below `__init__`)
    so a v1 pickle, whose `__dict__` lacks these keys, still exposes every v2
    field via normal attribute lookup instead of raising AttributeError.
    """

    # ── v2 metadata defaults (fallback for pre-v2 unpickled instances) ──────
    doc_id: str = ""
    title: str = ""
    heading_path: Tuple[str, ...] = ()
    heading_level: int = 0
    depth: int = 0
    layer: int = 0
    parent: Optional[int] = None
    line_start: int = 0
    line_end: int = 0
    body_line_start: int = 0
    body_line_end: int = 0
    n_tokens: int = 0
    subtree_tokens: int = 0
    kind: str = "leaf"

    def __init__(
        self,
        text: str,
        index: int,
        children: Set[int],
        embeddings,
        *,
        doc_id: str = "",
        title: str = "",
        heading_path: Tuple[str, ...] = (),
        heading_level: int = 0,
        depth: int = 0,
        layer: int = 0,
        parent: Optional[int] = None,
        line_start: int = 0,
        line_end: int = 0,
        body_line_start: int = 0,
        body_line_end: int = 0,
        n_tokens: int = 0,
        subtree_tokens: int = 0,
        kind: str = "leaf",
    ) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings
        self.doc_id = doc_id
        self.title = title
        self.heading_path = heading_path
        self.heading_level = heading_level
        self.depth = depth
        self.layer = layer
        self.parent = parent
        self.line_start = line_start
        self.line_end = line_end
        self.body_line_start = body_line_start
        self.body_line_end = body_line_end
        self.n_tokens = n_tokens
        self.subtree_tokens = subtree_tokens
        self.kind = kind


class Tree:
    """
    Represents the entire hierarchical tree structure.

    v2 adds `schema_version` plus document provenance (`doc_id`, `source_path`,
    `source_sha1`, `source_text`) and a `parent_of` reverse-children map. Same
    backward-compatibility approach as `Node`: class-level defaults so a v1
    pickle (schema_version absent from its `__dict__`) reads back as
    `schema_version == 1` rather than raising AttributeError. `parent_of`
    intentionally has no class-level default (a mutable `{}` there would be
    shared across every v1 instance) — check `schema_version` before relying
    on it being present.
    """

    schema_version: int = 1
    doc_id: str = ""
    source_path: str = ""
    source_sha1: str = ""
    source_text: str = ""

    def __init__(
        self,
        all_nodes,
        root_nodes,
        leaf_nodes,
        num_layers,
        layer_to_nodes,
        build_stats=None,
        *,
        schema_version: int = 2,
        doc_id: str = "",
        source_path: str = "",
        source_sha1: str = "",
        source_text: str = "",
        parent_of: Optional[Dict[int, int]] = None,
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
        self.build_stats = build_stats
        self.schema_version = schema_version
        self.doc_id = doc_id
        self.source_path = source_path
        self.source_sha1 = source_sha1
        self.source_text = source_text
        self.parent_of: Dict[int, int] = parent_of if parent_of is not None else {}
