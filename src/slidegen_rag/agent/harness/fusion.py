"""Reciprocal Rank Fusion — combines dense + BM25 rankings (and, within
`search`, rankings from multiple query reformulations) into one ordering.
Multi-query agreement is itself a ranking signal: a node surfaced by several
reformulations accumulates score from each."""

from __future__ import annotations

from slidegen_rag.agent.harness.doc_store import ScoredNode


def rrf_fuse(rankings: list[list[ScoredNode]], k: int = 60) -> list[ScoredNode]:
    """
    `rankings` is a list of independently-ranked ScoredNode lists (e.g. one
    dense + one BM25 ranking for a single query, or one ranking per query in
    a multi-query search). Each node's contributions across every ranking it
    appears in are summed: score(node) = sum(1 / (k + rank)) over every list
    it appears in, rank 0-based. Original per-list scores are discarded —
    RRF only uses rank position, which is what lets heterogeneous scales
    (cosine similarity vs. BM25) combine without normalization.
    """
    fused: dict[int, float] = {}
    node_by_id: dict[int, ScoredNode] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            node_id = scored.node.index
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (k + rank + 1)
            node_by_id.setdefault(node_id, scored)

    ordered_ids = sorted(fused.keys(), key=lambda nid: fused[nid], reverse=True)
    return [ScoredNode(node_by_id[nid].node, fused[nid]) for nid in ordered_ids]
