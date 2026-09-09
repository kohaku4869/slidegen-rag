"""
cluster_utils.py — RAPTOR paper clustering algorithm (Sarthi et al., "RAPTOR: Recursive
Abstractive Processing for Tree-Organized Retrieval").

Two-stage soft clustering:
  1. Global clustering: UMAP dimensionality reduction over ALL embeddings in the layer,
     then Gaussian Mixture Model (GMM) with the optimal number of components chosen by BIC.
  2. Local clustering: within each global cluster, a second UMAP + GMM pass refines the
     grouping (skipped when the global cluster is already small).

Cluster assignment is "soft": a node is assigned to every cluster whose GMM posterior
probability exceeds `threshold`, so a node may legitimately belong to more than one
cluster — this mirrors the paper's approach and differs from hard clustering (k-means).
"""

import logging
from typing import List

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

RANDOM_SEED = 224


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduces the dimensionality of the full embedding set via UMAP."""
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    )
    return reducer.fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    num_neighbors: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduces the dimensionality of a single global cluster's embeddings via UMAP."""
    num_neighbors = max(2, min(num_neighbors, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    )
    return reducer.fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """Selects the number of GMM components that minimizes the Bayesian Information Criterion."""
    max_clusters = max(1, min(max_clusters, len(embeddings) - 1))
    if max_clusters <= 1:
        return 1

    n_clusters_range = range(1, max_clusters + 1)
    bics = []
    for n in n_clusters_range:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    return n_clusters_range[int(np.argmin(bics))]


def GMM_cluster(
    embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED, max_clusters: int = 50
):
    """
    Fits a GMM with the BIC-optimal number of components and returns soft cluster
    membership per embedding (indices of components whose posterior probability
    exceeds `threshold`) plus the number of components used.
    """
    n_clusters = get_optimal_clusters(embeddings, max_clusters=max_clusters, random_state=random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED,
) -> List[np.ndarray]:
    """
    Two-stage (global + local) soft clustering of a layer's node embeddings.

    Returns:
        List[np.ndarray]: for each input embedding (same order as `embeddings`), the
        array of cluster ids it belongs to. A node with an empty array belongs to no
        cluster (should not normally happen, but is guarded against by the caller).
    """
    embeddings = np.asarray(embeddings)
    n_points = len(embeddings)

    if n_points <= dim + 1:
        # Too few points to meaningfully reduce dimensionality / cluster — one cluster.
        return [np.array([0]) for _ in range(n_points)]

    reduced_dim_global = min(dim, n_points - 2)
    reduced_global = global_cluster_embeddings(embeddings, reduced_dim_global)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_global, threshold, random_state=random_state, max_clusters=max_clusters
    )

    all_local_clusters = [np.array([]) for _ in range(n_points)]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_mask = np.array(
            [i in gc for gc in global_clusters], dtype=bool
        )
        indices_in_cluster = np.where(global_cluster_mask)[0]
        cluster_embeddings = embeddings[global_cluster_mask]

        if len(cluster_embeddings) == 0:
            continue

        if len(cluster_embeddings) <= dim + 1:
            local_clusters = [np.array([0]) for _ in cluster_embeddings]
            n_local_clusters = 1
        else:
            reduced_dim_local = min(dim, len(cluster_embeddings) - 2)
            reduced_local = local_cluster_embeddings(cluster_embeddings, reduced_dim_local)
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_local, threshold, random_state=random_state, max_clusters=max_clusters
            )

        for j in range(n_local_clusters):
            local_cluster_mask = np.array([j in lc for lc in local_clusters], dtype=bool)
            selected_indices = indices_in_cluster[local_cluster_mask]
            for idx in selected_indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    # Safety net: any node that ended up in no cluster (can happen with a very low
    # threshold/degenerate GMM fit) is assigned to its own singleton cluster.
    for idx, clusters in enumerate(all_local_clusters):
        if clusters.size == 0:
            all_local_clusters[idx] = np.array([total_clusters])
            total_clusters += 1

    return all_local_clusters
