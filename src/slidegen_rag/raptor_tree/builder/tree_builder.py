import collections
import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tiktoken

from slidegen_rag.raptor_tree.cluster_utils import RANDOM_SEED, perform_clustering
from slidegen_rag.raptor_tree.models.EmbeddingModels import BaseEmbeddingModel, EmbeddingModel
from slidegen_rag.raptor_tree.models.SummarizationModels import (BaseSummarizationModel,
                                  ChatNVIDIASummarizationModel)
from slidegen_rag.raptor_tree.stats import BuildStats, LayerStat
from slidegen_rag.raptor_tree.structure.tree_structure import Node, Tree
from slidegen_rag.raptor_tree.utils import get_embeddings, get_node_list, get_text, split_text

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        chunk_overlap=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
        reduction_dimension=None,
        cluster_threshold=None,
        max_clusters=None,
        max_length_in_cluster=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if chunk_overlap is None:
            chunk_overlap = 0
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be an integer and at least 0")
        self.chunk_overlap = chunk_overlap

        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model is None:
            summarization_model = ChatNVIDIASummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        self.summarization_model = summarization_model

        if embedding_models is None:
            embedding_models = {"SBERT": EmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError(
                "embedding_models must be a dictionary of model_name: instance pairs"
            )
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError(
                    "All embedding models must be an instance of BaseEmbeddingModel"
                )
        self.embedding_models = embedding_models

        if cluster_embedding_model is None:
            cluster_embedding_model = "SBERT"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )
        self.cluster_embedding_model = cluster_embedding_model

        if reduction_dimension is None:
            reduction_dimension = 10
        if not isinstance(reduction_dimension, int) or reduction_dimension < 1:
            raise ValueError("reduction_dimension must be an integer and at least 1")
        self.reduction_dimension = reduction_dimension

        if cluster_threshold is None:
            cluster_threshold = 0.1
        if not isinstance(cluster_threshold, (int, float)) or not (0 <= cluster_threshold <= 1):
            raise ValueError("cluster_threshold must be a number between 0 and 1")
        self.cluster_threshold = cluster_threshold

        if max_clusters is None:
            max_clusters = 50
        if not isinstance(max_clusters, int) or max_clusters < 1:
            raise ValueError("max_clusters must be an integer and at least 1")
        self.max_clusters = max_clusters

        if max_length_in_cluster is None:
            max_length_in_cluster = 3500
        if not isinstance(max_length_in_cluster, int) or max_length_in_cluster < 1:
            raise ValueError("max_length_in_cluster must be an integer and at least 1")
        self.max_length_in_cluster = max_length_in_cluster

    def log_config(self):
        config_log = """
        TreeBuilderConfig (RAPTOR):
            Tokenizer: {tokenizer}
            Max Tokens (leaf chunk): {max_tokens}
            Chunk Overlap: {chunk_overlap}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
            Reduction Dimension: {reduction_dimension}
            Cluster Threshold: {cluster_threshold}
            Max Clusters (BIC search cap): {max_clusters}
            Max Length In Cluster: {max_length_in_cluster}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            chunk_overlap=self.chunk_overlap,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
            reduction_dimension=self.reduction_dimension,
            cluster_threshold=self.cluster_threshold,
            max_clusters=self.max_clusters,
            max_length_in_cluster=self.max_length_in_cluster,
        )
        return config_log


class TreeBuilder:
    """
    Builds a RAPTOR tree: raw text is chunked into leaf nodes, then recursively
    clustered (GMM + UMAP, see cluster_utils.py) and summarized layer by layer
    until a single root remains — following the RAPTOR paper's algorithm rather
    than a document-structure heuristic.
    """

    def __init__(self, config) -> None:
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.chunk_overlap = config.chunk_overlap
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        self.reduction_dimension = config.reduction_dimension
        self.cluster_threshold = config.cluster_threshold
        self.max_clusters = config.max_clusters
        self.max_length_in_cluster = config.max_length_in_cluster
        self.last_build_stats: Optional[BuildStats] = None

        logging.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )

    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        """Creates a new node with the given index, text, and (optionally) children indices."""
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

    def batch_create_nodes(
        self,
        items: List[Tuple[int, str, Set[int]]],
    ) -> Dict[int, Node]:
        """Creates nodes for a batch of (index, text, children_indices) with batched embedding."""
        if not items:
            return {}

        texts = [text for _, text, _ in items]

        # Single batched call per embedding model (leverages GPU parallelism)
        all_embeddings: Dict[str, List] = {}
        for model_name, model in self.embedding_models.items():
            all_embeddings[model_name] = model.create_embeddings(texts)

        nodes: Dict[int, Node] = {}
        for i, (idx, text, children_indices) in enumerate(items):
            emb = {name: vecs[i] for name, vecs in all_embeddings.items()}
            nodes[idx] = Node(text, idx, children_indices, emb)

        return nodes

    def create_embedding(self, text) -> List[float]:
        """Generates an embedding for a single text using the cluster embedding model."""
        return self.embedding_models[self.cluster_embedding_model].create_embedding(text)

    def summarize(self, context, max_tokens=2048) -> str:
        return self.summarization_model.summarize(context, max_tokens)

    def batch_summarize(self, contexts, max_tokens=2048):
        return self.summarization_model.batch_summarize(contexts, max_tokens)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Strips Markdown noise (heading hashes, emphasis markers, bullet symbols).
        RAPTOR treats input as raw text and does not rely on document structure,
        unlike the Markdown-heading-based `tree` package.
        """
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"[*_`~]+", "", text)
        text = re.sub(r"^[ \t]*[-+•]\s+", "", text, flags=re.MULTILINE)
        return text.strip()

    def _num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_oversized_group(
        self, nodes: List[Node], random_state: int, depth: int = 0
    ) -> List[List[Node]]:
        """
        Recursively re-clusters a group of nodes whose combined text exceeds
        max_length_in_cluster, so a single summarization call never blows past the
        summarizer's context window. Falls back to an even split if clustering can
        no longer separate the group (e.g. near-identical embeddings) or the
        recursion depth cap is hit.
        """
        total_tokens = sum(self._num_tokens(n.text) for n in nodes)
        if total_tokens <= self.max_length_in_cluster or len(nodes) <= 1 or depth >= 4:
            return [nodes]

        embeddings = np.array(get_embeddings(nodes, self.cluster_embedding_model))
        sub_clusters = perform_clustering(
            embeddings,
            dim=self.reduction_dimension,
            threshold=self.cluster_threshold,
            max_clusters=self.max_clusters,
            random_state=random_state + depth + 1,
        )

        groups: Dict[int, List[Node]] = collections.defaultdict(list)
        for node, cluster_ids in zip(nodes, sub_clusters):
            for cid in cluster_ids:
                groups[int(cid)].append(node)

        if len(groups) <= 1:
            mid = len(nodes) // 2
            return [nodes[:mid], nodes[mid:]] if mid > 0 else [nodes]

        result: List[List[Node]] = []
        for group_nodes in groups.values():
            result.extend(self._split_oversized_group(group_nodes, random_state, depth + 1))
        return result

    def _summarize_and_create_parents(
        self, groups: List[List[Node]], next_index: int
    ) -> Tuple[Dict[int, Node], float, float]:
        """Batch-summarizes each node group into one parent node per group."""
        contexts = [get_text(group).strip() for group in groups]

        t_sum_start = time.perf_counter()
        summaries = self.batch_summarize(contexts, max_tokens=self.summarization_length)
        sum_time = time.perf_counter() - t_sum_start

        parent_items: List[Tuple[int, str, Set[int]]] = []
        for group, summary in zip(groups, summaries):
            children_indices = {n.index for n in group}
            parent_items.append((next_index, summary, children_indices))
            next_index += 1

        t_emb_start = time.perf_counter()
        parent_nodes = self.batch_create_nodes(parent_items)
        emb_time = time.perf_counter() - t_emb_start

        return parent_nodes, sum_time, emb_time

    def build_from_text(self, markdown_text: str, use_multithreading: bool = True) -> Tree:
        """
        Builds a RAPTOR Tree from raw text: chunk -> embed -> recursively cluster
        and summarize layer by layer until a single root node remains.

        Args:
            markdown_text (str): The input text (Markdown formatting is stripped).
            use_multithreading (bool, optional): Kept for API compatibility.

        Returns:
            Tree: The built Tree object.
        """
        t_build_start = time.perf_counter()

        t_chunk_start = time.perf_counter()
        cleaned_text = self._clean_text(markdown_text)
        leaf_texts = split_text(cleaned_text, self.tokenizer, self.max_tokens, self.chunk_overlap)
        if not leaf_texts:
            leaf_texts = [cleaned_text if cleaned_text else markdown_text]
        chunk_time = time.perf_counter() - t_chunk_start

        all_nodes: Dict[int, Node] = {}
        layer_to_nodes: Dict[int, List[Node]] = collections.defaultdict(list)
        layer_stats: List[LayerStat] = []

        leaf_items = [(i, text, set()) for i, text in enumerate(leaf_texts)]
        next_index = len(leaf_items)

        logging.info(f"Batch embedding {len(leaf_items)} leaf chunks")
        t_emb_start = time.perf_counter()
        leaf_nodes_created = self.batch_create_nodes(leaf_items)
        leaf_emb_time = time.perf_counter() - t_emb_start

        for node in leaf_nodes_created.values():
            all_nodes[node.index] = node
            layer_to_nodes[0].append(node)

        layer_stats.append(
            LayerStat(
                layer=0,
                name="Leaf Chunks",
                num_nodes=len(leaf_items),
                num_llm_calls=0,
                summarize_time=0.0,
                embedding_time=leaf_emb_time,
                total_time=chunk_time + leaf_emb_time,
            )
        )

        current_layer_nodes = get_node_list({n.index: n for n in layer_to_nodes[0]})
        final_layer_reached = 0

        for layer in range(1, self.num_layers + 1):
            if len(current_layer_nodes) <= self.reduction_dimension + 1:
                logging.info(
                    f"Stopping at layer {layer}: only {len(current_layer_nodes)} node(s) left "
                    f"(<= reduction_dimension + 1 = {self.reduction_dimension + 1})"
                )
                break

            embeddings = np.array(
                get_embeddings(current_layer_nodes, self.cluster_embedding_model)
            )

            logging.info(f"Clustering layer {layer} with {len(current_layer_nodes)} nodes")
            cluster_ids_per_node = perform_clustering(
                embeddings,
                dim=self.reduction_dimension,
                threshold=self.cluster_threshold,
                max_clusters=self.max_clusters,
                random_state=RANDOM_SEED + layer,
            )

            raw_groups: Dict[int, List[Node]] = collections.defaultdict(list)
            for node, cluster_ids in zip(current_layer_nodes, cluster_ids_per_node):
                for cid in cluster_ids:
                    raw_groups[int(cid)].append(node)

            groups: List[List[Node]] = []
            for group_nodes in raw_groups.values():
                groups.extend(
                    self._split_oversized_group(group_nodes, random_state=RANDOM_SEED + layer)
                )

            logging.info(f"Batch summarizing layer {layer} with {len(groups)} cluster(s)")
            parent_nodes, sum_time, emb_time = self._summarize_and_create_parents(
                groups, next_index
            )
            next_index += len(parent_nodes)

            for node in parent_nodes.values():
                all_nodes[node.index] = node
                layer_to_nodes[layer].append(node)

            layer_stats.append(
                LayerStat(
                    layer=layer,
                    name=f"Cluster Layer {layer}",
                    num_nodes=len(parent_nodes),
                    num_llm_calls=len(groups),
                    summarize_time=sum_time,
                    embedding_time=emb_time,
                    total_time=sum_time + emb_time,
                )
            )

            current_layer_nodes = get_node_list({n.index: n for n in layer_to_nodes[layer]})
            final_layer_reached = layer

            if len(current_layer_nodes) == 1:
                logging.info(f"Reached single root at layer {layer}")
                break

        # If we stopped early with more than one node remaining, summarize them
        # together into a single super-root so the tree always has one root.
        if len(current_layer_nodes) > 1:
            super_root_layer = final_layer_reached + 1
            combined_text = get_text(current_layer_nodes).strip()

            logging.info(f"Summarizing super root from {len(current_layer_nodes)} top-level nodes")
            t_super_sum_start = time.perf_counter()
            super_summary_list = self.batch_summarize(
                [combined_text], max_tokens=self.summarization_length
            )
            super_sum_time = time.perf_counter() - t_super_sum_start
            super_summary = super_summary_list[0] if super_summary_list else ""

            root_indices = {n.index for n in current_layer_nodes}
            t_super_emb_start = time.perf_counter()
            super_root_nodes = self.batch_create_nodes(
                [(next_index, super_summary, root_indices)]
            )
            super_emb_time = time.perf_counter() - t_super_emb_start
            super_root_node = super_root_nodes[next_index]

            all_nodes[next_index] = super_root_node
            layer_to_nodes[super_root_layer].append(super_root_node)
            final_layer_reached = super_root_layer

            layer_stats.append(
                LayerStat(
                    layer=super_root_layer,
                    name=f"Layer {super_root_layer} (Super Root)",
                    num_nodes=1,
                    num_llm_calls=1,
                    summarize_time=super_sum_time,
                    embedding_time=super_emb_time,
                    total_time=super_sum_time + super_emb_time,
                )
            )
            current_layer_nodes = [super_root_node]

        leaf_nodes_dict = {node.index: node for node in layer_to_nodes[0]}
        root_nodes_dict = {node.index: node for node in current_layer_nodes}
        num_layers = final_layer_reached + 1

        total_build_time = time.perf_counter() - t_build_start
        total_nodes = len(all_nodes)
        total_llm_calls = sum(l.num_llm_calls for l in layer_stats)
        total_summarize_time = sum(l.summarize_time for l in layer_stats)
        total_embedding_time = sum(l.embedding_time for l in layer_stats)

        build_stats = BuildStats(
            total_time=total_build_time,
            parse_time=chunk_time,
            total_summarize_time=total_summarize_time,
            total_embedding_time=total_embedding_time,
            total_nodes=total_nodes,
            total_llm_calls=total_llm_calls,
            num_layers=num_layers,
            layers=layer_stats,
        )
        self.last_build_stats = build_stats

        return Tree(
            all_nodes=all_nodes,
            root_nodes=root_nodes_dict,
            leaf_nodes=leaf_nodes_dict,
            num_layers=num_layers,
            layer_to_nodes=dict(layer_to_nodes),
            build_stats=build_stats,
        )
