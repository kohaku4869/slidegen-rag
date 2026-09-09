import collections
import logging
import os
import re
import time
from abc import abstractclassmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel, EmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import (BaseSummarizationModel,
                                  ChatNVIDIASummarizationModel)
from slidegen_rag.tree.stats import BuildStats, LayerStat
from slidegen_rag.tree.structure.tree_structure import Node, Tree
from slidegen_rag.tree.utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)



class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

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

    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log


class TreeBuilder:
    """
    The TreeBuilder class is responsible for building a hierarchical text abstraction
    structure, known as a "tree," using summarization models and
    embedding models.
    """

    def __init__(self, config) -> None:
        """Initializes the tokenizer, maximum tokens, number of layers, top-k value, threshold, and selection mode."""

        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        # self.split_threshold = config.split_threshold
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
        """Creates nodes for a batch of (index, text, children_indices) with batched embedding.

        Args:
            items: List of (index, text, children_indices) tuples.

        Returns:
            Dict mapping node index to Node object.
        """
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

    def parse_markdown_to_hierarchical_json(
        self, markdown_text: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parses a Markdown string into a hierarchical JSON structure.

        Rules discovered from real document analysis (pldc.md):
          1. Numeric depth (e.g. '1.1.2.1' -> 4) is authoritative; # count is unreliable.
          2. Headings containing chapter keywords (CHƯƠNG, PHẦN, etc.) are always level 1.
          3. All embedded markdown formatting symbols (*, _, `, ~) are stripped before processing.
          4. Lines starting with bullet markers (-, +, •) after a # are treated as content,
             not as section headings, because OCR tools often incorrectly apply # to list items.

        Returns preamble content (text before first heading) and a list of root node dicts.
        """
        # Keywords that signal a top-level chapter heading, regardless of # count
        CHAPTER_KEYWORDS = ('CHƯƠNG', 'PHẦN', 'CÂU HỎI ÔN TẬP', 'TÀI LIỆU THAM KHẢO')
        # Bullet markers that indicate a list item mistakenly tagged as a heading
        BULLET_MARKERS = ('-', '+', '•', '*')

        lines = markdown_text.strip().split('\n')
        root_nodes, ancestry_stack, preamble_lines = [], [], []
        first_heading_found = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            heading_match = re.match(r'^(#+)\s+(.*)', line)

            if heading_match:
                raw_text = heading_match.group(2).strip()

                # Rule 3: Strip all embedded markdown formatting symbols globally
                clean_text = re.sub(r'[*_`~\\]+', '', raw_text).strip()

                # Rule 4: If after cleaning the content starts with a bullet marker,
                # treat as a list item / content of the current section, not a new heading.
                if clean_text and clean_text[0] in BULLET_MARKERS:
                    if first_heading_found and ancestry_stack:
                        ancestry_stack[-1]['content'].append(clean_text)
                    elif not first_heading_found:
                        preamble_lines.append(clean_text)
                    continue

                first_heading_found = True

                # Rule 2: Chapter/part keywords always get level 1
                upper_text = clean_text.upper()
                if any(upper_text.startswith(kw) for kw in CHAPTER_KEYWORDS):
                    level = 1
                    clean_title = clean_text
                else:
                    # Rule 1: Derive level from numeric index (e.g. 1.1.2.1 -> 4)
                    # Also handle case where space appears between number and dot: "1.2.1.2 .Title"
                    normalized = re.sub(r'\s*\.\s*', '.', clean_text)
                    num_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s*', normalized)
                    if num_match:
                        num_str = num_match.group(1)
                        level = len(num_str.split('.'))
                        clean_title = normalized[num_match.end():].strip()
                    else:
                        # Fallback: use # count when no numeric index is present
                        level = len(heading_match.group(1))
                        clean_title = clean_text

                new_node = {"level": level, "title": clean_title, "content": [], "children": []}

                while ancestry_stack and ancestry_stack[-1]['level'] >= level:
                    ancestry_stack.pop()
                if not ancestry_stack:
                    root_nodes.append(new_node)
                else:
                    ancestry_stack[-1]['children'].append(new_node)
                ancestry_stack.append(new_node)
            else:
                if not first_heading_found:
                    preamble_lines.append(re.sub(r'[*_`~\\]+', '', line).strip())
                elif ancestry_stack:
                    ancestry_stack[-1]['content'].append(line)

        def _join_content_recursively(node):
            node['content'] = "\n".join(node.get('content', [])).strip()
            for child in node.get('children', []):
                _join_content_recursively(child)

        for root_node in root_nodes:
            _join_content_recursively(root_node)

        preamble_content = "\n".join(preamble_lines).strip()
        return preamble_content, root_nodes


    def _assign_json_layers(self, json_node: Dict[str, Any]) -> int:
        """
        Recursively assigns bottom-up layer numbers to JSON hierarchy dicts:
        - Nodes without children -> layer 0.
        - Nodes with children -> 1 + max(children layers).
        """
        children = json_node.get("children", [])
        if not children:
            json_node["layer"] = 0
            return 0

        child_layers = [self._assign_json_layers(c) for c in children]
        node_layer = 1 + max(child_layers)
        json_node["layer"] = node_layer
        return node_layer

    def _collect_json_nodes_by_layer(
        self, json_node: Dict[str, Any], layer_map: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        """
        Collects JSON hierarchy nodes into layer_map grouped by their assigned layer.
        """
        layer = json_node.get("layer", 0)
        layer_map[layer].append(json_node)
        for child in json_node.get("children", []):
            self._collect_json_nodes_by_layer(child, layer_map)

    def create_embedding(self, text) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_models[self.cluster_embedding_model].create_embedding(text)

    def summarize(self, context, max_tokens=2048) -> str:
        return self.summarization_model.summarize(context, max_tokens)

    def batch_summarize(self, contexts, max_tokens=2048):
        return self.summarization_model.batch_summarize(contexts, max_tokens)

    def build_from_text(self, markdown_text: str, use_multithreading: bool = True) -> Tree:
        """
        Builds a Tree structure from markdown text by parsing hierarchy and
        summarizing bottom-up layer by layer using batch_summarize until reaching a single root node.

        Args:
            markdown_text (str): The input Markdown text.
            use_multithreading (bool, optional): Kept for API compatibility.

        Returns:
            Tree: The built Tree object.
        """
        t_build_start = time.perf_counter()
        t_parse_start = time.perf_counter()
        preamble, root_jsons = self.parse_markdown_to_hierarchical_json(markdown_text)
        parse_time = time.perf_counter() - t_parse_start

        all_nodes: Dict[int, Node] = {}
        layer_to_nodes: Dict[int, List[Node]] = collections.defaultdict(list)
        layer_stats: List[LayerStat] = []
        next_index = 0

        if not root_jsons:
            text_content = preamble if preamble else markdown_text
            t_emb_start = time.perf_counter()
            node_idx, single_node = self.create_node(0, text_content, set())
            emb_time = time.perf_counter() - t_emb_start

            all_nodes[0] = single_node
            layer_to_nodes[0] = [single_node]
            root_dict = {0: single_node}
            leaf_dict = {0: single_node}

            l0_stat = LayerStat(
                layer=0,
                name="Leaf Nodes",
                num_nodes=1,
                num_llm_calls=0,
                summarize_time=0.0,
                embedding_time=emb_time,
                total_time=emb_time,
            )
            total_build_time = time.perf_counter() - t_build_start
            build_stats = BuildStats(
                total_time=total_build_time,
                parse_time=parse_time,
                total_summarize_time=0.0,
                total_embedding_time=emb_time,
                total_nodes=1,
                total_llm_calls=0,
                num_layers=1,
                layers=[l0_stat],
            )
            self.last_build_stats = build_stats

            return Tree(
                all_nodes=all_nodes,
                root_nodes=root_dict,
                leaf_nodes=leaf_dict,
                num_layers=1,
                layer_to_nodes=dict(layer_to_nodes),
                build_stats=build_stats,
            )

        # 1. Assign layer to all JSON hierarchy nodes
        for r_json in root_jsons:
            self._assign_json_layers(r_json)

        # 2. Group JSON nodes by layer
        layer_map: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r_json in root_jsons:
            self._collect_json_nodes_by_layer(r_json, layer_map)

        max_layer = max(layer_map.keys()) if layer_map else 0

        # 3. Process Layer 0 (Leaf Nodes - batch embedding, no LLM summarization)
        leaf_items: List[Tuple[int, str, Set[int]]] = []
        leaf_json_list: List[Dict[str, Any]] = []

        for leaf_json in layer_map[0]:
            title = leaf_json.get("title", "").strip()
            content = leaf_json.get("content", "").strip()
            leaf_text = f"{title}\n{content}".strip() if title else content

            leaf_items.append((next_index, leaf_text, set()))
            leaf_json_list.append(leaf_json)
            next_index += 1

        logging.info(f"Batch embedding Layer 0 with {len(leaf_items)} leaf nodes")
        t_l0_emb_start = time.perf_counter()
        leaf_nodes_created = self.batch_create_nodes(leaf_items)
        l0_emb_time = time.perf_counter() - t_l0_emb_start

        for leaf_json, (node_idx, _, _) in zip(leaf_json_list, leaf_items):
            node_obj = leaf_nodes_created[node_idx]
            leaf_json["node_obj"] = node_obj
            all_nodes[node_idx] = node_obj
            layer_to_nodes[0].append(node_obj)

        layer_stats.append(
            LayerStat(
                layer=0,
                name="Leaf Nodes",
                num_nodes=len(leaf_items),
                num_llm_calls=0,
                summarize_time=0.0,
                embedding_time=l0_emb_time,
                total_time=l0_emb_time,
            )
        )

        # 4. Process Layer 1..max_layer with batch_summarize + batch embedding per layer
        for current_layer in range(1, max_layer + 1):
            parent_jsons = layer_map[current_layer]
            if not parent_jsons:
                continue

            contexts_to_summarize: List[str] = []
            pending_parents_info: List[Tuple[Dict[str, Any], Set[int]]] = []

            for p_json in parent_jsons:
                title = p_json.get("title", "").strip()
                content = p_json.get("content", "").strip()
                child_nodes = [c["node_obj"] for c in p_json.get("children", []) if "node_obj" in c]
                children_indices = {c.index for c in child_nodes}

                children_texts = [f"Title: {c.text}" if "\n" not in c.text else c.text for c in child_nodes]
                context_for_summary = "\n\n".join(children_texts)

                if title:
                    context_for_summary = f"Topic: {title}\nContent:\n{context_for_summary}"
                if content:
                    context_for_summary += f"\nNote: {content}"

                contexts_to_summarize.append(context_for_summary)
                pending_parents_info.append((p_json, children_indices))

            # Batch summarize all parent nodes for current_layer in ONE call
            logging.info(f"Batch summarizing Layer {current_layer} with {len(contexts_to_summarize)} prompts")
            t_sum_start = time.perf_counter()
            summaries = self.batch_summarize(contexts_to_summarize, max_tokens=self.summarization_length)
            layer_sum_time = time.perf_counter() - t_sum_start

            # Collect all parent node data for batch embedding
            parent_items: List[Tuple[int, str, Set[int]]] = []
            for (p_json, children_indices), summary in zip(pending_parents_info, summaries):
                title = p_json.get("title", "").strip()
                parent_text = f"{title}: {summary}" if title else summary
                parent_items.append((next_index, parent_text, children_indices))
                next_index += 1

            # Batch embedding all parent nodes for current_layer in ONE call
            logging.info(f"Batch embedding Layer {current_layer} with {len(parent_items)} parent nodes")
            t_emb_start = time.perf_counter()
            parent_nodes_created = self.batch_create_nodes(parent_items)
            layer_emb_time = time.perf_counter() - t_emb_start

            for (p_json, _), (node_idx, _, _) in zip(pending_parents_info, parent_items):
                parent_node_obj = parent_nodes_created[node_idx]
                p_json["node_obj"] = parent_node_obj
                all_nodes[node_idx] = parent_node_obj
                layer_to_nodes[current_layer].append(parent_node_obj)

            layer_name = f"Layer {current_layer} Summaries"
            layer_stats.append(
                LayerStat(
                    layer=current_layer,
                    name=layer_name,
                    num_nodes=len(parent_items),
                    num_llm_calls=len(contexts_to_summarize),
                    summarize_time=layer_sum_time,
                    embedding_time=layer_emb_time,
                    total_time=layer_sum_time + layer_emb_time,
                )
            )

        # 5. Handle Super Root if top level has multiple root nodes
        top_root_nodes = [r["node_obj"] for r in root_jsons if "node_obj" in r]

        if len(top_root_nodes) == 1:
            final_root_node = top_root_nodes[0]
        else:
            root_indices = {r.index for r in top_root_nodes}
            root_texts = [r.text for r in top_root_nodes]
            combined_text = "\n\n".join(root_texts)

            logging.info(f"Summarizing super root from {len(top_root_nodes)} top root nodes")
            t_super_sum_start = time.perf_counter()
            super_summary_list = self.batch_summarize([combined_text], max_tokens=self.summarization_length)
            super_sum_time = time.perf_counter() - t_super_sum_start
            super_summary = super_summary_list[0] if super_summary_list else ""

            t_super_emb_start = time.perf_counter()
            super_root_nodes = self.batch_create_nodes(
                [(next_index, super_summary, root_indices)]
            )
            super_emb_time = time.perf_counter() - t_super_emb_start
            super_root_node = super_root_nodes[next_index]
            super_root_idx = next_index
            next_index += 1

            super_root_layer = max_layer + 1
            all_nodes[super_root_idx] = super_root_node
            layer_to_nodes[super_root_layer].append(super_root_node)
            final_root_node = super_root_node

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

        leaf_nodes_dict = {node.index: node for node in layer_to_nodes[0]}
        root_nodes_dict = {final_root_node.index: final_root_node}
        num_layers = max(layer_to_nodes.keys()) + 1

        total_build_time = time.perf_counter() - t_build_start
        total_nodes = len(all_nodes)
        total_llm_calls = sum(l.num_llm_calls for l in layer_stats)
        total_summarize_time = sum(l.summarize_time for l in layer_stats)
        total_embedding_time = sum(l.embedding_time for l in layer_stats)

        build_stats = BuildStats(
            total_time=total_build_time,
            parse_time=parse_time,
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


