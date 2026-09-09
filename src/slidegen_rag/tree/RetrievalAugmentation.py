import logging
import pickle

from slidegen_rag.tree import legacy_pickle  # noqa: F401 — installs unpickling shim on import
from slidegen_rag.tree.builder.tree_builder import TreeBuilder, TreeBuilderConfig
from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import BaseSummarizationModel
from slidegen_rag.tree.retriever.tree_retriever import TreeRetriever, TreeRetrieverConfig
from slidegen_rag.tree.structure.tree_structure import Node, Tree

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,
        embedding_model=None,
        summarization_model=None,
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="SBERT",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="SBERT",
    ):
        # Validate and propagate shared embedding_model
        if embedding_model is not None and not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError("embedding_model must be an instance of BaseEmbeddingModel")
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        # Validate and propagate shared summarization_model
        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError("summarization_model must be an instance of BaseSummarizationModel")
        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # Set TreeBuilderConfig
        if tree_builder_config is None:
            tree_builder_config = TreeBuilderConfig(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )
        elif not isinstance(tree_builder_config, TreeBuilderConfig):
            raise ValueError("tree_builder_config must be an instance of TreeBuilderConfig")

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError("tree_retriever_config must be an instance of TreeRetrieverConfig")

        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}

            {tree_retriever_config}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
        )
        return config_summary


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree and retrieving information.
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.

        Args:
            config (RetrievalAugmentationConfig): The configuration for the instance.
            tree: The tree instance or the path to a pickled tree file.
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError("config must be an instance of RetrievalAugmentationConfig")

        # Load tree from path if a string is given
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        self.tree_builder = TreeBuilder(config.tree_builder_config)
        self.tree_retriever_config = config.tree_retriever_config

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_documents(self, docs: str):
        """
        Builds the tree from input Markdown text and initializes the TreeRetriever.

        Args:
            docs (str): The input Markdown text to add to the tree.
        """
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Are you sure? (y/n): "
            )
            if user_input.lower() != "y":
                return

        self.tree = self.tree_builder.build_from_text(markdown_text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        question: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Retrieves relevant context from the tree based on the query.

        Args:
            question (str): The query to search for.
            start_layer (int): The layer to start retrieval from.
            num_layers (int): The number of layers to traverse.
            top_k (int): Maximum number of nodes to retrieve.
            max_tokens (int): Maximum tokens in the returned context.
            collapse_tree (bool): Whether to search across all layers. Defaults to True.
            return_layer_information (bool): Whether to return layer metadata.

        Returns:
            str or Tuple[str, List]: Context string, optionally with layer metadata.
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def save(self, path: str):
        """
        Saves the tree to a pickle file.

        Args:
            path (str): The file path to save the tree.
        """
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
