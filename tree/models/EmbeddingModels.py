import logging
import os
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

MODELS_DIR = Path("models_cache")


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass

    def create_embeddings(self, texts: List[str]) -> List:
        return [self.create_embedding(t) for t in texts]


class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="BAAI/bge-m3", cache_folder=MODELS_DIR):
        try:
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=cache_folder, 
                local_files_only=True
            )
        except Exception:
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=cache_folder, 
                local_files_only=False
            )


    def create_embedding(self, text):
        return self.model.encode(text)

    def create_embeddings(self, texts: List[str]) -> List:
        if not texts:
            return []
        if len(texts) == 1:
            return [self.model.encode(texts[0])]
        return self.model.encode(texts, batch_size=32, show_progress_bar=True).tolist()
