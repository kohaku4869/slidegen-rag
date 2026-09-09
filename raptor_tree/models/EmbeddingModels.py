import json
import logging
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

MODELS_DIR = Path("models_cache")


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str):
        pass

    def create_embeddings(self, texts: List[str]) -> List:
        return [self.create_embedding(t) for t in texts]


class EmbeddingModel(BaseEmbeddingModel):
    """
    Embedding Model supporting Remote-first execution (via ngrok/OpenAI-compatible endpoint)
    with automatic fallback to local SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        base_url: Optional[str] = None,
        cache_folder: Path = MODELS_DIR,
        timeout: int = 60,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.timeout = timeout
        self.batch_size = batch_size

        # Resolve remote base URL (from parameter, .env, or default)
        raw_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        raw_url = raw_url.strip() if raw_url else ""
        if raw_url:
            if raw_url.endswith("/v1/embeddings"):
                self.api_url = raw_url
            else:
                self.api_url = f"{raw_url.rstrip('/')}/v1/embeddings"
        else:
            self.api_url = None

        self._local_model = None

        if self.api_url:
            logging.info(f"[EmbeddingModel] Initialized with Remote URL: {self.api_url} (model={self.model_name})")
        else:
            logging.info(f"[EmbeddingModel] Initialized with Local SentenceTransformer (model={self.model_name})")

    def _get_local_model(self):
        """Lazy loader for local SentenceTransformer model."""
        if self._local_model is None:
            logging.info(f"[EmbeddingModel] Loading local SentenceTransformer model: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self._local_model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_folder,
                    local_files_only=True,
                    device=device,
                )
            except Exception:
                self._local_model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_folder,
                    local_files_only=False,
                    device=device,
                )
            logging.info(f"[EmbeddingModel] Local model loaded successfully on device={device}.")
        return self._local_model

    def _remote_encode_chunk(self, chunk: List[str]) -> List[List[float]]:
        """Sends a single chunk of texts to the remote embedding endpoint."""
        payload = json.dumps({
            "input": chunk,
            "model": self.model_name,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "SlideGenRAG/1.0",
        }

        req = urllib.request.Request(self.api_url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Remote API responded with status {response.status}")
            data = json.loads(response.read().decode("utf-8"))

        embeddings_data = data.get("data", [])
        if not embeddings_data:
            raise ValueError(f"Empty data received from remote embedding endpoint: {data}")

        # Ensure embeddings are returned in proper input index order
        embeddings_data.sort(key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in embeddings_data]

    def _remote_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Sends batched requests to remote embedding endpoint."""
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            chunk_res = self._remote_encode_chunk(chunk)
            results.extend(chunk_res)
        return results

    def _local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Encodes texts using local SentenceTransformer."""
        model = self._get_local_model()
        if len(texts) == 1:
            vec = model.encode(texts[0])
            return [vec.tolist() if hasattr(vec, "tolist") else list(vec)]
        
        vecs = model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return vecs.tolist() if hasattr(vecs, "tolist") else [list(v) for v in vecs]

    def create_embeddings(self, texts: List[str]) -> List:
        """
        Creates embeddings for a list of texts.
        Attempts remote endpoint first; automatically falls back to local model on error.
        """
        if not texts:
            return []

        if self.api_url:
            try:
                embeddings = self._remote_embeddings(texts)
                if len(embeddings) == len(texts):
                    logging.info(
                        f"[EmbeddingModel] Successfully generated {len(texts)} embeddings via Remote API ({self.api_url})"
                    )
                    return embeddings
            except Exception as e:
                logging.warning(
                    f"[EmbeddingModel] Remote embedding failed ({e}). Falling back to local SentenceTransformer..."
                )

        # Fallback to local
        return self._local_embeddings(texts)

    def create_embedding(self, text: str) -> List[float]:
        """Creates embedding for a single text."""
        res = self.create_embeddings([text])
        return res[0] if res else []
