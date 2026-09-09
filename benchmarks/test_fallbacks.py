"""
test_fallbacks.py — Verification of:
1. Remote Embedding (ngrok) + Local SentenceTransformer fallback.
2. Summarization token length overflow + Gemini 2.5 fallback.
"""

import logging
import os
import sys
import unittest
from pathlib import Path

# Fix Windows cp1252 encoding crash khi in UTF-8 ra stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


class TestEmbeddingFallback(unittest.TestCase):
    def test_remote_embedding_success(self):
        from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
        
        # Test with the configured ngrok remote endpoint
        model = EmbeddingModel(
            model_name="BAAI/bge-m3",
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://20d7-34-85-165-45.ngrok-free.app/"),
            timeout=30,
        )
        texts = ["Xin chào thế giới", "Hệ thống RAG phân cấp"]
        embeddings = model.create_embeddings(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 1024)
        self.assertEqual(len(embeddings[1]), 1024)
        print(" Remote Embedding succeeded: 2 vectors with dim 1024.")

    def test_remote_embedding_fallback_to_local(self):
        from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
        
        # Point to an invalid remote URL to force fallback to local
        model = EmbeddingModel(
            model_name="BAAI/bge-m3",
            base_url="http://invalid-fake-host-999.xyz/v1/embeddings",
            timeout=2,
        )
        texts = ["Kiểm tra fallback embedding"]
        embeddings = model.create_embeddings(texts)
        
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 1024)
        print(" Fallback to local SentenceTransformer succeeded: 1 vector with dim 1024.")


class TestSummarizationFallback(unittest.TestCase):
    def test_summarize_normal(self):
        from slidegen_rag.tree.models.SummarizationModels import ChatSummarizationModel
        
        model = ChatSummarizationModel()
        context = "Hà Nội là thủ đô của Việt Nam, có lịch sử ngàn năm văn hiến."
        summary = model.summarize(context, max_tokens=100)
        self.assertTrue(len(summary) > 0)
        print(f" Normal summarize succeeded: {summary[:50]}...")

    def test_summarize_max_token_fallback_to_gemini(self):
        from slidegen_rag.tree.models.SummarizationModels import ChatSummarizationModel
        
        model = ChatSummarizationModel()
        # Create an oversized context (~30,000 words / ~40,000 tokens) that exceeds vLLM 8192 token limit
        oversized_context = (
            "Chương 1: Tổng quan về hệ thống pháp luật đại cương Việt Nam và các nguyên tắc cơ bản.\n"
            + "Điều 1: Quy định chung về quyền và nghĩa vụ của công dân. " * 3000
        )
        print(f"Testing oversized context of length: {len(oversized_context)} chars...")
        
        summary = model.summarize(oversized_context, max_tokens=200)
        self.assertTrue(len(summary) > 0, "Summary should not be empty after fallback to Gemini 2.5")
        print(f" Gemini 2.5 Fallback succeeded on oversized context! Summary: {summary[:100]}...")


if __name__ == "__main__":
    unittest.main()
