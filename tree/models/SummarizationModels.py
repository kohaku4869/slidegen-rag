import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

    @abstractmethod
    def batch_summarize(self, contexts, max_tokens=150):
        pass


# Backward-compatible alias
ChatNVIDIASummarizationModel = None  # will be set after class definition


class ChatSummarizationModel(BaseSummarizationModel):
    """
    Summarization model wrapper backed by any OpenAI-compatible endpoint.

    Supports:
        - Single synchronous summarization (summarize)
        - Parallel synchronous batch summarization via ThreadPoolExecutor (batch_summarize)
        - Fully async batch summarization via asyncio.Semaphore (async_batch_summarize)

    All batch methods include automatic retry with exponential backoff for
    failed / rate-limited requests.
    """

    DEFAULT_BASE_URL = "http://171.226.10.154:8080/v1"

    def __init__(
        self,
        model: str = "default",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.4,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout: int = 30,
        max_retries: int = 5,
        retry_backoff_base: float = 2.0,
        max_workers: int = 20,
        request_delay: float = 0.0,
        rate_limit_backoff: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", self.DEFAULT_BASE_URL)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.rate_limit_backoff = rate_limit_backoff

        self.client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,  # we handle retries ourselves
        )

        logging.info(
            f"[ChatSummarizationModel] Initialized: base_url={self.base_url}, model={self.model}"
        )

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_messages(self, context: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": "Bạn là một người tóm tắt nội dung dựa trên những gì được cung cấp.",
            },
            {
                "role": "user",
                "content": (
                    "Viết tóm tắt dựa trên context sau, càng nhiều chi tiết quan trọng càng tốt,"
                    "chỉ trả lời bản tóm tắt,không trả lời thừa,dưới 1000 từ,"
                    f"không viết dưới dạng Markdown: {context}:"
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_rate_limited(self, exception: Exception) -> bool:
        msg = str(exception)
        return "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()

    # ------------------------------------------------------------------
    # Synchronous — single
    # ------------------------------------------------------------------

    def summarize(self, context: str, max_tokens: int = 1000, stop_sequence=None) -> str:
        """Summarize a single context synchronously."""
        try:
            response = self.client.invoke(
                self._build_messages(context), max_completion_tokens=max_tokens
            )
            return response.content
        except Exception as e:
            logging.error(f"[ChatSummarizationModel] summarize error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Synchronous — batch (ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def batch_summarize(self, contexts: List[str], max_tokens: int = 1000) -> List[str]:
        """
        Summarize a list of contexts in parallel using ThreadPoolExecutor.
        Failed requests are automatically retried with exponential backoff.
        """
        messages_list = [self._build_messages(c) for c in contexts]
        results = [""] * len(contexts)

        def invoke_one(idx):
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            try:
                resp = self.client.invoke(messages_list[idx], max_completion_tokens=max_tokens)
                if resp and resp.content and resp.content.strip():
                    return idx, resp.content, None
                return idx, "", None
            except Exception as e:
                return idx, "", e

        max_workers = min(self.max_workers, len(contexts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(invoke_one, i): i for i in range(len(contexts))}
            with tqdm(total=len(contexts), desc="Summarizing (sync)", unit="ctx") as pbar:
                for future in as_completed(futures):
                    idx, content, err = future.result()
                    results[idx] = content
                    if err and self._is_rate_limited(err):
                        logging.warning(f"Rate limit hit on context {idx}: {err}")
                    pbar.update(1)

        # Retry failed
        failed_indices = [i for i, r in enumerate(results) if not r.strip()]
        for attempt in range(self.max_retries):
            if not failed_indices:
                break
            logging.info(
                f"Retrying {len(failed_indices)} failed contexts (attempt {attempt + 1}/{self.max_retries})"
            )
            for idx in failed_indices[:]:
                try:
                    resp = self.client.invoke(messages_list[idx], max_completion_tokens=max_tokens)
                    if resp and resp.content and resp.content.strip():
                        results[idx] = resp.content
                        failed_indices.remove(idx)
                    if self.request_delay > 0:
                        time.sleep(self.request_delay)
                except Exception as e:
                    if self._is_rate_limited(e):
                        logging.warning(
                            f"Rate limited on context {idx}, sleeping {self.rate_limit_backoff}s"
                        )
                        time.sleep(self.rate_limit_backoff)
                    else:
                        logging.warning(
                            f"Retry {attempt + 1}/{self.max_retries} for context {idx} failed: {e}"
                        )
                        time.sleep(self.retry_backoff_base ** attempt)

        return results

    # ------------------------------------------------------------------
    # Asynchronous — batch (asyncio.Semaphore + tqdm.asyncio)
    # ------------------------------------------------------------------

    async def async_batch_summarize(
        self,
        contexts: List[str],
        max_tokens: int = 1000,
        num_workers: int = 15,
    ) -> List[str]:
        """
        Summarize a list of contexts concurrently using asyncio.

        Concurrency is controlled by asyncio.Semaphore (num_workers or self.max_workers).
        Failed requests are retried up to self.max_retries times with exponential backoff.
        Progress is displayed via tqdm.asyncio.

        Args:
            contexts:    List of text contexts to summarize.
            max_tokens:  Maximum completion tokens per request.
            num_workers: Max concurrent async requests (defaults to self.max_workers).

        Returns:
            List of summary strings in the same order as contexts.
        """
        concurrency = min(num_workers or self.max_workers, len(contexts))
        semaphore = asyncio.Semaphore(concurrency)
        messages_list = [self._build_messages(c) for c in contexts]
        results = [""] * len(contexts)

        async def invoke_one_async(idx: int) -> tuple:
            """Single async request with per-request retry and exponential backoff."""
            async with semaphore:
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)

                last_err = None
                for attempt in range(self.max_retries + 1):
                    try:
                        resp = await self.client.ainvoke(
                            messages_list[idx],
                            max_completion_tokens=max_tokens,
                        )
                        if resp and resp.content and resp.content.strip():
                            return idx, resp.content, None
                        return idx, "", None

                    except Exception as e:
                        last_err = e
                        if self._is_rate_limited(e):
                            sleep_time = self.rate_limit_backoff
                            logging.warning(
                                f"[async] Rate limit on context {idx} "
                                f"(attempt {attempt + 1}), sleeping {sleep_time}s: {e}"
                            )
                        else:
                            sleep_time = self.retry_backoff_base ** attempt
                            logging.warning(
                                f"[async] Error on context {idx} "
                                f"(attempt {attempt + 1}/{self.max_retries}), "
                                f"sleeping {sleep_time:.1f}s: {e}"
                            )

                        if attempt < self.max_retries:
                            await asyncio.sleep(sleep_time)

                logging.error(
                    f"[async] Context {idx} failed after {self.max_retries + 1} attempts: {last_err}"
                )
                return idx, "", last_err

        # Launch all tasks and update progress bar as each completes
        tasks = [asyncio.create_task(invoke_one_async(i)) for i in range(len(contexts))]

        with async_tqdm(total=len(contexts), desc="Summarizing (async)", unit="ctx") as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, content, err = await coro
                results[idx] = content
                pbar.update(1)

        failed = [i for i, r in enumerate(results) if not r.strip()]
        if failed:
            logging.warning(
                f"[async_batch_summarize] {len(failed)} context(s) produced empty results: {failed}"
            )

        return results


# Backward-compatible alias so callers still using the old name keep working
ChatNVIDIASummarizationModel = ChatSummarizationModel
