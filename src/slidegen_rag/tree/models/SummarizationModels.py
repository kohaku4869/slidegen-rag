import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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
    Summarization model wrapper backed by an OpenAI-compatible endpoint (e.g. vLLM)
    with automatic fallback to Google Gemini 2.5 upon context length / max token overflow.

    Supports:
        - Single synchronous summarization (summarize)
        - Parallel synchronous batch summarization via ThreadPoolExecutor (batch_summarize)
        - Fully async batch summarization via asyncio.Semaphore (async_batch_summarize)
    """

    DEFAULT_BASE_URL = "http://171.226.10.154:8080/v1"
    DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        model: str = "default",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        gemini_api_key: Optional[str] = None,
        temperature: float = 0.4,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 5,
        retry_backoff_base: float = 2.0,
        max_workers: int = 20,
        request_delay: float = 0.0,
        rate_limit_backoff: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", self.DEFAULT_BASE_URL)
        self.fallback_model = fallback_model
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

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
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,  # we handle retries and fallbacks ourselves
        )

        self._fallback_client: Optional[ChatGoogleGenerativeAI] = None

        logging.info(
            f"[ChatSummarizationModel] Initialized: base_url={self.base_url}, "
            f"primary_model={self.model}, fallback_model={self.fallback_model}"
        )

    def _get_fallback_client(self) -> Optional[ChatGoogleGenerativeAI]:
        """Lazy initialization of Google Gemini fallback client."""
        if self._fallback_client is None:
            if not self.gemini_api_key:
                logging.warning("[ChatSummarizationModel] GEMINI_API_KEY is not set. Fallback to Gemini disabled.")
                return None
            try:
                self._fallback_client = ChatGoogleGenerativeAI(
                    model=self.fallback_model,
                    google_api_key=self.gemini_api_key,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                logging.info(f"[ChatSummarizationModel] Fallback client ({self.fallback_model}) initialized.")
            except Exception as e:
                logging.error(f"[ChatSummarizationModel] Failed to initialize Gemini fallback client: {e}")
                return None
        return self._fallback_client

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
        if not exception:
            return False
        msg = str(exception)
        return "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()

    def _is_context_length_error(self, exception: Exception) -> bool:
        """Detects if the error is due to prompt exceeding model's max token/context limit."""
        if not exception:
            return False
        msg = str(exception).lower()
        context_error_keywords = [
            "context length",
            "context_length",
            "maximum context length",
            "longer than the model",
            "too many tokens",
            "max token",
            "token limit",
            "input tokens",
            "string_above_max_length",
            "badrequesterror",
            "vllm error",
            "400",
        ]
        return any(kw in msg for kw in context_error_keywords)

    def _invoke_fallback(self, context: str, max_tokens: int = 1000) -> str:
        """Synchronously invokes Gemini 2.5 fallback."""
        fb_client = self._get_fallback_client()
        if fb_client is None:
            return ""
        try:
            msgs = self._build_messages(context)
            lc_messages = [
                SystemMessage(content=msgs[0]["content"]),
                HumanMessage(content=msgs[1]["content"]),
            ]
            resp = fb_client.invoke(lc_messages, max_output_tokens=max_tokens)
            if resp and resp.content and str(resp.content).strip():
                return str(resp.content).strip()
            return ""
        except Exception as e:
            logging.error(f"[ChatSummarizationModel] Gemini fallback error: {e}")
            return ""

    async def _ainvoke_fallback(self, context: str, max_tokens: int = 1000) -> str:
        """Asynchronously invokes Gemini 2.5 fallback."""
        fb_client = self._get_fallback_client()
        if fb_client is None:
            return ""
        try:
            msgs = self._build_messages(context)
            lc_messages = [
                SystemMessage(content=msgs[0]["content"]),
                HumanMessage(content=msgs[1]["content"]),
            ]
            resp = await fb_client.ainvoke(lc_messages, max_output_tokens=max_tokens)
            if resp and resp.content and str(resp.content).strip():
                return str(resp.content).strip()
            return ""
        except Exception as e:
            logging.error(f"[ChatSummarizationModel] Async Gemini fallback error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Synchronous — single
    # ------------------------------------------------------------------

    def summarize(self, context: str, max_tokens: int = 1000, stop_sequence=None) -> str:
        """Summarize a single context synchronously with automatic fallback on context overflow."""
        try:
            response = self.client.invoke(
                self._build_messages(context), max_completion_tokens=max_tokens
            )
            return response.content
        except Exception as e:
            if self._is_context_length_error(e):
                logging.warning(
                    f"[ChatSummarizationModel] Context length exceeded on primary model ({e}). "
                    f"Falling back to {self.fallback_model}..."
                )
                fb_res = self._invoke_fallback(context, max_tokens)
                if fb_res:
                    return fb_res

            logging.error(f"[ChatSummarizationModel] summarize error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Batch Summarization (Async-backed with Sync fallback)
    # ------------------------------------------------------------------

    def batch_summarize(self, contexts: List[str], max_tokens: int = 1000) -> List[str]:
        """
        Summarize a list of contexts concurrently using asyncio event loop (async_batch_summarize).
        Falls back to ThreadPoolExecutor if an event loop is already active in the current thread.
        """
        if not contexts:
            return []
        try:
            return asyncio.run(
                self.async_batch_summarize(
                    contexts=contexts,
                    max_tokens=max_tokens,
                    num_workers=self.max_workers,
                )
            )
        except RuntimeError:
            # Event loop already running (e.g. within an async framework)
            return self._sync_threadpool_batch_summarize(contexts, max_tokens=max_tokens)

    def _sync_threadpool_batch_summarize(self, contexts: List[str], max_tokens: int = 1000) -> List[str]:
        """Synchronous batch summarization via ThreadPoolExecutor."""
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
                # Immediate fallback on max token / context length error
                if self._is_context_length_error(e):
                    logging.warning(
                        f"[ChatSummarizationModel] Context {idx} exceeded max token limit ({e}). "
                        f"Falling back to {self.fallback_model} immediately..."
                    )
                    fb_res = self._invoke_fallback(contexts[idx], max_tokens)
                    if fb_res:
                        return idx, fb_res, None
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

        # Retry failed (transient errors only)
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
                    if self._is_context_length_error(e):
                        logging.warning(
                            f"[ChatSummarizationModel] Retry for context {idx} failed due to context length ({e}). "
                            f"Falling back to {self.fallback_model}..."
                        )
                        fb_res = self._invoke_fallback(contexts[idx], max_tokens)
                        if fb_res:
                            results[idx] = fb_res
                            failed_indices.remove(idx)
                            continue

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

        # Final safety net: if still failed, try fallback
        final_failed = [i for i, r in enumerate(results) if not r.strip()]
        if final_failed:
            logging.info(f"[ChatSummarizationModel] Attempting final Gemini fallback for {len(final_failed)} remaining context(s)...")
            for idx in final_failed:
                fb_res = self._invoke_fallback(contexts[idx], max_tokens)
                if fb_res:
                    results[idx] = fb_res

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
        Falls back to Gemini 2.5 on context length errors.
        """
        concurrency = min(num_workers or self.max_workers, len(contexts))
        semaphore = asyncio.Semaphore(concurrency)
        messages_list = [self._build_messages(c) for c in contexts]
        results = [""] * len(contexts)

        async def invoke_one_async(idx: int) -> tuple:
            """Single async request with per-request retry, fallback and backoff."""
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
                        if self._is_context_length_error(e):
                            logging.warning(
                                f"[async] Context {idx} exceeded max token limit ({e}). "
                                f"Falling back to {self.fallback_model} immediately..."
                            )
                            fb_res = await self._ainvoke_fallback(contexts[idx], max_tokens)
                            if fb_res:
                                return idx, fb_res, None
                            return idx, "", e

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

                # Final async fallback attempt
                fb_res = await self._ainvoke_fallback(contexts[idx], max_tokens)
                if fb_res:
                    return idx, fb_res, None

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
