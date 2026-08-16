import logging
import os
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLMConfig

load_dotenv()

# Tắt warning khuyến cáo Automatic Function Calling (AFC) từ Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# Bảng giá ước tính Google Gemini (USD / 1,000,000 tokens)
GEMINI_PRICING = {
    "default": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gemini-2.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gemini-2.0-flash": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-1.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gemini-1.5-pro": {"input_per_m": 1.25, "output_per_m": 5.00},
    "gemini-2.5-pro": {"input_per_m": 1.25, "output_per_m": 5.00},
    "gemini-3.0": {"input_per_m": 0.5, "output_per_m": 3.0}
}


class TokenTracker:
    """Theo dõi số lượng token và tính toán chi phí sử dụng LLM."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.reasoning_tokens: int = 0
        self.total_tokens: int = 0
        self.total_calls: int = 0

    def add_usage(self, usage: dict):
        if not usage or not isinstance(usage, dict):
            return
        inp = usage.get("input_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        tot = usage.get("total_tokens", 0) or (inp + out)
        reasoning = (usage.get("output_token_details") or {}).get("reasoning", 0) or 0

        self.input_tokens += inp
        self.output_tokens += out
        self.reasoning_tokens += reasoning
        self.total_tokens += tot
        self.total_calls += 1

    def get_summary(self, model_name: str = "gemini-3.0") -> dict:
        pricing = GEMINI_PRICING.get(model_name, GEMINI_PRICING["gemini-3.0"])
        input_cost = (self.input_tokens / 1_000_000) * pricing["input_per_m"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output_per_m"]
        total_cost = input_cost + output_cost
        vnd_cost = total_cost * 25400  # Tỷ giá tham khảo USD -> VNĐ

        return {
            "calls": self.total_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": total_cost,
            "total_cost_vnd": vnd_cost,
        }

    def format_report(self, model_name: str = "gemini-2.5-flash") -> str:
        summary = self.get_summary(model_name)
        reasoning_str = f" (gồm {summary['reasoning_tokens']:,} reasoning)" if summary['reasoning_tokens'] else ""
        return (
            "📊 [Thống kê Token & Chi phí Agent]\n"
            f"   • Số lần gọi LLM : {summary['calls']} lượt\n"
            f"   • Input Tokens   : {summary['input_tokens']:,}\n"
            f"   • Output Tokens  : {summary['output_tokens']:,}{reasoning_str}\n"
            f"   • Tổng Tokens    : {summary['total_tokens']:,}\n"
            f"   • Chi phí ước tính: ${summary['total_cost_usd']:.6f} USD (~{summary['total_cost_vnd']:.2f} VNĐ)"
        )


class LLMModel:
    """
    Wrapper around ChatGoogleGenerativeAI (Google Gemini) for LangChain Agent.

    Features:
        - Tool binding (bind_tools)
        - Synchronous invoke
        - Streaming (stream)
        - Async invoke (ainvoke)
        - Async streaming (astream)
        - Token & Price Tracking
    """

    DEFAULT_MODEL = LLMConfig.MODEL
    tracker = TokenTracker()

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initializes the LLM client using ChatGoogleGenerativeAI for Google Gemini.
        """
        self.model = model or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        self.temperature = temperature if temperature is not None else LLMConfig.TEMPERATURE
        self.top_p = top_p if top_p is not None else LLMConfig.TOP_P
        self.max_tokens = max_tokens if max_tokens is not None else LLMConfig.MAX_TOKENS
        self.timeout = timeout if timeout is not None else LLMConfig.TIMEOUT
        self.max_retries = max_retries if max_retries is not None else LLMConfig.MAX_RETRIES

        self._client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        logging.info(
            f"[LLMModel] Initialized ChatGoogleGenerativeAI with model={self.model}"
        )

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------

    @classmethod
    def reset_usage(cls):
        """Reset bộ đếm token trước khi bắt đầu câu hỏi mới."""
        cls.tracker.reset()

    @classmethod
    def get_usage_summary(cls, model_name: Optional[str] = None) -> dict:
        return cls.tracker.get_summary(model_name or cls.DEFAULT_MODEL)

    @classmethod
    def get_usage_report(cls, model_name: Optional[str] = None) -> str:
        return cls.tracker.format_report(model_name or cls.DEFAULT_MODEL)

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> BaseChatModel:
        """
        Binds a list of LangChain tools to the LLM, enabling function/tool calling.

        Args:
            tools (list[BaseTool]): List of tools to bind.
            **kwargs: Additional kwargs forwarded to client.bind_tools().

        Returns:
            BaseChatModel: A new runnable with tools bound.
        """
        bound = self._client.bind_tools(tools, **kwargs)
        logging.info(f"[LLMModel] Bound {len(tools)} tools: {[t.name for t in tools]}")
        return bound

    # ------------------------------------------------------------------
    # Synchronous
    # ------------------------------------------------------------------

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        """
        Sends messages to the LLM and returns a single response.

        Args:
            messages (list[BaseMessage]): Conversation history / prompt messages.
            **kwargs: Additional kwargs forwarded to client.invoke().

        Returns:
            BaseMessage: The LLM response message.
        """
        try:
            response = self._client.invoke(messages, **kwargs)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                LLMModel.tracker.add_usage(usage)
            logging.info(
                f"[LLMModel] invoke | tokens_used={usage or {}}"
            )
            return response
        except Exception as e:
            logging.error(f"[LLMModel] invoke error: {e}")
            raise

    def stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[BaseMessage]:
        """
        Streams LLM output chunk-by-chunk.

        Args:
            messages (list[BaseMessage]): Conversation history / prompt messages.
            **kwargs: Additional kwargs forwarded to client.stream().

        Yields:
            BaseMessage: Streamed response chunks.
        """
        try:
            for chunk in self._client.stream(messages, **kwargs):
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    LLMModel.tracker.add_usage(usage)
                yield chunk
        except Exception as e:
            logging.error(f"[LLMModel] stream error: {e}")
            raise

    # ------------------------------------------------------------------
    # Asynchronous
    # ------------------------------------------------------------------

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        """
        Asynchronously sends messages to the LLM and returns a single response.

        Args:
            messages (list[BaseMessage]): Conversation history / prompt messages.
            **kwargs: Additional kwargs forwarded to client.ainvoke().

        Returns:
            BaseMessage: The LLM response message.
        """
        try:
            response = await self._client.ainvoke(messages, **kwargs)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                LLMModel.tracker.add_usage(usage)
            logging.info(
                f"[LLMModel] ainvoke | tokens_used={usage or {}}"
            )
            return response
        except Exception as e:
            logging.error(f"[LLMModel] ainvoke error: {e}")
            raise

    async def astream(self, messages: list[BaseMessage], **kwargs: Any):
        """
        Asynchronously streams LLM output chunk-by-chunk.

        Args:
            messages (list[BaseMessage]): Conversation history / prompt messages.
            **kwargs: Additional kwargs forwarded to client.astream().

        Yields:
            BaseMessage: Streamed response chunks.
        """
        try:
            async for chunk in self._client.astream(messages, **kwargs):
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    LLMModel.tracker.add_usage(usage)
                yield chunk
        except Exception as e:
            logging.error(f"[LLMModel] astream error: {e}")
            raise

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def client(self) -> ChatGoogleGenerativeAI:
        """Returns the underlying ChatGoogleGenerativeAI client for advanced usage."""
        return self._client

    def __repr__(self) -> str:
        return (
            f"LLMModel(model={self.model!r}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
