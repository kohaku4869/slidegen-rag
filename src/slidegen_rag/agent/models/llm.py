import logging
import os
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from slidegen_rag.config import LLMConfig

load_dotenv()

# Tắt warning khuyến cáo Automatic Function Calling (AFC) từ Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# Bảng giá ước tính Google Gemini (USD / 1,000,000 tokens)
GEMINI_PRICING = {
    "default": {"input_per_m": 0.30, "output_per_m": 2.50},
    "gemini-2.5-flash": {"input_per_m": 0.30, "output_per_m": 2.50},
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
        self.local_calls: int = 0
        self.gemini_calls: int = 0

    def add_usage(self, usage: dict, is_local: bool = False):
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
        if is_local:
            self.local_calls += 1
        else:
            self.gemini_calls += 1

    def get_summary(self, model_name: str = "gemini-2.5-flash") -> dict:
        pricing = GEMINI_PRICING.get(model_name, GEMINI_PRICING["gemini-2.5-flash"])
        input_cost = (self.input_tokens / 1_000_000) * pricing["input_per_m"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output_per_m"]
        total_cost = input_cost + output_cost
        vnd_cost = total_cost * 25400  # Tỷ giá tham khảo USD -> VNĐ

        return {
            "calls": self.total_calls,
            "local_calls": self.local_calls,
            "gemini_calls": self.gemini_calls,
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
            f"   • Số lần gọi LLM : {summary['calls']} lượt (Gemini: {summary['gemini_calls']}, Local: {summary['local_calls']})\n"
            f"   • Input Tokens   : {summary['input_tokens']:,}\n"
            f"   • Output Tokens  : {summary['output_tokens']:,}{reasoning_str}\n"
            f"   • Tổng Tokens    : {summary['total_tokens']:,}\n"
            f"   • Chi phí ước tính: ${summary['total_cost_usd']:.6f} USD (~{summary['total_cost_vnd']:.2f} VNĐ)"
        )


class LLMModel:
    """
    Unified LLM wrapper hỗ trợ đa Provider (Google Gemini & Local OpenAI-compatible)
    kèm cơ chế tự động Fallback, Tool Binding, Streaming và Token Tracking.

    Providers:
        - "gemini": Google Gemini (mặc định cho các bước tư duy sâu như retrieve, context_eval, generate).
        - "local": Model tự host qua LLM_BASE_URL (thích hợp cho các bước nhẹ như yes/no grading).
    """

    DEFAULT_MODEL = LLMConfig.MODEL
    tracker = TokenTracker()

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        fallback_to_gemini: bool = True,
    ):
        self.provider = provider.lower()
        self.temperature = temperature if temperature is not None else LLMConfig.TEMPERATURE
        self.top_p = top_p if top_p is not None else LLMConfig.TOP_P
        self.max_tokens = max_tokens if max_tokens is not None else LLMConfig.MAX_TOKENS
        self.timeout = timeout if timeout is not None else LLMConfig.TIMEOUT
        self.max_retries = max_retries if max_retries is not None else LLMConfig.MAX_RETRIES
        self.fallback_to_gemini = fallback_to_gemini

        self._client: BaseChatModel | None = None
        self._fallback_client: BaseChatModel | None = None

        if self.provider in ("local", "openai"):
            self.base_url = base_url or os.getenv("LLM_BASE_URL", "").strip()
            self.api_key = api_key or os.getenv("LLM_API_KEY", "").strip()
            self.model = model or "default"

            if self.base_url and self.api_key:
                try:
                    self._client = ChatOpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
                    logging.info(f"[LLMModel] Initialized Local ChatOpenAI @ {self.base_url}")
                except Exception as e:
                    logging.warning(f"[LLMModel] Không thể tạo client Local OpenAI: {e}")

            if self.fallback_to_gemini:
                self._fallback_client = self._create_gemini_client(
                    model=LLMConfig.MODEL,
                    temperature=self.temperature,
                    max_tokens=max_tokens or 512,
                )

            # Nếu không có local base_url thì chuyển hẳn sang gemini
            if self._client is None:
                logging.info("[LLMModel] Không có Local URL hợp lệ, sử dụng Gemini làm primary.")
                self._client = self._fallback_client or self._create_gemini_client(
                    model=LLMConfig.MODEL,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.provider = "gemini"
        else:
            # Gemini Provider
            self.model = model or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
            self.api_key = (
                api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )
            self._client = self._create_gemini_client(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logging.info(f"[LLMModel] Initialized ChatGoogleGenerativeAI with model={self.model}")

    def _create_gemini_client(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatGoogleGenerativeAI:
        key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temperature,
            top_p=self.top_p,
            max_output_tokens=max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
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
        """
        bound = self._client.bind_tools(tools, **kwargs)
        logging.info(f"[LLMModel] Bound {len(tools)} tools: {[t.name for t in tools]}")
        return bound

    # ------------------------------------------------------------------
    # Synchronous
    # ------------------------------------------------------------------

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        try:
            response = self._client.invoke(messages, **kwargs)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                LLMModel.tracker.add_usage(usage, is_local=(self.provider == "local"))
            logging.info(f"[LLMModel] invoke ({self.provider}) | tokens_used={usage or {}}")
            return response
        except Exception as e:
            if self._fallback_client is not None:
                logging.warning(f"[LLMModel] Primary ({self.provider}) failed: {e}. Fallback to Gemini...")
                response = self._fallback_client.invoke(messages, **kwargs)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    LLMModel.tracker.add_usage(usage, is_local=False)
                return response
            logging.error(f"[LLMModel] invoke error: {e}")
            raise

    def stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[BaseMessage]:
        try:
            for chunk in self._client.stream(messages, **kwargs):
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    LLMModel.tracker.add_usage(usage, is_local=(self.provider == "local"))
                yield chunk
        except Exception as e:
            if self._fallback_client is not None:
                logging.warning(f"[LLMModel] Stream primary ({self.provider}) failed: {e}. Fallback to Gemini...")
                for chunk in self._fallback_client.stream(messages, **kwargs):
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        LLMModel.tracker.add_usage(usage, is_local=False)
                    yield chunk
                return
            logging.error(f"[LLMModel] stream error: {e}")
            raise

    # ------------------------------------------------------------------
    # Asynchronous
    # ------------------------------------------------------------------

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        try:
            response = await self._client.ainvoke(messages, **kwargs)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                LLMModel.tracker.add_usage(usage, is_local=(self.provider == "local"))
            logging.info(f"[LLMModel] ainvoke ({self.provider}) | tokens_used={usage or {}}")
            return response
        except Exception as e:
            if self._fallback_client is not None:
                logging.warning(f"[LLMModel] ainvoke primary ({self.provider}) failed: {e}. Fallback to Gemini...")
                response = await self._fallback_client.ainvoke(messages, **kwargs)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    LLMModel.tracker.add_usage(usage, is_local=False)
                return response
            logging.error(f"[LLMModel] ainvoke error: {e}")
            raise

    async def astream(self, messages: list[BaseMessage], **kwargs: Any):
        try:
            async for chunk in self._client.astream(messages, **kwargs):
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    LLMModel.tracker.add_usage(usage, is_local=(self.provider == "local"))
                yield chunk
        except Exception as e:
            if self._fallback_client is not None:
                logging.warning(f"[LLMModel] astream primary ({self.provider}) failed: {e}. Fallback to Gemini...")
                async for chunk in self._fallback_client.astream(messages, **kwargs):
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        LLMModel.tracker.add_usage(usage, is_local=False)
                    yield chunk
                return
            logging.error(f"[LLMModel] astream error: {e}")
            raise

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def client(self) -> BaseChatModel:
        """Returns the underlying chat model client."""
        return self._client

    def __repr__(self) -> str:
        return (
            f"LLMModel(provider={self.provider!r}, model={self.model!r}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens})"
        )
