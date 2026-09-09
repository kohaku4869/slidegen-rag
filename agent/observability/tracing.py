"""
LangSmith tracing bootstrap cho CRAG Agent.

Nguyên tắc: fail-soft tuyệt đối. Thiếu API key, lỗi mạng, hoặc bất kỳ lỗi nào
liên quan LangSmith đều KHÔNG được làm crash hoặc làm chậm lượt chạy Agent —
chỉ log cảnh báo và tự tắt tracing.

Sử dụng:
    from agent.observability.tracing import init_langsmith, build_run_config

    init_langsmith()  # gọi 1 lần lúc khởi tạo graph (agent/graph/builder.py)

    config, tracer = build_run_config("crag:câu hỏi", tags=["cli"])
    result = await graph.ainvoke(initial_state, config=config)
    url = resolve_run_url(tracer)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from config import LangSmithConfig

logger = logging.getLogger(__name__)

_initialized = False
_tracing_enabled = False


def init_langsmith(force: Optional[bool] = None) -> bool:
    """
    Bật/tắt LangSmith tracing dựa trên LangSmithConfig (.env).

    Set cả LANGCHAIN_TRACING_V2 lẫn LANGSMITH_TRACING vào os.environ để tương
    thích mọi phiên bản langchain-core, sau đó xoá lru_cache của langsmith
    (get_env_var, get_tracer_project) vì hai hàm này cache theo tiến trình —
    sửa os.environ muộn mà không clear cache thì vô tác dụng.

    Idempotent — gọi nhiều lần an toàn. Không bao giờ raise.

    Args:
        force: ép bật (True) / tắt (False) tracing, bỏ qua .env. None = theo
               LangSmithConfig.ENABLED (mặc định).

    Returns:
        True nếu tracing đang bật.
    """
    global _initialized, _tracing_enabled

    enabled = LangSmithConfig.ENABLED if force is None else force

    if enabled and not LangSmithConfig.API_KEY:
        logger.warning(
            "[tracing] LANGSMITH_TRACING=true nhưng thiếu LANGSMITH_API_KEY — "
            "tắt tracing, Agent vẫn chạy bình thường."
        )
        enabled = False

    try:
        if enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = LangSmithConfig.ENDPOINT
            os.environ["LANGSMITH_ENDPOINT"] = LangSmithConfig.ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = LangSmithConfig.API_KEY
            os.environ["LANGSMITH_API_KEY"] = LangSmithConfig.API_KEY
            os.environ["LANGCHAIN_PROJECT"] = LangSmithConfig.PROJECT
            os.environ["LANGSMITH_PROJECT"] = LangSmithConfig.PROJECT
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"

        try:
            from langsmith import utils as ls_utils

            ls_utils.get_env_var.cache_clear()
            ls_utils.get_tracer_project.cache_clear()
        except Exception:
            pass  # cache clear là best-effort, không có cũng không sao

        _tracing_enabled = enabled
        _initialized = True

        if enabled:
            logger.info(
                f"[tracing] LangSmith tracing BẬT | project={LangSmithConfig.PROJECT} "
                f"| endpoint={LangSmithConfig.ENDPOINT}"
            )
        else:
            logger.info("[tracing] LangSmith tracing TẮT")

        return enabled
    except Exception as e:
        logger.warning(f"[tracing] init_langsmith lỗi, tự tắt tracing: {e}")
        _tracing_enabled = False
        _initialized = True
        return False


def is_tracing_enabled() -> bool:
    """Trạng thái tracing hiện tại (sau init_langsmith)."""
    if not _initialized:
        return init_langsmith()
    return _tracing_enabled


def get_tracer(project: Optional[str] = None) -> Optional[Any]:
    """
    Tạo một LangChainTracer mới, dùng để tự tay đưa vào config["callbacks"]
    (thay vì dựa vào tracing_v2_enabled() contextvar).

    Returns:
        LangChainTracer nếu tracing đang bật, None nếu tắt hoặc lỗi khởi tạo.
    """
    if not is_tracing_enabled():
        return None
    try:
        from langsmith import Client
        from langchain_core.tracers.langchain import LangChainTracer

        client = Client(api_url=LangSmithConfig.ENDPOINT, api_key=LangSmithConfig.API_KEY)
        return LangChainTracer(project_name=project or LangSmithConfig.PROJECT, client=client)
    except Exception as e:
        logger.warning(f"[tracing] Không tạo được LangChainTracer: {e}")
        return None


def build_run_config(
    run_name: str,
    *,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    callbacks: Optional[list] = None,
    run_id: Optional[Any] = None,
    recursion_limit: int = 50,
) -> tuple[dict, Optional[Any]]:
    """
    Dựng RunnableConfig cho một lượt graph.ainvoke/astream.

    Gộp tracer riêng (nếu tracing bật) vào danh sách callbacks — không double
    log vì LangChain dedup theo isinstance(handler, LangChainTracer).

    Returns:
        (config, tracer) — tracer để sau đó gọi resolve_run_url(tracer).
    """
    tracer = get_tracer()
    cbs = list(callbacks or [])
    if tracer is not None:
        cbs.append(tracer)

    config: dict = {
        "run_name": run_name,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
        "recursion_limit": recursion_limit,
    }
    if cbs:
        config["callbacks"] = cbs
    if run_id is not None:
        config["run_id"] = run_id

    return config, tracer


def resolve_run_url(tracer: Optional[Any], *, wait: bool = True) -> Optional[str]:
    """
    Lấy URL trace LangSmith của root run gắn với `tracer`.

    Flush tracer trước khi đọc (nếu wait=True) để tránh race giữa việc ghi
    run lên server và việc đọc lại nó. Bọc try/except toàn bộ — lỗi mạng /
    404 / run chưa kịp ghi đều trả None thay vì raise.
    """
    if tracer is None:
        return None
    try:
        if wait:
            from langchain_core.tracers.langchain import wait_for_all_tracers

            wait_for_all_tracers()

        latest_run = getattr(tracer, "latest_run", None)
        run_id = getattr(latest_run, "id", None)
        if not run_id:
            return None

        client = getattr(tracer, "client", None)
        if client is None:
            from langsmith import Client

            client = Client(api_url=LangSmithConfig.ENDPOINT, api_key=LangSmithConfig.API_KEY)

        run = client.read_run(run_id)
        return getattr(run, "url", None)
    except Exception as e:
        logger.debug(f"[tracing] resolve_run_url thất bại (bỏ qua): {e}")
        return None
