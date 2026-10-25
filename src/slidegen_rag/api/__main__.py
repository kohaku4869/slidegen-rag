"""Entry point cho console script `slidegen-api` (xem pyproject.toml).

Dev với auto-reload: `uv run uvicorn slidegen_rag.api.app:app --reload`
(không dùng console script này, vì `reload=True` fork tiến trình theo cách
không hợp với hàm `main()` gọi trực tiếp).
"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("slidegen_rag.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
