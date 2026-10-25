"""Entry point cho console script `slidegen-ui` (xem pyproject.toml) — tương
đương `streamlit run ui/app.py`. Tham số thêm được chuyển thẳng cho
Streamlit, vd. `uv run slidegen-ui --server.port 8600`.

UI KHÔNG cần `slidegen-api`: nó gọi `slidegen_rag.services` trực tiếp trong
cùng process. Chỉ cần `DATABASE_URL` (Postgres)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
