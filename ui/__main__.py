"""Entry point cho console script `slidegen-ui` (xem pyproject.toml) — tương
đương `streamlit run ui/app.py`. Tham số thêm được chuyển thẳng cho
Streamlit, vd. `uv run slidegen-ui --server.port 8600`.

UI KHÔNG cần `slidegen-api`: nó gọi `slidegen_rag.services` trực tiếp trong
cùng process. Chỉ cần `DATABASE_URL` (Postgres)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    from slidegen_rag.config import MODEL_CHOICES, apply_model_choice

    # `--model` là cờ của app (không phải của Streamlit): tách ra trước khi
    # chuyển phần còn lại cho Streamlit. Streamlit chạy cùng process nên env
    # đặt ở đây áp cho mọi lần rerun.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="gemini")
    known, streamlit_args = parser.parse_known_args(sys.argv[1:])
    apply_model_choice(known.model)

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *streamlit_args]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
