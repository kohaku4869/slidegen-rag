"""
gemini_token_report.py — tổng hợp token/chi phí Gemini đã dùng từ 1 file
trajectory.txt đã lưu (data/test/<topic>/trajectory.txt). Đọc lại log text,
KHÔNG gọi lại LLM/API nào — dùng để soát chi phí của các lần chạy trước mà
không cần chạy lại benchmark.

Hỗ trợ cả 2 format dòng token trong trajectory.txt (xem
agent/observability/trajectory.py::render_trajectory):
    - Mới: "Tokens Gemini: in=X out=Y tổng=Z | Chi phí ước tính: $A USD (~B VNĐ)"
      — sau khi TokenTracker tách riêng Gemini/local (agent/models/llm.py),
      số này CHỈ tính token Gemini, đáng tin cậy cho việc theo dõi chi phí.
    - Cũ: "Tokens: in=X out=Y tổng=Z | Chi phí ước tính: $A USD (~B VNĐ)"
      — từ trước khi tách, có thể LẪN token của local model (grader) vào
      cùng 1 tổng; số ở đây chỉ mang tính tham khảo tương đối, không chính
      xác 100% cho riêng Gemini. Script đánh dấu [cũ] cho các câu dùng format
      này để người đọc biết mà không tưởng nhầm là số chính xác.

Usage:
    uv run python scripts/gemini_token_report.py --topic pldc_v2
    uv run python scripts/gemini_token_report.py --file data/test/pldc_v2/trajectory.txt
"""

import argparse
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_QUESTION_BLOCK_RE = re.compile(r"=== CÂU HỎI (\d+) ===")
_TOKENS_GEMINI_RE = re.compile(
    r"Tokens Gemini: in=([\d,]+) out=([\d,]+) tổng=([\d,]+) \| "
    r"Chi phí ước tính: \$([\d.]+) USD"
)
_TOKENS_LEGACY_RE = re.compile(
    r"^Tokens: in=([\d,]+) out=([\d,]+) tổng=([\d,]+) \| "
    r"Chi phí ước tính: \$([\d.]+) USD",
    re.MULTILINE,
)


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def parse_trajectory_file(text: str) -> list[dict]:
    """Trả về 1 dict/câu hỏi có dữ liệu token:
    {"num", "input_tokens", "output_tokens", "total_tokens", "cost_usd",
    "legacy"} — "legacy"=True nếu phải fallback về dòng "Tokens:" cũ (số có
    thể lẫn token local, xem module docstring)."""
    blocks = _QUESTION_BLOCK_RE.split(text)
    rows = []
    # re.split với 1 capturing group trả về [prefix, num1, block1, num2, block2, ...]
    for i in range(1, len(blocks), 2):
        num = int(blocks[i])
        block_text = blocks[i + 1]
        m = _TOKENS_GEMINI_RE.search(block_text)
        legacy = False
        if m is None:
            m = _TOKENS_LEGACY_RE.search(block_text)
            legacy = True
        if m is None:
            continue
        rows.append(
            {
                "num": num,
                "input_tokens": _to_int(m.group(1)),
                "output_tokens": _to_int(m.group(2)),
                "total_tokens": _to_int(m.group(3)),
                "cost_usd": float(m.group(4)),
                "legacy": legacy,
            }
        )
    return rows


def format_report(path: Path, rows: list[dict]) -> str:
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    total_tok = sum(r["total_tokens"] for r in rows)
    total_cost = sum(r["cost_usd"] for r in rows)
    n_legacy = sum(1 for r in rows if r["legacy"])

    lines = [f"File: {path}", f"Số câu hỏi có dữ liệu token: {len(rows)}"]
    if n_legacy:
        lines.append(
            f"⚠ {n_legacy} câu dùng format 'Tokens:' cũ (trước khi tách Gemini/local) "
            f"— số của các câu này có thể lẫn token local, không hoàn toàn chính xác."
        )
    lines.append("")
    lines.append("Chi tiết theo câu hỏi:")
    for r in rows:
        tag = " [cũ]" if r["legacy"] else ""
        lines.append(
            f"  Câu {r['num']:>3}: in={r['input_tokens']:>7,} out={r['output_tokens']:>7,} "
            f"tổng={r['total_tokens']:>8,} chi phí=${r['cost_usd']:.6f}{tag}"
        )
    lines.append("")
    lines.append("Tổng cộng:")
    lines.append(f"  Input tokens : {total_in:,}")
    lines.append(f"  Output tokens: {total_out:,}")
    lines.append(f"  Tổng tokens  : {total_tok:,}")
    lines.append(
        f"  Chi phí ước tính: ${total_cost:.4f} USD (~{total_cost * 25400:.0f} VNĐ)"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tổng hợp token/chi phí Gemini từ 1 file trajectory.txt đã lưu (không gọi lại LLM)."
    )
    parser.add_argument(
        "--topic", help="Tên topic dưới data/test/<topic>/trajectory.txt"
    )
    parser.add_argument("--file", help="Đường dẫn trực tiếp tới 1 file trajectory.txt")
    args = parser.parse_args()

    if not args.topic and not args.file:
        parser.error("Cần truyền --topic hoặc --file")

    path = (
        Path(args.file)
        if args.file
        else Path("data/test") / args.topic / "trajectory.txt"
    )
    if not path.exists():
        print(f"Không tìm thấy file: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    rows = parse_trajectory_file(text)
    if not rows:
        print(f"Không tìm thấy dòng 'Tokens' nào trong {path}", file=sys.stderr)
        return 1

    print(format_report(path, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
