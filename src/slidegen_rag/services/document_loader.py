"""document_loader.py — Chuyển file nguồn (.md/.txt/.docx/.pdf) thành
Markdown trước khi đưa vào `TreeBuilder` (chỉ hiểu Markdown). Dùng chung bởi
CLI, UI và REST API.

- `.docx`: giữ cấu trúc — style Heading N/Title -> `#`, danh sách -> `- `,
  bảng -> bảng Markdown.
- `.pdf`: extract text đơn giản (pypdf), KHÔNG OCR, không nhận diện heading;
  mỗi trang thành một mục `## Trang N` dưới tiêu đề `# <tên file>` để cây có
  cấu trúc tối thiểu. PDF scan (không có text layer) báo lỗi rõ ràng.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = (".md", ".markdown", ".txt", ".docx", ".pdf")


class UnsupportedDocumentError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def to_markdown(filename: str, data: bytes) -> str:
    """Raise `UnsupportedDocumentError` (đuôi file lạ) hoặc
    `EmptyDocumentError` (không trích được text)."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    if ext in (".md", ".markdown", ".txt"):
        text = data.decode("utf-8", errors="ignore")
    elif ext == ".docx":
        text = _docx_to_markdown(data)
    elif ext == ".pdf":
        text = _pdf_to_markdown(data, title=stem)
    else:
        raise UnsupportedDocumentError(
            f"Định dạng '{ext or filename}' chưa hỗ trợ — dùng: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if not text.strip():
        raise EmptyDocumentError(f"Không trích được nội dung văn bản từ '{filename}'")
    return text


def load_file(path: str | Path) -> str:
    p = Path(path)
    return to_markdown(p.name, p.read_bytes())


# ── PDF ──────────────────────────────────────────────────────────────────────


def _pdf_to_markdown(data: bytes, *, title: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    sections: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(f"## Trang {i}\n\n{_clean_pdf_text(text)}")
    if not sections:
        return ""
    return f"# {title}\n\n" + "\n\n".join(sections) + "\n"


def _clean_pdf_text(text: str) -> str:
    # Dòng bắt đầu bằng '#' trong PDF không phải heading Markdown.
    text = re.sub(r"(?m)^#", r"\\#", text)
    return re.sub(r"\n{3,}", "\n\n", text)


# ── DOCX ─────────────────────────────────────────────────────────────────────


def _docx_to_markdown(data: bytes) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    # `p.style` của python-docx tra cứu rất chậm (~ms/đoạn); dựng bảng id->tên một lần.
    style_names = {s.style_id: s.name for s in doc.styles}
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            md = _paragraph_to_markdown(Paragraph(child, doc), style_names)
        elif tag == "tbl":
            md = _table_to_markdown(Table(child, doc))
        else:
            continue
        if md:
            blocks.append(md)
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _paragraph_to_markdown(p, style_names: dict[str, str]) -> str:
    text = p.text.strip()
    if not text:
        return ""
    ppr = p._p.pPr
    style_id = ppr.pStyle.val if ppr is not None and ppr.pStyle is not None else None
    style = style_names.get(style_id, "") if style_id else ""
    if style == "Title":
        return f"# {text}"
    m = re.match(r"Heading (\d)", style)
    if m:
        return f"{'#' * min(int(m.group(1)), 6)} {text}"
    if style.startswith("List Number"):
        return f"1. {text}"
    if style.startswith("List"):
        return f"- {text}"
    return text


def _table_to_markdown(table) -> str:
    rows = [
        [re.sub(r"\s+", " ", c.text).strip().replace("|", "\\|") for c in row.cells]
        for row in table.rows
    ]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
