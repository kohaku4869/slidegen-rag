const fs = require("fs");
const L = require("./lib");
const { D, h1, FONT } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, TableOfContents, Footer, PageNumber, LevelFormat } = D;

const C = [...require("./content1"), ...require("./content2")];
C.push(h1("TÀI LIỆU THAM KHẢO"));
[
  "Sarthi P. et al., “RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval”, ICLR 2024.",
  "Yan S.-Q. et al., “Corrective Retrieval Augmented Generation”, 2024.",
  "Lewis P. et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”, NeurIPS 2020.",
  "Chen J. et al., “BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings”, 2024.",
  "Tài liệu LangGraph: https://langchain-ai.github.io/langgraph/",
  "SQLAlchemy: https://docs.sqlalchemy.org/ ; Streamlit: https://docs.streamlit.io/",
  "Bài báo AgenticRAG (arXiv:2605.05538) được nêu trong tài liệu thiết kế của dự án.",
].forEach((t) => C.push(new Paragraph({ numbering: { reference: "num2", level: 0 }, children: L.runs(t), spacing: { after: 80 } })));

const center = (t, size, bold) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: t, font: FONT, size, bold })] });
const gap = (n) => new Paragraph({ spacing: { before: n }, children: [] });
const cover = [
  center("[TÊN TRƯỜNG]", 28, true), center("[KHOA / VIỆN]", 28, true), gap(1200),
  center("BÁO CÁO BÀI TẬP LỚN", 36, true), center("MÔN: [TÊN MÔN HỌC] (PYTHON)", 28, true), gap(600),
  center("ĐỀ TÀI", 28, true),
  center("SLIDEGEN-RAG: HỆ THỐNG HỎI ĐÁP TÀI LIỆU DÀI", 34, true),
  center("BẰNG AGENTIC RAG VỚI CHỈ MỤC CÂY HEADING", 34, true), gap(800),
  ...[["Giảng viên hướng dẫn", "[Tên giảng viên]"], ["Lớp", "[Mã lớp]"],
    ["Nhóm thực hiện", "Đinh Tuấn Quang (agent, retrieval, evaluation)"],
    ["", "Nguyễn Duy Long (backend, memory)"], ["", "Lê Anh Tú (giao diện Streamlit)"]].map(([a, b]) =>
    new Paragraph({ indent: { left: 1200 }, spacing: { after: 80 },
      children: [new TextRun({ text: a ? a + ": " : "", font: FONT, size: 26, bold: true }), new TextRun({ text: b, font: FONT, size: 26 })] })),
  gap(1000), center("[Địa danh], tháng 10 năm 2026", 26, false),
];
const head = (t) => new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: t, font: FONT, size: 30, bold: true })] });
const toc = [head("MỤC LỤC"), new TableOfContents("Mục lục", { hyperlink: true, headingStyleRange: "1-2" })];
const lists = [head("DANH MỤC HÌNH"),
  ...L.figures.map((t) => new Paragraph({ children: L.runs(t, { size: 24 }), spacing: { after: 60 } })),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 240 }, children: [new TextRun({ text: "DANH MỤC BẢNG", font: FONT, size: 30, bold: true })] }),
  ...L.tables.map((t) => new Paragraph({ children: L.runs(t, { size: 24 }), spacing: { after: 60 } }))];

const hs = (id, name, size, o, lvl) => ({ id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { size, bold: true, font: FONT, color: "1F3864", ...o }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: lvl, keepNext: true, keepLines: true } });
const ind = (l, h) => ({ paragraph: { indent: { left: l, hanging: h } } });
const doc = new Document({
  creator: "Nhóm HRAG", title: "Báo cáo bài tập lớn HRAG",
  features: { updateFields: true },
  styles: { default: { document: { run: { font: FONT, size: 26 } } },
    paragraphStyles: [hs("Heading1", "Heading 1", 32, {}, 0), hs("Heading2", "Heading 2", 28, {}, 1), hs("Heading3", "Heading 3", 26, { italics: true, color: "000000" }, 2)] },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: ind(720, 360) }, { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: ind(1080, 360) }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: ind(720, 360) }] },
    { reference: "num2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: ind(720, 540) }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1701, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] }) },
    children: [...toc, ...C],
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(__dirname + "/_body.docx", b); console.log("written", b.length, "figs", L.figures.length, "tables", L.tables.length); });
