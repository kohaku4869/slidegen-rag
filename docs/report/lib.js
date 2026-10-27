const fs = require("fs");
const path = require("path");
const D = require("docx");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, HeadingLevel,
  BorderStyle, ShadingType, LevelFormat } = D;

const FONT = "Times New Roman";
const SZ = 26; // 13pt
const ASSETS = path.join(__dirname, "assets");
let figN = 0, tabN = 0;
const figures = [], tables = [];

function runs(text, opts = {}) {
  // **bold** and `code` inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: SZ, ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: SZ, ...opts }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 22, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size: SZ, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: SZ, ...opts }));
  return out;
}
const p = (text, o = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 312 }, indent: o.noIndent ? undefined : { firstLine: 480 }, ...o.para });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: true });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60, line: 300 } });
const num = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(t), spacing: { after: 60, line: 300 } });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, italics: true, font: FONT, size: 24 })] });
const code = (t) => t.split("\n").map((l) => new Paragraph({ keepNext: true, keepLines: true, spacing: { after: 0, line: 252 },
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6" }, indent: { left: 240 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 20 })] }));

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fig(file, title, widthPx = 600, maxHOpt = 520) {
  const f = path.join(ASSETS, file);
  if (!fs.existsSync(f)) { console.warn("MISSING FIGURE", file); return [p(`[Thiếu ảnh: ${file}]`, { noIndent: true })]; }
  const { w, h } = pngSize(f);
  const maxH = maxHOpt; let W = widthPx, H = Math.round((widthPx * h) / w);
  if (H > maxH) { W = Math.round((W * maxH) / H); H = maxH; }
  figN++; figures.push(`Hình ${figN}. ${title}`);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: W, height: H },
      altText: { title, description: title, name: file } })] }), caption(`Hình ${figN}. ${title}`)];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "9CA3AF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(title, header, rows, widths) {
  tabN++; tables.push(`Bảng ${tabN}. ${title}`);
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, head, keep) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "DBE4F0" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ keepNext: keep, children: runs(String(t), { size: 22, bold: head ? true : undefined }) })] });
  return [new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 },
    children: [new TextRun({ text: `Bảng ${tabN}. ${title}`, italics: true, font: FONT, size: 24 })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, widths[i], true, true)) }),
        ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], false, ri < rows.length - 1)) }))] }),
    new Paragraph({ spacing: { after: 160 }, children: [] })];
}
module.exports = { D, p, h1, h2, h3, bullet, num, code, fig, table, caption, runs, FONT, SZ, figures, tables };
