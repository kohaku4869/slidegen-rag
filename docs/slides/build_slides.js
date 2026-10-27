const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const sizeOf = (f) => { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
const A = (f) => path.join(__dirname, "..", "report", "assets", f);
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "HRAG - Hierarchical Agentic RAG";
pres.author = "Nhóm HRAG";
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
const NAVY = "0F2D40", TEAL = "1C6E8C", AMBER = "F2A93B", INK = "1B2A33", MUTED = "5B6B75", LIGHT = "F3F6F8", WHITE = "FFFFFF", CARD = "1A4560";
const H = "Cambria", B = "Calibri";

function img(slide, file, x, y, w, h, name) {
  const s = sizeOf(A(file)); const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  slide.addImage({ path: A(file), x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, altText: name, objectName: name });
}
function title(slide, t, color = INK) {
  slide.addText(t, { x: 0.5, y: 0.3, w: 9, h: 0.75, fontFace: H, fontSize: 28, bold: true, color, margin: 0, isTextBox: true, valign: "middle", objectName: "Title" });
}
const content = (t) => { const s = pres.addSlide(); s.background = { color: WHITE }; title(s, t); return s; };
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });

// ---- 1. Tiêu đề và vấn đề
{
  const s = pres.addSlide(); s.background = { color: NAVY };
  s.addText("HRAG", { x: 0.6, y: 0.55, w: 8.8, h: 1.0, fontFace: H, fontSize: 60, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Hierarchical Agentic RAG: hỏi đáp tài liệu dài có trích dẫn", { x: 0.6, y: 1.55, w: 8.8, h: 0.5, fontFace: B, fontSize: 20, color: "CADCFC", margin: 0, isTextBox: true });
  s.addText("Bài tập lớn Python · Khoa CNTT1 · Đinh Tuấn Quang, Nguyễn Duy Long, Lê Anh Tú", { x: 0.6, y: 2.15, w: 8.8, h: 0.4, fontFace: B, fontSize: 14, color: "9FB6C6", margin: 0, isTextBox: true });
  s.addText("Naive RAG gặp vấn đề gì với giáo trình dài?", { x: 0.6, y: 3.0, w: 8.8, h: 0.4, fontFace: B, fontSize: 16, bold: true, color: AMBER, margin: 0, isTextBox: true });
  const probs = [["Cắt đoạn cố định", "Mất cấu trúc chương, mục"], ["Chỉ tìm một lần", "Không tự kiểm tra kết quả"], ["Không có trí nhớ", "Câu hỏi nối tiếp bị hiểu sai"]];
  probs.forEach(([a, b], i) => {
    const x = 0.6 + i * 3.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.5, w: 2.8, h: 1.3, fill: { color: CARD }, rectRadius: 0.1, objectName: "ProblemCard" + (i + 1) });
    s.addText([{ text: a, options: { bold: true, fontSize: 17, color: WHITE, breakLine: true } }, { text: b, options: { fontSize: 14, color: "CADCFC" } }],
      { x: x + 0.2, y: 3.6, w: 2.4, h: 1.1, fontFace: B, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addNotes("Quang mở đầu. Giới thiệu tên HRAG. Bài toán: hỏi đáp trên giáo trình Pháp luật đại cương dài khoảng 534 nghìn ký tự. Naive RAG cắt văn bản thành đoạn cố định nên mất cấu trúc, chỉ tìm một lần nên không biết kết quả đã đủ chưa, và không nhớ hội thoại. Nhóm xây HRAG để giải quyết ba vấn đề này.");
}

// ---- 2. Kiến trúc
{
  const s = content("Kiến trúc 4 lớp, dùng chung một tầng dịch vụ");
  const rows = [["Giao diện", "DBE4F0", ["Streamlit UI", "Dashboard Gradio", "Dòng lệnh"]], ["Backend", "E3EFE0", ["Tầng services: hỏi đáp, lập chỉ mục, job, cache cây"]],
    ["Lõi RAG", "FBEFD5", ["Agent LangGraph", "Cây chỉ mục", "Bộ nhớ hội thoại"]], ["Lưu trữ và mô hình", "EDE3F2", ["PostgreSQL, file", "Gemini, LLM tự host", "Embedding bge-m3"]]];
  rows.forEach(([t, c, items], i) => {
    const y = 1.2 + i * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 5.8, h: 0.9, fill: { color: c }, rectRadius: 0.08, objectName: "Layer" + (i + 1) });
    s.addText(t, { x: 0.6, y, w: 1.2, h: 0.9, fontFace: B, fontSize: 13, bold: true, color: NAVY, margin: 0, valign: "middle", isTextBox: true });
    const iw = (4.3 - 0.1 * (items.length - 1)) / items.length;
    items.forEach((it, j) => {
      const x = 1.9 + j * (iw + 0.1);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.12, w: iw, h: 0.66, fill: { color: WHITE }, rectRadius: 0.06, objectName: "LayerItem" + (i + 1) + "_" + (j + 1) });
      s.addText(it, { x, y: y + 0.12, w: iw, h: 0.66, fontFace: B, fontSize: 12, color: INK, align: "center", valign: "middle", margin: 3, isTextBox: true });
    });
  });
  const m = [["Đinh Tuấn Quang", "Agent, truy hồi, đánh giá"], ["Nguyễn Duy Long", "Backend, cơ sở dữ liệu, bộ nhớ"], ["Lê Anh Tú", "Giao diện Streamlit"]];
  m.forEach(([n, r], i) => {
    const y = 1.3 + i * 1.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.5, y, w: 3.0, h: 1.1, fill: { color: LIGHT }, rectRadius: 0.1, shadow: shadow(), objectName: "MemberCard" + (i + 1) });
    s.addText([{ text: n, options: { bold: true, fontSize: 16, color: NAVY, breakLine: true } }, { text: r, options: { fontSize: 14, color: MUTED } }],
      { x: 6.7, y: y + 0.1, w: 2.7, h: 0.9, fontFace: B, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addNotes("Quang giới thiệu nhanh kiến trúc: giao diện, backend dạng tầng services, lõi RAG gồm agent, cây chỉ mục và bộ nhớ, lớp lưu trữ gồm PostgreSQL và file. Điểm đáng chú ý là Streamlit gọi trực tiếp tầng services, không qua HTTP. Giới thiệu phân công: Quang lo agent và đánh giá, Long lo backend, cơ sở dữ liệu và bộ nhớ, Tú lo giao diện.");
}

// ---- 3. Cây heading
{
  const s = content("Chỉ mục cây theo heading, không cắt đoạn");
  img(s, "diag_tree.png", 0.5, 1.2, 3.9, 4.0, "Cấu trúc cây chỉ mục");
  [["320", "nút"], ["5", "tầng"], ["103", "lần gọi LLM"]].forEach(([n, l], i) => {
    const x = 4.8 + i * 1.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 1.5, h: 1.2, fill: { color: NAVY }, rectRadius: 0.1, objectName: "StatTile" + (i + 1) });
    s.addText([{ text: n, options: { fontSize: 32, bold: true, color: AMBER, fontFace: H, breakLine: true } }, { text: l, options: { fontSize: 13, color: WHITE, fontFace: B } }],
      { x, y: 1.3, w: 1.5, h: 1.2, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText([
    { text: "Lá là nguyên văn từng mục, nút cha là bản tóm tắt tính từ dưới lên", options: { bullet: true, breakLine: true } },
    { text: "Mỗi nút nhớ số dòng trong tài liệu nguồn nên trích dẫn chính xác", options: { bullet: true, breakLine: true } },
    { text: "Đi lên, xuống, sang bên trong cây để lấy thêm ngữ cảnh", options: { bullet: true } },
  ], { x: 4.8, y: 2.8, w: 4.7, h: 2.3, fontFace: B, fontSize: 16, color: INK, paraSpaceAfter: 10, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Quang: thay vì cắt đoạn cố định hay gom cụm như RAPTOR, nhóm dùng chính heading của tài liệu làm cấu trúc cây. Giáo trình được lập thành 320 nút, 5 tầng, tốn 103 lần gọi LLM để tóm tắt, khoảng 5 phút. Lá giữ nguyên văn, nút cha là tóm tắt. Mỗi nút giữ số dòng nguồn nên trích dẫn được chính xác.");
}

// ---- 4. Agent
{
  const s = content("Agent đa khía cạnh, tự kiểm tra và sửa lỗi");
  const fl = ["Chọn công cụ tìm kiếm", "Chạy công cụ song song", "Chấm điểm từng đoạn", "Đã đủ ngữ cảnh chưa?"];
  fl.forEach((t, i) => {
    const y = 1.3 + i * 0.98;
    s.addShape(i === 3 ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 2.7, h: 0.66, fill: { color: i === 3 ? AMBER : LIGHT }, rectRadius: 0.08, shadow: shadow(), objectName: "LoopBox" + (i + 1) });
    s.addText(t, { x: 0.6, y, w: 2.7, h: 0.66, fontFace: B, fontSize: 14, bold: true, color: NAVY, align: "center", valign: "middle", margin: 2, isTextBox: true });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: 1.95, y: y + 0.68, w: 0, h: 0.28, line: { color: TEAL, width: 2, endArrowType: "triangle" }, objectName: "LoopArrow" + (i + 1) });
  });
  s.addShape(pres.shapes.LINE, { x: 3.3, y: 4.5, w: 0.3, h: 0, line: { color: TEAL, width: 2 }, objectName: "LoopBack1" });
  s.addShape(pres.shapes.LINE, { x: 3.6, y: 1.63, w: 0, h: 2.87, line: { color: TEAL, width: 2 }, objectName: "LoopBack2" });
  s.addShape(pres.shapes.LINE, { x: 3.3, y: 1.63, w: 0.3, h: 0, flipH: true, line: { color: TEAL, width: 2, endArrowType: "triangle" }, objectName: "LoopBack3" });
  s.addText("chưa đủ: lặp lại", { x: 3.7, y: 2.9, w: 1.0, h: 0.5, fontFace: B, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
  const steps = [["Tách", "Câu hỏi được tách thành tối đa 4 khía cạnh, chạy song song"], ["Truy hồi", "Mỗi khía cạnh chọn 1 trong 5 công cụ, chấm điểm đoạn, tự đánh giá đã đủ chưa"], ["Trả lời", "Dừng khi đủ ngữ cảnh hoặc hết 5 vòng, rồi sinh câu trả lời có trích dẫn"]];
  steps.forEach(([a, b], i) => {
    const y = 1.3 + i * 1.35;
    s.addShape(pres.shapes.OVAL, { x: 4.85, y: y + 0.1, w: 0.6, h: 0.6, fill: { color: TEAL }, objectName: "StepNum" + (i + 1) });
    s.addText(String(i + 1), { x: 4.85, y: y + 0.1, w: 0.6, h: 0.6, align: "center", valign: "middle", fontFace: H, fontSize: 20, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText([{ text: a, options: { bold: true, fontSize: 18, color: NAVY, breakLine: true } }, { text: b, options: { fontSize: 15, color: INK } }],
      { x: 5.65, y, w: 3.85, h: 1.1, fontFace: B, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Quang: agent xây bằng LangGraph. Câu hỏi so sánh như so sánh quy phạm pháp luật và đạo đức được tách thành nhiều khía cạnh, mỗi khía cạnh chạy một vòng truy hồi riêng: chọn công cụ tìm kiếm, chấm điểm đoạn tìm được, rồi tự đánh giá đã đủ chưa. Nếu chưa đủ thì tìm tiếp, tối đa 5 vòng. Cuối cùng gộp ngữ cảnh và sinh câu trả lời có trích dẫn.");
}

// ---- 5. Kết quả
{
  const s = content("Điểm LLM-judge tăng từ 6,3 lên 8,54");
  const labels = ["Accuracy", "Completeness", "Relevance", "Faithfulness", "Tổng"];
  const data = [
    { name: "Naive RAG", labels, values: [5.8, 4.9, 6.5, 7.9, 6.3] },
    { name: "RAPTOR + CRAG", labels, values: [6.38, 5.47, 7.08, 8.80, 6.93] },
    { name: "Heading + CRAG", labels, values: [7.27, 6.92, 8.10, 9.27, 7.89] },
    { name: "HRAG (agent)", labels, values: [8.22, 8.38, 8.42, 9.15, 8.54] },
  ];
  s.addChart(pres.charts.BAR, data, { x: 0.4, y: 1.15, w: 6.5, h: 4.2, barDir: "col", barGrouping: "clustered",
    chartColors: ["B8C2CA", "F2A93B", "5FA8C4", NAVY], showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: INK, legendFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 10, valAxisMajorUnit: 2, valAxisLabelColor: MUTED, catAxisLabelColor: INK, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E3E8EC", size: 0.5 }, catGridLine: { style: "none" },
    showValue: false, objectName: "ScoreChart" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.3, w: 2.4, h: 2.0, fill: { color: NAVY }, rectRadius: 0.1, objectName: "BigStat" });
  s.addText([{ text: "8,54", options: { fontSize: 48, bold: true, color: AMBER, fontFace: H, breakLine: true } }, { text: "trên thang 10, HRAG", options: { fontSize: 14, color: WHITE, fontFace: B } }],
    { x: 7.2, y: 1.3, w: 2.4, h: 2.0, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText([{ text: "+2,2 điểm so với Naive RAG", options: { bold: true, color: NAVY, fontSize: 15, breakLine: true } },
    { text: "30 câu hỏi pháp luật, judge LLM, độ nhiễu khoảng ±0,06", options: { color: MUTED, fontSize: 13 } }],
    { x: 7.2, y: 3.5, w: 2.4, h: 1.5, fontFace: B, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
  s.addNotes("Quang: đánh giá trên 30 câu hỏi pháp luật tiếng Việt, mỗi câu chấm bốn tiêu chí bằng judge LLM. Naive RAG 6,3; RAPTOR kèm CRAG 6,93; cây heading kèm CRAG 7,89; HRAG đạt 8,54. Cải thiện lớn nhất ở Completeness nhờ tách khía cạnh. Độ nhiễu của judge khoảng 0,06 nên các chênh lệch này đều lớn hơn nhiễu. Chỉ có một tài liệu thử nghiệm nên cần mở rộng thêm.");
}

// ---- 6. Backend, bộ nhớ, CSDL
{
  const s = content("Backend, cơ sở dữ liệu và bộ nhớ hội thoại");
  const cols = [
    ["1", "tầng services dùng chung", ["Giao diện, dòng lệnh và dashboard cùng gọi", "Lập chỉ mục chạy nền, có tiến độ và huỷ"]],
    ["9", "bảng PostgreSQL 18", ["SQLAlchemy bất đồng bộ, Alembic migration", "Project, tài liệu, phiên chat, tin nhắn, trích dẫn"]],
    ["5", "lượt hội thoại gần nhất", ["Nén lịch sử cũ bằng LLM", "Viết lại câu hỏi nối tiếp thành câu đầy đủ"]],
  ];
  cols.forEach(([n, l, bl], i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.9, h: 3.3, fill: { color: LIGHT }, rectRadius: 0.1, shadow: shadow(), objectName: "Card" + (i + 1) });
    s.addText([{ text: n, options: { fontSize: 44, bold: true, color: TEAL, fontFace: H, breakLine: true } }, { text: l, options: { fontSize: 15, bold: true, color: NAVY } }],
      { x: x + 0.2, y: 1.4, w: 2.5, h: 1.4, fontFace: B, margin: 0, valign: "top", isTextBox: true });
    s.addText(bl.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < bl.length - 1 } })),
      { x: x + 0.2, y: 2.9, w: 2.5, h: 1.6, fontFace: B, fontSize: 15, color: INK, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Long trình bày. Backend là một tầng services thuần Python dùng chung cho giao diện và dòng lệnh. Dữ liệu lưu ở PostgreSQL 18 với 9 bảng, truy cập bất đồng bộ bằng SQLAlchemy và quản lý cấu trúc bằng Alembic. Bộ nhớ ngắn hạn giữ 5 lượt gần nhất, nén lịch sử cũ bằng LLM, và viết lại câu hỏi nối tiếp để agent hiểu được ngữ cảnh.");
}

// ---- 7. Demo
{
  const s = content("Demo trên giáo trình Pháp luật đại cương");
  const steps = ["Mở project, xem cây chỉ mục của tài liệu", "Hỏi câu so sánh để thấy các khía cạnh chạy song song", "Bấm vào trích dẫn để xem đoạn nguồn", "Hỏi câu nối tiếp để thấy bộ nhớ hoạt động"];
  steps.forEach((t, i) => {
    const y = 1.3 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: AMBER }, objectName: "DemoNum" + (i + 1) });
    s.addText(String(i + 1), { x: 0.5, y: y + 0.05, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: H, fontSize: 18, bold: true, color: NAVY, margin: 0, isTextBox: true });
    s.addText(t, { x: 1.2, y, w: 3.6, h: 0.8, fontFace: B, fontSize: 16, color: INK, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.3, w: 4.4, h: 3.1, fill: { color: LIGHT }, shadow: shadow(), objectName: "ScreenshotFrame" });
  img(s, "ui_chat_answer.png", 5.15, 1.35, 4.3, 3.0, "Ảnh chụp màn hình giao diện chat");
  s.addText("Câu hỏi gợi ý: “So sánh quy phạm pháp luật và quy phạm đạo đức”", { x: 5.1, y: 4.55, w: 4.4, h: 0.6, fontFace: B, fontSize: 13, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("Tú thực hiện demo, khoảng 2,5 phút. Kịch bản: (1) mở project Pháp luật đại cương, lướt qua cây chỉ mục; (2) hỏi So sánh quy phạm pháp luật và quy phạm đạo đức, vừa chờ vừa chỉ hộp tiến trình các khía cạnh chạy song song; (3) bấm một chip trích dẫn để xem đoạn nguồn; (4) hỏi Vậy quy phạm pháp luật thì sao để cho thấy bộ nhớ hội thoại. Chuẩn bị: kiểm tra dịch vụ embedding còn hoạt động, chạy thử trước một lượt, và có video quay sẵn dự phòng.");
}

// ---- 8. Kết luận
{
  const s = pres.addSlide(); s.background = { color: NAVY };
  s.addText("Kết luận", { x: 0.6, y: 0.4, w: 8.8, h: 0.8, fontFace: H, fontSize: 32, bold: true, color: WHITE, margin: 0, isTextBox: true });
  const cols = [["Đã làm được", ["Cây heading giữ cấu trúc và trích dẫn tới dòng nguồn", "Agent đa khía cạnh đạt 8,54/10", "Hệ thống đủ backend, cơ sở dữ liệu, bộ nhớ, giao diện"]],
    ["Hạn chế", ["Mới thử trên một tài liệu, 30 câu hỏi", "Chưa streaming, chưa có xác thực", "Chưa có bộ nhớ dài hạn"]],
    ["Hướng phát triển", ["Streaming câu trả lời", "pgvector thay cho pickle", "Mở rộng nhiều tài liệu, sinh slide từ câu trả lời"]]];
  cols.forEach(([h, bl], i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.4, w: 2.9, h: 3.0, fill: { color: CARD }, rectRadius: 0.1, objectName: "ConclCard" + (i + 1) });
    s.addText(h, { x: x + 0.2, y: 1.5, w: 2.5, h: 0.5, fontFace: H, fontSize: 18, bold: true, color: AMBER, margin: 0, isTextBox: true });
    s.addText(bl.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < bl.length - 1 } })),
      { x: x + 0.2, y: 2.05, w: 2.5, h: 2.25, fontFace: B, fontSize: 14, color: WHITE, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Xin cảm ơn thầy cô đã lắng nghe. Mời thầy cô đặt câu hỏi.", { x: 0.6, y: 4.75, w: 8.8, h: 0.5, fontFace: B, fontSize: 18, color: "CADCFC", margin: 0, isTextBox: true });
  s.addNotes("Cả nhóm kết luận. Kết quả: cây heading giữ cấu trúc, agent đa khía cạnh đạt 8,54 trên 10, hệ thống hoàn chỉnh từ backend đến giao diện. Hạn chế: mới thử một tài liệu, chưa streaming, chưa có xác thực. Hướng phát triển: streaming, pgvector, mở rộng nhiều tài liệu và sinh slide từ câu trả lời. Mời thầy cô đặt câu hỏi. Slide dự phòng khi bị hỏi: RAPTOR, chi phí và độ trễ, thư viện, xem báo cáo.");
}

pres.writeFile({ fileName: path.join(__dirname, "HRAG_thuyet_trinh.pptx") }).then((f) => console.log("written", f));
