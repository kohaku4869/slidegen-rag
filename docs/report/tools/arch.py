from playwright.sync_api import sync_playwright
import os
rows = [
 ("Frontend", "#DBE4F0", ["Streamlit UI", "Gradio dashboard (đánh giá)", "Dòng lệnh (CLI)"]),
 ("Backend", "#E3EFE0", ["Tầng services (hỏi đáp, lập chỉ mục, job, bộ nhớ đệm cây)"]),
 ("RAG core", "#FBEFD5", ["LangGraph agent (decompose, facets, generate)", "Tree index (builder + retriever)", "ConversationManager (short-term memory)"]),
]
bottom = [
 ("Storage", "#EDE3F2", ["PostgreSQL 18 (catalog, sessions, messages)", "File tài liệu nguồn và chỉ mục cây"]),
 ("External models", "#F6E0E0", ["Gemini (agent, generate)", "Self-hosted LLM (grader, summarizer, generate)", "bge-m3 embeddings"]),
]
def box(title, color, items):
    chips = "".join(f"<div class='chip'>{i}</div>" for i in items)
    return f"<div class='layer' style='background:{color}'><div class='t'>{title}</div><div class='row'>{chips}</div></div>"
arrow = "<div class='arrow'>&#9660;</div>"
html = """<html><body style='margin:0;background:white;font-family:Segoe UI,Arial,sans-serif'>
<style>
#a{width:1100px;padding:20px;background:white}
.layer{border:2px solid #6b7280;border-radius:12px;padding:12px 16px}
.t{font-weight:700;font-size:26px;margin-bottom:8px;color:#1F3864}
.row{display:flex;gap:12px}
.chip{flex:1;background:white;border:1.5px solid #9CA3AF;border-radius:8px;padding:12px 10px;text-align:center;font-size:23px;line-height:1.25}
.arrow{text-align:center;font-size:30px;color:#4b5563;line-height:36px}
.two{display:flex;gap:16px}.two .layer{flex:1}
.two .row{flex-direction:column}
</style><div id='a'>""" + arrow.join(box(*r) for r in rows) + arrow + "<div class='two'>" + "".join(box(*b) for b in bottom) + "</div></div></body></html>"
open("arch.html", "w", encoding="utf-8").write(html)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=3)
    pg.goto("file:///" + os.path.abspath("arch.html").replace("\\", "/"))
    pg.locator("#a").screenshot(path="../assets/diag_arch.png")
    b.close()
