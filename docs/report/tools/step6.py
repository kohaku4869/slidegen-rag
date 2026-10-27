import sys, time; sys.path.insert(0, "docs/report/tools")
from common import *
from playwright.sync_api import sync_playwright
Q1 = "So sánh quy phạm pháp luật và quy phạm đạo đức."
Q2 = "Vậy quy phạm pháp luật có những đặc điểm nào khác biệt nhất?"
def ask(pg, q, tag):
    box = pg.locator("[data-testid=stChatInput] textarea")
    box.fill(q); box.press("Enter")
    t0 = time.time(); n = 0
    while time.time() - t0 < 420:
        pg.wait_for_timeout(4000)
        if n < 2 and time.time() - t0 > 6 + n * 8:
            pg.screenshot(path=A + f"ui_chat_running_{tag}{n}.png"); n += 1
        if pg.get_by_text("Chi tiết chạy").count() > (0 if tag == "1" else 1):
            break
    pg.wait_for_timeout(2500)
    print(tag, "elapsed", round(time.time() - t0))
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    open_project(pg)
    pg.get_by_role("button", name="Chat mới").click()
    pg.locator("[data-testid=stChatInput] textarea").wait_for(timeout=60000)
    pg.wait_for_timeout(2000)
    pg.screenshot(path=A + "ui_chat_empty.png")
    ask(pg, Q1, "1")
    pg.screenshot(path=A + "ui_chat_answer.png")
    print(pg.url)
    ask(pg, Q2, "2")
    pg.screenshot(path=A + "ui_chat_followup.png")
    b.close()
