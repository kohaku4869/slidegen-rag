import sys, time; sys.path.insert(0, "docs/report/tools")
from common import *
from playwright.sync_api import sync_playwright
Q = "So sánh trách nhiệm hình sự và trách nhiệm hành chính, nêu các loại hình phạt và biện pháp xử lý tương ứng."
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    open_project(pg)
    pg.get_by_role("button", name="Chat mới").click()
    box = pg.locator("[data-testid=stChatInput] textarea"); box.wait_for(timeout=60000)
    pg.wait_for_timeout(1500)
    box.fill(Q); box.press("Enter")
    best = 0; t0 = time.time(); opened = False
    while time.time() - t0 < 300:
        pg.wait_for_timeout(700)
        hdr = pg.get_by_text("Agent đang suy luận")
        if hdr.count():
            if not opened:
                try: hdr.last.click(timeout=1000); opened = True
                except Exception: pass
            n = pg.locator("code, [data-testid=stStatus] code").count()
            if n > best:
                best = n; pg.screenshot(path=A + "ui_chat_running.png")
        elif opened or time.time() - t0 > 8:
            if pg.get_by_text("Chi tiết chạy").count(): break
    print("steps", best, "elapsed", round(time.time() - t0))
    pg.wait_for_timeout(2000)
    pg.screenshot(path=A + "ui_chat_answer2.png")
    pg.get_by_text("Chi tiết chạy").last.scroll_into_view_if_needed()
    pg.get_by_text("Chi tiết chạy").last.click(); pg.wait_for_timeout(1500)
    pg.get_by_text("Chi tiết chạy").last.scroll_into_view_if_needed()
    pg.mouse.wheel(0, 150); pg.wait_for_timeout(500)
    pg.screenshot(path=A + "ui_chat_details.png")
    pg.goto("http://localhost:8501"); pg.get_by_role("button", name="Mở").first.wait_for(timeout=60000)
    pg.get_by_role("link", name="Settings").click(); pg.wait_for_timeout(4000)
    pg.screenshot(path=A + "ui_settings.png")
    b.close()
