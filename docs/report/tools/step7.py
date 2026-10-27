import sys, time; sys.path.insert(0, "docs/report/tools")
from common import *
from playwright.sync_api import sync_playwright
Q3 = "Hãy so sánh trách nhiệm hình sự và trách nhiệm hành chính."
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    open_project(pg)
    pg.get_by_role("tab", name="Phiên chat").click()
    pg.wait_for_timeout(1500)
    pg.screenshot(path=A + "ui_sessions_tab.png")
    pg.get_by_role("button", name="Mở").first.click()
    pg.locator("[data-testid=stChatInput] textarea").wait_for(timeout=60000)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=A + "ui_chat_history.png")
    # details expander of last answer
    pg.get_by_text("Chi tiết chạy").last.click()
    pg.wait_for_timeout(1500)
    pg.mouse.wheel(0, 2000); pg.wait_for_timeout(800)
    pg.screenshot(path=A + "ui_chat_details.png")
    # citation chip
    chips = pg.locator("[data-testid=stPopover] button")
    print("chips", chips.count())
    if chips.count():
        chips.last.scroll_into_view_if_needed(); chips.last.click(); pg.wait_for_timeout(1500)
        pg.screenshot(path=A + "ui_chat_citation.png")
        pg.keyboard.press("Escape")
    # new question with expanded status
    box = pg.locator("[data-testid=stChatInput] textarea")
    box.fill(Q3); box.press("Enter")
    t0 = time.time(); shots = 0
    while time.time() - t0 < 300:
        pg.wait_for_timeout(3000)
        st = pg.locator("[data-testid=stExpander] summary, [data-testid=stStatus] summary")
        try:
            hdr = pg.get_by_text("Agent đang suy luận")
            if hdr.count() and shots < 3:
                if shots == 0 or True:
                    try: hdr.last.click(timeout=1500)
                    except Exception: pass
                pg.wait_for_timeout(800)
                pg.screenshot(path=A + f"ui_chat_running_x{shots}.png"); shots += 1
        except Exception as e:
            print("err", e)
        if pg.get_by_text("Chi tiết chạy").count() >= 3:
            break
    pg.wait_for_timeout(2000)
    print("elapsed", round(time.time() - t0), "shots", shots)
    b.close()
