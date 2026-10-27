from playwright.sync_api import sync_playwright
A = "docs/report/assets/"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1100})
    pg.goto("http://localhost:7860")
    pg.get_by_role("tab", name="So sánh 2 lần chạy").click()
    pg.wait_for_timeout(1500)
    box = pg.get_by_role("tabpanel").filter(has_text="Chủ đề (Topic)").last
    pg.get_by_label("Chủ đề (Topic)").last.click()
    pg.wait_for_timeout(500)
    pg.get_by_role("option", name="pldc_v2").click()
    pg.wait_for_timeout(2500)
    pg.get_by_role("button", name="So sánh").click()
    pg.wait_for_timeout(5000)
    pg.screenshot(path=A + "gradio_results.png", full_page=True)
    b.close()
