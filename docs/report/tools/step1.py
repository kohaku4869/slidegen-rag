from playwright.sync_api import sync_playwright
A = "docs/report/assets/"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto("http://localhost:8501")
    pg.get_by_role("button", name="Tạo project").wait_for(timeout=90000)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=A + "ui_projects.png")
    pg.get_by_role("button", name="Tạo project").click()
    pg.wait_for_timeout(1500)
    pg.screenshot(path=A + "ui_create_dialog.png")
    pg.get_by_role("dialog").get_by_role("textbox").first.fill("Pháp luật đại cương")
    pg.wait_for_timeout(500)
    print([x.inner_text() for x in pg.get_by_role("dialog").get_by_role("button").all()])
    b.close()
