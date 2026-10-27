from playwright.sync_api import sync_playwright
A = "docs/report/assets/"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto("http://localhost:8501")
    pg.get_by_role("button", name="Mở").first.wait_for(timeout=90000)
    pg.get_by_role("button", name="Mở").first.click()
    pg.wait_for_timeout(3000)
    print(pg.url)
    pg.screenshot(path=A + "ui_project_empty.png")
    pg.get_by_role("button", name="Upload tài liệu").first.click()
    pg.wait_for_timeout(1500)
    pg.locator("input[type=file]").set_input_files("data/src/pldc.md")
    pg.wait_for_timeout(2500)
    pg.screenshot(path=A + "ui_upload_dialog.png")
    print([x.inner_text() for x in pg.get_by_role("dialog").get_by_role("button").all()])
    b.close()
