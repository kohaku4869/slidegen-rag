from playwright.sync_api import sync_playwright
A = "docs/report/assets/"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto("http://localhost:8501")
    pg.get_by_role("button", name="Tạo project").wait_for(timeout=90000)
    pg.get_by_role("button", name="Tạo project").click()
    d = pg.get_by_role("dialog")
    d.get_by_role("textbox").first.fill("Pháp luật đại cương")
    d.get_by_role("textbox").first.press("Enter")
    d.get_by_role("button", name="Tạo", exact=True).click()
    pg.wait_for_timeout(3000)
    print(pg.url)
    pg.screenshot(path=A + "ui_projects.png")
    names = pg.locator("h3, h2").all_inner_texts()
    print(names)
    b.close()
