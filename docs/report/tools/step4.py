from playwright.sync_api import sync_playwright
A = "docs/report/assets/"
PID = "01a17403-6bbe-7d4d-8e35-734710be129f"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(f"http://localhost:8501/project?active_project_id={PID}")
    pg.get_by_role("button", name="Upload tài liệu").first.wait_for(timeout=90000)
    pg.get_by_role("button", name="Upload tài liệu").first.click()
    pg.wait_for_timeout(1500)
    pg.locator("input[type=file]").set_input_files("data/src/pldc.md")
    pg.wait_for_timeout(2500)
    pg.get_by_role("dialog").get_by_role("button", name="Upload", exact=True).click()
    for i in range(3):
        pg.wait_for_timeout(8000)
        pg.screenshot(path=A + f"ui_ingest_{i}.png")
    b.close()
