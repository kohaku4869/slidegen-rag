import sys; sys.path.insert(0, "docs/report/tools")
from common import *
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto("http://localhost:8501"); pg.get_by_role("button", name="Mở").first.wait_for(timeout=60000)
    pg.wait_for_timeout(1000)
    pg.get_by_text("Settings", exact=True).first.click()
    pg.wait_for_timeout(2000)
    pg.get_by_text("Postgres").first.wait_for(timeout=30000)
    pg.wait_for_timeout(3000)
    pg.screenshot(path=A + "ui_settings.png")
    b.close()
