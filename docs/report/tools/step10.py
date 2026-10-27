import sys; sys.path.insert(0, "docs/report/tools")
from common import *
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    open_project(pg)
    pg.get_by_role("button", name="Chi tiết").first.click(); pg.wait_for_timeout(4000)
    pg.get_by_role("tab", name="Cây").click(); pg.wait_for_timeout(2500)
    # expand first expander arrow then pick a node
    pg.locator("button[data-testid=stBaseButton-secondary]").first.click(); pg.wait_for_timeout(2500)
    pg.get_by_role("button", name="CÂU HỎI ÔN TẬP CHƯƠNG 2").first.click(); pg.wait_for_timeout(2500)
    pg.screenshot(path=A + "ui_document_tree.png")
    b.close()
