import sys; sys.path.insert(0, "docs/report/tools")
from common import *
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    open_project(pg)
    pg.screenshot(path=A + "ui_project_ready.png")
    pg.get_by_role("button", name="Chi tiết").first.click()
    pg.wait_for_timeout(5000)
    pg.screenshot(path=A + "ui_document_overview.png")
    print([t.inner_text() for t in pg.get_by_role("tab").all()])
    for name, f in [("Cây", "ui_document_tree.png"), ("Nguồn", "ui_document_source.png")]:
        pg.get_by_role("tab", name=name).click()
        pg.wait_for_timeout(3000)
        pg.screenshot(path=A + f)
    b.close()
