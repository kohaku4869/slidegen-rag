A = "docs/report/assets/"
def open_project(pg):
    pg.goto("http://localhost:8501")
    pg.get_by_role("button", name="Mở").first.wait_for(timeout=120000)
    pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="Mở").first.click()
    pg.get_by_role("button", name="Chi tiết").first.wait_for(timeout=60000)
    pg.wait_for_timeout(1500)
