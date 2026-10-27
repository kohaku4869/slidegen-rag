import sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
wait = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(url)
    try:
        pg.wait_for_selector("[data-testid=stAppViewContainer] h1, [data-testid=stMain] h1, .gradio-container", timeout=60000)
    except Exception as e:
        print("selector timeout", e)
    pg.wait_for_timeout(wait)
    pg.screenshot(path=out, full_page=False)
    b.close()
