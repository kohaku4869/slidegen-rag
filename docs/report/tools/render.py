import json, sys
from playwright.sync_api import sync_playwright
d = json.load(open("diagrams.json", encoding="utf-8"))
html = "<html><body style='margin:0;background:white'>" + "".join(
    f"<div id='{k}' style='display:inline-block;padding:16px;background:white'><pre class='mermaid'>{v}</pre></div><br>" for k, v in d.items()
) + """<script type='module'>import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
mermaid.initialize({startOnLoad:false,theme:"neutral",themeVariables:{fontSize:"18px"},securityLevel:'loose',flowchart:{htmlLabels:true,useMaxWidth:false},sequence:{useMaxWidth:false},er:{useMaxWidth:false}});
await mermaid.run(); window.done=true;</script></body></html>"""
open("diagrams.html", "w", encoding="utf-8").write(html)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=4)
    pg.goto("file:///" + __import__("os").path.abspath("diagrams.html").replace("\\", "/"))
    pg.wait_for_function("window.done===true", timeout=60000)
    for k in d:
        pg.locator("#" + k).screenshot(path=f"../assets/diag_{k}.png")
    b.close()
