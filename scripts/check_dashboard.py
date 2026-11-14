"""TC-18: drive the real dashboard in Edge (Playwright, channel=msedge) and save screenshots.

Usage: python tc18_browser.py <base_url> <out_dir>
Prints one line per check; exits 1 if any check fails.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE, OUT = sys.argv[1].rstrip("/"), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, locale="zh-TW")
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(BASE + "/")
    page.wait_for_selector("#go:not([disabled])", timeout=15000)
    check("dropdowns load from /series", page.locator("#store option").count() >= 1,
          f"stores={page.locator('#store option').count()} items={page.locator('#item option').count()}")
    page.wait_for_selector("#compare table", timeout=15000)
    cmp_text = page.inner_text("#compare")
    check("compare table shows all 6 strategies",
          all(n in cmp_text for n in ["季節性", "28 天移動平均", "TSB", "LightGBM", "分位數", "完美預測"]))
    check("compare shows bootstrap CI table", "bootstrap" in cmp_text and "95%" in cmp_text)
    check("synthetic data is labelled", "合成資料" in page.inner_text("body"))

    # main flow: pick the 2nd item, change inputs, query
    items = page.locator("#item option")
    if items.count() > 1:
        page.select_option("#item", index=1)
    item = page.input_value("#item")
    page.fill("#on_hand", "3")
    page.fill("#lead", "10")
    page.fill("#sl", "0.97")
    page.fill("#moq", "12")
    page.click("#go")
    page.wait_for_selector("#chart svg", timeout=15000)
    page.wait_for_function("!document.querySelector('#rec .state')", timeout=15000)
    rec = page.inner_text("#rec")
    check("forecast chart rendered", page.locator("#chart svg").count() == 1, f"item={item}")
    check("replenishment advice rendered", any(ch.isdigit() for ch in rec), rec.replace("\n", " | ")[:160])
    page.screenshot(path=str(OUT / "tc18-desktop.png"), full_page=True)

    # boundary: lead time 40 must be rejected in the page, with a field message
    page.fill("#lead", "40")
    page.click("#go")
    page.wait_for_timeout(500)
    err = page.locator("#lead ~ .field-err").inner_text()
    check("invalid lead time shows field error", bool(err.strip()), err)
    page.screenshot(path=str(OUT / "tc18-invalid-input.png"), full_page=False)

    # reload keeps working
    page.reload()
    page.wait_for_selector("#go:not([disabled])", timeout=15000)
    page.wait_for_selector("#compare table", timeout=15000)
    check("page works after reload", True)

    # phone width: no horizontal page scroll
    mob = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, locale="zh-TW").new_page()
    mob.on("pageerror", lambda e: errors.append(str(e)))
    mob.goto(BASE + "/")
    mob.wait_for_selector("#go:not([disabled])", timeout=15000)
    mob.click("#go")
    mob.wait_for_selector("#chart svg", timeout=15000)
    mob.wait_for_selector("#compare table", timeout=15000)
    sw, cw = mob.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
    check("no horizontal scroll at 390px", sw <= cw, f"scrollWidth={sw} clientWidth={cw}")
    mob.screenshot(path=str(OUT / "tc18-mobile.png"), full_page=True)

    check("no console / page errors", not errors, "; ".join(errors)[:300])

    # LLM assistant box. (a) whatever the real server says (503 without a key) must render, not crash
    page.goto(BASE + "/")
    page.wait_for_selector("#go:not([disabled])", timeout=15000)
    page.fill("#chat-q", "哪個預測方法總成本最低？")
    page.click("#chat-go")
    page.wait_for_function("!document.querySelector('#chat-out .state')", timeout=120000)
    out = page.inner_text("#chat-out")
    real_ok = page.locator("#chat-out .answer").count() == 1
    check("chat: real /chat renders an answer or a clear error",
          real_ok or ("503" in out and "LLM 助理目前無法使用" in out), out.replace("\n", " ")[:120])
    if not real_ok:
        page.screenshot(path=str(OUT / "chat-no-key.png"), full_page=False, clip=None)
    # (b) a canned 200 reply (route-mocked) renders answer + tool list
    canned = {"answer": "建議訂 21 件。\n（目前是合成資料）", "served_by": "claude-opus-5-5",
              "data_source": "synthetic", "data_note": "合成資料",
              "tool_calls": [{"name": "compute_replenishment", "input": {}, "ok": True}]}
    page.route("**/chat", lambda r: r.fulfill(status=200, content_type="application/json",
                                              body=__import__("json").dumps(canned, ensure_ascii=False)))
    page.locator("#chat-examples button").first.click()
    page.wait_for_selector("#chat-out .answer", timeout=15000)
    check("chat: example chip asks and renders answer + tools",
          "建議訂 21 件" in page.inner_text("#chat-out .answer") and "算補貨" in page.inner_text("#chat-out .tools"))
    page.locator("#chat-out").scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "chat-mocked.png"), full_page=True)
    browser.close()

print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
