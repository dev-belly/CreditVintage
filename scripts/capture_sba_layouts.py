"""Check real Chrome layouts and capture fully painted report sections."""

from __future__ import annotations

import csv
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from playwright.sync_api import Browser, expect, sync_playwright


class PreviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, message: str, *args: Any) -> None:
        pass


def check_brief(
    browser: Browser, root: Path, destination: Path, locale: str, base_url: str
) -> dict[str, object]:
    """Compare the rendered cards to independently ranked held-out predictions."""
    report_url = base_url + "/sba-brief/" + ("en/" if locale == "en" else "")
    copied_text = "Link copied" if locale == "en" else "链接已复制"
    fallback_text = "address-bar" if locale == "en" else "请复制地址栏"
    with (root / "_site" / "sba-credit" / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        predictions = list(csv.DictReader(handle))
    events = sum(int(row["chargeoff_24m"]) for row in predictions)
    expected = {}
    for method in ("constant", "logistic", "isotonic"):
        ranking = sorted(predictions, key=lambda row: (-float(row[method]), row["record_id"]))
        for capacity in (5, 10, 20):
            reviewed = len(predictions) * capacity // 100
            captured = sum(int(row["chargeoff_24m"]) for row in ranking[:reviewed])
            expected[method, capacity] = (reviewed, captured)
    errors: list[str] = []
    context = browser.new_context()
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    checked = 0
    observed_widths = []
    for width, height, label in ((1440, 1100, "desktop"), (390, 1500, "mobile")):
        page.set_viewport_size({"width": width, "height": height})
        page.goto(report_url, wait_until="load")
        expect(page.locator("html")).to_have_attribute("lang", locale)
        observed_widths.append(page.evaluate("window.innerWidth"))
        if observed_widths[-1] != width:
            raise ValueError("The actual brief viewport differs from the requested size")
        for capacity in (5, 10, 20):
            page.locator(f'button[data-capacity="{capacity}"]').click()
            expect(page.locator(f'button[data-capacity="{capacity}"]')).to_have_attribute(
                "aria-pressed", "true"
            )
            for method in ("constant", "logistic", "isotonic"):
                reviewed, captured = expected[method, capacity]
                card = page.locator(f'article[data-method="{method}"]')
                expect(card.locator(".captured")).to_have_text(str(captured))
                expect(card.locator(".precision")).to_have_text(f"{captured / reviewed:.2%}")
                expect(card.locator(".capture")).to_have_text(f"{captured / events:.2%}")
                expect(card.locator(".outside")).to_contain_text(str(events - captured))
                if card.locator("meter").evaluate("meter => meter.value") != captured:
                    raise ValueError("The brief's event meter differs from its count")
                checked += 1
            expect(page.locator("#selection-status")).to_contain_text(f"{reviewed:,}")
            expect(page.locator("#denominator-explanation")).to_contain_text(f"/ {events}")
            if page.evaluate("location.hash") != f"#capacity={capacity}":
                raise ValueError("The selected capacity is absent from the scenario URL")
            expect(page.locator("[data-language-link]")).to_have_attribute(
                "href", re.compile(f"#capacity={capacity}$")
            )
        page.locator('button[data-capacity="10"]').focus()
        page.keyboard.press("Space")
        expect(page.locator('article[data-method="logistic"] .captured')).to_have_text("26")
        page.locator("#all-comparisons summary").click()
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            raise ValueError("The interactive brief overflows the viewport")
        page.locator("#all-comparisons summary").click()
        page.screenshot(path=str(destination / f"brief-{locale}-{label}.png"), full_page=True)
    # Direct links restore a comparison, including subsequent hash navigation.
    page.goto(report_url + "#capacity=20", wait_until="load")
    expect(page.locator('button[data-capacity="20"]')).to_have_attribute("aria-pressed", "true")
    page.evaluate("location.hash = 'capacity=5'")
    expect(page.locator('button[data-capacity="5"]')).to_have_attribute("aria-pressed", "true")
    # Actual HTTP navigation checks both relative language links and retained state.
    for language in ("zh-CN", "en") if locale == "en" else ("en", "zh-CN"):
        page.locator("[data-language-link]").click()
        expect(page.locator("html")).to_have_attribute("lang", language)
        expect(page.locator('button[data-capacity="5"]')).to_have_attribute("aria-pressed", "true")
    page.goto(report_url + "#capacity=999", wait_until="load")
    expect(page.locator('button[data-capacity="10"]')).to_have_attribute("aria-pressed", "true")
    # Test both clipboard paths without writing to a runner's actual clipboard.
    page.evaluate(
        "Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{"
        "writeText:async text => {window.copiedScenario = text}}})"
    )
    page.locator("#copy-scenario").click()
    expect(page.locator("#share-status")).to_contain_text(copied_text)
    if not page.evaluate("window.copiedScenario.endsWith('#capacity=10')"):
        raise ValueError("The copied scenario has the wrong capacity")
    page.evaluate(
        "Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{"
        "writeText:async () => {throw new Error('clipboard unavailable')}}})"
    )
    page.locator("#copy-scenario").click()
    expect(page.locator("#share-status")).to_contain_text(fallback_text)
    # A slow copy of an older capacity must not announce success for a newer view.
    page.evaluate(
        "Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{"
        "writeText:() => new Promise(resolve => {window.finishCopy = resolve})}})"
    )
    page.locator("#copy-scenario").click()
    page.locator('button[data-capacity="5"]').click()
    page.evaluate("window.finishCopy()")
    expect(page.locator("#share-status")).to_have_text("")
    if errors:
        raise ValueError(f"Brief JavaScript errors: {errors}")
    context.close()
    plain = browser.new_context(java_script_enabled=False, viewport={"width": 390, "height": 1500})
    page = plain.new_page()
    page.goto(report_url, wait_until="load")
    expect(page.locator('button[data-capacity="10"]')).to_be_disabled()
    expect(page.locator('article[data-method="logistic"] .captured')).to_have_text("26")
    page.locator("#all-comparisons summary").click()
    rows = page.locator("#all-comparisons tbody tr")
    if rows.count() != 9:
        raise ValueError("The no-script table must retain every fixed comparison")
    for index, (method, capacity) in enumerate(expected):
        reviewed, captured = expected[method, capacity]
        cells = rows.nth(index).locator("td")
        expect(cells.nth(2)).to_have_text(f"{reviewed:,}")
        expect(cells.nth(3)).to_have_text(f"{captured} / {events}")
        expect(cells.nth(4)).to_have_text(f"{captured / reviewed:.2%}")
        expect(cells.nth(5)).to_have_text(f"{captured / events:.2%}")
    plain.close()
    return {
        "language": locale,
        "card_selections": checked,
        "observed_viewport_widths": observed_widths,
        "copied_link_and_fallback": True,
        "stale_clipboard_result_ignored": True,
        "without_javascript_comparisons": 9,
        "language_round_trip_preserves_capacity": True,
        "javascript_errors": errors,
    }


def capture() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    report = root / "_site" / "sba-credit" / "index.html"
    destination = root / "sba-layout"
    destination.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(report.as_uri(), wait_until="load")
        page.screenshot(path=str(destination / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 2200})
        page.goto(report.as_uri(), wait_until="load")
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            raise ValueError("SBA mobile page overflows the viewport")
        if page.locator("#models tbody tr").count() != 3:
            raise ValueError("SBA model rows are missing")
        cells = page.locator("#models td")
        if cells.count() != 9 or not cells.evaluate_all(
            "cells => cells.every(cell => getComputedStyle(cell).display === 'grid' "
            "&& cell.getBoundingClientRect().width < 390)"
        ):
            raise ValueError("Mobile model metrics must all fit without horizontal scrolling")
        page.screenshot(path=str(destination / "mobile-top.png"))
        # Element captures wait for scroll/paint; fragment URLs in CLI captures
        # produced blank deep sections despite an otherwise successful build.
        for section in ("models", "evidence"):
            page.locator(f"#{section}").screenshot(
                path=str(destination / f"mobile-{section}.png"), animations="disabled"
            )
        handler = partial(PreviewHandler, directory=str(root / "_site"))
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                brief = {
                    locale: check_brief(browser, root, destination, locale, base_url)
                    for locale in ("zh-CN", "en")
                }
            finally:
                server.shutdown()
                thread.join()
        browser.close()
    return {
        "status": "verified",
        "browser": "Google Chrome",
        "mobile_width": 390,
        "models": 3,
        "metric_cells": 9,
        "page_overflow": False,
        "screenshots": 8,
        "brief": brief,
    }


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
