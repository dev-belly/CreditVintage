"""Check real Chrome layouts and capture fully painted report sections."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


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
        browser.close()
    return {
        "status": "verified",
        "browser": "Google Chrome",
        "mobile_width": 390,
        "models": 3,
        "metric_cells": 9,
        "page_overflow": False,
        "screenshots": 4,
    }


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
