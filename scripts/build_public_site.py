"""Publish the existing reports and a freshly verified synthetic lineage bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from creditvintage.artifacts import verify_result
from creditvintage.lineage import verify_lineage, write_demo
from creditvintage.monitoring import verify_monitor

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Choose an empty build destination: {destination}")
    shutil.copytree(ROOT / "docs", destination)
    (destination / ".nojekyll").touch()
    verify_result(destination / "demo")
    verify_monitor(destination / "monitor")
    summary = write_demo(destination / "lineage")
    checked = verify_lineage(destination / "lineage")
    if not (destination / "index.html").is_file():
        raise ValueError("Site entry page is missing")
    return {"summary": summary, "verification": checked}


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "_site"), indent=2))
