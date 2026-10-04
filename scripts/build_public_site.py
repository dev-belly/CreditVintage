"""Publish the existing reports and a freshly verified synthetic lineage bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from creditvintage.artifacts import verify_result
from creditvintage.lineage import BUNDLE_FILES, verify_lineage, write_demo
from creditvintage.monitoring import verify_monitor

ROOT = Path(__file__).resolve().parents[1]


def archive_evidence(directory: Path) -> None:
    """Make a deterministic archive containing only the verifier's fixed inventory."""
    names = sorted((*BUNDLE_FILES, "manifest.json"))
    with ZipFile(directory / "evidence.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            item = ZipInfo(f"lineage/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, (directory / name).read_bytes())
    with ZipFile(directory / "evidence.zip") as archive:
        if archive.namelist() != [f"lineage/{name}" for name in names]:
            raise ValueError("Evidence archive inventory mismatch")
        for name in names:
            if archive.read(f"lineage/{name}") != (directory / name).read_bytes():
                raise ValueError(f"Evidence archive bytes mismatch: {name}")


def build(destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Choose an empty build destination: {destination}")
    shutil.copytree(ROOT / "docs", destination)
    (destination / ".nojekyll").touch()
    verify_result(destination / "demo")
    verify_monitor(destination / "monitor")
    summary = write_demo(destination / "lineage")
    checked = verify_lineage(destination / "lineage")
    archive_evidence(destination / "lineage")
    if not (destination / "index.html").is_file():
        raise ValueError("Site entry page is missing")
    return {"summary": summary, "verification": checked}


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "_site"), indent=2))
