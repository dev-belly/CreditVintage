"""Publish the existing reports and a freshly verified synthetic lineage bundle."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from creditvintage.artifacts import verify_result
from creditvintage.benchmark import benchmark, verify_benchmark, write_benchmark
from creditvintage.business import verify_analysis, write_analysis
from creditvintage.card_credit import verify_card_benchmark, write_card_benchmark
from creditvintage.feature_monitoring import (
    monitor_features,
    verify_feature_monitor,
    write_feature_monitor,
)
from creditvintage.lineage import verify_lineage, write_demo
from creditvintage.monitoring import verify_monitor
from creditvintage.public_credit import verify_public_benchmark, write_public_benchmark
from creditvintage.sample import generate

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Choose an empty build destination: {destination}")
    shutil.copytree(ROOT / "docs", destination)
    (destination / ".nojekyll").touch()
    verify_result(destination / "demo")
    verify_monitor(destination / "monitor")
    applications, features, _ = generate()
    write_feature_monitor(
        monitor_features(applications, features, date(2024, 10, 1), data_kind="synthetic"),
        destination / "feature-monitor",
    )
    feature_monitor = verify_feature_monitor(destination / "feature-monitor")
    write_analysis(destination / "demo", destination / "analysis")
    analysis = verify_analysis(destination / "analysis")
    summary = write_demo(destination / "lineage")
    checked = verify_lineage(destination / "lineage")
    write_benchmark(benchmark(*generate(), data_kind="synthetic"), destination / "benchmark")
    comparison = verify_benchmark(destination / "benchmark")
    write_public_benchmark(destination / "public-credit")
    public_credit = verify_public_benchmark(destination / "public-credit")
    write_card_benchmark(destination / "card-credit")
    card_credit = verify_card_benchmark(destination / "card-credit")
    if not (destination / "index.html").is_file():
        raise ValueError("Site entry page is missing")
    return {
        "summary": summary,
        "verification": checked,
        "analysis": analysis,
        "benchmark": comparison,
        "public_credit": public_credit,
        "card_credit": card_credit,
        "feature_monitor": feature_monitor,
    }


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "_site"), indent=2))
