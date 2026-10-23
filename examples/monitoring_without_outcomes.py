"""Run a synthetic monitoring case with all current-cohort outcomes removed.

Install CreditVintage first, then run this script from any working directory.
The new output folder contains editable inputs, preflight JSON and a verified
monitoring report. No private data or optional integration is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from creditvintage.configuration import load_config
from creditvintage.core import load_csv_bundle
from creditvintage.inputs import initialize_inputs
from creditvintage.monitoring import monitor, preflight_monitor, verify_monitor, write_monitor


def run(directory: Path) -> dict[str, object]:
    if directory.exists():
        raise ValueError("Choose a new output directory; existing inputs are never overwritten")
    inputs = directory / "inputs"
    initialize_inputs(inputs)
    config = load_config(inputs / "cohorts.json")
    applications, features, performance = load_csv_bundle(
        inputs / "applications.csv", inputs / "features.csv", inputs / "performance.csv"
    )
    current_ids = {
        app.application_id
        for app in applications
        if config.test_start <= app.decision_at <= config.test_end
    }
    path = inputs / "performance.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, reports = reader.fieldnames, list(reader)
    assert columns is not None
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in reports if row["application_id"] not in current_ids)
    performance = [row for row in performance if row.application_id not in current_ids]
    cutoff = date(2024, 10, 1)
    checked = preflight_monitor(
        applications, features, performance, cutoff, config, data_kind="synthetic"
    )
    (directory / "preflight.json").write_text(
        json.dumps(checked, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    write_monitor(
        monitor(applications, features, performance, cutoff, config, data_kind="synthetic"),
        directory / "report",
    )
    verified = verify_monitor(directory / "report")
    return {
        "data_kind": "synthetic",
        "as_of": cutoff.isoformat(),
        "current_outcome_rows": 0,
        "preflight": checked,
        "verification": verified,
        "report": str(directory / "report" / "index.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    try:
        result = run(args.out)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Monitoring example: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
