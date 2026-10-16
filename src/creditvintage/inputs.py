"""Create an editable CSV example and diagnose inputs without fitting a model."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.configuration import load_config, preflight
from creditvintage.core import (
    FEATURE_NAMES,
    Application,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    snapshot,
)
from creditvintage.model import Config, _rows_between
from creditvintage.sample import generate

INPUT_FILES = ("applications.csv", "features.csv", "performance.csv", "cohorts.json")
_DATES = {"decision_at", "observed_at", "available_at", "as_of"}
_INTEGERS = {"principal_cents", "days_past_due"}


def _write_records[R: (Application, FeatureEvent, PerformanceEvent)](
    destination: Path, filename: str, records: list[R]
) -> None:
    _csv(
        destination / filename,
        [asdict(record) for record in records],
        tuple(item.name for item in fields(records[0])),
    )


def initialize_inputs(destination: Path) -> dict[str, Any]:
    """Write a complete synthetic example; existing destinations are never reused."""
    if destination.exists():
        raise DataContractError("Choose a new input directory; existing files are not overwritten")
    config = Config()
    applications, features, performance = generate(config.seed, 20)
    checked = preflight(applications, features, performance, config, data_kind="synthetic")
    destination.mkdir(parents=True, exist_ok=False)
    _write_records(destination, "applications.csv", applications)
    _write_records(destination, "features.csv", features)
    _write_records(destination, "performance.csv", performance)
    _json(destination / "cohorts.json", checked["config"])
    manifest = {
        "schema": "creditvintage-input-example-v1",
        "data_kind": "synthetic",
        "description": "Generated example records, not observed bank or customer data",
        "seed": config.seed,
        "input_sha256": checked["input_sha256"],
        "input_rows": checked["input_rows"],
        "files_sha256": {name: _hash(destination / name) for name in INPUT_FILES},
    }
    _json(destination / "case.json", manifest)
    return {"status": "created", "out": str(destination), **manifest, "model_fitted": False}


@dataclass
class _Issues:
    limit: int
    total: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        filename: str,
        line: int | None,
        code: str,
        message: str,
        *,
        column: str | None = None,
        stage: str | None = None,
    ) -> None:
        self.total += 1
        if len(self.rows) < self.limit:
            self.rows.append(
                {
                    "file": filename,
                    "line": line,
                    "column": column,
                    "stage": stage,
                    "code": code,
                    "message": message,
                }
            )


def _read_file[R: (Application, FeatureEvent, PerformanceEvent)](
    directory: Path, filename: str, record_type: type[R], issues: _Issues
) -> tuple[list[tuple[int, R]], dict[str, int]]:
    columns = tuple(item.name for item in fields(record_type))
    records: list[tuple[int, R]] = []
    count = 0
    line = 1
    try:
        with (directory / filename).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames and reader.fieldnames[0].startswith("\ufeff"):
                issues.add(
                    filename, 1, "encoding_bom", "Save the CSV as UTF-8 without a byte-order mark"
                )
                return records, {"rows": 0, "valid_rows": 0}
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(columns)
                or set(reader.fieldnames) != set(columns)
            ):
                issues.add(filename, 1, "columns", "Expected exactly: " + ", ".join(columns))
                return records, {"rows": 0, "valid_rows": 0}
            try:
                for row in reader:
                    line = reader.line_num
                    count += 1
                    if set(row) != set(columns) or any(value is None for value in row.values()):
                        issues.add(filename, line, "row_shape", "Row does not match the CSV header")
                        continue
                    values: dict[str, Any] = {}
                    before = issues.total
                    for name in columns:
                        try:
                            if name in _DATES:
                                values[name] = date.fromisoformat(row[name])
                            elif name in _INTEGERS:
                                values[name] = int(row[name])
                            elif name == "value":
                                values[name] = float(row[name])
                            else:
                                values[name] = row[name]
                        except ValueError:
                            expected = (
                                "a calendar date"
                                if name in _DATES
                                else "an integer"
                                if name in _INTEGERS
                                else "a numeric value"
                            )
                            issues.add(
                                filename, line, "value_type", f"Expected {expected}", column=name
                            )
                    if issues.total != before:
                        continue
                    try:
                        records.append((line, record_type(**values)))
                    except DataContractError as exc:
                        issues.add(filename, line, "record_contract", str(exc))
            except csv.Error:
                line = reader.reader.line_num
                issues.add(filename, line, "csv_quoting", "Malformed CSV quoting")
    except UnicodeError:
        issues.add(filename, None, "encoding", "Expected UTF-8 CSV text")
    except OSError:
        issues.add(filename, None, "unreadable_file", "File is missing or cannot be read")
    return records, {"rows": count, "valid_rows": len(records)}


def _relationships(
    applications: list[tuple[int, Application]],
    features: list[tuple[int, FeatureEvent]],
    performance: list[tuple[int, PerformanceEvent]],
    config: Config | None,
    issues: _Issues,
) -> None:
    by_id: dict[str, tuple[int, Application]] = {}
    for line, app in applications:
        if app.application_id in by_id:
            issues.add(
                "applications.csv", line, "duplicate_application", "Duplicate application ID"
            )
        else:
            by_id[app.application_id] = (line, app)
    feature_keys: dict[tuple[str, str, date], int] = {}
    available_features: dict[str, set[str]] = defaultdict(set)
    for line, event in features:
        source = by_id.get(event.application_id)
        if source is None:
            issues.add("features.csv", line, "unknown_application", "No matching application row")
        elif event.available_at <= source[1].decision_at:
            available_features[event.application_id].add(event.name)
        key = (event.application_id, event.name, event.available_at)
        if key in feature_keys:
            issues.add(
                "features.csv", line, "ambiguous_feature", "Duplicate same-day feature revision"
            )
        feature_keys[key] = line
    report_keys: set[tuple[str, date, date]] = set()
    for line, report in performance:
        source = by_id.get(report.application_id)
        if source is None:
            issues.add(
                "performance.csv", line, "unknown_application", "No matching application row"
            )
        elif report.as_of <= source[1].decision_at:
            issues.add(
                "performance.csv", line, "performance_date", "Report must follow the decision"
            )
        report_key = (report.application_id, report.as_of, report.available_at)
        if report_key in report_keys:
            issues.add(
                "performance.csv", line, "duplicate_performance", "Duplicate performance revision"
            )
        report_keys.add(report_key)
    if config is not None:
        for line, app in applications:
            if app.decision_at <= config.evaluation_as_of:
                for name in FEATURE_NAMES:
                    if name not in available_features[app.application_id]:
                        issues.add(
                            "applications.csv",
                            line,
                            "missing_decision_feature",
                            f"No {name} value available by the decision date",
                        )


def check_inputs(directory: Path, *, max_issues: int = 100) -> dict[str, Any]:
    """Collect CSV failures, then check relationships and actual evaluation gates."""
    if type(max_issues) is not int or not 1 <= max_issues <= 1000:
        raise DataContractError("max_issues must be an integer from 1 to 1000")
    issues = _Issues(max_issues)
    config = None
    try:
        config = load_config(directory / "cohorts.json")
    except (DataContractError, OSError, ValueError) as exc:
        issues.add("cohorts.json", None, "config", str(exc))
    before_rows = issues.total
    applications, app_counts = _read_file(directory, "applications.csv", Application, issues)
    features, feature_counts = _read_file(directory, "features.csv", FeatureEvent, issues)
    performance, report_counts = _read_file(directory, "performance.csv", PerformanceEvent, issues)
    row_errors = issues.total != before_rows
    checks = {
        "configuration": "passed" if config is not None else "failed",
        "csv_rows": "failed" if row_errors else "passed",
        "relationships": "blocked",
        "evaluation_cohorts": "blocked",
    }
    verified = None
    if not row_errors:
        before_relationships = issues.total
        _relationships(applications, features, performance, config, issues)
        checks["relationships"] = "failed" if issues.total != before_relationships else "passed"
        if config is None and checks["relationships"] == "passed":
            checks["relationships"] = "blocked"
        if issues.total == 0 and config is not None:
            apps = [record for _, record in applications]
            feats = [record for _, record in features]
            reports = [record for _, record in performance]
            locations = {app.application_id: line for line, app in applications}
            for stage, start, end, cutoff in (
                ("train", config.train_start, config.train_end, config.calibration_start),
                (
                    "calibration",
                    config.calibration_start,
                    config.calibration_end,
                    config.test_start,
                ),
                ("test", config.test_start, config.test_end, config.evaluation_as_of),
            ):
                rows = snapshot(apps, feats, reports, cutoff)
                incomplete = [
                    row
                    for row in rows
                    if start <= row.application.decision_at <= end and row.label is None
                ]
                for row in incomplete:
                    issues.add(
                        "applications.csv",
                        locations[row.application.application_id],
                        row.status,
                        f"{stage} outcome is incomplete at {cutoff.isoformat()}",
                        stage=stage,
                    )
                if not incomplete:
                    try:
                        _rows_between(rows, start, end, stage)
                    except DataContractError as exc:
                        issues.add("cohorts.json", None, "cohort_contract", str(exc), stage=stage)
            checks["evaluation_cohorts"] = "failed" if issues.total else "passed"
            if issues.total == 0:
                verified = preflight(apps, feats, reports, config, data_kind="user_supplied")
    return {
        "schema": "creditvintage-input-diagnostics-v1",
        "status": "invalid_inputs" if issues.total else "ready_for_evaluation",
        "model_fitted": False,
        "input_rows": {
            "applications.csv": app_counts,
            "features.csv": feature_counts,
            "performance.csv": report_counts,
        },
        "checks": checks,
        "issue_count": issues.total,
        "issues_truncated": issues.total > len(issues.rows),
        "issues": issues.rows,
        "preflight": verified,
        "scope": "Documented input and maturity contract; not source authenticity or model quality",
    }
