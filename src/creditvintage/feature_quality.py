"""Decision-time feature coverage and visible delay, with portable event replay."""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import (
    FEATURE_NAMES,
    Application,
    DataContractError,
    FeatureEvent,
    _decision_feature,
    _index_features,
)
from creditvintage.feature_inputs import decode_inputs, encode_inputs
from creditvintage.model import Config, _digest
from creditvintage.validation import read_json_object

SCHEMA = "creditvintage-feature-quality-v1"
STATUSES = ("available", "late_for_decision", "observed_after_decision", "no_visible_event")
AVAILABILITY_COLUMNS = (
    "population",
    "application_id",
    "decision_at",
    "vintage",
    "feature",
    "status",
    "value",
    "observed_at",
    "available_at",
    "observation_age_days",
    "availability_lag_days",
    "late_days",
)
COVERAGE_COLUMNS = (
    "population",
    "vintage",
    "feature",
    "applications",
    *STATUSES,
    "coverage_rate",
    "median_observation_age_days",
    "median_availability_lag_days",
    "max_availability_lag_days",
    "median_late_days",
    "max_late_days",
)
QUALITY_FILES = ("inputs.json", "quality.json", "availability.csv", "coverage.csv", "index.html")


@dataclass(frozen=True)
class FeatureQualityResult:
    inputs: dict[str, Any]
    summary: dict[str, Any]
    availability: list[dict[str, Any]]
    coverage: list[dict[str, Any]]


def _availability(
    app: Application,
    name: str,
    events: list[FeatureEvent],
    as_of: date,
    population: str,
) -> dict[str, Any]:
    visible = [event for event in events if event.available_at <= as_of]
    selected = _decision_feature(visible, app.decision_at)
    status = "available"
    if selected is None:
        before_decision = [event for event in visible if event.observed_at <= app.decision_at]
        if before_decision:
            status = "late_for_decision"
            selected = min(before_decision, key=lambda event: event.available_at)
        elif visible:
            status = "observed_after_decision"
            selected = min(visible, key=lambda event: event.available_at)
        else:
            status = "no_visible_event"
    return {
        "population": population,
        "application_id": app.application_id,
        "decision_at": app.decision_at.isoformat(),
        "vintage": app.decision_at.strftime("%Y-%m"),
        "feature": name,
        "status": status,
        "value": selected.value if selected else None,
        "observed_at": selected.observed_at.isoformat() if selected else None,
        "available_at": selected.available_at.isoformat() if selected else None,
        "observation_age_days": (app.decision_at - selected.observed_at).days
        if selected and status == "available"
        else None,
        "availability_lag_days": (selected.available_at - selected.observed_at).days
        if selected
        else None,
        "late_days": (selected.available_at - app.decision_at).days
        if selected and status == "late_for_decision"
        else None,
    }


def _coverage(rows: list[dict[str, Any]], population: str, vintage: str) -> list[dict[str, Any]]:
    result = []
    for name in FEATURE_NAMES:
        selected = [row for row in rows if row["feature"] == name]
        counts = Counter(row["status"] for row in selected)
        on_time = [row for row in selected if row["status"] == "available"]
        ages = [row["observation_age_days"] for row in on_time]
        lags = [row["availability_lag_days"] for row in on_time]
        late = [row["late_days"] for row in selected if row["status"] == "late_for_decision"]
        result.append(
            {
                "population": population,
                "vintage": vintage,
                "feature": name,
                "applications": len(selected),
                **{status: counts[status] for status in STATUSES},
                "coverage_rate": round(counts["available"] / len(selected), 8)
                if selected
                else None,
                "median_observation_age_days": median(ages) if ages else None,
                "median_availability_lag_days": median(lags) if lags else None,
                "max_availability_lag_days": max(lags) if lags else None,
                "median_late_days": median(late) if late else None,
                "max_late_days": max(late) if late else None,
            }
        )
    return result


def monitor_feature_quality(
    applications: list[Application],
    features: list[FeatureEvent],
    as_of: date,
    config: Config | None = None,
    *,
    data_kind: str = "user_supplied",
) -> FeatureQualityResult:
    """Keep every expected feature slot; diagnose absence without changing strict scoring."""
    config = config or Config()
    if type(as_of) is not date or as_of < config.test_start:
        raise DataContractError("Feature-quality cutoff must be on or after the test window starts")
    if data_kind not in {"synthetic", "user_supplied"}:
        raise DataContractError("Unknown data kind")
    _, index = _index_features(applications, features)
    availability, coverage = [], []
    populations: dict[str, Any] = {}
    for population, start, end in (
        ("reference", config.calibration_start, config.calibration_end),
        ("current", config.test_start, config.test_end),
    ):
        window = [app for app in applications if start <= app.decision_at <= end]
        visible = sorted(
            (app for app in window if app.decision_at <= as_of),
            key=lambda app: (app.decision_at, app.application_id),
        )
        rows = [
            _availability(app, name, index.get((app.application_id, name), []), as_of, population)
            for app in visible
            for name in FEATURE_NAMES
        ]
        incomplete = {row["application_id"] for row in rows if row["status"] != "available"}
        populations[population] = {
            "applications": len(visible),
            "complete_applications": len(visible) - len(incomplete),
            "incomplete_applications": len(incomplete),
            "applications_not_yet_visible": len(window) - len(visible),
            "status": "no_visible_applications"
            if not visible
            else "incomplete_coverage"
            if incomplete
            else "complete_coverage",
        }
        coverage.extend(_coverage(rows, population, "all"))
        for vintage in sorted({row["vintage"] for row in rows}):
            coverage.extend(
                _coverage(
                    [row for row in rows if row["vintage"] == vintage],
                    population,
                    vintage,
                )
            )
        availability.extend(rows)
    summary = {
        "schema": SCHEMA,
        "kind": "decision_time_feature_quality",
        "data_kind": data_kind,
        "as_of": as_of.isoformat(),
        "model_fitted": False,
        "outcomes_required": False,
        "input_sha256": _digest(applications, features, []),
        "reference_window": [
            config.calibration_start.isoformat(),
            config.calibration_end.isoformat(),
        ],
        "current_window": [config.test_start.isoformat(), config.test_end.isoformat()],
        "populations": populations,
        "applications_outside_windows": len(applications)
        - sum(
            item["applications"] + item["applications_not_yet_visible"]
            for item in populations.values()
        ),
        "coverage": [row for row in coverage if row["vintage"] == "all"],
        "by_vintage": [row for row in coverage if row["vintage"] != "all"],
        "scope": (
            "Expected feature slots, coverage and visible timing; "
            "no imputation, model fitting or risk conclusion"
        ),
    }
    source = encode_inputs(
        applications, features, as_of, config, schema=SCHEMA, data_kind=data_kind
    )
    return FeatureQualityResult(source, summary, availability, coverage)


def _render(result: FeatureQualityResult) -> str:
    summary = result.summary
    current = summary["populations"]["current"]

    def available_count(row: dict[str, Any]) -> str:
        return (
            f"{row['available']}/{row['applications']}"
            if row["applications"]
            else "No applications"
        )

    rows = "\n".join(
        f"<tr><th>{row['population']}</th><th>{row['feature']}</th>"
        f"<td>{available_count(row)}</td>"
        f"<td>{row['late_for_decision']}</td><td>{row['observed_after_decision']}</td>"
        f"<td>{row['no_visible_event']}</td></tr>"
        for row in summary["coverage"]
    )
    monthly = "\n".join(
        f"<tr><th>{row['vintage']}</th><th>{row['feature']}</th>"
        f"<td>{row['available']}/{row['applications']}</td><td>{row['late_for_decision']}</td>"
        f"<td>{row['observed_after_decision']}</td><td>{row['no_visible_event']}</td></tr>"
        for row in summary["by_vintage"]
        if row["population"] == "current"
    )
    labels = {
        "late_for_decision": "Arrived after decision",
        "observed_after_decision": "Observed after decision",
        "no_visible_event": "No visible event",
    }
    missing = [row for row in result.availability if row["status"] != "available"]
    detail = (
        "\n".join(
            f"<tr><th>{html.escape(row['application_id'])}</th><td>{row['feature']}</td>"
            f"<td>{labels[row['status']]}</td><td>{row['decision_at']}</td>"
            f"<td>{row['observed_at'] or '—'}</td><td>{row['available_at'] or '—'}</td></tr>"
            for row in missing[:50]
        )
        or '<tr><td colspan="6">No unavailable feature slots in these populations.</td></tr>'
    )

    def days(value: Any) -> str:
        return str(value) if value is not None else "—"

    timing = "\n".join(
        f"<tr><th>{row['feature']}</th><td>{row['available']}</td>"
        f"<td>{days(row['median_availability_lag_days'])}</td>"
        f"<td>{days(row['max_availability_lag_days'])}</td><td>{days(row['median_late_days'])}</td>"
        f"<td>{days(row['max_late_days'])}</td></tr>"
        for row in summary["coverage"]
        if row["population"] == "current"
    )
    template = files("creditvintage").joinpath("feature_quality.html").read_text(encoding="utf-8")
    replacements = {
        "@@SOURCE@@": "Synthetic example"
        if summary["data_kind"] == "synthetic"
        else "User-supplied data",
        "@@AS_OF@@": summary["as_of"],
        "@@CURRENT@@": str(current["applications"]),
        "@@COMPLETE@@": str(current["complete_applications"]),
        "@@INCOMPLETE@@": str(current["incomplete_applications"]),
        "@@FUTURE@@": str(current["applications_not_yet_visible"]),
        "@@OVERVIEW@@": rows,
        "@@MONTHLY@@": monthly,
        "@@TIMING@@": timing,
        "@@DETAIL@@": detail,
        "@@SHOWN@@": str(min(50, len(missing))),
        "@@MISSING@@": str(len(missing)),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _write(result: FeatureQualityResult, destination: Path) -> dict[str, Any]:
    _json(destination / "inputs.json", result.inputs)
    _json(destination / "quality.json", result.summary)
    _csv(destination / "availability.csv", result.availability, AVAILABILITY_COLUMNS)
    _csv(destination / "coverage.csv", result.coverage, COVERAGE_COLUMNS)
    (destination / "index.html").write_text(_render(result), encoding="utf-8", newline="")
    manifest = {
        "schema": SCHEMA,
        "input_sha256": result.summary["input_sha256"],
        "files_sha256": {name: _hash(destination / name) for name in QUALITY_FILES},
    }
    _json(destination / "manifest.json", manifest)
    return manifest


def write_feature_quality(result: FeatureQualityResult, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise DataContractError(
            "Feature-quality destination already exists; choose a new directory"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".feature-quality-") as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        manifest = _write(result, staged)
        if destination.exists():
            raise DataContractError("Feature-quality destination already exists")
        staged.rename(destination)
    return manifest


def verify_feature_quality(destination: Path) -> dict[str, Any]:
    try:
        source = read_json_object(destination / "inputs.json")
        applications, features, as_of, config, data_kind = decode_inputs(source, schema=SCHEMA)
        result = monitor_feature_quality(applications, features, as_of, config, data_kind=data_kind)
        manifest = read_json_object(destination / "manifest.json")
        with TemporaryDirectory(prefix="creditvintage-quality-replay-") as temporary:
            expected = Path(temporary)
            if manifest != _write(result, expected):
                raise DataContractError("Feature-quality manifest disagrees with event replay")
            for name in QUALITY_FILES:
                if (destination / name).read_bytes() != (expected / name).read_bytes():
                    raise DataContractError(
                        f"Feature-quality artifact disagrees with event replay: {name}"
                    )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify feature quality: {exc}") from exc
    return {
        "status": "verified",
        "populations": result.summary["populations"],
        "feature_slots": len(result.availability),
        "model_fitted": False,
        "input_sha256": result.summary["input_sha256"],
    }
