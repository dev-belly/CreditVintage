"""Model-free marginal input distributions, replayable from decision-time events."""

from __future__ import annotations

import html
import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
from datetime import date
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.configuration import DATE_FIELDS
from creditvintage.core import (
    FEATURE_NAMES,
    Application,
    CohortRow,
    DataContractError,
    FeatureEvent,
    parse_calendar_date,
    snapshot,
)
from creditvintage.model import Config, _digest
from creditvintage.validation import read_json_object

SCHEMA = "creditvintage-feature-monitor-v1"
NUMERIC_INPUTS = (*FEATURE_NAMES, "principal_cents")
FEATURE_COLUMNS = ("application_id", "decision_at", "segment", *NUMERIC_INPUTS)
DISTRIBUTION_COLUMNS = (
    "feature",
    "count_kind",
    "value",
    "reference_count",
    "current_count",
    "reference_share",
    "current_share",
)
FEATURE_MONITOR_FILES = (
    "inputs.json",
    "monitor.json",
    "reference_features.csv",
    "current_features.csv",
    "distributions.csv",
    "index.html",
)
MINIMUM_POPULATION = 20


@dataclass(frozen=True)
class FeatureMonitorResult:
    inputs: dict[str, Any]
    summary: dict[str, Any]
    reference: list[dict[str, Any]]
    current: list[dict[str, Any]]
    distributions: list[dict[str, Any]]


def _numeric_comparison(
    name: str, reference: list[int | float], current: list[int | float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exact cumulative counts; no quantile bins, smoothing, p-values or thresholds."""
    try:
        if any(not math.isfinite(value) for value in [*reference, *current]):
            raise DataContractError(f"Feature monitoring requires finite {name}")
    except OverflowError as exc:
        raise DataContractError(f"Feature monitoring requires finite {name}") from exc
    expected, actual = sorted(reference), sorted(current)
    denominator = len(expected) * len(actual)
    largest, location = -1, expected[0]
    at_reference, at_current = 0, 0
    distribution = []
    for value in sorted(set(expected) | set(actual)):
        reference_count = bisect_right(expected, value)
        current_count = bisect_right(actual, value)
        numerator = abs(reference_count * len(actual) - current_count * len(expected))
        if numerator > largest:
            largest, location = numerator, value
            at_reference, at_current = reference_count, current_count
        distribution.append(
            {
                "feature": name,
                "count_kind": "cumulative",
                "value": value,
                "reference_count": reference_count,
                "current_count": current_count,
                "reference_share": round(reference_count / len(expected), 8),
                "current_share": round(current_count / len(actual), 8),
            }
        )

    def describe(values: list[int | float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": round(sum(values) / len(values), 8),
        }

    return {
        "feature": name,
        "method": "empirical_ks_distance",
        "distance": round(largest / denominator, 6),
        "at_value": location,
        "reference_cdf_at_value": round(at_reference / len(expected), 8),
        "current_cdf_at_value": round(at_current / len(actual), 8),
        "reference": describe(reference),
        "current": describe(current),
    }, distribution


def _comparisons(
    reference: list[dict[str, Any]], current: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, distributions = [], []
    for name in NUMERIC_INPUTS:
        metric, distribution = _numeric_comparison(
            name, [row[name] for row in reference], [row[name] for row in current]
        )
        metrics.append(metric)
        distributions.extend(distribution)
    numerator = 0
    for segment in ("retail", "small_business"):
        expected = sum(row["segment"] == segment for row in reference)
        actual = sum(row["segment"] == segment for row in current)
        numerator += abs(expected * len(current) - actual * len(reference))
        distributions.append(
            {
                "feature": "segment",
                "count_kind": "category",
                "value": segment,
                "reference_count": expected,
                "current_count": actual,
                "reference_share": round(expected / len(reference), 8),
                "current_share": round(actual / len(current), 8),
            }
        )
    metrics.append(
        {
            "feature": "segment",
            "method": "total_variation_distance",
            "distance": round(numerator / (2 * len(reference) * len(current)), 6),
            "reference": {"count": len(reference)},
            "current": {"count": len(current)},
        }
    )
    return metrics, distributions


def _feature_rows(rows: list[CohortRow]) -> list[dict[str, Any]]:
    return [
        {
            "application_id": row.application.application_id,
            "decision_at": row.application.decision_at.isoformat(),
            "segment": row.application.segment,
            **dict(zip(FEATURE_NAMES, row.features, strict=True)),
            "principal_cents": row.application.principal_cents,
        }
        for row in rows
    ]


def monitor_features(
    applications: list[Application],
    features: list[FeatureEvent],
    as_of: date,
    config: Config | None = None,
    *,
    data_kind: str = "user_supplied",
) -> FeatureMonitorResult:
    """Compare calibration-window inputs with visible current inputs; never read labels."""
    config = config or Config()
    if type(as_of) is not date or as_of < config.test_start:
        raise DataContractError("Feature monitor cutoff must be on or after the test window starts")
    if data_kind not in {"synthetic", "user_supplied"}:
        raise DataContractError("Unknown data kind")
    # Reuse the real snapshot contract, including missing features, duplicates,
    # and available_at <= decision_at. No separate temporal selection algorithm.
    rows = snapshot(applications, features, [], as_of)
    reference = _feature_rows(
        [
            row
            for row in rows
            if config.calibration_start <= row.application.decision_at <= config.calibration_end
        ]
    )
    current = _feature_rows(
        [row for row in rows if config.test_start <= row.application.decision_at <= config.test_end]
    )
    if len(reference) < MINIMUM_POPULATION or len(current) < MINIMUM_POPULATION:
        raise DataContractError("Feature monitoring needs at least 20 applications per population")
    metrics, distributions = _comparisons(reference, current)
    monthly = []
    for month in sorted({row["decision_at"][:7] for row in current}):
        selected = [row for row in current if row["decision_at"].startswith(month)]
        sufficient = len(selected) >= MINIMUM_POPULATION
        monthly.append(
            {
                "vintage": month,
                "applications": len(selected),
                "status": "compared" if sufficient else "insufficient_sample",
                "distances": {
                    item["feature"]: item["distance"]
                    for item in _comparisons(reference, selected)[0]
                }
                if sufficient
                else None,
            }
        )
    current_window = [
        app for app in applications if config.test_start <= app.decision_at <= config.test_end
    ]
    inputs = json.loads(
        json.dumps(
            {
                "schema": SCHEMA,
                "data_kind": data_kind,
                "as_of": as_of,
                "config": asdict(config),
                "applications": [
                    asdict(row) for row in sorted(applications, key=lambda row: row.application_id)
                ],
                "features": [
                    asdict(row)
                    for row in sorted(
                        features, key=lambda row: (row.application_id, row.name, row.available_at)
                    )
                ],
            },
            default=str,
            allow_nan=False,
        )
    )
    summary = {
        "schema": SCHEMA,
        "kind": "label_free_feature_monitor",
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
        "reference_applications": len(reference),
        "current_applications": len(current),
        "current_applications_not_yet_visible": len(current_window) - len(current),
        "applications_outside_windows": len(applications) - len(reference) - len(current_window),
        "minimum_population": MINIMUM_POPULATION,
        "features": metrics,
        "by_vintage": monthly,
        "scope": (
            "Marginal input distributions; no labels, fitted model, "
            "significance test or decision threshold"
        ),
    }
    return FeatureMonitorResult(inputs, summary, reference, current, distributions)


def _render(result: FeatureMonitorResult) -> str:
    summary = result.summary
    feature_rows = "\n".join(
        f"<tr><th scope='row'>{html.escape(item['feature'])}</th>"
        "<td>"
        + ("KS distance" if item["method"] == "empirical_ks_distance" else "Total variation")
        + "</td>"
        f"<td><strong>{item['distance']:.3f}</strong></td>"
        f"<td>{item['reference'].get('mean', '—')}</td>"
        f"<td>{item['current'].get('mean', '—')}</td></tr>"
        for item in summary["features"]
    )
    monthly_rows = "\n".join(
        f"<tr><th scope='row'>{item['vintage']}</th><td>{item['applications']}</td>"
        + (
            "".join(
                f"<td>{item['distances'][name]:.3f}</td>" for name in (*NUMERIC_INPUTS, "segment")
            )
            if item["distances"] is not None
            else '<td colspan="5">Insufficient sample (&lt;20)</td>'
        )
        + "</tr>"
        for item in summary["by_vintage"]
    )
    template = files("creditvintage").joinpath("feature_monitor.html").read_text(encoding="utf-8")
    replacements = {
        "@@AS_OF@@": summary["as_of"],
        "@@REFERENCE@@": str(summary["reference_applications"]),
        "@@CURRENT@@": str(summary["current_applications"]),
        "@@FUTURE@@": str(summary["current_applications_not_yet_visible"]),
        "@@SOURCE@@": "Synthetic example"
        if summary["data_kind"] == "synthetic"
        else "User-supplied data",
        "@@REFERENCE_WINDOW@@": " to ".join(summary["reference_window"]),
        "@@CURRENT_WINDOW@@": " to ".join(summary["current_window"]),
        "@@FEATURE_ROWS@@": feature_rows,
        "@@MONTHLY_ROWS@@": monthly_rows,
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _write(result: FeatureMonitorResult, destination: Path) -> dict[str, Any]:
    _json(destination / "inputs.json", result.inputs)
    _json(destination / "monitor.json", result.summary)
    _csv(destination / "reference_features.csv", result.reference, FEATURE_COLUMNS)
    _csv(destination / "current_features.csv", result.current, FEATURE_COLUMNS)
    _csv(destination / "distributions.csv", result.distributions, DISTRIBUTION_COLUMNS)
    (destination / "index.html").write_text(_render(result), encoding="utf-8", newline="")
    manifest = {
        "schema": SCHEMA,
        "input_sha256": result.summary["input_sha256"],
        "files_sha256": {name: _hash(destination / name) for name in FEATURE_MONITOR_FILES},
    }
    _json(destination / "manifest.json", manifest)
    return manifest


def write_feature_monitor(result: FeatureMonitorResult, destination: Path) -> dict[str, Any]:
    """Stage a complete report, refusing to overwrite an existing destination."""
    if destination.exists():
        raise DataContractError(
            "Feature-monitor destination already exists; choose a new directory"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".feature-monitor-") as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        manifest = _write(result, staged)
        if destination.exists():
            raise DataContractError("Feature-monitor destination already exists")
        staged.rename(destination)
    return manifest


def _replay(source: dict[str, Any]) -> FeatureMonitorResult:
    if set(source) != {"schema", "data_kind", "as_of", "config", "applications", "features"}:
        raise DataContractError("Unexpected feature-monitor input fields")
    if source["schema"] != SCHEMA or not isinstance(source["config"], dict):
        raise DataContractError("Unexpected feature-monitor input schema")
    config = dict(source["config"])
    if set(config) != {item.name for item in fields(Config)}:
        raise DataContractError("Unexpected feature-monitor config fields")
    for name in DATE_FIELDS:
        config[name] = parse_calendar_date(config[name], name)
    for name in ("applications", "features"):
        if not isinstance(source[name], list) or any(
            not isinstance(row, dict) for row in source[name]
        ):
            raise DataContractError(f"Expected feature-monitor {name} records")
    applications = []
    for original in source["applications"]:
        row = dict(original)
        row["decision_at"] = parse_calendar_date(row["decision_at"], "decision_at")
        applications.append(Application(**row))
    features = []
    for original in source["features"]:
        row = dict(original)
        for name in ("observed_at", "available_at"):
            row[name] = parse_calendar_date(row[name], name)
        features.append(FeatureEvent(**row))
    return monitor_features(
        applications,
        features,
        parse_calendar_date(source["as_of"], "as_of"),
        Config(**config),
        data_kind=source["data_kind"],
    )


def verify_feature_monitor(destination: Path) -> dict[str, Any]:
    """Rebuild snapshots, counts, distances and HTML from the retained input events."""
    try:
        manifest = read_json_object(destination / "manifest.json")
        result = _replay(read_json_object(destination / "inputs.json"))
        with TemporaryDirectory(prefix="creditvintage-feature-replay-") as temporary:
            expected = Path(temporary)
            expected_manifest = _write(result, expected)
            if manifest != expected_manifest:
                raise DataContractError("Feature-monitor manifest disagrees with event replay")
            for name in FEATURE_MONITOR_FILES:
                if (destination / name).read_bytes() != (expected / name).read_bytes():
                    raise DataContractError(
                        f"Feature-monitor artifact disagrees with event replay: {name}"
                    )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify feature monitor: {exc}") from exc
    return {
        "status": "verified",
        "reference_applications": len(result.reference),
        "current_applications": len(result.current),
        "monitored_inputs": len(result.summary["features"]),
        "model_fitted": False,
        "input_sha256": result.summary["input_sha256"],
    }
