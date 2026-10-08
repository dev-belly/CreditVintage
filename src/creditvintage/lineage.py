"""Checked PITBridge scalar features, source lineage and full model-input replay."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import tempfile
from dataclasses import asdict
from datetime import date, timedelta
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from creditvintage.artifacts import PREDICTION_COLUMNS, verify_result, write_result
from creditvintage.core import (
    _ID,
    FEATURE_NAMES,
    Application,
    DataContractError,
    FeatureEvent,
    load_csv_bundle,
)
from creditvintage.model import Config, Result, evaluate
from creditvintage.sample import generate
from creditvintage.validation import read_csv_rows, read_json_object

SOURCE = "credit_application"
MAX_AGE_DAYS = 7
PIT_COMMIT = "ed19dc698534f45a2b646fb4976ff6b01966b0cc"
LINEAGE_COLUMNS = (
    "application_id",
    "decision_at_utc",
    "feature",
    "value",
    "source",
    "record_id",
    "revision",
    "event_at",
    "published_at",
    "ingested_at",
    "available_at",
)
INPUT_COLUMNS = {
    "applications.csv": ("application_id", "decision_at", "principal_cents", "segment"),
    "features.csv": ("application_id", "name", "value", "observed_at", "available_at"),
    "performance.csv": ("application_id", "as_of", "available_at", "days_past_due"),
}
BUNDLE_FILES = (
    "applications.csv",
    "features.csv",
    "performance.csv",
    "lineage.csv",
    "summary.json",
    "index.html",
    "pit/inputs.json",
    "pit/snapshots.csv",
    "pit/comparison.csv",
    "pit/summary.json",
    "pit/report.html",
    "pit/manifest.json",
    "evaluation/summary.json",
    "evaluation/predictions.csv",
    "evaluation/calibration_bins.csv",
    "evaluation/vintage_metrics.csv",
    "evaluation/index.html",
    "evaluation/manifest.json",
)


def _pit() -> tuple[Any, Any]:
    try:
        return importlib.import_module("pitbridge.core"), importlib.import_module(
            "pitbridge.bundle"
        )
    except ModuleNotFoundError as exc:
        raise DataContractError(
            "Install PITBridge first; see the source-to-prediction setup"
        ) from exc


def cutoff(day: date) -> str:
    """CreditVintage's date-only decisions explicitly mean end of that UTC day."""
    return f"{day.isoformat()}T23:59:59.999999+00:00"


def adapt_bundle(
    applications: list[Application], pit_directory: Path
) -> tuple[list[FeatureEvent], list[dict[str, Any]]]:
    core, bundle = _pit()
    bundle.verify_bundle(pit_directory)
    observations, decisions, specs = bundle.decode_inputs(
        bundle.read_json(pit_directory / "inputs.json")
    )
    apps = {app.application_id: app for app in applications}
    if not apps or len(apps) != len(applications):
        raise DataContractError("Applications must have unique IDs and cannot be empty")
    if {(s.source, s.feature, s.max_age_days) for s in specs} != {
        (SOURCE, name, MAX_AGE_DAYS) for name in FEATURE_NAMES
    }:
        raise DataContractError(
            "PIT feature contract requires three credit features with a 7-day TTL"
        )
    expected_decisions = {(key, key, cutoff(app.decision_at)) for key, app in apps.items()}
    if {(d.decision_id, d.entity_id, d.decision_at) for d in decisions} != expected_decisions:
        raise DataContractError(
            "PIT decisions must match application IDs, entities and UTC end-of-day cutoffs"
        )
    if any(row.entity_id not in apps for row in observations):
        raise DataContractError("Source observation references an unknown application")
    rows = core.build_snapshot(observations, decisions, specs)
    features: list[FeatureEvent] = []
    lineage: list[dict[str, Any]] = []
    for row in rows:
        if row.status != "selected":
            raise DataContractError(
                f"Unavailable PIT feature: {row.decision_id}/{row.feature}: {row.status}"
            )
        if not _ID.fullmatch(row.record_id):
            raise DataContractError("Selected record ID must be safe for CSV export")
        features.append(
            FeatureEvent(
                row.decision_id,
                row.feature,
                row.value,
                date.fromisoformat(row.event_at[:10]),
                date.fromisoformat(row.available_at[:10]),
            )
        )
        lineage.append(
            dict(
                zip(
                    LINEAGE_COLUMNS,
                    (
                        row.decision_id,
                        row.decision_at,
                        row.feature,
                        row.value,
                        row.source,
                        row.record_id,
                        row.revision,
                        row.event_at,
                        row.published_at,
                        row.ingested_at,
                        row.available_at,
                    ),
                    strict=True,
                )
            )
        )
    return features, lineage


def _csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _serial(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in asdict(row).items()
        }
        for row in rows
    ]


def _source_data(apps: list[Application], features: list[FeatureEvent]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = [
        {
            "record_id": f"{f.application_id}-{f.name}-{f.observed_at.isoformat()}",
            "entity_id": f.application_id,
            "source": SOURCE,
            "feature": f.name,
            "event_at": f"{f.observed_at.isoformat()}T00:00:00Z",
            "published_at": f"{f.available_at.isoformat()}T08:00:00Z",
            "ingested_at": f"{f.available_at.isoformat()}T09:00:00Z",
            "revision": 1,
            "value": f.value,
            "deleted": False,
        }
        for f in features
    ]
    # Historical corrections were published before the decision but arrived after it.
    # The adapter must use max(publication, ingestion), not publication alone.
    for app in apps[::10]:
        original = next(
            row
            for row in observations
            if row["entity_id"] == app.application_id and row["feature"] == "utilization"
        )
        observations.append(
            {
                **original,
                "record_id": original["record_id"] + "-r2",
                "revision": 2,
                "value": 0.01,
                "published_at": f"{app.decision_at.isoformat()}T12:00:00Z",
                "ingested_at": f"{(app.decision_at + timedelta(days=1)).isoformat()}T00:00:00Z",
            }
        )
    return {
        "data_kind": "synthetic",
        "observations": observations,
        "decisions": [
            {
                "decision_id": app.application_id,
                "entity_id": app.application_id,
                "decision_at": cutoff(app.decision_at),
            }
            for app in apps
        ],
        "specs": [
            {"source": SOURCE, "feature": name, "max_age_days": MAX_AGE_DAYS}
            for name in FEATURE_NAMES
        ],
    }


def render_lineage(summary: dict[str, Any], lineage: list[dict[str, Any]], result: Result) -> str:
    explorer_version = summary.get("explorer_version", 1)
    if type(explorer_version) is not int or explorer_version not in (1, 2):
        raise DataContractError("Unsupported lineage explorer version")
    predicted = {row["application_id"]: row for row in result.predictions}
    test_lineage = [row for row in lineage if row["application_id"] in predicted]
    options = "".join(
        f'<option value="{escape(key, quote=True)}">{escape(key)}</option>'
        for key in sorted(predicted)
    )
    rows = "".join(
        f'<tr data-application="{escape(row["application_id"], quote=True)}">'
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in LINEAGE_COLUMNS)
        + "</tr>"
        for row in test_lineage
    )
    payload = json.dumps(predicted, sort_keys=True, allow_nan=False).replace("<", "\\u003c")
    template = files("creditvintage").joinpath("lineage.html").read_text(encoding="utf-8")
    for key, value in {
        "@@ARCHIVE_LINK@@": (
            '<a href="evidence.zip" download>Complete evidence ZIP</a>'
            if explorer_version == 2
            else ""
        ),
        "@@APPLICATIONS@@": str(summary["applications"]),
        "@@FEATURES@@": str(summary["feature_rows"]),
        "@@EXCLUDED@@": str(summary["future_observations"]),
        "@@OPTIONS@@": options,
        "@@ROWS@@": rows,
        "@@PREDICTIONS@@": payload,
    }.items():
        template = template.replace(key, value)
    return template


def _archive_evidence(directory: Path) -> None:
    """Package the fixed evidence inventory for local and published explorers."""
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


def write_demo(destination: Path, seed: int = 20260927, per_vintage: int = 20) -> dict[str, Any]:
    _, bundle = _pit()
    apps, original_features, reports = generate(seed, per_vintage)
    destination.mkdir(parents=True, exist_ok=True)
    data = _source_data(apps, original_features)
    bundle.write_bundle(data, destination / "pit")
    features, lineage = adapt_bundle(apps, destination / "pit")
    input_rows: dict[str, list[Any]] = {
        "applications.csv": apps,
        "features.csv": features,
        "performance.csv": reports,
    }
    for name, records in input_rows.items():
        _csv(destination / name, _serial(records), INPUT_COLUMNS[name])
    # Evaluate the exported inputs, so the report is tied to the saved CSVs.
    inputs = load_csv_bundle(*(destination / name for name in INPUT_COLUMNS))
    result = evaluate(*inputs, Config(seed=seed), data_kind="synthetic")
    write_result(result, destination / "evaluation")
    _csv(destination / "lineage.csv", lineage, LINEAGE_COLUMNS)
    observations, decisions, _ = bundle.decode_inputs(
        bundle.read_json(destination / "pit/inputs.json")
    )
    decision_times = {decision.entity_id: decision.decision_at for decision in decisions}
    summary = {
        "schema_version": 1,
        "explorer_version": 2,
        "data_kind": "synthetic",
        "seed": seed,
        "applications": len(apps),
        "feature_rows": len(features),
        "future_observations": sum(
            row.available_at > decision_times[row.entity_id] for row in observations
        ),
        "pitbridge_reference_commit": PIT_COMMIT,
        "decision_convention": "end_of_day_utc",
        "max_age_days": MAX_AGE_DAYS,
        "units": {
            "utilization": "fraction",
            "debt_to_income": "fraction",
            "prior_delinquencies": "count",
            "principal": "USD cents",
        },
    }
    _json(destination / "summary.json", summary)
    (destination / "index.html").write_text(
        render_lineage(summary, lineage, result), encoding="utf-8"
    )
    _json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "engine": "creditvintage-lineage/1",
            "files_sha256": {
                name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                for name in BUNDLE_FILES
            },
        },
    )
    _archive_evidence(destination)
    return summary


def verify_lineage(destination: Path) -> dict[str, Any]:
    """Reject malformed evidence with the same contract errors as other verifiers."""
    try:
        return _verify_lineage(destination)
    except DataContractError:
        raise
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify lineage: {exc}") from exc


def _verify_lineage(destination: Path) -> dict[str, Any]:
    manifest = read_json_object(destination / "manifest.json")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("engine") != "creditvintage-lineage/1"
        or not isinstance(manifest.get("files_sha256"), dict)
        or set(manifest.get("files_sha256", {})) != set(BUNDLE_FILES)
    ):
        raise DataContractError("Unsupported lineage manifest or unexpected artifact paths")
    for name in BUNDLE_FILES:
        if (
            hashlib.sha256((destination / name).read_bytes()).hexdigest()
            != manifest["files_sha256"][name]
        ):
            raise DataContractError(f"Lineage digest mismatch: {name}")
    summary = read_json_object(destination / "summary.json")
    explorer_version = summary.get("explorer_version", 1)
    if (
        type(summary.get("schema_version")) is not int
        or summary["schema_version"] != 1
        or type(explorer_version) is not int
        or explorer_version not in (1, 2)
        or summary.get("data_kind") != "synthetic"
        or summary.get("decision_convention") != "end_of_day_utc"
        or summary.get("max_age_days") != MAX_AGE_DAYS
        or summary.get("pitbridge_reference_commit") != PIT_COMMIT
        or type(summary.get("seed")) is not int
    ):
        raise DataContractError("Unsupported lineage demo contract")
    for name in ("applications", "feature_rows", "future_observations"):
        if type(summary.get(name)) is not int or summary[name] < 0:
            raise DataContractError("Lineage counts must be nonnegative integers")
    apps, exported_features, reports = load_csv_bundle(
        *(destination / name for name in INPUT_COLUMNS)
    )
    features, lineage = adapt_bundle(apps, destination / "pit")
    if features != exported_features:
        raise DataContractError("Exported model features disagree with PIT source replay")
    with tempfile.TemporaryDirectory() as name:
        expected_csv = Path(name) / "lineage.csv"
        _csv(expected_csv, lineage, LINEAGE_COLUMNS)
        if expected_csv.read_bytes() != (destination / "lineage.csv").read_bytes():
            raise DataContractError("Feature lineage disagrees with source replay")
    if len(apps) != summary["applications"] or len(features) != summary["feature_rows"]:
        raise DataContractError("Lineage counts disagree with inputs")
    _, bundle = _pit()
    source_data = bundle.read_json(destination / "pit/inputs.json")
    if source_data.get("data_kind") != "synthetic":
        raise DataContractError("Lineage demo source must be marked synthetic")
    observations, decisions, _ = bundle.decode_inputs(source_data)
    decision_times = {d.entity_id: d.decision_at for d in decisions}
    if (
        sum(o.available_at > decision_times[o.entity_id] for o in observations)
        != summary["future_observations"]
    ):
        raise DataContractError("Future-source count disagrees with source replay")
    expected_units = {
        "utilization": "fraction",
        "debt_to_income": "fraction",
        "prior_delinquencies": "count",
        "principal": "USD cents",
    }
    if summary.get("units") != expected_units:
        raise DataContractError("Feature units disagree with adapter contract")
    verify_result(destination / "evaluation")
    replayed = evaluate(
        apps, features, reports, Config(seed=summary["seed"]), data_kind="synthetic"
    )
    saved = read_json_object(destination / "evaluation/summary.json")
    for field in ("project", "data_kind", "label_rule", "model", "config", "cohorts"):
        if saved.get(field) != replayed.summary[field]:
            raise DataContractError(
                f"Evaluation metadata disagrees with full model replay: {field}"
            )
    model_manifest = read_json_object(destination / "evaluation/manifest.json")
    if replayed.input_sha256 != model_manifest["input_sha256"]:
        raise DataContractError("Model input digest disagrees with replay")
    predicted = read_csv_rows(destination / "evaluation/predictions.csv", PREDICTION_COLUMNS)
    if [row["application_id"] for row in replayed.predictions] != [
        row["application_id"] for row in predicted
    ]:
        raise DataContractError("Prediction identities disagree with full model replay")
    for field in ("raw_pd", "calibrated_pd"):
        if not np.allclose(
            [row[field] for row in replayed.predictions],
            [float(row[field]) for row in predicted],
            rtol=1e-8,
            atol=1e-10,
        ):
            raise DataContractError(f"Predictions disagree with full model replay: {field}")
    for field in (
        "decision_at",
        "vintage",
        "segment",
        "principal_cents",
        "label",
        "review_selected",
    ):
        if [str(row[field]) for row in replayed.predictions] != [row[field] for row in predicted]:
            raise DataContractError(f"Prediction metadata disagrees with replay: {field}")
    rendered = render_lineage(
        summary,
        lineage,
        Result(
            saved, replayed.predictions, replayed.deciles, replayed.vintages, replayed.input_sha256
        ),
    )
    if (destination / "index.html").read_text(encoding="utf-8") != rendered:
        raise DataContractError("Lineage explorer disagrees with full model replay")
    return {
        "verified": True,
        "applications": len(apps),
        "source_features": len(features),
        "pit_replay": True,
        "model_replay": True,
        "prediction_tolerance": 1e-8,
    }
