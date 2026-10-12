"""Verified SQL cohort analysis with an independent Python aggregation oracle."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import sqlite3
import tempfile
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from creditvintage.artifacts import PREDICTION_COLUMNS, RESULT_FILES, verify_result
from creditvintage.core import DataContractError
from creditvintage.validation import read_csv_rows, read_json_object

SOURCE_FILES = (*RESULT_FILES, "manifest.json")
SQL_FILES = ("business_schema.sql", "business_groups.sql", "business_queue.sql")
GROUP_COLUMNS = (
    "scope",
    "vintage",
    "segment",
    "loans",
    "events",
    "event_rate",
    "mean_pd",
    "event_pd_gap",
    "principal_cents",
    "exposure_weighted_pd",
    "reviewed",
    "reviewed_events",
    "review_share",
    "review_precision",
    "event_capture",
    "previous_event_rate",
    "event_rate_change",
    "scenario_expected_loss_usd",
    "scenario_event_loss_proxy_usd",
)
QUEUE_COLUMNS = (
    "review_rank",
    "application_id",
    "decision_at",
    "vintage",
    "segment",
    "principal_cents",
    "raw_pd",
    "calibrated_pd",
)


def _python_groups(rows: list[dict[str, Any]], lgd: float) -> list[dict[str, Any]]:
    """Count every loan directly; this path neither executes nor parses SQL."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in (
            ("portfolio", "all", "all"),
            ("vintage", row["vintage"], "all"),
            ("segment", "all", row["segment"]),
            ("vintage_segment", row["vintage"], row["segment"]),
        ):
            groups[key].append(row)
    result = []
    previous: dict[tuple[str, str], float] = {}
    for (scope, vintage, segment), members in sorted(groups.items()):
        n = len(members)
        events = sum(row["label"] for row in members)
        selected = [row for row in members if row["review_selected"] == 1]
        reviewed_events = sum(row["label"] for row in selected)
        principal = sum(row["principal_cents"] for row in members)
        weighted = math.fsum(row["principal_cents"] * row["calibrated_pd"] for row in members)
        mean_pd = math.fsum(row["calibrated_pd"] for row in members) / n
        event_rate = events / n
        previous_rate = previous.get((scope, segment))
        if scope in {"vintage", "vintage_segment"}:
            previous[(scope, segment)] = event_rate
        result.append(
            {
                "scope": scope,
                "vintage": vintage,
                "segment": segment,
                "loans": n,
                "events": events,
                "event_rate": event_rate,
                "mean_pd": mean_pd,
                "event_pd_gap": event_rate - mean_pd,
                "principal_cents": principal,
                "exposure_weighted_pd": weighted / principal,
                "reviewed": len(selected),
                "reviewed_events": reviewed_events,
                "review_share": len(selected) / n,
                "review_precision": reviewed_events / len(selected) if selected else None,
                "event_capture": reviewed_events / events if events else None,
                "previous_event_rate": previous_rate,
                "event_rate_change": event_rate - previous_rate
                if previous_rate is not None
                else None,
                "scenario_expected_loss_usd": weighted * lgd / 100,
                "scenario_event_loss_proxy_usd": sum(
                    row["principal_cents"] for row in members if row["label"] == 1
                )
                * lgd
                / 100,
            }
        )
    return result


def _aggregate(
    predictions: list[dict[str, str]], lgd: float, queries: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = [
        {
            key: int(value)
            if key in {"label", "review_selected", "principal_cents"}
            else float(value)
            if key in {"raw_pd", "calibrated_pd"}
            else value
            for key, value in row.items()
        }
        for row in predictions
    ]
    if not rows or sum(row["principal_cents"] for row in rows) > 2**63 - 1:
        raise DataContractError("SQL analysis needs loans with total cents within SQLite int64")
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(queries["business_schema.sql"])
        connection.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row[key] for key in PREDICTION_COLUMNS) for row in rows],
        )
        groups = [
            dict(row) for row in connection.execute(queries["business_groups.sql"], {"lgd": lgd})
        ]
        queue = [dict(row) for row in connection.execute(queries["business_queue.sql"])]
    except (sqlite3.Error, OverflowError) as exc:
        raise DataContractError(f"Cannot compute SQL analysis: {exc}") from exc
    finally:
        connection.close()
    expected = _python_groups(rows, lgd)
    if len(groups) != len(expected):
        raise DataContractError("SQL group coverage disagrees with Python replay")
    exact = {
        "scope",
        "vintage",
        "segment",
        "loans",
        "events",
        "principal_cents",
        "reviewed",
        "reviewed_events",
    }
    for actual, oracle in zip(groups, expected, strict=True):
        if tuple(actual) != GROUP_COLUMNS:
            raise DataContractError("Unexpected SQL group columns")
        for key in GROUP_COLUMNS:
            a, b = actual[key], oracle[key]
            if key in exact or a is None or b is None:
                valid = a == b
            else:
                tolerance = 0.010001 if key.endswith("usd") else 0.000000011
                valid = math.isfinite(a) and abs(a - b) <= tolerance
            if not valid:
                raise DataContractError(f"SQL {key} disagrees with Python replay")
    ranked = sorted(
        (row for row in rows if row["review_selected"] == 1),
        key=lambda row: (-row["calibrated_pd"], row["application_id"]),
    )
    expected_queue = [
        {"review_rank": rank, **{key: row[key] for key in QUEUE_COLUMNS[1:]}}
        for rank, row in enumerate(ranked, 1)
    ]
    if queue != expected_queue or any(tuple(row) != QUEUE_COLUMNS for row in queue):
        raise DataContractError("SQL review queue disagrees with Python replay")
    return groups, queue


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _rate(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.1f} pp" if signed else f"{value * 100:.1f}%"


def _render(summary: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    portfolio = summary["portfolio"]
    body = []
    for row in groups:
        cells = (
            row["vintage"],
            row["segment"],
            row["loans"],
            row["events"],
            _rate(row["event_rate"]),
            _rate(row["mean_pd"]),
            _rate(row["event_pd_gap"], True),
            _rate(row["event_rate_change"], True),
            f"{row['reviewed']} / {row['loans']}",
            _rate(row["review_precision"]),
            _rate(row["event_capture"]),
            f"${row['principal_cents'] / 100:,.2f}",
        )
        initial_visibility = "" if row["scope"] == "vintage" else " hidden"
        body.append(
            f'<tr data-scope="{row["scope"]}"{initial_visibility}>'
            + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells)
            + "</tr>"
        )
    replacements = {
        "@@SOURCE@@": "SYNTHETIC EXAMPLE" if summary["data_kind"] == "synthetic" else "USER DATA",
        "@@LOANS@@": str(portfolio["loans"]),
        "@@EVENT_RATE@@": _rate(portfolio["event_rate"]),
        "@@MEAN_PD@@": _rate(portfolio["mean_pd"]),
        "@@CAPTURE@@": _rate(portfolio["event_capture"]),
        "@@REVIEWED@@": str(portfolio["reviewed"]),
        "@@EVENTS@@": str(portfolio["events"]),
        "@@REVIEWED_EVENTS@@": str(portfolio["reviewed_events"]),
        "@@PRECISION@@": _rate(portfolio["review_precision"]),
        "@@ROWS@@": "\n".join(body),
        "@@CUTOFF@@": html.escape(summary["evaluation_as_of"]),
    }
    template = files("creditvintage").joinpath("business.html").read_text(encoding="utf-8")
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _build(source: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Freeze captured source bytes before verification and all subsequent reads."""
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary)
        for name in SOURCE_FILES:
            (snapshot / name).write_bytes(source[name])
        checked = verify_result(snapshot)
        original = read_json_object(snapshot / "summary.json")
        predictions = read_csv_rows(snapshot / "predictions.csv", PREDICTION_COLUMNS)
    queries = {
        name: files("creditvintage").joinpath(name).read_text(encoding="utf-8")
        for name in SQL_FILES
    }
    groups, queue = _aggregate(predictions, original["config"]["scenario_lgd"], queries)
    summary = {
        "project": "CreditVintage SQL business analysis",
        "schema_version": 1,
        "data_kind": original["data_kind"],
        "input_sha256": checked["input_sha256"],
        "evaluation_as_of": original["config"]["evaluation_as_of"],
        "label_rule": original["label_rule"],
        "scenario_lgd": original["config"]["scenario_lgd"],
        "review_fraction": original["config"]["review_fraction"],
        "group_rows": len(groups),
        "queue_rows": len(queue),
        "portfolio": next(row for row in groups if row["scope"] == "portfolio"),
    }
    output = {f"source/{name}": content for name, content in source.items()}
    output.update({name: query.encode("utf-8") for name, query in queries.items()})
    output["summary.json"] = _json_bytes(summary)
    output["groups.csv"] = _csv_bytes(groups, GROUP_COLUMNS)
    output["review_queue.csv"] = _csv_bytes(queue, QUEUE_COLUMNS)
    output["index.html"] = _render(summary, groups).encode("utf-8")
    output["manifest.json"] = _json_bytes(
        {
            "schema_version": 1,
            "files_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(output.items())
            },
        }
    )
    return output, summary


def write_analysis(report: Path, destination: Path) -> dict[str, Any]:
    """Analyze a verified mature report; never overwrite an existing directory."""
    if destination.exists():
        raise DataContractError(f"Choose a new analysis directory: {destination}")
    source = {name: (report / name).read_bytes() for name in SOURCE_FILES}
    output, summary = _build(source)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in output.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return summary


def verify_analysis(destination: Path) -> dict[str, Any]:
    """Rebuild SQL, groups, label-free queue and HTML from the embedded source."""
    try:
        source = {name: (destination / "source" / name).read_bytes() for name in SOURCE_FILES}
        expected, summary = _build(source)
        actual_files = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual_files != set(expected):
            raise DataContractError("Unexpected SQL analysis file inventory")
        for name, content in expected.items():
            if (destination / name).read_bytes() != content:
                raise DataContractError(f"SQL analysis disagrees with source replay: {name}")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify SQL analysis: {exc}") from exc
    return {
        "status": "verified",
        "loans": summary["portfolio"]["loans"],
        "groups": summary["group_rows"],
        "reviewed": summary["queue_rows"],
    }
