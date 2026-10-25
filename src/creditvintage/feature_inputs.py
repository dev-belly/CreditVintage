"""Canonical two-table event evidence shared by the two input diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import date
from typing import Any

from creditvintage.configuration import DATE_FIELDS
from creditvintage.core import Application, DataContractError, FeatureEvent, parse_calendar_date
from creditvintage.model import Config


def encode_inputs(
    applications: list[Application],
    features: list[FeatureEvent],
    as_of: date,
    config: Config,
    *,
    schema: str,
    data_kind: str,
) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        json.dumps(
            {
                "schema": schema,
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
    return result


def decode_inputs(
    source: dict[str, Any],
    *,
    schema: str,
) -> tuple[list[Application], list[FeatureEvent], date, Config, str]:
    if set(source) != {"schema", "data_kind", "as_of", "config", "applications", "features"}:
        raise DataContractError("Unexpected feature-event input fields")
    if source["schema"] != schema or not isinstance(source["config"], dict):
        raise DataContractError("Unexpected feature-event input schema")
    config = dict(source["config"])
    if set(config) != {item.name for item in fields(Config)}:
        raise DataContractError("Unexpected feature-event config fields")
    for name in DATE_FIELDS:
        config[name] = parse_calendar_date(config[name], name)
    for name in ("applications", "features"):
        if not isinstance(source[name], list) or any(
            not isinstance(row, dict) for row in source[name]
        ):
            raise DataContractError(f"Expected feature-event {name} records")
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
    return (
        applications,
        features,
        parse_calendar_date(source["as_of"], "as_of"),
        Config(**config),
        source["data_kind"],
    )
