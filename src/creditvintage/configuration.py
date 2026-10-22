"""Explicit calendar windows and a model-free check for supplied event data."""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from typing import Any

from creditvintage.core import (
    Application,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    parse_calendar_date,
    snapshot,
)
from creditvintage.model import Config, _digest, _rows_between
from creditvintage.validation import read_json_object

DATE_FIELDS = (
    "train_start",
    "train_end",
    "calibration_start",
    "calibration_end",
    "test_start",
    "test_end",
    "evaluation_as_of",
)


def load_config(path: Path | None = None, *, seed: int | None = None) -> Config:
    """Require all seven dates in supplied JSON; an explicit CLI seed takes precedence."""
    values: dict[str, Any] = {}
    if path is not None:
        values = read_json_object(path)
        unknown = set(values) - {field.name for field in fields(Config)}
        if unknown:
            raise DataContractError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        missing = set(DATE_FIELDS) - set(values)
        if missing:
            raise DataContractError(f"Missing config dates: {', '.join(sorted(missing))}")
        for name in DATE_FIELDS:
            values[name] = parse_calendar_date(values[name], f"Config {name}")
    if seed is not None:
        values["seed"] = seed
    return Config(**values)


def preflight(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
    config: Config,
    *,
    data_kind: str,
) -> dict[str, Any]:
    """Use evaluation's actual maturity gates, with no fitting or filesystem writes."""
    if data_kind not in {"synthetic", "user_supplied"}:
        raise DataContractError("Unknown data kind")
    cohorts: dict[str, Any] = {}
    for stage, start, end, cutoff in (
        ("train", config.train_start, config.train_end, config.calibration_start),
        ("calibration", config.calibration_start, config.calibration_end, config.test_start),
        ("test", config.test_start, config.test_end, config.evaluation_as_of),
    ):
        rows = _rows_between(
            snapshot(applications, features, performance, cutoff), start, end, stage
        )
        events = sum(row.label == 1 for row in rows)
        cohorts[stage] = {
            "applications": len(rows),
            "events": events,
            "non_events": len(rows) - events,
            "decision_start": start.isoformat(),
            "decision_end": end.isoformat(),
            "outcomes_visible_by": cutoff.isoformat(),
        }
    return {
        "status": "ready_for_evaluation",
        "data_kind": data_kind,
        "input_sha256": _digest(applications, features, performance),
        "config": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in asdict(config).items()
        },
        "input_rows": {
            "applications": len(applications),
            "features": len(features),
            "performance": len(performance),
        },
        "applications_outside_windows": len(applications)
        - sum(cohort["applications"] for cohort in cohorts.values()),
        "cohorts": cohorts,
        "model_fitted": False,
        "scope": "Input and maturity checks; not model quality or source authenticity",
    }
