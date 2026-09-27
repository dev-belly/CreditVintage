"""Point-in-time application features and fully observed loan outcomes."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

FEATURE_NAMES = ("utilization", "debt_to_income", "prior_delinquencies")
REPORT_DAYS = (30, 60, 90, 120, 150, 180)
HORIZON_DAYS = 180
REPORT_LAG_DAYS = 2
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class DataContractError(ValueError):
    """The input cannot support a trustworthy point-in-time evaluation."""


@dataclass(frozen=True)
class Application:
    application_id: str
    decision_at: date
    principal_cents: int
    segment: Literal["retail", "small_business"]

    def __post_init__(self) -> None:
        if not isinstance(self.application_id, str) or not _ID.fullmatch(self.application_id):
            raise DataContractError("Application ID must be 1–128 safe ASCII characters")
        if type(self.decision_at) is not date:
            raise DataContractError("Application needs an ID and decision date")
        if type(self.principal_cents) is not int or self.principal_cents <= 0:
            raise DataContractError("Principal must be positive integer cents")
        if self.segment not in {"retail", "small_business"}:
            raise DataContractError("Unknown segment")


@dataclass(frozen=True)
class FeatureEvent:
    application_id: str
    name: str
    value: float
    observed_at: date
    available_at: date

    def __post_init__(self) -> None:
        if type(self.observed_at) is not date or type(self.available_at) is not date:
            raise DataContractError("Feature needs calendar dates")
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise DataContractError("Feature needs a finite numeric value")
        object.__setattr__(self, "value", float(self.value))
        if self.name not in FEATURE_NAMES:
            raise DataContractError("Unknown feature or non-finite value")
        if self.observed_at > self.available_at:
            raise DataContractError("Feature cannot be available before observation")
        if self.name in {"utilization", "debt_to_income"} and not 0 <= self.value <= 1:
            raise DataContractError("Rate features must be between zero and one")
        if self.name == "prior_delinquencies" and (
            not 0 <= self.value <= 20 or not self.value.is_integer()
        ):
            raise DataContractError("Prior delinquency count must be an integer from 0 to 20")


@dataclass(frozen=True)
class PerformanceEvent:
    application_id: str
    as_of: date
    available_at: date
    days_past_due: int

    def __post_init__(self) -> None:
        if (
            type(self.as_of) is not date
            or type(self.available_at) is not date
            or self.as_of > self.available_at
            or type(self.days_past_due) is not int
            or not 0 <= self.days_past_due <= 365
        ):
            raise DataContractError("Invalid performance report date or DPD")


@dataclass(frozen=True)
class CohortRow:
    application: Application
    features: tuple[float, float, float]
    label: int | None
    status: str


def snapshot(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
    as_of: date,
) -> list[CohortRow]:
    """Only use records available by the decision and evaluation cutoffs.

    A negative label requires every scheduled 30-day report through day 180.
    A known 90+ DPD event is positive even when another report is missing.
    All labels wait until the full horizon and reporting lag have elapsed.
    """
    by_id = {app.application_id: app for app in applications}
    if len(by_id) != len(applications):
        raise DataContractError("Duplicate application ID")
    feature_index: dict[tuple[str, str], list[FeatureEvent]] = defaultdict(list)
    report_index: dict[str, list[PerformanceEvent]] = defaultdict(list)
    feature_keys: set[tuple[str, str, date]] = set()
    report_keys: set[tuple[str, date, date]] = set()
    for feature_event in features:
        if feature_event.application_id not in by_id:
            raise DataContractError("Feature references unknown application")
        key = (
            feature_event.application_id,
            feature_event.name,
            feature_event.available_at,
        )
        if key in feature_keys:
            raise DataContractError("Ambiguous same-day feature revision")
        feature_keys.add(key)
        feature_index[(feature_event.application_id, feature_event.name)].append(feature_event)
    for report_event in performance:
        app = by_id.get(report_event.application_id)
        if app is None or report_event.as_of <= app.decision_at:
            raise DataContractError("Performance references unknown or future application")
        report_key = (
            report_event.application_id,
            report_event.as_of,
            report_event.available_at,
        )
        if report_key in report_keys:
            raise DataContractError("Duplicate performance revision")
        report_keys.add(report_key)
        report_index[report_event.application_id].append(report_event)

    rows: list[CohortRow] = []
    for app in sorted(applications, key=lambda item: (item.decision_at, item.application_id)):
        if app.decision_at > as_of:
            continue
        values: list[float] = []
        for name in FEATURE_NAMES:
            candidates = [
                event
                for event in feature_index[(app.application_id, name)]
                if event.available_at <= app.decision_at and event.observed_at <= app.decision_at
            ]
            if not candidates:
                raise DataContractError(f"Missing decision-time {name}: {app.application_id}")
            values.append(max(candidates, key=lambda event: event.available_at).value)

        label: int | None = None
        status = "not_mature"
        if app.decision_at + timedelta(days=HORIZON_DAYS + REPORT_LAG_DAYS) <= as_of:
            latest_reports: dict[date, PerformanceEvent] = {}
            for report in report_index[app.application_id]:
                if report.available_at > as_of or report.as_of > app.decision_at + timedelta(
                    days=HORIZON_DAYS
                ):
                    continue
                current = latest_reports.get(report.as_of)
                if current is None or report.available_at > current.available_at:
                    latest_reports[report.as_of] = report
            if any(event.days_past_due >= 90 for event in latest_reports.values()):
                label, status = 1, "default"
            elif all(
                app.decision_at + timedelta(days=day) in latest_reports for day in REPORT_DAYS
            ):
                label, status = 0, "observed_nondefault"
            else:
                status = "missing_performance"
        rows.append(CohortRow(app, (values[0], values[1], values[2]), label, status))
    return rows


def load_csv_bundle(
    applications_path: Path, features_path: Path, performance_path: Path
) -> tuple[list[Application], list[FeatureEvent], list[PerformanceEvent]]:
    """Read the documented three-file contract without silently filling gaps."""

    def read(path: Path, columns: set[str]) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(columns)
                or set(reader.fieldnames) != columns
            ):
                raise DataContractError(f"Unexpected CSV columns: {path.name}")
            rows = list(reader)
            if any(
                set(row) != columns or any(value is None for value in row.values()) for row in rows
            ):
                raise DataContractError(f"Malformed CSV row: {path.name}")
            return rows

    try:
        apps = [
            Application(
                row["application_id"],
                date.fromisoformat(row["decision_at"]),
                int(row["principal_cents"]),
                row["segment"],  # type: ignore[arg-type]
            )
            for row in read(
                applications_path,
                {"application_id", "decision_at", "principal_cents", "segment"},
            )
        ]
        feats = [
            FeatureEvent(
                row["application_id"],
                row["name"],
                float(row["value"]),
                date.fromisoformat(row["observed_at"]),
                date.fromisoformat(row["available_at"]),
            )
            for row in read(
                features_path,
                {"application_id", "name", "value", "observed_at", "available_at"},
            )
        ]
        reports = [
            PerformanceEvent(
                row["application_id"],
                date.fromisoformat(row["as_of"]),
                date.fromisoformat(row["available_at"]),
                int(row["days_past_due"]),
            )
            for row in read(
                performance_path,
                {"application_id", "as_of", "available_at", "days_past_due"},
            )
        ]
    except (ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Invalid CSV value: {exc}") from exc
    return apps, feats, reports
