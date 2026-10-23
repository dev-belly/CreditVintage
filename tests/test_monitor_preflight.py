from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta

import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditvintage.cli import main
from creditvintage.configuration import preflight
from creditvintage.core import DataContractError, PerformanceEvent
from creditvintage.inputs import initialize_inputs
from creditvintage.model import Config
from creditvintage.monitoring import monitor, preflight_monitor
from creditvintage.sample import generate

CUTOFF = date(2024, 10, 1)
CONFIG = Config(seed=7)


@pytest.fixture
def inputs():
    apps, features, reports = generate(seed=7, per_vintage=25)
    historical = {app.application_id for app in apps if app.decision_at < CONFIG.test_start}
    return apps, features, [report for report in reports if report.application_id in historical]


@pytest.fixture
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Monitoring preflight must not fit models")

    for model in (LogisticRegression, IsotonicRegression):
        monkeypatch.setattr(model, "fit", forbidden)


def test_unlabelled_current_cohort_passes_without_fitting_or_mutating_inputs(inputs, no_fit):
    before = [[asdict(row) for row in rows] for rows in inputs]
    result = preflight_monitor(*inputs, CUTOFF, CONFIG, data_kind="synthetic")
    assert result["status"] == "ready_for_monitoring" and result["model_fitted"] is False
    assert [
        result["cohorts"][stage]["applications"] for stage in ("train", "calibration", "current")
    ] == [300, 150, 100]
    assert result["current_applications_not_yet_visible"] == 50
    assert result["applications_outside_windows"] == 0
    assert result["performance_scope"] == "train_and_calibration_only"
    assert result["cohorts"]["current"]["outcomes_required"] is False
    assert set(result["cohorts"]["current"]) == {
        "applications",
        "decision_start",
        "decision_end",
        "visible_by",
        "outcomes_required",
    }
    assert "psi" not in result and "scores" not in result
    assert before == [[asdict(row) for row in rows] for rows in inputs]
    with pytest.raises(DataContractError, match="test has unlabelled loans"):
        preflight(*inputs, CONFIG, data_kind="synthetic")


def test_preflight_counts_match_the_actual_monitor(inputs):
    checked = preflight_monitor(*inputs, CUTOFF, CONFIG, data_kind="synthetic")
    actual = monitor(*inputs, CUTOFF, CONFIG)
    assert checked["cohorts"]["calibration"]["applications"] == actual.summary["reference_loans"]
    assert checked["cohorts"]["current"]["applications"] == actual.summary["current_loans"]
    assert checked["input_sha256"] == actual.input_sha256


def test_unused_current_outcomes_change_source_digest_but_not_readiness(inputs, no_fit):
    apps, features, reports = inputs
    current = next(app for app in apps if app.decision_at == CONFIG.test_start)
    extra = PerformanceEvent(current.application_id, current.decision_at, current.decision_at, 90)
    before = preflight_monitor(*inputs, CUTOFF, CONFIG, data_kind="synthetic")
    after = preflight_monitor(
        apps, features, reports + [extra], CUTOFF, CONFIG, data_kind="synthetic"
    )
    assert after["cohorts"] == before["cohorts"] and after["status"] == before["status"]
    assert after["input_sha256"] != before["input_sha256"]
    assert after["input_rows"]["performance"] == before["input_rows"]["performance"] + 1


@pytest.mark.parametrize("year,stage", [(2022, "train"), (2023, "calibration")])
def test_missing_historical_outcomes_still_fail_the_real_maturity_gate(inputs, no_fit, year, stage):
    apps, features, reports = inputs
    identity = next(app.application_id for app in apps if app.decision_at.year == year)
    incomplete = [report for report in reports if report.application_id != identity]
    with pytest.raises(DataContractError, match=stage + " has unlabelled loans"):
        preflight_monitor(apps, features, incomplete, CUTOFF, CONFIG, data_kind="synthetic")


@pytest.mark.parametrize("year", [2022, 2023])
def test_historical_cohorts_still_need_both_outcomes(inputs, no_fit, year):
    apps, features, reports = inputs
    identities = {app.application_id for app in apps if app.decision_at.year == year}
    single_class = [
        replace(report, days_past_due=0) if report.application_id in identities else report
        for report in reports
    ]
    with pytest.raises(DataContractError, match="at least 20 loans and both outcomes"):
        preflight_monitor(apps, features, single_class, CUTOFF, CONFIG, data_kind="synthetic")


@pytest.mark.parametrize("size", [19, 20])
def test_current_minimum_is_twenty_and_failure_precedes_model_fitting(inputs, no_fit, size):
    apps, features, reports = inputs
    current = [app for app in apps if app.decision_at == CONFIG.test_start][:size]
    selected = [app for app in apps if app.decision_at < CONFIG.test_start] + current
    identities = {app.application_id for app in selected}
    features = [row for row in features if row.application_id in identities]
    if size == 19:
        for operation in (
            lambda: preflight_monitor(
                selected, features, reports, CUTOFF, CONFIG, data_kind="synthetic"
            ),
            lambda: monitor(selected, features, reports, CUTOFF, CONFIG),
        ):
            with pytest.raises(DataContractError, match="at least 20 visible applications"):
                operation()
    else:
        checked = preflight_monitor(
            selected, features, reports, CUTOFF, CONFIG, data_kind="synthetic"
        )
        assert checked["cohorts"]["current"]["applications"] == 20


def test_future_application_features_are_not_required_before_the_cutoff(inputs, no_fit):
    apps, features, reports = inputs
    future = {app.application_id for app in apps if app.decision_at > CUTOFF}
    visible_features = [row for row in features if row.application_id not in future]
    result = preflight_monitor(
        apps, visible_features, reports, CUTOFF, CONFIG, data_kind="synthetic"
    )
    assert result["cohorts"]["current"]["applications"] == 100
    assert result["current_applications_not_yet_visible"] == 50


def test_missing_visible_features_fail_without_fitting(inputs, no_fit):
    apps, features, reports = inputs
    current = next(app for app in apps if app.decision_at == CONFIG.test_start)
    incomplete = [
        row
        for row in features
        if not (row.application_id == current.application_id and row.name == "utilization")
    ]
    with pytest.raises(DataContractError, match="Missing decision-time utilization"):
        preflight_monitor(apps, incomplete, reports, CUTOFF, CONFIG, data_kind="synthetic")


@pytest.mark.parametrize("cutoff", [date(2024, 6, 30), "2024-10-01", None, datetime(2024, 10, 1)])
def test_invalid_cutoffs_fail_without_fitting(inputs, no_fit, cutoff):
    with pytest.raises(DataContractError, match="Monitor cutoff"):
        preflight_monitor(*inputs, cutoff, CONFIG, data_kind="synthetic")


def test_custom_calendar_and_future_count_are_applied(inputs, no_fit):
    shift = timedelta(days=1461)
    apps, features, reports = inputs
    apps = [replace(row, decision_at=row.decision_at - shift) for row in apps]
    features = [
        replace(row, observed_at=row.observed_at - shift, available_at=row.available_at - shift)
        for row in features
    ]
    reports = [
        replace(row, as_of=row.as_of - shift, available_at=row.available_at - shift)
        for row in reports
    ]
    config = replace(
        CONFIG,
        **{
            name: value - shift for name, value in asdict(CONFIG).items() if isinstance(value, date)
        },
    )
    checked = preflight_monitor(
        apps, features, reports, CUTOFF - shift, config, data_kind="user_supplied"
    )
    assert checked["as_of"] == "2020-10-01" and checked["data_kind"] == "user_supplied"
    assert checked["cohorts"]["current"]["applications"] == 100
    assert checked["current_applications_not_yet_visible"] == 50


def test_cli_synthetic_monitoring_preflight_writes_only_json(tmp_path, monkeypatch, no_fit, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["preflight", "--as-of", "2024-10-01", "--seed", "7", "--per-vintage", "25"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_for_monitoring" and result["data_kind"] == "synthetic"
    assert result["cohorts"]["current"]["applications"] == 100
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cutoff", ["20241001", "2024-W40-2"])
def test_cli_monitoring_preflight_keeps_the_strict_date_contract(cutoff, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["preflight", "--as-of", cutoff])
    assert exc.value.code == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err


def test_cli_supplied_unlabelled_inputs_reach_monitor_and_leave_inputs_unchanged(tmp_path, capsys):
    directory = tmp_path / "inputs"
    initialize_inputs(directory)
    path = directory / "performance.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(row for row in rows if not row["application_id"].startswith("CV2024"))
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    flags = ["--config", str(directory / "cohorts.json")]
    for name in ("applications", "features", "performance"):
        flags += ["--" + name, str(directory / (name + ".csv"))]
    assert main(["preflight", *flags, "--as-of", "2024-10-01"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["cohorts"]["current"]["applications"] == 80
    assert main(["preflight", *flags]) == 2
    assert "test has unlabelled loans" in capsys.readouterr().err
    output = tmp_path / "monitor"
    assert main(["monitor", *flags, "--as-of", "2024-10-01", "--out", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["current_loans"] == 80
    assert main(["verify-monitor", str(output)]) == 0
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
