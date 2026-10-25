from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditvintage.artifacts import _hash, _json
from creditvintage.cli import main
from creditvintage.core import FEATURE_NAMES, Application, DataContractError, FeatureEvent, snapshot
from creditvintage.feature_monitoring import monitor_features, write_feature_monitor
from creditvintage.feature_quality import (
    QUALITY_FILES,
    monitor_feature_quality,
    verify_feature_quality,
    write_feature_quality,
)
from creditvintage.inputs import initialize_inputs
from creditvintage.model import Config
from creditvintage.sample import generate, generate_feature_quality

CUTOFF = date(2024, 10, 1)
DECISION = date(2024, 7, 1)


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Feature coverage diagnostics must never fit models")

    for model in (LogisticRegression, IsotonicRegression):
        monkeypatch.setattr(model, "fit", forbidden)


@pytest.fixture
def inputs():
    apps = [Application("reference", date(2023, 7, 1), 10000, "retail")]
    apps += [
        Application(name, DECISION, 20000, "retail")
        for name in ("available", "absent", "late", "post", "future-record")
    ]
    apps.append(Application("future-app", date(2024, 11, 1), 20000, "retail"))
    features = []
    for app in apps[:-1]:
        for name, value in zip(FEATURE_NAMES, (0.2, 0.3, 0), strict=True):
            if name == "utilization" and app.application_id == "absent":
                continue
            observed, available = (
                app.decision_at - timedelta(days=5),
                app.decision_at - timedelta(days=1),
            )
            if name == "utilization":
                if app.application_id == "late":
                    available = DECISION + timedelta(days=2)
                elif app.application_id == "post":
                    observed, available = DECISION + timedelta(days=1), DECISION + timedelta(days=3)
                elif app.application_id == "future-record":
                    available = CUTOFF + timedelta(days=1)
            features.append(FeatureEvent(app.application_id, name, value, observed, available))
    return apps, features


def test_exact_coverage_denominators_reasons_and_delay(inputs):
    before = [[asdict(row) for row in rows] for rows in inputs]
    result = monitor_feature_quality(*inputs, CUTOFF)
    current = result.summary["populations"]["current"]
    assert current == {
        "applications": 5,
        "complete_applications": 1,
        "incomplete_applications": 4,
        "applications_not_yet_visible": 1,
        "status": "incomplete_coverage",
    }
    assert result.summary["populations"]["reference"]["complete_applications"] == 1
    group = next(
        row
        for row in result.summary["coverage"]
        if row["population"] == "current" and row["feature"] == "utilization"
    )
    assert [
        group[key]
        for key in (
            "applications",
            "available",
            "late_for_decision",
            "observed_after_decision",
            "no_visible_event",
        )
    ] == [5, 1, 1, 1, 2]
    assert group["coverage_rate"] == 0.2
    assert group["median_availability_lag_days"] == group["max_availability_lag_days"] == 4
    assert group["median_late_days"] == group["max_late_days"] == 2
    assert group["median_observation_age_days"] == 5
    assert len(result.availability) == 18  # six visible applications, three expected slots each.
    assert result.summary["model_fitted"] is False and result.summary["outcomes_required"] is False
    assert "performance" not in result.inputs
    assert before == [[asdict(row) for row in rows] for rows in inputs]


def test_unknown_events_do_not_leak_from_after_report_cutoff(inputs):
    apps, features = inputs
    before = monitor_feature_quality(apps, features, CUTOFF)
    hidden = [event for event in features if event.available_at > CUTOFF]
    after = monitor_feature_quality(
        apps, [event for event in features if event not in hidden], CUTOFF
    )
    assert after.availability == before.availability and after.coverage == before.coverage
    assert after.summary["input_sha256"] != before.summary["input_sha256"]
    later = monitor_feature_quality(apps, features, CUTOFF + timedelta(days=1))
    slot = next(
        row
        for row in later.availability
        if row["application_id"] == "future-record" and row["feature"] == "utilization"
    )
    assert slot["status"] == "late_for_decision" and slot["late_days"] == 93
    assert later.summary["populations"]["current"]["complete_applications"] == 1


def test_first_visible_late_arrival_is_not_the_latest_revision(inputs):
    apps, features = inputs
    features.append(
        FeatureEvent("late", "utilization", 0.4, DECISION, DECISION + timedelta(days=9))
    )
    r = monitor_feature_quality(apps, features, CUTOFF)
    slot = next(
        row
        for row in r.availability
        if row["application_id"] == "late" and row["feature"] == "utilization"
    )
    assert slot["late_days"] == 2 and slot["value"] == 0.2


def test_known_predecision_observation_takes_priority_over_a_postdecision_record(inputs):
    apps, features = inputs
    features.append(
        FeatureEvent(
            "late", "utilization", 0.9, DECISION + timedelta(days=1), DECISION + timedelta(days=1)
        )
    )
    result = monitor_feature_quality(apps, features, CUTOFF)
    slot = next(
        row
        for row in result.availability
        if row["application_id"] == "late" and row["feature"] == "utilization"
    )
    assert slot["status"] == "late_for_decision" and slot["late_days"] == 2


def test_one_application_with_two_missing_inputs_is_not_counted_twice(inputs):
    apps, features = inputs
    features = [
        event
        for event in features
        if not (event.application_id == "absent" and event.name == "debt_to_income")
    ]
    result = monitor_feature_quality(apps, features, CUTOFF)
    assert result.summary["populations"]["current"]["incomplete_applications"] == 4
    assert sum(row["status"] != "available" for row in result.availability) == 5


def test_v05_feature_monitor_bytes_match_the_public_wheel_fixture(tmp_path):
    apps, features, _ = generate(seed=7, per_vintage=10)
    result = monitor_features(apps, features, CUTOFF, Config(seed=7), data_kind="synthetic")
    manifest = write_feature_monitor(result, tmp_path / "legacy-compatible")
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "feature_monitor_v05_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest == expected


def test_later_correction_does_not_mark_an_on_time_input_as_late(inputs):
    apps, features = inputs
    expected = monitor_feature_quality(apps, features, CUTOFF)
    features = features + [
        FeatureEvent("available", "utilization", 0.8, DECISION, DECISION + timedelta(days=10))
    ]
    actual = monitor_feature_quality(apps, features, CUTOFF)
    assert actual.availability == expected.availability and actual.coverage == expected.coverage


def test_reason_changes_only_when_a_late_event_is_visible(inputs):
    before = monitor_feature_quality(*inputs, DECISION)
    later = monitor_feature_quality(*inputs, DECISION + timedelta(days=2))

    def slot(result):
        return next(
            row
            for row in result.availability
            if row["application_id"] == "late" and row["feature"] == "utilization"
        )

    assert slot(before)["status"] == "no_visible_event"
    assert slot(later)["status"] == "late_for_decision"
    assert before.summary["populations"] == later.summary["populations"]


def test_monthly_counts_do_not_discard_small_groups_or_average_rates(inputs):
    apps, features = inputs
    for i in range(2):
        app = Application(f"aug-{i}", date(2024, 8, 1), 20000, "retail")
        apps.append(app)
        for name, value in zip(FEATURE_NAMES, (0.2, 0.3, 0), strict=True):
            features.append(
                FeatureEvent(app.application_id, name, value, app.decision_at, app.decision_at)
            )
    r = monitor_feature_quality(apps, features, CUTOFF)
    rows = [
        row
        for row in r.coverage
        if row["population"] == "current" and row["feature"] == "utilization"
    ]
    assert [(row["vintage"], row["applications"], row["available"]) for row in rows] == [
        ("all", 7, 3),
        ("2024-07", 5, 1),
        ("2024-08", 2, 2),
    ]
    assert rows[0]["coverage_rate"] == round(3 / 7, 8)
    assert rows[2]["median_availability_lag_days"] == 0


def test_zero_populations_have_unknown_rates_and_no_false_complete_status(tmp_path):
    r = monitor_feature_quality([], [], CUTOFF)
    assert r.availability == [] and len(r.coverage) == 6
    assert all(row["applications"] == 0 and row["coverage_rate"] is None for row in r.coverage)
    assert all(row["median_late_days"] is None for row in r.coverage)
    assert all(
        pop["status"] == "no_visible_applications" for pop in r.summary["populations"].values()
    )
    write_feature_quality(r, tmp_path / "empty")
    assert verify_feature_quality(tmp_path / "empty")["feature_slots"] == 0
    page = (tmp_path / "empty" / "index.html").read_text(encoding="utf-8")
    assert "No applications" in page and "0/0" not in page


@pytest.mark.parametrize("invalid", [None, "2024-10-01", datetime(2024, 10, 1), date(2024, 6, 30)])
def test_cutoffs_keep_the_calendar_contract(inputs, invalid):
    with pytest.raises(DataContractError, match="Feature-quality cutoff"):
        monitor_feature_quality(*inputs, invalid)


@pytest.mark.parametrize("problem", ["duplicate_app", "duplicate_revision", "unknown_id"])
def test_identity_and_revision_errors_still_fail(inputs, problem):
    apps, features = inputs
    if problem == "duplicate_app":
        apps = apps + [apps[0]]
    elif problem == "duplicate_revision":
        features = features + [features[0]]
    else:
        features = features + [replace(features[0], application_id="unknown")]
    with pytest.raises(DataContractError):
        monitor_feature_quality(apps, features, CUTOFF)


def test_complete_data_matches_strict_snapshot_and_incomplete_data_still_fails(inputs):
    apps, features, _ = generate(seed=7, per_vintage=10)
    result = monitor_feature_quality(apps, features, CUTOFF)
    snapshots = {
        row.application.application_id: row.features for row in snapshot(apps, features, [], CUTOFF)
    }
    for row in result.availability:
        assert row["status"] == "available"
        assert row["value"] == snapshots[row["application_id"]][FEATURE_NAMES.index(row["feature"])]
    with pytest.raises(DataContractError, match="Missing decision-time"):
        snapshot(*inputs, [], CUTOFF)
    broken_apps, broken_features = generate_feature_quality(per_vintage=10)
    with pytest.raises(DataContractError, match="Missing decision-time"):
        monitor_features(broken_apps, broken_features, CUTOFF)


def test_row_reordering_and_custom_calendar_preserve_coverage(inputs):
    apps, features = inputs
    expected = monitor_feature_quality(apps, features, CUTOFF)
    assert (
        monitor_feature_quality(list(reversed(apps)), list(reversed(features)), CUTOFF) == expected
    )
    offset = timedelta(days=1461)
    config = Config()
    shifted = replace(
        config,
        **{
            name: value - offset
            for name, value in asdict(config).items()
            if isinstance(value, date)
        },
    )
    actual = monitor_feature_quality(
        [replace(app, decision_at=app.decision_at - offset) for app in apps],
        [
            replace(
                event,
                observed_at=event.observed_at - offset,
                available_at=event.available_at - offset,
            )
            for event in features
        ],
        CUTOFF - offset,
        shifted,
    )
    assert actual.summary["populations"] == expected.summary["populations"]
    assert actual.summary["coverage"] == expected.summary["coverage"]


def test_report_replays_incomplete_coverage_and_preserves_existing_destination(inputs, tmp_path):
    result = monitor_feature_quality(*inputs, CUTOFF, data_kind="synthetic")
    target = tmp_path / "report"
    write_feature_quality(result, target)
    checked = verify_feature_quality(target)
    assert checked["status"] == "verified" and checked["feature_slots"] == 18
    assert checked["populations"]["current"]["incomplete_applications"] == 4
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(DataContractError, match="already exists"):
        write_feature_quality(result, target)
    assert before == {path.name: path.read_bytes() for path in target.iterdir()}
    assert set(before) == {*QUALITY_FILES, "manifest.json"}
    page = (target / "index.html").read_text(encoding="utf-8")
    assert "@@" not in page and "Synthetic example" in page and "Showing 4 of 4" in page


@pytest.mark.parametrize("filename", QUALITY_FILES)
def test_rehashed_false_coverage_events_ledger_and_page_are_rejected(inputs, tmp_path, filename):
    target = tmp_path / "report"
    write_feature_quality(monitor_feature_quality(*inputs, CUTOFF), target)
    path = target / filename
    if filename == "inputs.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["features"][0]["available_at"] = "2024-10-02"
        _json(path, value)
    elif filename == "quality.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["populations"]["current"]["complete_applications"] = 5
        _json(path, value)
    elif filename.endswith(".csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        rows[1][-1] = "999"
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
    else:
        path.write_text(
            path.read_text(encoding="utf-8").replace("What was missing", "Nothing was missing"),
            encoding="utf-8",
        )
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_sha256"][filename] = _hash(path)
    _json(target / "manifest.json", manifest)
    with pytest.raises(DataContractError, match="Cannot verify feature quality"):
        verify_feature_quality(target)


def test_cli_builtin_case_has_four_deliberate_gaps(tmp_path, capsys):
    target = tmp_path / "report"
    assert (
        main(
            [
                "monitor-feature-quality",
                "--as-of",
                "2024-10-01",
                "--per-vintage",
                "20",
                "--out",
                str(target),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["data_kind"] == "synthetic" and result["model_fitted"] is False
    assert result["populations"]["reference"]["applications"] == 120
    assert result["populations"]["current"]["applications"] == 80
    assert result["populations"]["current"]["complete_applications"] == 76
    assert result["populations"]["current"]["applications_not_yet_visible"] == 40
    assert main(["verify-feature-quality", str(target)]) == 0


def test_supplied_two_csvs_can_diagnose_a_gap_without_performance(tmp_path, capsys):
    root = tmp_path / "inputs"
    initialize_inputs(root)
    (root / "performance.csv").unlink()
    path = root / "features.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = [
            row
            for row in reader
            if (row["application_id"], row["name"]) != ("CV202407-0000", "utilization")
        ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    flags = [
        "--applications",
        str(root / "applications.csv"),
        "--features",
        str(path),
        "--config",
        str(root / "cohorts.json"),
        "--as-of",
        "2024-10-01",
    ]
    assert main(["monitor-feature-quality", *flags, "--out", str(tmp_path / "quality")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data_kind"] == "user_supplied"
    assert result["populations"]["current"]["complete_applications"] == 79
    assert main(["monitor-features", *flags, "--out", str(tmp_path / "distribution")]) == 2
    assert "Missing decision-time" in capsys.readouterr().err
    assert not (tmp_path / "distribution").exists()
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
