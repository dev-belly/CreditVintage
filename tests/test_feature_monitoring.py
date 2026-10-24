from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta

import pytest
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditvintage.artifacts import _hash, _json
from creditvintage.cli import main
from creditvintage.core import Application, DataContractError, FeatureEvent, load_csv_bundle
from creditvintage.feature_monitoring import (
    FEATURE_MONITOR_FILES,
    monitor_features,
    verify_feature_monitor,
    write_feature_monitor,
)
from creditvintage.inputs import initialize_inputs
from creditvintage.model import Config
from creditvintage.sample import generate

CUTOFF = date(2024, 10, 1)


@pytest.fixture
def inputs():
    applications, features = [], []
    for population, decision in (("reference", date(2023, 7, 1)), ("current", date(2024, 7, 1))):
        for i in range(20):
            current = population == "current"
            app = Application(
                f"{population}-{i:02}",
                decision,
                20000 if current else 10000,
                "small_business" if current and i >= 10 else "retail",
            )
            applications.append(app)
            for name, value in (
                ("utilization", float(i >= (5 if current else 10))),
                ("debt_to_income", 0.3),
                ("prior_delinquencies", 0),
            ):
                features.append(FeatureEvent(app.application_id, name, value, decision, decision))
    return applications, features


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Feature monitoring must not fit a model")

    for cls in (LogisticRegression, IsotonicRegression):
        monkeypatch.setattr(cls, "fit", forbidden)


def test_hand_calculated_distances_and_cumulative_counts(inputs):
    before = [[asdict(row) for row in records] for records in inputs]
    result = monitor_features(*inputs, CUTOFF)
    metrics = {item["feature"]: item for item in result.summary["features"]}
    # F_ref(0)=10/20 and F_cur(0)=5/20; all remaining mass is at 1.
    assert metrics["utilization"]["distance"] == 0.25
    assert metrics["utilization"]["at_value"] == 0
    assert metrics["utilization"]["reference_cdf_at_value"] == 0.5
    assert metrics["utilization"]["current_cdf_at_value"] == 0.25
    assert metrics["debt_to_income"]["distance"] == 0
    assert metrics["prior_delinquencies"]["distance"] == 0
    assert metrics["principal_cents"]["distance"] == 1
    # Reference is all retail, current is 50/50: TV = (0.5 + 0.5)/2.
    assert metrics["segment"]["distance"] == 0.5
    counts = [row for row in result.distributions if row["feature"] == "utilization"]
    assert [(row["value"], row["reference_count"], row["current_count"]) for row in counts] == [
        (0, 10, 5),
        (1, 20, 20),
    ]
    assert result.summary["model_fitted"] is False
    assert result.summary["outcomes_required"] is False
    assert "performance" not in result.inputs
    assert "label" not in result.current[0]
    assert before == [[asdict(row) for row in records] for records in inputs]


@pytest.mark.parametrize(
    "reference,current,expected", [(0.2, 0.8, 1), (0.8, 0.2, 1), (0.2, 0.2, 0)]
)
def test_constant_numeric_populations_detect_both_shift_directions(
    inputs, reference, current, expected
):
    applications, features = inputs
    features = [
        replace(event, value=reference if event.application_id.startswith("reference") else current)
        if event.name == "utilization"
        else event
        for event in features
    ]
    result = monitor_features(applications, features, CUTOFF)
    assert result.summary["features"][0]["distance"] == expected


def test_principal_comparisons_preserve_integer_differences_above_binary64_precision(inputs):
    applications, features = inputs
    applications = [
        replace(
            app, principal_cents=2**53 + (1 if app.application_id.startswith("reference") else 2)
        )
        for app in applications
    ]
    result = monitor_features(applications, features, CUTOFF)
    metric = next(
        item for item in result.summary["features"] if item["feature"] == "principal_cents"
    )
    assert metric["distance"] == 1 and metric["at_value"] == 2**53 + 1
    assert result.reference[0]["principal_cents"] == 2**53 + 1


@pytest.mark.parametrize("seed", [7, 19, 20260927])
def test_numeric_distances_agree_with_the_independent_scipy_implementation(seed):
    applications, features, _ = generate(seed=seed, per_vintage=20)
    result = monitor_features(applications, features, CUTOFF)
    for item in result.summary["features"]:
        if item["method"] == "empirical_ks_distance":
            name = item["feature"]
            expected = ks_2samp(
                [row[name] for row in result.reference], [row[name] for row in result.current]
            ).statistic
            assert item["distance"] == round(float(expected), 6)


def test_future_features_and_late_corrections_cannot_change_decision_time_values(inputs):
    applications, features = inputs
    original = monitor_features(*inputs, CUTOFF)
    app = next(app for app in applications if app.application_id == "current-00")
    late = FeatureEvent(app.application_id, "utilization", 1, app.decision_at, date(2024, 8, 1))
    future = Application("future", date(2024, 12, 1), 30000, "retail")
    actual = monitor_features(applications + [future], features + [late], CUTOFF)
    assert actual.current == original.current
    assert actual.summary["features"] == original.summary["features"]
    assert actual.summary["current_applications_not_yet_visible"] == 1
    assert actual.summary["input_sha256"] != original.summary["input_sha256"]


def test_small_months_have_no_distance_instead_of_zero(inputs):
    applications, features = inputs
    moved = {
        app.application_id for app in applications if app.application_id.startswith("current-1")
    }
    applications = [
        replace(app, decision_at=date(2024, 8, 1)) if app.application_id in moved else app
        for app in applications
    ]
    features = [
        replace(event, observed_at=date(2024, 8, 1), available_at=date(2024, 8, 1))
        if event.application_id in moved
        else event
        for event in features
    ]
    months = monitor_features(applications, features, CUTOFF).summary["by_vintage"]
    assert [row["applications"] for row in months] == [10, 10]
    assert all(
        row["distances"] is None and row["status"] == "insufficient_sample" for row in months
    )


@pytest.mark.parametrize("population,size", [("reference", 19), ("current", 19), ("current", 20)])
def test_population_minimum_includes_exactly_twenty(inputs, population, size):
    applications, features = inputs
    retained = [
        app
        for app in applications
        if not app.application_id.startswith(population) or int(app.application_id[-2:]) < size
    ]
    identifiers = {app.application_id for app in retained}
    features = [event for event in features if event.application_id in identifiers]
    if size == 19:
        with pytest.raises(DataContractError, match="at least 20"):
            monitor_features(retained, features, CUTOFF)
    else:
        assert monitor_features(retained, features, CUTOFF).summary["current_applications"] == 20


@pytest.mark.parametrize("cutoff", [None, "2024-10-01", datetime(2024, 10, 1), date(2024, 6, 30)])
def test_invalid_cutoffs_fail_without_fitting(inputs, cutoff):
    with pytest.raises(DataContractError, match="Feature monitor cutoff"):
        monitor_features(*inputs, cutoff)


@pytest.mark.parametrize("problem", ["missing", "duplicate_revision", "unknown_application"])
def test_snapshot_validation_is_preserved(inputs, problem):
    applications, features = inputs
    if problem == "missing":
        features = features[1:]
    elif problem == "duplicate_revision":
        features = features + [features[0]]
    else:
        features = features + [replace(features[0], application_id="unknown")]
    with pytest.raises(DataContractError):
        monitor_features(applications, features, CUTOFF)


def test_custom_calendar_and_row_reordering_preserve_results(inputs):
    applications, features = inputs
    offset = timedelta(days=1461)
    config = Config()
    shifted_config = replace(
        config,
        **{
            name: value - offset
            for name, value in asdict(config).items()
            if isinstance(value, date)
        },
    )
    shifted = monitor_features(
        [replace(app, decision_at=app.decision_at - offset) for app in reversed(applications)],
        [
            replace(
                event,
                observed_at=event.observed_at - offset,
                available_at=event.available_at - offset,
            )
            for event in reversed(features)
        ],
        CUTOFF - offset,
        shifted_config,
    )
    expected = monitor_features(*inputs, CUTOFF)
    assert shifted.summary["features"] == expected.summary["features"]
    assert shifted.summary["as_of"] == "2020-10-01"
    reordered = monitor_features(list(reversed(applications)), list(reversed(features)), CUTOFF)
    assert reordered == expected


def test_complete_report_replays_from_events_and_refuses_existing_destination(inputs, tmp_path):
    report = tmp_path / "report"
    result = monitor_features(*inputs, CUTOFF, data_kind="synthetic")
    write_feature_monitor(result, report)
    verified = verify_feature_monitor(report)
    assert verified["status"] == "verified" and verified["monitored_inputs"] == 5
    assert verified["reference_applications"] == verified["current_applications"] == 20
    assert set(path.name for path in report.iterdir()) == {*FEATURE_MONITOR_FILES, "manifest.json"}
    before = {path.name: path.read_bytes() for path in report.iterdir()}
    with pytest.raises(DataContractError, match="already exists"):
        write_feature_monitor(result, report)
    assert before == {path.name: path.read_bytes() for path in report.iterdir()}
    page = (report / "index.html").read_text(encoding="utf-8")
    assert "@@" not in page and "Synthetic example" in page
    assert "KS distance" in page and "not statistical significance" in page


@pytest.mark.parametrize("filename", FEATURE_MONITOR_FILES)
def test_rehashing_false_evidence_cannot_bypass_event_replay(inputs, tmp_path, filename):
    write_feature_monitor(monitor_features(*inputs, CUTOFF), tmp_path / "report")
    report = tmp_path / "report"
    path = report / filename
    if filename == "inputs.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        content["features"][0]["value"] = 0.9
        _json(path, content)
    elif filename == "monitor.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        content["features"][0]["distance"] = 0.99
        _json(path, content)
    elif filename.endswith(".csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        rows[1][-1] = "999"
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
    else:
        path.write_text(
            path.read_text(encoding="utf-8").replace("Which inputs", "Unchanged inputs"),
            encoding="utf-8",
        )
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_sha256"][filename] = _hash(path)
    _json(report / "manifest.json", manifest)
    with pytest.raises(DataContractError, match="Cannot verify feature monitor"):
        verify_feature_monitor(report)


def test_cli_only_needs_two_csvs_and_does_not_load_performance(tmp_path, capsys):
    source = tmp_path / "inputs"
    initialize_inputs(source)
    (source / "performance.csv").unlink()
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    destination = tmp_path / "report"
    flags = [
        "--applications",
        str(source / "applications.csv"),
        "--features",
        str(source / "features.csv"),
        "--config",
        str(source / "cohorts.json"),
    ]
    assert (
        main(["monitor-features", *flags, "--as-of", "2024-10-01", "--out", str(destination)]) == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["reference_applications"] == 120 and result["current_applications"] == 80
    assert result["current_applications_not_yet_visible"] == 40
    assert result["data_kind"] == "user_supplied" and result["model_fitted"] is False
    assert main(["verify-feature-monitor", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}


def test_cli_rejects_incomplete_or_invalid_inputs_before_output(tmp_path, capsys):
    source = tmp_path / "inputs"
    initialize_inputs(source)
    destination = tmp_path / "report"
    flags = ["--as-of", "2024-10-01", "--out", str(destination)]
    assert (
        main(["monitor-features", *flags, "--applications", str(source / "applications.csv")]) == 2
    )
    assert "both application and feature" in capsys.readouterr().err
    feature_path = source / "features.csv"
    with feature_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    with feature_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([rows[0], *rows[2:]])
    assert (
        main(
            [
                "monitor-features",
                *flags,
                "--applications",
                str(source / "applications.csv"),
                "--features",
                str(feature_path),
            ]
        )
        == 2
    )
    assert "Missing decision-time" in capsys.readouterr().err and not destination.exists()


def test_shared_three_table_loader_still_reads_outcome_records(tmp_path):
    initialize_inputs(tmp_path / "inputs")
    apps, features, outcomes = load_csv_bundle(
        *[
            tmp_path / "inputs" / name
            for name in ("applications.csv", "features.csv", "performance.csv")
        ]
    )
    assert len(apps) == 480 and features and outcomes
