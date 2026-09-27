from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from creditvintage.artifacts import verify_result, write_result
from creditvintage.cli import main
from creditvintage.core import (
    REPORT_DAYS,
    Application,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    load_csv_bundle,
    snapshot,
)
from creditvintage.model import Config, evaluate
from creditvintage.sample import generate

DECISION = date(2022, 1, 1)


def one_loan() -> tuple[list[Application], list[FeatureEvent], list[PerformanceEvent]]:
    apps = [Application("loan-1", DECISION, 100_000, "retail")]
    features = [
        FeatureEvent("loan-1", name, value, DECISION - timedelta(days=5), DECISION)
        for name, value in (
            ("utilization", 0.2),
            ("debt_to_income", 0.3),
            ("prior_delinquencies", 0.0),
        )
    ]
    reports = [
        PerformanceEvent(
            "loan-1", DECISION + timedelta(days=day), DECISION + timedelta(days=day + 2), 0
        )
        for day in REPORT_DAYS
    ]
    return apps, features, reports


def test_feature_publication_after_decision_cannot_change_snapshot() -> None:
    apps, features, reports = one_loan()
    cutoff = DECISION + timedelta(days=182)
    before = snapshot(apps, features, reports, cutoff)[0]
    features.append(FeatureEvent("loan-1", "utilization", 0.99, DECISION, cutoff))
    after = snapshot(apps, features, reports, cutoff)[0]
    assert before.features == after.features == (0.2, 0.3, 0.0)


def test_full_horizon_and_reporting_coverage_are_required() -> None:
    apps, features, reports = one_loan()
    early = snapshot(apps, features, reports, DECISION + timedelta(days=181))[0]
    assert (early.label, early.status) == (None, "not_mature")
    mature = snapshot(apps, features, reports, DECISION + timedelta(days=182))[0]
    assert (mature.label, mature.status) == (0, "observed_nondefault")
    reports.pop(2)
    missing = snapshot(apps, features, reports, DECISION + timedelta(days=182))[0]
    assert (missing.label, missing.status) == (None, "missing_performance")


def test_late_performance_revision_is_excluded_until_available() -> None:
    apps, features, reports = one_loan()
    cutoff = DECISION + timedelta(days=182)
    reports.append(
        PerformanceEvent("loan-1", DECISION + timedelta(days=90), cutoff + timedelta(days=1), 90)
    )
    assert snapshot(apps, features, reports, cutoff)[0].label == 0
    assert snapshot(apps, features, reports, cutoff + timedelta(days=1))[0].label == 1


def test_confirmed_default_is_positive_even_with_missing_other_report() -> None:
    apps, features, reports = one_loan()
    reports.pop(0)
    reports[1] = replace(reports[1], days_past_due=90)
    row = snapshot(apps, features, reports, DECISION + timedelta(days=182))[0]
    assert (row.label, row.status) == (1, "default")


def test_ambiguous_revisions_and_duplicate_applications_fail() -> None:
    apps, features, reports = one_loan()
    with pytest.raises(DataContractError, match="Duplicate application"):
        snapshot(apps + apps, features, reports, DECISION + timedelta(days=182))
    with pytest.raises(DataContractError, match="Ambiguous"):
        snapshot(apps, features + [features[0]], reports, DECISION + timedelta(days=182))


def test_invalid_amount_dpd_and_spreadsheet_formula_id_are_rejected() -> None:
    with pytest.raises(DataContractError, match="integer cents"):
        Application("fractional", DECISION, 100.5, "retail")
    with pytest.raises(DataContractError, match="safe ASCII"):
        Application("=1+1", DECISION, 100, "retail")
    with pytest.raises(DataContractError, match="DPD"):
        PerformanceEvent("loan-1", DECISION, DECISION, 90.5)


def test_csv_contract_rejects_duplicate_headers_and_extra_fields(tmp_path) -> None:
    apps = tmp_path / "applications.csv"
    feats = tmp_path / "features.csv"
    perf = tmp_path / "performance.csv"
    apps.write_text(
        "application_id,decision_at,principal_cents,segment,segment\n"
        "loan-1,2022-01-01,10000,retail,retail\n"
    )
    feats.write_text("application_id,name,value,observed_at,available_at\n")
    perf.write_text("application_id,as_of,available_at,days_past_due\n")
    with pytest.raises(DataContractError, match="Unexpected CSV columns"):
        load_csv_bundle(apps, feats, perf)
    apps.write_text(
        "application_id,decision_at,principal_cents,segment\n"
        "loan-1,2022-01-01,10000,retail,unexpected\n"
    )
    with pytest.raises(DataContractError, match="Malformed CSV row"):
        load_csv_bundle(apps, feats, perf)


def test_chronology_requires_maturity_gaps() -> None:
    with pytest.raises(DataContractError, match="maturation gaps"):
        Config(calibration_start=date(2023, 1, 1))
    with pytest.raises(DataContractError, match="maturation gaps"):
        Config(test_start=date(2024, 1, 1))


def test_synthetic_run_is_deterministic_and_conserves_test_counts() -> None:
    data = generate(seed=7, per_vintage=25)
    first = evaluate(*data, Config(seed=7))
    second = evaluate(*generate(seed=7, per_vintage=25), Config(seed=7))
    assert first == second
    assert first.summary["cohorts"]["train"]["loans"] == 300
    assert first.summary["cohorts"]["calibration"]["loans"] == 150
    assert first.summary["cohorts"]["test"]["loans"] == 150
    assert sum(bucket["loans"] for bucket in first.deciles) == 150
    assert sum(vintage["loans"] for vintage in first.vintages) == 150
    assert sum(row["review_selected"] for row in first.predictions) == 15
    assert first.summary["test_metrics"]["constant_train_rate"]["roc_auc"] == 0.5


def test_missing_train_outcome_fails_instead_of_becoming_good() -> None:
    apps, features, reports = generate(seed=7, per_vintage=25)
    cohort = snapshot(apps, features, reports, date(2023, 7, 1))
    nondefault = next(row.application for row in cohort if row.label == 0)
    reports = [
        report
        for report in reports
        if not (
            report.application_id == nondefault.application_id
            and report.as_of == nondefault.decision_at + timedelta(days=30)
        )
    ]
    with pytest.raises(DataContractError, match="train has unlabelled loans"):
        evaluate(apps, features, reports, Config(seed=7))


def test_export_verifier_recomputes_metrics_even_if_hashes_are_rewritten(tmp_path) -> None:
    result = evaluate(*generate(seed=7, per_vintage=25), Config(seed=7))
    write_result(result, tmp_path)
    assert verify_result(tmp_path)["loans"] == 150
    predictions = tmp_path / "predictions.csv"
    with predictions.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    assert columns is not None
    rows[0]["calibrated_pd"] = "0.999999"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DataContractError, match="digest mismatch"):
        verify_result(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["predictions.csv"] = hashlib.sha256(
        predictions.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataContractError, match="Metrics disagree"):
        verify_result(tmp_path)


def test_cli_run_verify_and_partial_input_failure(tmp_path, capsys) -> None:
    output = tmp_path / "run"
    assert main(["run", "--seed", "7", "--per-vintage", "25", "--out", str(output)]) == 0
    capsys.readouterr()
    assert (output / "index.html").exists()
    assert main(["verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["loans"] == 150
    assert main(["run", "--applications", "only-one.csv"]) == 2
    assert "all three CSV" in capsys.readouterr().err


def test_supplied_csv_path_matches_same_synthetic_events(tmp_path, capsys) -> None:
    generated = generate(seed=7, per_vintage=25)
    paths = [tmp_path / name for name in ("applications.csv", "features.csv", "performance.csv")]
    for path, items in zip(paths, generated, strict=True):
        with path.open("w", newline="", encoding="utf-8") as handle:
            rows = [asdict(item) for item in items]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output = tmp_path / "external"
    assert (
        main(
            [
                "run",
                "--seed",
                "7",
                "--applications",
                str(paths[0]),
                "--features",
                str(paths[1]),
                "--performance",
                str(paths[2]),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    expected_sha = evaluate(*generated, Config(seed=7)).input_sha256
    assert verify_result(output)["input_sha256"] == expected_sha
    assert json.loads((output / "summary.json").read_text())["data_kind"] == "user_supplied"
    assert "User-supplied data" in (output / "index.html").read_text()
