from __future__ import annotations

import csv
import hashlib
import json
from datetime import date

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditvintage.cli import main
from creditvintage.configuration import load_config, preflight
from creditvintage.core import DataContractError, load_csv_bundle, snapshot
from creditvintage.inputs import INPUT_FILES, check_inputs, initialize_inputs


@pytest.fixture
def case(tmp_path):
    path = tmp_path / "inputs"
    initialize_inputs(path)
    return path


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_complete_example_is_readable_by_actual_loader_and_never_fits(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Input checks must not fit models")

    for model in (LogisticRegression, IsotonicRegression, HistGradientBoostingClassifier):
        monkeypatch.setattr(model, "fit", forbidden)
    path = tmp_path / "case"
    assert main(["init-inputs", "--out", str(path)]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["data_kind"] == "synthetic" and created["model_fitted"] is False
    assert created["input_rows"] == {"applications": 480, "features": 1584, "performance": 2880}
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    assert main(["check-inputs", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_for_evaluation"
    assert result["issue_count"] == 0 and result["model_fitted"] is False
    assert set(result["checks"].values()) == {"passed"}
    actual = preflight(
        *load_csv_bundle(*(path / name for name in INPUT_FILES[:3])),
        load_config(path / "cohorts.json"),
        data_kind="user_supplied",
    )
    assert result["preflight"] == actual
    assert [actual["cohorts"][s]["applications"] for s in ("train", "calibration", "test")] == [
        240,
        120,
        120,
    ]
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    with pytest.raises(DataContractError, match="not overwritten"):
        initialize_inputs(path)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}


def test_collects_independent_field_and_domain_errors_across_every_file(case):
    columns, rows = read(case / "applications.csv")
    rows[0]["principal_cents"] = "-1"
    rows[1]["decision_at"], rows[1]["principal_cents"] = "bad date", "bad integer"
    write(case / "applications.csv", columns, rows)
    columns, rows = read(case / "features.csv")
    rows[0]["value"] = "nan"
    write(case / "features.csv", columns, rows)
    columns, rows = read(case / "performance.csv")
    rows[0]["days_past_due"] = "999"
    write(case / "performance.csv", columns, rows)
    result = check_inputs(case)
    assert result["issue_count"] == 5
    assert {i["file"] for i in result["issues"]} == set(INPUT_FILES[:3])
    assert {i["line"] for i in result["issues"]} == {2, 3}
    assert {i["column"] for i in result["issues"] if i["code"] == "value_type"} == {
        "decision_at",
        "principal_cents",
    }
    assert result["checks"]["relationships"] == "blocked"
    assert result["checks"]["evaluation_cohorts"] == "blocked"


@pytest.mark.parametrize("filename", INPUT_FILES[:3])
def test_missing_files_do_not_stop_diagnosis_of_other_files(case, filename):
    (case / filename).unlink()
    result = check_inputs(case)
    assert result["status"] == "invalid_inputs"
    assert result["issues"][0]["file"] == filename
    assert result["issues"][0]["code"] == "unreadable_file"
    assert result["preflight"] is None


@pytest.mark.parametrize("header", ["", "application_id,application_id", "\ufeffapplication_id"])
def test_empty_duplicate_and_bom_headers_fail_without_guessing(case, header):
    (case / "applications.csv").write_text(header, encoding="utf-8")
    result = check_inputs(case)
    expected = "encoding_bom" if header.startswith("\ufeff") else "columns"
    assert result["issues"][0]["code"] == expected
    assert result["issues"][0]["line"] == 1


def test_duplicate_records_orphan_references_and_bad_report_dates_are_collected(case):
    columns, rows = read(case / "applications.csv")
    rows.append(dict(rows[0]))
    write(case / "applications.csv", columns, rows)
    columns, rows = read(case / "features.csv")
    rows += [dict(rows[0]), {**rows[0], "application_id": "UNKNOWN"}]
    write(case / "features.csv", columns, rows)
    columns, rows = read(case / "performance.csv")
    rows += [dict(rows[0]), {**rows[0], "application_id": "UNKNOWN"}]
    rows[1]["as_of"] = "2022-01-01"
    write(case / "performance.csv", columns, rows)
    result = check_inputs(case)
    assert result["issue_count"] == 6
    assert {i["code"] for i in result["issues"]} == {
        "duplicate_application",
        "ambiguous_feature",
        "unknown_application",
        "performance_date",
        "duplicate_performance",
    }
    assert result["checks"]["evaluation_cohorts"] == "blocked"


def test_locates_every_missing_decision_time_feature_without_using_late_revisions(case):
    _, apps = read(case / "applications.csv")
    first, second = apps[0]["application_id"], apps[10]["application_id"]
    columns, rows = read(case / "features.csv")
    rows = [
        row
        for row in rows
        if not (row["application_id"] == first and row["name"] == "utilization")
        and not (row["application_id"] == second and row["available_at"] <= "2022-01-01")
    ]
    write(case / "features.csv", columns, rows)
    result = check_inputs(case)
    assert result["issue_count"] == 4
    assert {i["code"] for i in result["issues"]} == {"missing_decision_feature"}
    assert {i["line"] for i in result["issues"]} == {2, 12}


def test_reports_all_incomplete_non_events_across_three_cohorts(case):
    apps, feats, reports = load_csv_bundle(*(case / name for name in INPUT_FILES[:3]))
    eligible = [row for row in snapshot(apps, feats, reports, date(2025, 7, 1)) if row.label == 0]
    chosen = {
        next(
            row.application.application_id
            for row in eligible
            if row.application.decision_at.year == year
        )
        for year in (2022, 2023, 2024)
    }
    columns, rows = read(case / "performance.csv")
    removed = set()
    kept = []
    for row in rows:
        if row["application_id"] in chosen and row["application_id"] not in removed:
            removed.add(row["application_id"])
        else:
            kept.append(row)
    write(case / "performance.csv", columns, kept)
    result = check_inputs(case)
    assert result["issue_count"] == 3
    assert {i["stage"] for i in result["issues"]} == {"train", "calibration", "test"}
    assert {i["code"] for i in result["issues"]} == {"missing_performance"}
    assert all(i["file"] == "applications.csv" and i["line"] >= 2 for i in result["issues"])


def test_checks_minimum_size_for_each_cohort_with_actual_evaluation_gate(case):
    columns, apps = read(case / "applications.csv")
    chosen = [
        row
        for year in ("2022", "2023", "2024")
        for row in [a for a in apps if a["decision_at"].startswith(year)][:10]
    ]
    ids = {row["application_id"] for row in chosen}
    write(case / "applications.csv", columns, chosen)
    for filename in ("features.csv", "performance.csv"):
        columns, rows = read(case / filename)
        write(case / filename, columns, [r for r in rows if r["application_id"] in ids])
    result = check_inputs(case)
    assert result["issue_count"] == 3
    assert {i["stage"] for i in result["issues"]} == {"train", "calibration", "test"}
    assert {i["code"] for i in result["issues"]} == {"cohort_contract"}


def test_bad_configuration_still_scans_rows_and_never_claims_readiness(case):
    (case / "cohorts.json").write_text('{"train_start":"2022-01-01"}', encoding="utf-8")
    result = check_inputs(case)
    assert result["status"] == "invalid_inputs" and result["preflight"] is None
    assert result["checks"]["csv_rows"] == "passed"
    assert result["checks"]["evaluation_cohorts"] == "blocked"
    assert result["issues"][0]["code"] == "config"


def test_malformed_csv_and_wrong_width_report_physical_lines(case):
    (case / "features.csv").write_text(
        'application_id,name,value,observed_at,available_at\n"unclosed\n', encoding="utf-8"
    )
    with (case / "performance.csv").open("a", encoding="utf-8", newline="") as handle:
        handle.write("too,few,columns\n")
    result = check_inputs(case)
    assert {(i["file"], i["line"], i["code"]) for i in result["issues"]} == {
        ("features.csv", 2, "csv_quoting"),
        ("performance.csv", 2882, "row_shape"),
    }


def test_unreadable_encoding_does_not_invent_a_line_location(case):
    (case / "features.csv").write_bytes(b"\xff\xfe\x00not UTF-8")
    result = check_inputs(case)
    issue = next(i for i in result["issues"] if i["file"] == "features.csv")
    assert issue["code"] == "encoding" and issue["line"] is None


def test_quoted_multiline_record_reports_its_physical_end_line(case):
    (case / "applications.csv").write_text(
        'application_id,decision_at,principal_cents,segment\n"bad\nid",2022-01-01,100,retail\n',
        encoding="utf-8",
    )
    result = check_inputs(case)
    assert result["issues"][0]["code"] == "record_contract"
    assert result["issues"][0]["line"] == 3


def test_issue_limit_bounds_output_without_hiding_total_or_exit_status(case, capsys):
    columns, rows = read(case / "features.csv")
    for row in rows[:12]:
        row["value"] = "not-a-number"
    write(case / "features.csv", columns, rows)
    assert main(["check-inputs", str(case), "--max-issues", "3"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["issue_count"] == 12 and len(result["issues"]) == 3
    assert result["issues_truncated"] is True
    assert all("not-a-number" not in i["message"] for i in result["issues"])


@pytest.mark.parametrize("limit", [0, -1, 1001, True])
def test_invalid_issue_limit_fails(case, limit):
    with pytest.raises(DataContractError, match="1 to 1000"):
        check_inputs(case, max_issues=limit)


def test_crlf_and_future_excluded_application_follow_actual_preflight(case):
    columns, apps = read(case / "applications.csv")
    apps.append({**apps[0], "application_id": "FUTURE", "decision_at": "2030-01-01"})
    write(case / "applications.csv", columns, apps)
    for name in INPUT_FILES[:3]:
        path = case / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    result = check_inputs(case)
    assert result["status"] == "ready_for_evaluation"
    assert result["preflight"]["applications_outside_windows"] == 1
