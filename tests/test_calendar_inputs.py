from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from creditvintage.cli import main
from creditvintage.core import DataContractError, load_csv_bundle
from creditvintage.inputs import INPUT_FILES, check_inputs, initialize_inputs


@pytest.fixture
def case(tmp_path):
    directory = tmp_path / "inputs"
    initialize_inputs(directory)
    return directory


def change_date(directory, filename, column, form):
    path = directory / filename
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    original = date.fromisoformat(rows[0][column])
    year, week, weekday = original.isocalendar()
    rows[0][column] = (
        original.isoformat().replace("-", "")
        if form == "compact"
        else f"{year:04d}-W{week:02d}-{weekday}"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("filename", "column"),
    [
        ("applications.csv", "decision_at"),
        ("features.csv", "observed_at"),
        ("features.csv", "available_at"),
        ("performance.csv", "as_of"),
        ("performance.csv", "available_at"),
    ],
)
@pytest.mark.parametrize("form", ["compact", "week"])
def test_diagnostics_and_loader_reject_alternate_date_formats(case, filename, column, form):
    change_date(case, filename, column, form)
    before = {path.name: path.read_bytes() for path in case.iterdir()}
    result = check_inputs(case)
    assert result["status"] == "invalid_inputs"
    assert result["issue_count"] == 1
    issue = result["issues"][0]
    assert (issue["file"], issue["line"], issue["column"], issue["code"]) == (
        filename,
        2,
        column,
        "value_type",
    )
    assert "YYYY-MM-DD" in issue["message"]
    assert result["checks"]["evaluation_cohorts"] == "blocked"
    with pytest.raises(DataContractError, match="YYYY-MM-DD"):
        load_csv_bundle(*(case / name for name in INPUT_FILES[:3]))
    assert before == {path.name: path.read_bytes() for path in case.iterdir()}


def test_date_diagnostics_collect_every_file_and_keep_json_exit_status(case, capsys):
    for filename, column in (
        ("applications.csv", "decision_at"),
        ("features.csv", "observed_at"),
        ("performance.csv", "available_at"),
    ):
        change_date(case, filename, column, "compact")
    assert main(["check-inputs", str(case)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["issue_count"] == 3
    assert {issue["file"] for issue in result["issues"]} == set(INPUT_FILES[:3])
    assert result["model_fitted"] is False and result["preflight"] is None


@pytest.mark.parametrize("command", ["preflight", "run", "benchmark", "monitor"])
def test_each_event_command_rejects_bad_dates_without_output(case, tmp_path, capsys, command):
    change_date(case, "applications.csv", "decision_at", "compact")
    arguments = [command, "--config", str(case / "cohorts.json")]
    for filename in INPUT_FILES[:3]:
        arguments.extend(["--" + filename.removesuffix(".csv"), str(case / filename)])
    output = tmp_path / "report"
    if command != "preflight":
        arguments.extend(["--out", str(output)])
    if command == "monitor":
        arguments.extend(["--as-of", "2024-10-01"])
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert "YYYY-MM-DD" in captured.err and "decision_at" in captured.err
    assert "Traceback" not in captured.err and not output.exists()


@pytest.mark.parametrize("cutoff", ["20241001", "2024-W40-2"])
def test_monitor_rejects_alternate_cutoff_before_creating_output(tmp_path, capsys, cutoff):
    output = tmp_path / "monitor"
    with pytest.raises(SystemExit) as exc:
        main(["monitor", "--as-of", cutoff, "--out", str(output)])
    assert exc.value.code == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err
    assert not output.exists()
