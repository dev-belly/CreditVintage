"""Reject unsupported numbers before readiness, fitting or report creation."""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from creditvintage.cli import main
from creditvintage.core import Application, DataContractError, FeatureEvent
from creditvintage.inputs import INPUT_FILES, check_inputs, initialize_inputs
from creditvintage.model import Config


@pytest.mark.parametrize("principal", [0, -1, True, 1.5, 2**63, 10**400])
def test_principal_must_fit_positive_sqlite_integer_cents(principal):
    with pytest.raises(DataContractError, match="integer cents"):
        Application("loan", date(2024, 1, 1), principal, "retail")


@pytest.mark.parametrize("principal", [1, 2**53 + 1, 2**63 - 1])
def test_supported_integer_cents_are_preserved_exactly(principal):
    app = Application("loan", date(2024, 1, 1), principal, "retail")
    assert app.principal_cents == principal
    assert type(app.principal_cents) is int


@pytest.mark.parametrize("name", ["utilization", "debt_to_income", "prior_delinquencies"])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_enormous_integer_features_raise_a_contract_error(name, value):
    with pytest.raises(DataContractError, match="finite numeric value"):
        FeatureEvent("loan", name, value, date(2024, 1, 1), date(2024, 1, 1))


@pytest.mark.parametrize("field", ["scenario_lgd", "review_fraction"])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_enormous_integer_config_fractions_raise_a_contract_error(field, value):
    with pytest.raises(DataContractError, match=field + " must be a finite number"):
        Config(**{field: value})


@pytest.fixture
def case(tmp_path):
    directory = tmp_path / "inputs"
    initialize_inputs(directory)
    return directory


def replace_principal(directory, value):
    path = directory / "applications.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    rows[0]["principal_cents"] = str(value)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_diagnostics_name_oversized_principal_and_block_evaluation(case):
    replace_principal(case, 2**63)
    before = {path.name: path.read_bytes() for path in case.iterdir()}
    result = check_inputs(case)
    assert result["status"] == "invalid_inputs"
    assert result["issue_count"] == 1
    issue = result["issues"][0]
    assert (issue["file"], issue["line"], issue["code"]) == (
        "applications.csv",
        2,
        "record_contract",
    )
    assert "integer cents" in issue["message"]
    assert result["checks"]["evaluation_cohorts"] == "blocked"
    assert result["model_fitted"] is False and result["preflight"] is None
    assert before == {path.name: path.read_bytes() for path in case.iterdir()}


@pytest.mark.parametrize("field", ["scenario_lgd", "review_fraction"])
def test_config_diagnostics_collect_huge_fraction_without_traceback(case, capsys, field):
    path = case / "cohorts.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(config | {field: 10**400}), encoding="utf-8")
    assert main(["check-inputs", str(case)]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["issue_count"] == 1
    assert result["issues"][0]["file"] == "cohorts.json"
    assert result["issues"][0]["code"] == "config"
    assert field in result["issues"][0]["message"]
    assert result["checks"]["evaluation_cohorts"] == "blocked"
    assert "Traceback" not in captured.err and result["model_fitted"] is False


@pytest.mark.parametrize("command", ["preflight", "run", "benchmark", "monitor"])
def test_event_commands_reject_oversized_cents_before_fitting(case, tmp_path, capsys, command):
    replace_principal(case, 10**400)
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
    assert "integer cents" in captured.err
    assert "Traceback" not in captured.err and not output.exists()
