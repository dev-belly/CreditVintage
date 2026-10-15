from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from creditvintage.artifacts import verify_result
from creditvintage.cli import main
from creditvintage.configuration import load_config, preflight
from creditvintage.core import DataContractError
from creditvintage.model import Config
from creditvintage.sample import generate


def config_document(config=None):
    config = config or Config()
    return {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in asdict(config).items()
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"train_strat": "2022-01-01"}, "Unknown config keys"),
        ({"train_start": "20220101"}, "YYYY-MM-DD"),
        ({"test_end": "2024-02-30"}, "YYYY-MM-DD"),
        ({"evaluation_as_of": 20250701}, "YYYY-MM-DD"),
        ({"calibration_start": "2023-01-01"}, "maturation gaps"),
        ({"review_fraction": True}, "finite number"),
        ({"scenario_lgd": "0.45"}, "finite number"),
        ({"review_fraction": 0}, "finite number"),
        ({"seed": True}, "uint32"),
    ],
)
def test_invalid_configuration_cannot_silently_use_default_windows(tmp_path, change, message):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_document() | change), encoding="utf-8")
    with pytest.raises(DataContractError, match=message):
        load_config(path)


def test_all_dates_required_and_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"seed": 7}', encoding="utf-8")
    with pytest.raises(DataContractError, match="Missing config dates"):
        load_config(path)
    path.write_text('{"seed": 7, "seed": 8}', encoding="utf-8")
    with pytest.raises(DataContractError, match="Duplicate JSON key"):
        load_config(path)
    path.write_text(json.dumps(config_document() | {"scenario_lgd": float("nan")}))
    with pytest.raises(DataContractError, match="finite number"):
        load_config(path)


def test_explicit_seed_override_and_unchanged_default(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_document(Config(seed=7))), encoding="utf-8")
    assert load_config() == Config()
    assert load_config(path).seed == 7
    assert load_config(path, seed=11).seed == 11


def test_preflight_reports_excluded_applications_without_fitting(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Preflight must not fit a model")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden_fit)
    config = Config(train_start=date(2022, 2, 1))
    result = preflight(*generate(seed=7, per_vintage=25), config, data_kind="synthetic")
    assert result["model_fitted"] is False
    assert result["cohorts"]["train"]["applications"] == 275
    assert result["cohorts"]["calibration"]["applications"] == 150
    assert result["cohorts"]["test"]["applications"] == 150
    assert result["applications_outside_windows"] == 25


def test_preflight_rejects_missing_non_event_coverage():
    apps, features, reports = generate(seed=7, per_vintage=25)
    from creditvintage.core import snapshot

    non_event = next(
        row.application.application_id
        for row in snapshot(apps, features, reports, Config().calibration_start)
        if row.label == 0
    )
    reports = [row for row in reports if row.application_id != non_event]
    with pytest.raises(DataContractError, match="train has unlabelled loans"):
        preflight(apps, features, reports, Config(), data_kind="user_supplied")


@pytest.fixture
def shifted_inputs(tmp_path):
    # A different calendar must actually reach the model, not only its metadata.
    shift = timedelta(days=1461)
    apps, features, reports = generate(seed=7, per_vintage=10)
    apps = [replace(row, decision_at=row.decision_at - shift) for row in apps]
    features = [
        replace(row, observed_at=row.observed_at - shift, available_at=row.available_at - shift)
        for row in features
    ]
    reports = [
        replace(row, as_of=row.as_of - shift, available_at=row.available_at - shift)
        for row in reports
    ]
    flags = []
    for name, rows in (("applications", apps), ("features", features), ("performance", reports)):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        flags += [f"--{name}", str(path)]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                key: (value - shift).isoformat() if isinstance(value, date) else value
                for key, value in asdict(Config(seed=7)).items()
            }
        ),
        encoding="utf-8",
    )
    return flags + ["--config", str(config)]


def test_supplied_custom_calendar_reaches_run_monitor_and_benchmark(
    shifted_inputs, tmp_path, capsys
):
    assert main(["preflight", *shifted_inputs]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["cohorts"]["test"]["applications"] == 60
    assert checked["config"]["test_start"] == "2020-07-01"
    assert checked["applications_outside_windows"] == 0
    out = tmp_path / "evaluation"
    assert main(["run", *shifted_inputs, "--out", str(out)]) == 0
    assert verify_result(out)["loans"] == 60
    summary = json.loads((out / "summary.json").read_text())
    assert summary["config"]["test_start"] == "2020-07-01"
    for command, flags in (("benchmark", []), ("monitor", ["--as-of", "2020-10-01"])):
        destination = tmp_path / command
        assert main([command, *shifted_inputs, *flags, "--out", str(destination)]) == 0
        assert main([f"verify-{command}", str(destination)]) == 0


def test_bad_config_fails_without_output_or_traceback(tmp_path, capsys):
    config = tmp_path / "bad.json"
    config.write_text('{"train_start": "yesterday"}', encoding="utf-8")
    output = tmp_path / "must-not-exist"
    assert main(["run", "--config", str(config), "--out", str(output)]) == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert "Missing config dates" in captured.err
    assert "Traceback" not in captured.err
