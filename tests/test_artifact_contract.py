"""Published reports must reject invalid claims even after their hashes are updated."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from creditvintage.artifacts import verify_result
from creditvintage.core import DataContractError
from creditvintage.monitoring import verify_monitor


def _report(tmp_path: Path, kind: str) -> Path:
    destination = tmp_path / kind
    shutil.copytree(Path("docs") / kind, destination)
    return destination


def _rehash(directory: Path, name: str) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files_sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def _edit_json(directory: Path, name: str, keys: tuple[str, ...], value: Any) -> None:
    path = directory / name
    document = json.loads(path.read_text())
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(document))
    _rehash(directory, name)


def _edit_csv(directory: Path, name: str, key: str, value: str) -> None:
    path = directory / name
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    assert columns is not None
    rows[0][key] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _rehash(directory, name)


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("test_metrics", "calibrated", "brier"), float("nan")),
        (("cohorts", "test", "default_rate"), float("nan")),
        (("config", "scenario_lgd"), float("nan")),
        (("test_portfolio", "mean_pd"), "NaN"),
        (("test_portfolio", "scenario_expected_loss_usd"), float("inf")),
    ],
)
def test_rehashed_result_rejects_nonfinite_json_claims(
    tmp_path: Path, keys: tuple[str, ...], value: Any
) -> None:
    directory = _report(tmp_path, "demo")
    _edit_json(directory, "summary.json", keys, value)
    with pytest.raises(DataContractError, match="finite"):
        verify_result(directory)


@pytest.mark.parametrize(
    ("kind", "name", "column", "value"),
    [
        ("demo", "calibration_bins.csv", "mean_pd", "NaN"),
        ("demo", "vintage_metrics.csv", "scenario_expected_loss_usd", "nan"),
        ("monitor", "score_bins.csv", "psi_component", "NaN"),
        ("monitor", "score_bins.csv", "lower", "1e309"),
    ],
)
def test_rehashed_report_rejects_nonfinite_csv_claims(
    tmp_path: Path, kind: str, name: str, column: str, value: str
) -> None:
    directory = _report(tmp_path, kind)
    _edit_csv(directory, name, column, value)
    verify = verify_result if kind == "demo" else verify_monitor
    with pytest.raises(DataContractError, match="finite"):
        verify(directory)


def test_rehashed_result_requires_the_full_maturation_gap(tmp_path: Path) -> None:
    directory = _report(tmp_path, "demo")
    _edit_json(directory, "summary.json", ("config", "evaluation_as_of"), "2024-12-31")
    with pytest.raises(DataContractError, match="maturation gaps"):
        verify_result(directory)


def test_rehashed_review_capacity_must_match_the_recorded_config(tmp_path: Path) -> None:
    directory = _report(tmp_path, "demo")
    _edit_json(directory, "summary.json", ("config", "review_fraction"), 0.2)
    with pytest.raises(DataContractError, match="Review capacity"):
        verify_result(directory)


def test_rehashed_result_rejects_fractional_portfolio_counts(tmp_path: Path) -> None:
    directory = _report(tmp_path, "demo")
    _edit_json(directory, "summary.json", ("test_portfolio", "loans"), 480.5)
    with pytest.raises(DataContractError, match="Cohort count"):
        verify_result(directory)


@pytest.mark.parametrize(("column", "value"), [("application_id", "=1+1"), ("segment", "unknown")])
def test_rehashed_monitor_rejects_invalid_application_identity(
    tmp_path: Path, column: str, value: str
) -> None:
    directory = _report(tmp_path, "monitor")
    _edit_csv(directory, "monitor_scores.csv", column, value)
    with pytest.raises(DataContractError, match="identity or segment"):
        verify_monitor(directory)


def test_rehashed_monitor_requires_reference_labels_to_mature(tmp_path: Path) -> None:
    directory = _report(tmp_path, "monitor")
    _edit_json(directory, "monitor.json", ("reference_window",), ["2023-07-01", "2024-06-30"])
    with pytest.raises(DataContractError, match="maturation gap"):
        verify_monitor(directory)


@pytest.mark.parametrize("name", ["calibration_bins.csv", "vintage_metrics.csv"])
def test_rehashed_aggregate_csv_rejects_duplicate_headers(tmp_path: Path, name: str) -> None:
    directory = _report(tmp_path, "demo")
    path = directory / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    column = rows[0].index("mean_pd")
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([row + [row[column]] for row in rows])
    _rehash(directory, name)
    with pytest.raises(DataContractError, match="Unexpected CSV columns"):
        verify_result(directory)


def test_original_published_reports_still_verify() -> None:
    assert verify_result(Path("docs/demo"))["loans"] == 480
    assert verify_monitor(Path("docs/monitor"))["loans"] == 320


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("roc_auc_95_interval", [0.99, 1.0]),
        ("brier_95_interval", [0.0, 0.001]),
        ("valid_replicates", 299),
        ("requested_replicates", 20),
        ("unit", "independent_loan"),
    ],
)
def test_rehashed_bootstrap_claims_must_replay_from_prediction_rows(
    tmp_path: Path, key: str, value: Any
) -> None:
    directory = _report(tmp_path, "demo")
    _edit_json(directory, "summary.json", ("vintage_bootstrap", key), value)
    with pytest.raises(DataContractError, match="bootstrap"):
        verify_result(directory)


@pytest.mark.parametrize("kind", ["demo", "monitor"])
def test_rehashed_html_must_match_verified_report_evidence(tmp_path: Path, kind: str) -> None:
    directory = _report(tmp_path, kind)
    (directory / "index.html").write_text("<h1>Invented perfect model performance</h1>")
    _rehash(directory, "index.html")
    verify = verify_result if kind == "demo" else verify_monitor
    with pytest.raises(DataContractError, match="HTML"):
        verify(directory)


@pytest.mark.parametrize(
    ("kind", "name", "key"),
    [
        ("demo", "summary.json", "data_kind"),
        ("monitor", "monitor.json", "data_kind"),
        ("demo", "manifest.json", "input_sha256"),
        ("monitor", "manifest.json", "input_sha256"),
        ("demo", "summary.json", "roc_auc"),
    ],
)
def test_duplicate_json_keys_cannot_make_report_claims_ambiguous(
    tmp_path: Path, kind: str, name: str, key: str
) -> None:
    directory = _report(tmp_path, kind)
    path = directory / name
    content = path.read_text()
    marker = f'"{key}":'
    replacement = f'"{key}": "conflicting earlier claim", {marker}'
    assert marker in content
    path.write_text(content.replace(marker, replacement, 1))
    if name != "manifest.json":
        _rehash(directory, name)
    verify = verify_result if kind == "demo" else verify_monitor
    with pytest.raises(DataContractError, match="Duplicate JSON key"):
        verify(directory)
