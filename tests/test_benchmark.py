from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from datetime import date

import pytest

from creditvintage.benchmark import (
    METHODS,
    MODEL_FEATURES,
    POINTS_FACTOR,
    POINTS_OFFSET,
    benchmark,
    verify_benchmark,
    write_benchmark,
)
from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.model import Config
from creditvintage.sample import generate


@pytest.fixture(scope="module")
def compared():
    return benchmark(
        *generate(seed=7, per_vintage=25),
        Config(seed=7),
        data_kind="synthetic",
        bootstrap_replicates=20,
    )


def test_all_methods_receive_same_budget_and_group_counts_reconcile(compared):
    for capacity in compared.summary["capacities"]:
        for method in METHODS:
            rows = [
                row
                for row in compared.policies
                if row["method"] == method and row["capacity"] == capacity
            ]
            portfolio = next(row for row in rows if row["group_type"] == "portfolio")
            assert portfolio["reviewed"] == math.ceil(150 * capacity)
            assert (
                portfolio["false_positives"] + portfolio["captured_events"] == portfolio["reviewed"]
            )
            for group_type in ("vintage", "segment"):
                members = [row for row in rows if row["group_type"] == group_type]
                for field in ("applications", "events", "reviewed", "captured_events"):
                    assert sum(row[field] for row in members) == portfolio[field]


def test_held_out_outcomes_cannot_change_scores_bins_or_explanations(compared):
    apps, features, reports = generate(seed=7, per_vintage=25)
    test_ids = {app.application_id for app in apps if app.decision_at >= date(2024, 7, 1)}
    good_id = next(
        app_id
        for app_id in sorted(test_ids)
        if all(row.days_past_due < 90 for row in reports if row.application_id == app_id)
    )
    changed = [
        replace(row, days_past_due=90) if row.application_id == good_id else row for row in reports
    ]
    other = benchmark(apps, features, changed, Config(seed=7), bootstrap_replicates=20)
    assert other.summary["woe_scorecard"] == compared.summary["woe_scorecard"]
    assert other.explanations == compared.explanations
    assert [
        (row["method"], row["application_id"], row["score"], row["review_rank"])
        for row in other.predictions
    ] == [
        (row["method"], row["application_id"], row["score"], row["review_rank"])
        for row in compared.predictions
    ]


def test_future_feature_revisions_cannot_change_predictions(compared):
    apps, features, reports = generate(seed=7, per_vintage=25)
    decisions = {app.application_id: app.decision_at for app in apps}
    changed = [
        replace(row, value=0.0) if row.available_at > decisions[row.application_id] else row
        for row in features
    ]
    other = benchmark(apps, changed, reports, Config(seed=7), bootstrap_replicates=20)
    assert other.predictions == compared.predictions
    assert other.explanations == compared.explanations


def test_log_odds_and_scorecard_points_reconstruct_probabilities(compared):
    for row in compared.predictions:
        if row["method"] == "raw_logistic":
            reasons = [
                reason
                for reason in compared.explanations
                if reason["application_id"] == row["application_id"]
            ]
            assert {reason["feature"] for reason in reasons} == set(MODEL_FEATURES)
            logit = reasons[0]["baseline_log_odds"] + sum(
                reason["log_odds_contribution"] for reason in reasons
            )
            assert 1 / (1 + math.exp(-logit)) == pytest.approx(row["score"], abs=1e-9)
        elif row["method"] == "woe_logistic":
            logit = (POINTS_OFFSET - row["scorecard_points"]) / POINTS_FACTOR
            assert 1 / (1 + math.exp(-logit)) == pytest.approx(row["score"], abs=1e-9)
    assert all("label" not in row for row in compared.explanations)


def rehash(directory, name):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_verifier_detects_rehashed_policy_tampering(compared, tmp_path):
    write_benchmark(compared, tmp_path)
    assert verify_benchmark(tmp_path)["applications"] == 150
    path = tmp_path / "policy_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
    rows[0]["captured_events"] = str(int(rows[0]["captured_events"]) + 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    rehash(tmp_path, path.name)
    with pytest.raises(DataContractError, match="captured_events disagrees"):
        verify_benchmark(tmp_path)


def test_verifier_detects_rehashed_explanation_tampering(compared, tmp_path):
    write_benchmark(compared, tmp_path)
    path = tmp_path / "explanations.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
    rows[0]["log_odds_contribution"] = "99"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    rehash(tmp_path, path.name)
    with pytest.raises(DataContractError, match="algebra mismatch"):
        verify_benchmark(tmp_path)


@pytest.mark.parametrize("capacities", [(), (0.1, 0.1), (0,), (float("nan"),)])
def test_invalid_budgets_fail_before_fitting(capacities):
    with pytest.raises(DataContractError):
        benchmark(*generate(seed=7, per_vintage=25), capacities=capacities)


def test_cli_benchmark_and_verify(tmp_path, capsys):
    assert main(["benchmark", "--seed", "7", "--per-vintage", "25", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["verify-benchmark", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["methods"] == 5


@pytest.mark.parametrize("artifact", ["predictions.csv", "summary.json"])
def test_verifier_rejects_rehashed_scorecard_or_interval_changes(compared, tmp_path, artifact):
    write_benchmark(compared, tmp_path)
    path = tmp_path / artifact
    if artifact == "summary.json":
        value = json.loads(path.read_text())
        value["paired_bootstrap"]["intervals"][0]["capture_difference_95_interval"][0] += 0.1
        path.write_text(json.dumps(value))
        message = "Paired intervals disagree"
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows, columns = list(reader), reader.fieldnames
        row = next(row for row in rows if row["method"] == "woe_logistic")
        row["scorecard_points"] = str(float(row["scorecard_points"]) + 20)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        message = "Scorecard points do not reconstruct"
    rehash(tmp_path, artifact)
    with pytest.raises(DataContractError, match=message):
        verify_benchmark(tmp_path)
