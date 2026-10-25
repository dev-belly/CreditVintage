from __future__ import annotations

import copy
import json
import shutil
from datetime import date, timedelta

import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditvintage.artifacts import _hash, _json
from creditvintage.cli import main
from creditvintage.core import FEATURE_NAMES, Application, DataContractError, FeatureEvent
from creditvintage.feature_policy import (
    CHECK_FILES,
    check_feature_quality,
    evaluate_feature_policy,
    example_policy,
    initialize_policy,
    policy_digest,
    validate_policy,
    verify_feature_check,
    write_feature_check,
)
from creditvintage.feature_quality import monitor_feature_quality, write_feature_quality
from creditvintage.sample import generate, generate_feature_quality

CUTOFF = date(2024, 10, 1)


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Coverage policies must not fit models")

    for model in (LogisticRegression, IsotonicRegression):
        monkeypatch.setattr(model, "fit", forbidden)


def one_rule(**changes):
    document = example_policy()
    document["rules"] = [document["rules"][0]]
    document["rules"][0].update(changes)
    return document


@pytest.fixture
def quality():
    return monitor_feature_quality(*generate_feature_quality(per_vintage=20), CUTOFF)


def test_default_policy_fails_each_missing_feature_without_dropping_applications(quality):
    before = copy.deepcopy(quality)
    policy = example_policy()
    result = evaluate_feature_policy(quality, policy)
    assert result.summary["status"] == "fail"
    assert result.summary["counts"] == {"pass": 0, "fail": 3, "insufficient_data": 0}
    assert [r["available"] for r in result.summary["checks"]] == [78, 79, 79]
    assert {r["applications"] for r in result.summary["checks"]} == {80}
    assert all(r["reason"] == "below_minimum_coverage" for r in result.summary["checks"])
    assert quality == before and policy == example_policy()


def test_a_partial_or_tolerant_policy_is_not_scoring_readiness(quality):
    result = evaluate_feature_policy(quality, one_rule(min_coverage="0.975"))
    assert result.summary["status"] == "pass"  # Exactly 78 / 80, with two unavailable slots.
    above_exact = evaluate_feature_policy(quality, one_rule(min_coverage="0.975000000000000001"))
    assert above_exact.summary["status"] == "fail"
    assert quality.summary["populations"]["current"]["incomplete_applications"] == 4
    assert "not scoring readiness" in result.summary["scope"]


def test_exact_fraction_ignores_rounded_display_and_binary_float_thresholds():
    apps = [Application(str(i), date(2024, 7, 1), 10000, "retail") for i in range(3)]
    features = [
        FeatureEvent(
            app.application_id,
            name,
            value,
            app.decision_at - timedelta(days=2),
            app.decision_at - timedelta(days=1),
        )
        for i, app in enumerate(apps)
        for name, value in zip(FEATURE_NAMES, (0.2, 0.3, 0), strict=True)
        if not (i == 2 and name == "utilization")
    ]
    result = monitor_feature_quality(apps, features, CUTOFF)
    # The displayed 0.66666667 is above exact 2/3, so equality to that display must fail.
    failed = evaluate_feature_policy(
        result, one_rule(min_applications=1, min_coverage="0.66666667")
    )
    passed = evaluate_feature_policy(
        result, one_rule(min_applications=1, min_coverage="0.666666666666666666")
    )
    assert failed.summary["checks"][0]["coverage_rate"] == 0.66666667
    assert failed.summary["status"] == "fail" and passed.summary["status"] == "pass"


def test_monthly_rule_retains_small_and_empty_months(quality):
    july = evaluate_feature_policy(quality, one_rule(vintage="2024-07", min_applications=21))
    assert july.summary["status"] == "insufficient_data"
    assert july.summary["checks"][0]["applications"] == 20
    empty = evaluate_feature_policy(quality, one_rule(vintage="2024-12", min_coverage="0"))
    row = empty.summary["checks"][0]
    assert row["applications"] == row["available"] == 0 and row["coverage_rate"] is None
    assert row["status"] == "insufficient_data"
    reference = evaluate_feature_policy(
        quality, one_rule(population="reference", vintage="2023-07")
    )
    assert reference.summary["status"] == "pass"


def test_empty_population_cannot_pass_even_zero_required_coverage():
    result = evaluate_feature_policy(
        monitor_feature_quality([], [], CUTOFF), one_rule(min_applications=1, min_coverage="0")
    )
    assert result.summary["status"] == "insufficient_data"
    assert result.summary["checks"][0]["coverage_rate"] is None


def test_fail_has_precedence_over_insufficient_and_pass(quality):
    policy = example_policy()
    policy["rules"][1].update(vintage="2024-12")
    policy["rules"][2].update(min_coverage="0.9")
    result = evaluate_feature_policy(quality, policy)
    assert result.summary["counts"] == {"pass": 1, "fail": 1, "insufficient_data": 1}
    assert result.summary["status"] == "fail"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", "=formula"),
        ("id", "a" * 65),
        ("id", "中文"),
        ("id", []),
        ("population", "test"),
        ("population", []),
        ("feature", "renamed"),
        ("feature", []),
        ("vintage", "2024-13"),
        ("vintage", "0000-01"),
        ("vintage", "2024-7"),
        ("vintage", 202407),
        ("min_applications", 0),
        ("min_applications", True),
        ("min_applications", 20.0),
        ("min_coverage", 0.95),
        ("min_coverage", True),
        ("min_coverage", "NaN"),
        ("min_coverage", "1e-1"),
        ("min_coverage", " 0.95"),
        ("min_coverage", "1.01"),
        ("min_coverage", "-0.1"),
        ("min_coverage", "0.９５"),
        ("min_coverage", "0.1234567890123456789"),
    ],
)
def test_invalid_policy_fields_are_rejected(key, value):
    with pytest.raises(DataContractError):
        validate_policy(one_rule(**{key: value}))


def test_policy_schema_no_silent_defaults_duplicates_or_unknown_keys(quality):
    documents = [
        {},
        {"schema": "wrong", "rules": []},
        {**example_policy(), "extra": True},
        {"schema": example_policy()["schema"], "rules": []},
        {"schema": example_policy()["schema"], "rules": ["not a rule"]},
    ]
    missing = one_rule()
    del missing["rules"][0]["min_coverage"]
    unknown = one_rule(threshold="0.9")
    duplicate = one_rule()
    duplicate["rules"].append(dict(duplicate["rules"][0]))
    for document in [*documents, missing, unknown, duplicate]:
        with pytest.raises(DataContractError):
            evaluate_feature_policy(quality, document)
    with pytest.raises(DataContractError, match="outside"):
        evaluate_feature_policy(quality, one_rule(vintage="2025-01"))


def test_self_contained_replay_policy_identity_and_source_immutability(tmp_path, quality):
    source, policy_path, output = tmp_path / "source", tmp_path / "policy.json", tmp_path / "check"
    write_feature_quality(quality, source)
    initialize_policy(policy_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    verdict = check_feature_quality(source, policy_path, output)
    assert verdict["status"] == "fail"
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    shutil.rmtree(source)
    checked = verify_feature_check(output, expected_policy_sha256=policy_digest(example_policy()))
    assert checked["status"] == "verified" and checked["policy_status"] == "fail"
    assert checked["model_fitted"] is False
    assert set(CHECK_FILES) | {"manifest.json"} == {
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("target", ["check.json", "checks.csv", "index.html", "policy.json"])
def test_rehashed_false_check_or_policy_cannot_verify(tmp_path, quality, target):
    output = tmp_path / "check"
    write_feature_check(evaluate_feature_policy(quality, example_policy()), output)
    path = output / target
    if target == "check.json":
        contents = json.loads(path.read_text())
        contents["status"] = "pass"
        _json(path, contents)
    elif target == "policy.json":
        contents = json.loads(path.read_text())
        contents["rules"][0]["min_coverage"] = "0.9"
        _json(path, contents)
    else:
        path.write_text(path.read_text() + "changed\n")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files_sha256"][target] = _hash(path)
    _json(output / "manifest.json", manifest)
    with pytest.raises(DataContractError, match="replay"):
        verify_feature_check(output)


def test_corrupt_source_is_rejected_before_any_check_output(tmp_path, quality):
    source, output, policy = tmp_path / "source", tmp_path / "check", tmp_path / "policy.json"
    write_feature_quality(quality, source)
    initialize_policy(policy)
    summary = json.loads((source / "quality.json").read_text())
    summary["populations"]["current"]["complete_applications"] = 80
    _json(source / "quality.json", summary)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files_sha256"]["quality.json"] = _hash(source / "quality.json")
    _json(source / "manifest.json", manifest)
    with pytest.raises(DataContractError):
        check_feature_quality(source, policy, output)
    assert not output.exists()


def test_external_policy_pin_detects_a_fully_regenerated_replacement(tmp_path, quality):
    trusted = policy_digest(example_policy())
    tolerant = example_policy()
    for rule in tolerant["rules"]:
        rule["min_coverage"] = "0.9"
    output = tmp_path / "replacement"
    write_feature_check(evaluate_feature_policy(quality, tolerant), output)
    assert verify_feature_check(output)["policy_status"] == "pass"
    with pytest.raises(DataContractError, match="trusted expected digest"):
        verify_feature_check(output, expected_policy_sha256=trusted)
    with pytest.raises(DataContractError, match="64 lowercase"):
        verify_feature_check(output, expected_policy_sha256="not-a-digest")


def test_cli_exit_codes_failed_verified_passed_insufficient_and_invalid(tmp_path, capsys, quality):
    source, output, policy = tmp_path / "source", tmp_path / "check", tmp_path / "policy.json"
    write_feature_quality(quality, source)
    assert main(["init-feature-policy", "--out", str(policy)]) == 0
    init = json.loads(capsys.readouterr().out)
    assert (
        main(["check-feature-quality", str(source), "--policy", str(policy), "--out", str(output)])
        == 3
    )
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "fail"
    assert (
        main(
            ["verify-feature-check", str(output), "--expected-policy-sha256", init["policy_sha256"]]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["policy_status"] == "fail"
    assert main(["verify-feature-check", str(output), "--expected-policy-sha256", "0" * 64]) == 2
    assert "trusted" in capsys.readouterr().err
    original = policy.read_bytes()
    assert main(["init-feature-policy", "--out", str(policy)]) == 2
    assert policy.read_bytes() == original
    capsys.readouterr()
    assert (
        main(["check-feature-quality", str(source), "--policy", str(policy), "--out", str(output)])
        == 2
    )
    capsys.readouterr()
    _json(policy, one_rule(min_applications=100))
    assert (
        main(
            [
                "check-feature-quality",
                str(source),
                "--policy",
                str(policy),
                "--out",
                str(tmp_path / "small"),
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_data"
    healthy = tmp_path / "healthy"
    apps, features, _ = generate(per_vintage=20)
    write_feature_quality(monitor_feature_quality(apps, features, CUTOFF), healthy)
    _json(policy, example_policy())
    assert (
        main(
            [
                "check-feature-quality",
                str(healthy),
                "--policy",
                str(policy),
                "--out",
                str(tmp_path / "pass"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    policy.write_text('{"schema": "a", "schema": "b"}')
    assert (
        main(
            [
                "check-feature-quality",
                str(source),
                "--policy",
                str(policy),
                "--out",
                str(tmp_path / "invalid"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "invalid").exists()
    assert "Duplicate JSON key" in capsys.readouterr().err
