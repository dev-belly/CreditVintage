"""Date, leakage, rare-event uncertainty and refitted evidence contracts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.sba_credit import (
    METHODS,
    SBA_FILES,
    fit_scores,
    metrics,
    month_bootstrap,
    verify_sba_benchmark,
    write_sba_benchmark,
)
from creditvintage.sba_source import (
    DATA_SHA256,
    SOURCES,
    WINDOWS,
    anniversary,
    borrower_key,
    convert_record,
    load_sba,
)


@pytest.fixture(scope="module")
def loans():
    return load_sba()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("sba") / "evidence"
    write_sba_benchmark(path)
    return path


def source_row(**changes):
    return {
        "AsOfDate": "2026-06-30",
        "Program": " 7A",
        "ApprovalDate": "2012-12-20",
        "FirstDisbursementDate": "2013-01-01",
        "PaidInFullDate": "2017-01-01",
        "ChargeOffDate": "",
        "LoanStatus": "P I F",
        "GrossApproval": "100000.0",
        "SBAGuaranteedApproval": "75000.0",
        "InitialInterestRate": "7.5",
        "ProcessingMethod": "Standard",
        "BorrName": "Example Loan Business",
        "BorrState": "CA",
        "BorrZip": "90210",
        **changes,
    }


def reseal(path):
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files_sha256"] = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in SBA_FILES
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    with zipfile.ZipFile(path / "evidence.zip", "w") as archive:
        for name in (*SBA_FILES, "manifest.json"):
            archive.write(path / name, name)


def test_frozen_dates_native_target_and_groups(loans):
    assert len(loans) == 35000
    assert len({x.borrower_group_sha256 for x in loans}) == 35000
    for stage, (_, _, size) in WINDOWS.items():
        subset = [x for x in loans if x.stage == stage]
        assert len(subset) == size
        assert max(x.horizon_end for x in subset) < (
            WINDOWS["calibration"][0]
            if stage == "train"
            else WINDOWS["test"][0]
            if stage == "calibration"
            else date(2026, 6, 30)
        )
    assert [sum(x.chargeoff_24m for x in loans if x.stage == s) for s in WINDOWS] == [816, 42, 89]
    assert sum(not x.initial_interest_rate_pct for x in loans if x.stage == "train") > 15000


@pytest.mark.parametrize(
    "value,expected",
    [
        (date(2016, 2, 29), date(2018, 2, 28)),
        (date(2013, 1, 31), date(2015, 1, 31)),
        (date(2008, 2, 28), date(2010, 2, 28)),
    ],
)
def test_calendar_anniversary(value, expected):
    assert anniversary(value) == expected


@pytest.mark.parametrize(
    "charged,target",
    [
        ("2013-01-01", 1),
        ("2015-01-01", 1),
        ("2015-01-02", 0),
        ("2025-12-31", 0),
    ],
)
def test_native_chargeoff_boundary_and_late_lifetime(charged, target):
    loan = convert_record(source_row(LoanStatus="CHGOFF", ChargeOffDate=charged), "10", 1)
    assert loan.chargeoff_24m == target


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"LoanStatus": "CHGOFF"}, "chargeoff_date_missing"),
        ({"FirstDisbursementDate": ""}, "missing_funding_or_approval"),
        ({"FirstDisbursementDate": "2013/01/01"}, "invalid_date"),
        ({"ApprovalDate": "2013-02-01"}, "invalid_date_order"),
        ({"FirstDisbursementDate": "2027-01-01"}, "invalid_date_order"),
        ({"FirstDisbursementDate": "2014-01-01"}, "outside_frozen_windows"),
        ({"PaidInFullDate": "2012-01-01"}, "invalid_outcome_date"),
        ({"ChargeOffDate": "2027-01-01"}, "invalid_outcome_date"),
        ({"LoanStatus": "CANCLD"}, "unusable_loan_status"),
        ({"GrossApproval": "NaN"}, "invalid_money"),
        ({"GrossApproval": "0"}, "invalid_money"),
        ({"GrossApproval": "100000.001"}, "invalid_money"),
        ({"SBAGuaranteedApproval": "110000"}, "guarantee_exceeds_gross"),
        ({"InitialInterestRate": "inf"}, "invalid_initial_rate"),
        ({"InitialInterestRate": "-1"}, "invalid_initial_rate"),
        ({"InitialInterestRate": "101"}, "invalid_initial_rate"),
        ({"BorrName": ""}, "missing_borrower_group"),
        ({"ProcessingMethod": ""}, "missing_processing_method"),
    ],
)
def test_original_record_exclusions_are_explicit(changes, reason):
    with pytest.raises(ValueError, match=reason):
        convert_record(source_row(**changes), "10", 1)


@pytest.mark.parametrize("changes", [{"Program": "504"}, {"AsOfDate": "2026-03-31"}])
def test_other_program_or_snapshot_is_hard_error(changes):
    with pytest.raises(DataContractError, match="snapshot"):
        convert_record(source_row(**changes), "10", 1)


def test_missing_rate_and_exempt_remain_observed_source_conventions():
    loan = convert_record(source_row(InitialInterestRate="", LoanStatus="EXEMPT"), "10", 1)
    assert loan.initial_interest_rate_pct == ""
    assert loan.chargeoff_24m == 0
    assert borrower_key("Example, Inc.", "ca", "90210") == borrower_key(
        "EXAMPLE INC", "CA", "90210"
    )


def test_test_outcomes_and_future_fields_cannot_change_scores(loans):
    before, audit = fit_scores(loans)
    changed = [
        replace(
            x,
            chargeoff_24m=1 - x.chargeoff_24m,
            charged_off_at=None if x.chargeoff_24m else x.funded_at,
            source_loan_status="FUTURE_STATUS",
            paid_in_full_at=date(2026, 1, 1),
        )
        if x.stage == "test"
        else x
        for x in loans
    ]
    after, changed_audit = fit_scores(changed)
    assert audit == changed_audit
    for method in METHODS:
        np.testing.assert_array_equal(before[method], after[method])


def test_heldout_category_and_extreme_value_do_not_fit_transforms(loans):
    _, audit = fit_scores(loans)
    changed = [
        replace(x, initial_interest_rate_pct="1000000", processing_method="HELDOUT_ONLY")
        if x.stage == "test"
        else x
        for x in loans
    ]
    _, changed_audit = fit_scores(changed)
    assert audit == changed_audit
    assert "HELDOUT_ONLY" not in audit["approval_processing_categories"]


@pytest.mark.parametrize("mutation", ["duplicate", "group", "cohort", "target", "horizon"])
def test_protocol_guards_before_fitting(loans, mutation):
    changed = list(loans)
    if mutation == "duplicate":
        changed[-1] = changed[0]
    if mutation == "group":
        changed[-1] = replace(changed[-1], borrower_group_sha256=changed[0].borrower_group_sha256)
    if mutation == "cohort":
        changed[-1] = replace(changed[-1], stage="train")
    if mutation == "target":
        changed[-1] = replace(changed[-1], chargeoff_24m=1 - changed[-1].chargeoff_24m)
    if mutation == "horizon":
        changed[-1] = replace(changed[-1], horizon_end=date(2020, 1, 1))
    with pytest.raises(DataContractError):
        fit_scores(changed)


def test_independent_auc_and_plain_brier():
    y = np.asarray([1, 0, 1, 0, 0])
    p = np.asarray([0.6, 0.3, 0.3, 0.1, 0.1])
    values = metrics(y, p)
    assert values["roc_auc"] == pytest.approx(mannwhitneyu(p[y == 1], p[y == 0]).statistic / 6)
    assert values["brier"] == pytest.approx(sum((p - y) ** 2) / 5)


def test_paired_month_bootstrap_hand_values_and_one_class_retention():
    y = np.asarray([0, 1, 0, 1])
    scores = {m: np.full(4, 0.2) for m in METHODS}
    rows, stated = month_bootstrap(y, scores, ["a", "a", "b", "b"], iterations=20)
    assert len(rows) == 60
    for row in rows:
        assert row["roc_auc"] == 0.5 and row["average_precision"] == 0.5
        assert row["brier"] == pytest.approx(0.34)
    assert stated["paired_isotonic_minus_logistic_brier_95"] == [0, 0]
    rows, stated = month_bootstrap(y, scores, ["a", "b", "a", "b"], iterations=20)
    assert len(rows) == 60
    assert any(row["roc_auc"] is None for row in rows)
    assert 0 < stated["methods"]["constant"]["roc_auc"]["valid_replicates"] < 20


def test_complete_evidence_refits_and_queue_is_outcome_free(bundle):
    checked = verify_sba_benchmark(bundle)
    assert checked["refitted"] and checked["test_records"] == 10000 and checked["test_events"] == 89
    with (bundle / "review_queue.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["method", "record_id", "rank", "score"]
        assert len(list(reader)) == 3000
    assert (
        hashlib.sha256(gzip.decompress((bundle / "source.csv.gz").read_bytes())).hexdigest()
        == DATA_SHA256
    )


def test_source_digest_rejects_replaced_and_truncated_gzip():
    with pytest.raises(DataContractError):
        load_sba(gzip.compress(b"fabricated"))
    with pytest.raises(DataContractError):
        load_sba(b"not gzip")


def test_rehashed_false_metrics_fail_refitting(bundle, tmp_path):
    path = tmp_path / "forged"
    shutil.copytree(bundle, path)
    summary = json.loads((path / "summary.json").read_text())
    summary["test_metrics"]["logistic"]["roc_auc"] = 0.99
    (path / "summary.json").write_text(json.dumps(summary))
    reseal(path)
    with pytest.raises(DataContractError, match="SBA summary"):
        verify_sba_benchmark(path)


def test_rehashed_nonfinite_prediction_is_rejected(bundle, tmp_path):
    path = tmp_path / "nonfinite"
    shutil.copytree(bundle, path)
    p = path / "predictions.csv"
    lines = p.read_text().splitlines()
    parts = lines[1].split(",")
    parts[4] = "nan"
    lines[1] = ",".join(parts)
    p.write_text("\n".join(lines) + "\n")
    reseal(path)
    with pytest.raises(DataContractError, match="predictions"):
        verify_sba_benchmark(path)


def test_extra_zip_member_and_html_fabrication_rejected(bundle, tmp_path):
    path = tmp_path / "zip"
    shutil.copytree(bundle, path)
    with zipfile.ZipFile(path / "evidence.zip", "a") as archive:
        archive.writestr("../extra", "bad")
    with pytest.raises(DataContractError, match="ZIP inventory"):
        verify_sba_benchmark(path)
    (path / "index.html").write_text("Fake successful study")
    reseal(path)
    with pytest.raises(DataContractError, match="HTML"):
        verify_sba_benchmark(path)


def test_output_never_overwrites_prior_evidence(bundle, capsys):
    before = (bundle / "manifest.json").read_bytes()
    assert main(["sba-benchmark", "--out", str(bundle)]) == 2
    assert (bundle / "manifest.json").read_bytes() == before
    assert "new SBA destination" in capsys.readouterr().err


def test_converter_rejects_wrong_original_hash_before_output(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_sba", Path(__file__).parents[1] / "scripts/prepare_sba.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare = module.prepare

    source = tmp_path / "raw"
    source.mkdir()
    (source / SOURCES[0][1]).write_text("wrong snapshot")
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="source digest"):
        prepare(source, output)
    assert not output.exists()
