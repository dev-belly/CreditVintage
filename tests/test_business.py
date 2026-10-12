"""Business denominators, SQL independence and portable evidence regressions."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from creditvintage import business
from creditvintage.business import SQL_FILES, _aggregate, verify_analysis, write_analysis
from creditvintage.cli import main
from creditvintage.core import DataContractError

REPORT = Path(__file__).resolve().parents[1] / "docs" / "demo"


def _queries() -> dict[str, str]:
    return {
        name: files("creditvintage").joinpath(name).read_text(encoding="utf-8")
        for name in SQL_FILES
    }


def _rows() -> list[dict[str, str]]:
    fixture = (
        ("A", "2024-07", "retail", 10000, 1, 0.8, 1),
        ("B", "2024-07", "retail", 90000, 1, 0.2, 0),
        ("C", "2024-08", "retail", 40000, 1, 0.6, 0),
        ("D", "2024-08", "small_business", 60000, 0, 0.6, 1),
    )
    return [
        {
            "application_id": identity,
            "decision_at": vintage + "-01",
            "vintage": vintage,
            "segment": segment,
            "principal_cents": str(principal),
            "label": str(label),
            "raw_pd": str(probability),
            "calibrated_pd": str(probability),
            "review_selected": str(selected),
        }
        for identity, vintage, segment, principal, label, probability, selected in fixture
    ]


def test_count_and_exposure_denominators_are_distinct() -> None:
    groups, queue = _aggregate(_rows(), 0.45, _queries())
    portfolio = next(row for row in groups if row["scope"] == "portfolio")
    assert portfolio["event_rate"] == 0.75
    assert portfolio["mean_pd"] == 0.55
    assert portfolio["exposure_weighted_pd"] == 0.43
    assert portfolio["review_precision"] == 0.5
    assert portfolio["event_capture"] == pytest.approx(1 / 3, abs=1e-8)
    assert portfolio["scenario_expected_loss_usd"] == 387.0
    assert portfolio["scenario_event_loss_proxy_usd"] == 630.0
    assert [row["application_id"] for row in queue] == ["A", "D"]


def test_zero_denominators_are_null_and_previous_vintage_is_observed() -> None:
    rows = _rows()
    for row in rows[2:]:
        row["vintage"] = "2024-10"
        row["decision_at"] = "2024-10-01"
    groups, _ = _aggregate(rows, 0.45, _queries())
    unreviewed = next(
        row
        for row in groups
        if row["scope"] == "vintage_segment"
        and row["vintage"] == "2024-10"
        and row["segment"] == "retail"
    )
    assert unreviewed["review_precision"] is None
    assert unreviewed["event_capture"] == 0
    no_events = next(
        row for row in groups if row["scope"] == "segment" and row["segment"] == "small_business"
    )
    assert no_events["review_precision"] == 0
    assert no_events["event_capture"] is None
    october = next(
        row for row in groups if row["scope"] == "vintage" and row["vintage"] == "2024-10"
    )
    assert october["previous_event_rate"] == 1
    assert october["event_rate_change"] == -0.5


def test_queue_ties_use_identity_and_export_no_outcomes() -> None:
    rows = _rows()
    for row in rows:
        row["review_selected"] = "1"
        row["calibrated_pd"] = "0.5"
    _, queue = _aggregate(list(reversed(rows)), 0.45, _queries())
    assert [row["application_id"] for row in queue] == ["A", "B", "C", "D"]
    assert [row["review_rank"] for row in queue] == [1, 2, 3, 4]
    assert all("label" not in row for row in queue)


@pytest.mark.parametrize(
    "before,after",
    [
        ("SUM(label) AS events", "COUNT(*) AS events"),
        ("NULLIF(events, 0)", "NULLIF(loans, 0)"),
        ("AVG(calibrated_pd) AS mean_pd", "AVG(raw_pd) AS mean_pd"),
    ],
)
def test_python_oracle_rejects_wrong_sql_denominators(before: str, after: str) -> None:
    rows = _rows()
    rows[0]["raw_pd"] = "0.1"
    queries = _queries()
    queries["business_groups.sql"] = queries["business_groups.sql"].replace(before, after)
    with pytest.raises(DataContractError, match="disagrees with Python replay"):
        _aggregate(rows, 0.45, queries)


def test_sql_integer_exposure_overflow_fails_cleanly() -> None:
    rows = _rows()
    rows[0]["principal_cents"] = str(2**63 - 1)
    with pytest.raises(DataContractError, match="SQLite int64"):
        _aggregate(rows, 0.45, _queries())


def test_analysis_is_portable_and_preserves_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "report"
    shutil.copytree(REPORT, source)
    target = tmp_path / "analysis"
    summary = write_analysis(source, target)
    assert summary["group_rows"] == 21
    assert summary["queue_rows"] == 48
    assert summary["portfolio"]["reviewed_events"] == 25
    for name in business.SOURCE_FILES:
        assert (target / "source" / name).read_bytes() == (source / name).read_bytes()
    shutil.rmtree(source)
    assert verify_analysis(target) == {
        "status": "verified",
        "loans": 480,
        "groups": 21,
        "reviewed": 48,
    }
    assert all(b"\r" not in path.read_bytes() for path in target.rglob("*") if path.is_file())


def test_source_is_frozen_before_verification_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "report"
    shutil.copytree(REPORT, source)
    original_verify = business.verify_result

    def concurrent_update(snapshot: Path) -> dict:
        checked = original_verify(snapshot)
        (source / "predictions.csv").write_text("inconsistent concurrent save", encoding="utf-8")
        return checked

    monkeypatch.setattr(business, "verify_result", concurrent_update)
    target = tmp_path / "analysis"
    write_analysis(source, target)
    assert (target / "source" / "predictions.csv").read_bytes() == (
        REPORT / "predictions.csv"
    ).read_bytes()
    assert verify_analysis(target)["loans"] == 480


def test_invalid_source_never_creates_output(tmp_path: Path) -> None:
    source = tmp_path / "report"
    shutil.copytree(REPORT, source)
    (source / "predictions.csv").write_text("malformed", encoding="utf-8")
    target = tmp_path / "analysis"
    with pytest.raises(DataContractError, match="digest mismatch"):
        write_analysis(source, target)
    assert not target.exists()


def test_existing_analysis_cannot_be_overwritten(tmp_path: Path) -> None:
    (tmp_path / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(DataContractError, match="new analysis directory"):
        write_analysis(REPORT, tmp_path)
    assert (tmp_path / "sentinel").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "name", ["groups.csv", "review_queue.csv", "summary.json", "index.html", "business_groups.sql"]
)
def test_rehashed_analysis_still_must_replay(tmp_path: Path, name: str) -> None:
    target = tmp_path / "analysis"
    write_analysis(REPORT, target)
    path = target / name
    if name == "groups.csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = list(reader)
        rows[0]["events"] = "0"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    elif name == "summary.json":
        document = json.loads(path.read_text(encoding="utf-8"))
        document["portfolio"]["event_capture"] = 1.0
        path.write_bytes(business._json_bytes(document))
    elif name == "index.html":
        path.write_text(
            path.read_text(encoding="utf-8").replace("23.4%", "100.0%"),
            encoding="utf-8",
            newline="",
        )
    else:
        path.write_bytes(path.read_bytes() + b"invented evidence\n")
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_bytes(business._json_bytes(manifest))
    with pytest.raises(DataContractError, match=name.replace(".", r"\.")):
        verify_analysis(target)


def test_unrelated_exports_cannot_enter_analysis(tmp_path: Path) -> None:
    target = tmp_path / "analysis"
    write_analysis(REPORT, target)
    (target / "unrelated.csv").write_text("unrelated", encoding="utf-8")
    with pytest.raises(DataContractError, match="file inventory"):
        verify_analysis(target)


def test_public_cli_generates_and_checks_analysis(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    target = tmp_path / "analysis"
    assert main(["analyze", str(REPORT), "--out", str(target)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["portfolio"]["events"] == 107
    assert main(["verify-analysis", str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert main(["analyze", str(REPORT), "--out", str(target)]) == 2
    assert "new analysis directory" in capsys.readouterr().err
