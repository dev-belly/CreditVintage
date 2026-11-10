"""A matching digest cannot make malformed review-case data valid."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from creditvintage import review_bundle
from creditvintage.business import QUEUE_COLUMNS, write_analysis
from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.review_bundle import export_reviews, verify_review_bundle
from creditvintage.reviews import initialize_reviews, record_review, review_status

Rows = list[dict[str, str]]


@pytest.fixture(scope="module")
def seed_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("review-queue-contract")
    write_analysis(Path(__file__).resolve().parents[1] / "docs" / "demo", root / "analysis")
    database = root / "seed.sqlite"
    initialize_reviews(root / "analysis", database)
    return database


@pytest.fixture
def workspace(seed_workspace: Path, tmp_path: Path) -> Path:
    database = tmp_path / "reviews.sqlite"
    shutil.copyfile(seed_workspace, database)
    return database


def _csv_bytes(rows: Rows) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=QUEUE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _change_database(database: Path, change: Callable[[Rows], None]) -> None:
    # Model an administrator rewriting both frozen bytes and their digest.
    with sqlite3.connect(database) as connection:
        original = connection.execute("SELECT queue_csv FROM source").fetchone()[0]
        rows = list(csv.DictReader(io.StringIO(original.decode("utf-8"))))
        change(rows)
        queue = _csv_bytes(rows)
        connection.execute("DROP TRIGGER source_no_update")
        connection.execute(
            "UPDATE source SET queue_csv=?, queue_sha256=?",
            (queue, hashlib.sha256(queue).hexdigest()),
        )


def _change_bundle(directory: Path, change: Callable[[Rows], None]) -> None:
    # Rebuild matching hashes and derived status without calling the replay
    # implementation under test. Every seed case is queued with no journal.
    assert json.loads((directory / "events.json").read_bytes()) == []
    with (directory / "review_queue.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    change(rows)
    queue = _csv_bytes(rows)
    (directory / "review_queue.csv").write_bytes(queue)
    source = json.loads((directory / "source.json").read_bytes())
    source["queue_sha256"] = hashlib.sha256(queue).hexdigest()
    status = json.loads((directory / "status.json").read_bytes())
    status["source"] = source
    status["cases"] = [{**case, **row} for case, row in zip(status["cases"], rows, strict=True)]
    (directory / "source.json").write_bytes(review_bundle._json(source))
    (directory / "status.json").write_bytes(review_bundle._json(status))
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["files"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in review_bundle.FILES
    }
    (directory / "manifest.json").write_bytes(review_bundle._json(manifest))


def _snapshot(path: Path) -> dict[str, bytes]:
    if path.is_file():
        return {path.name: path.read_bytes()}
    return {member.name: member.read_bytes() for member in path.iterdir()}


BAD_VALUES = [
    ("application_id", "bad id"),
    ("application_id", "a" * 129),
    ("decision_at", "20240701"),
    ("decision_at", "2024-02-30"),
    ("decision_at", "2024-07-01T00:00:00"),
    ("vintage", "2000-01"),
    ("segment", "corporate"),
    ("principal_cents", "0"),
    ("principal_cents", "-1"),
    ("principal_cents", "1.25"),
    ("principal_cents", str(2**63)),
    *[
        (field, value)
        for field in ("raw_pd", "calibrated_pd")
        for value in ("nan", "inf", "-inf", "-0.1", "1.1", "not-a-number")
    ],
]


@pytest.mark.parametrize("channel", ["workspace", "bundle"])
@pytest.mark.parametrize("field,value", BAD_VALUES)
def test_matching_hashes_do_not_admit_invalid_case_fields(
    workspace: Path, tmp_path: Path, channel: str, field: str, value: str
) -> None:
    def change(rows: Rows) -> None:
        rows[0][field] = value

    target = workspace
    if channel == "workspace":
        _change_database(workspace, change)
    else:
        target = tmp_path / "bundle"
        export_reviews(workspace, target)
        _change_bundle(target, change)
    before = _snapshot(target)
    with pytest.raises(DataContractError, match="queue"):
        review_status(target) if channel == "workspace" else verify_review_bundle(target)
    assert _snapshot(target) == before


@pytest.mark.parametrize("channel", ["workspace", "bundle"])
@pytest.mark.parametrize("ties", [False, True], ids=["score-order", "identity-ties"])
def test_queue_priorities_must_follow_descending_score_then_identity(
    workspace: Path, tmp_path: Path, channel: str, ties: bool
) -> None:
    def change(rows: Rows) -> None:
        if ties:
            for row in rows:
                row["calibrated_pd"] = "0.5"
            rows.sort(key=lambda row: row["application_id"], reverse=True)
        else:
            rows.reverse()
        for rank, row in enumerate(rows, 1):
            row["review_rank"] = str(rank)

    if channel == "workspace":
        _change_database(workspace, change)
        with pytest.raises(DataContractError, match="queue"):
            review_status(workspace)
    else:
        directory = tmp_path / "bundle"
        export_reviews(workspace, directory)
        _change_bundle(directory, change)
        with pytest.raises(DataContractError, match="queue"):
            verify_review_bundle(directory)


@pytest.mark.parametrize("channel", ["workspace", "bundle"])
def test_legal_ties_use_identity_order(workspace: Path, tmp_path: Path, channel: str) -> None:
    def change(rows: Rows) -> None:
        rows.sort(key=lambda row: row["application_id"])
        for rank, row in enumerate(rows, 1):
            row.update(calibrated_pd="0.5", review_rank=str(rank))

    if channel == "workspace":
        _change_database(workspace, change)
        actual = review_status(workspace)
    else:
        directory = tmp_path / "bundle"
        export_reviews(workspace, directory)
        _change_bundle(directory, change)
        actual = verify_review_bundle(directory)
    identities = [case["application_id"] for case in actual["cases"]]
    assert identities == sorted(identities)
    assert actual["selected_cases"] == actual["counts"]["queued"] == 48


@pytest.mark.parametrize(
    "field,value,index",
    [
        ("raw_pd", "0", 0),
        ("raw_pd", "1", 0),
        ("calibrated_pd", "1", 0),
        ("calibrated_pd", "0", -1),
        ("principal_cents", "1", 0),
        ("principal_cents", str(2**63 - 1), 0),
    ],
)
def test_valid_closed_bounds_preserve_original_field_text(
    workspace: Path, tmp_path: Path, field: str, value: str, index: int
) -> None:
    directory = tmp_path / "bundle"
    export_reviews(workspace, directory)

    def change(rows: Rows) -> None:
        rows[index][field] = value

    _change_bundle(directory, change)
    actual = verify_review_bundle(directory)
    assert actual["cases"][index][field] == value
    assert actual["checkpoint_matched"] is False


@pytest.mark.parametrize("operation", ["claim", "export"])
def test_invalid_queue_cannot_append_events_or_create_delivery(
    workspace: Path, tmp_path: Path, operation: str
) -> None:
    identity = review_status(workspace)["cases"][0]["application_id"]
    _change_database(workspace, lambda rows: rows[0].update(raw_pd="nan"))
    before = workspace.read_bytes()
    directory = tmp_path / "must-not-exist"
    with pytest.raises(DataContractError, match="queue"):
        if operation == "claim":
            record_review(workspace, identity, "claim", "alice", 0, "invalid-claim")
        else:
            export_reviews(workspace, directory)
    with sqlite3.connect(workspace) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert not directory.exists()
    assert workspace.read_bytes() == before


@pytest.mark.parametrize("channel", ["workspace", "bundle"])
def test_cli_reports_invalid_queue_without_traceback(
    workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], channel: str
) -> None:
    if channel == "workspace":
        _change_database(workspace, lambda rows: rows[0].update(raw_pd="nan"))
        arguments = ["review-status", str(workspace)]
    else:
        directory = tmp_path / "bundle"
        export_reviews(workspace, directory)
        _change_bundle(directory, lambda rows: rows[0].update(raw_pd="nan"))
        arguments = ["verify-reviews", str(directory)]
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "queue" in captured.err and "Traceback" not in captured.err


def test_valid_consistent_rewrite_still_requires_external_checkpoint(
    workspace: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "bundle"
    original = export_reviews(workspace, directory)
    _change_bundle(directory, lambda rows: rows[0].update(raw_pd="0.125"))
    assert verify_review_bundle(directory)["checkpoint_matched"] is False
    with pytest.raises(DataContractError, match="checkpoint"):
        verify_review_bundle(directory, expected_sha256=original["bundle_sha256"])


@pytest.mark.parametrize(
    "decision_at,vintage",
    [("0001-01-01", "1-01"), ("0001-01-01", "0001-01"), ("9999-12-31", "9999-12")],
)
def test_valid_calendar_bounds_and_legacy_year_padding_remain_readable(
    workspace: Path, tmp_path: Path, decision_at: str, vintage: str
) -> None:
    directory = tmp_path / "bundle"
    export_reviews(workspace, directory)
    _change_bundle(directory, lambda rows: rows[0].update(decision_at=decision_at, vintage=vintage))
    actual = verify_review_bundle(directory)
    assert actual["cases"][0]["decision_at"] == decision_at
    assert actual["cases"][0]["vintage"] == vintage
