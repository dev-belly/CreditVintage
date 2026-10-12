"""Workflow regressions: competing reviewers, retries, history and label separation."""

from __future__ import annotations

import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from creditvintage import reviews
from creditvintage.business import write_analysis
from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.reviews import initialize_reviews, record_review, review_status

REPORT = Path(__file__).resolve().parents[1] / "docs" / "demo"


@pytest.fixture(scope="module")
def analysis(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("analysis") / "verified"
    write_analysis(REPORT, output)
    return output


@pytest.fixture
def workspace(analysis: Path, tmp_path: Path) -> tuple[Path, str]:
    database = tmp_path / "reviews.sqlite"
    initialize_reviews(analysis, database)
    return database, review_status(database)["cases"][0]["application_id"]


def test_complete_workflow_preserves_reason_and_evidence(workspace: tuple[Path, str]) -> None:
    database, identity = workspace
    record_review(database, identity, "claim", "alice", 0, "claim-1")
    record_review(
        database,
        identity,
        "resolve",
        "alice",
        1,
        "resolve-1",
        note="Income document needs a second review",
        evidence="synthetic-doc:income-001",
        outcome="escalated",
    )
    result = review_status(database)
    assert result["selected_cases"] == 48
    assert result["counts"] == {"queued": 47, "in_review": 0, "resolved": 1}
    assert result["outcomes"] == {"cleared": 0, "escalated": 1}
    assert result["completion_rate"] == 1 / 48
    assert result["cases"][0]["version"] == 2
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT note,evidence FROM events WHERE sequence=2"
        ).fetchone() == (
            "Income document needs a second review",
            "synthetic-doc:income-001",
        )


def test_retry_same_request_does_not_append_even_after_later_transition(
    workspace: tuple[Path, str],
) -> None:
    database, identity = workspace
    accepted = record_review(database, identity, "claim", "alice", 0, "retry")
    record_review(database, identity, "release", "alice", 1, "release", note="Shift ended")
    retry = record_review(database, identity, "claim", "alice", 0, "retry")
    assert retry["status"] == "replayed"
    assert retry["event"] == accepted["event"]
    assert review_status(database)["events"] == 2
    with pytest.raises(DataContractError, match="different content"):
        record_review(database, identity, "claim", "bob", 0, "retry")


def test_two_competing_claims_only_one_commits(workspace: tuple[Path, str]) -> None:
    database, identity = workspace

    def claim(actor: str) -> str:
        try:
            record_review(database, identity, "claim", actor, 0, actor)
            return "accepted"
        except DataContractError as error:
            assert "Stale" in str(error)
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(claim, ("alice", "bob"))) == ["accepted", "stale"]
    assert review_status(database)["events"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"actor": "bob"},
        {"expected_version": 0},
        {"note": ""},
        {"evidence": ""},
        {"outcome": "approved"},
        {"expected_version": True},
        {"request_id": " bad "},
        {"note": "bad\x00note"},
    ],
)
def test_invalid_resolution_never_appends(
    workspace: tuple[Path, str],
    changes: dict[str, object],
) -> None:
    database, identity = workspace
    record_review(database, identity, "claim", "alice", 0, "claim")
    arguments = dict(
        database=database,
        application_id=identity,
        action="resolve",
        actor="alice",
        expected_version=1,
        request_id="resolve",
        note="Checked supplied evidence",
        evidence="synthetic-doc:001",
        outcome="cleared",
    )
    arguments.update(changes)
    with pytest.raises(DataContractError):
        record_review(**arguments)
    assert review_status(database)["events"] == 1
    assert review_status(database)["cases"][0]["status"] == "in_review"


def test_reopen_keeps_history_and_requires_fresh_owner(workspace: tuple[Path, str]) -> None:
    database, identity = workspace
    record_review(database, identity, "claim", "alice", 0, "a")
    record_review(
        database,
        identity,
        "resolve",
        "alice",
        1,
        "b",
        note="Checked",
        evidence="synthetic-doc:001",
        outcome="cleared",
    )
    record_review(database, identity, "reopen", "supervisor", 2, "c", note="New document arrived")
    status = review_status(database)
    assert status["events"] == 3
    assert status["counts"]["resolved"] == 0
    assert status["cases"][0]["outcome"] is None
    assert status["cases"][0]["owner"] is None
    with pytest.raises(DataContractError, match="current owner"):
        record_review(
            database,
            identity,
            "resolve",
            "alice",
            3,
            "d",
            note="Checked",
            evidence="synthetic-doc:002",
            outcome="cleared",
        )
    record_review(database, identity, "claim", "bob", 3, "e")


def test_no_future_labels_or_source_files_in_workspace(workspace: tuple[Path, str]) -> None:
    database, _ = workspace
    with sqlite3.connect(database) as connection:
        queue = connection.execute("SELECT queue_csv FROM source").fetchone()[0]
        columns = queue.decode("utf-8").splitlines()[0].split(",")
        assert "label" not in columns
        assert "review_selected" not in columns
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall() == [("source",), ("events",)]


def test_bad_analysis_fails_before_database_creation(analysis: Path, tmp_path: Path) -> None:
    copied = tmp_path / "analysis"
    shutil.copytree(analysis, copied)
    (copied / "review_queue.csv").write_text("label\n1\n", encoding="utf-8")
    database = tmp_path / "never.sqlite"
    with pytest.raises(DataContractError):
        initialize_reviews(copied, database)
    assert not database.exists()


def test_append_only_guards_and_replay_reject_invalid_history(workspace: tuple[Path, str]) -> None:
    database, identity = workspace
    record_review(database, identity, "claim", "alice", 0, "claim")
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="Append-only"):
            connection.execute("DELETE FROM events")
        with pytest.raises(sqlite3.IntegrityError, match="Frozen"):
            connection.execute("UPDATE source SET queue_sha256='x'")
        # A database administrator can disable guards; state replay is still semantic.
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET expected_version=7")
    with pytest.raises(DataContractError, match="Stale"):
        review_status(database)


def test_cli_missing_database_and_invalid_transition_return_2(
    workspace: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    database, identity = workspace
    missing = tmp_path / "missing.sqlite"
    assert main(["review-status", str(missing)]) == 2
    assert not missing.exists()
    assert (
        main(
            [
                "review",
                str(database),
                identity,
                "resolve",
                "--actor",
                "alice",
                "--version",
                "0",
                "--request-id",
                "r",
                "--note",
                "Checked",
                "--evidence",
                "synthetic-doc:001",
                "--outcome",
                "cleared",
            ]
        )
        == 2
    )
    assert "current owner" in capsys.readouterr().err


def test_existing_database_is_never_replaced(workspace: tuple[Path, str], analysis: Path) -> None:
    database, identity = workspace
    record_review(database, identity, "claim", "alice", 0, "claim")
    before = database.read_bytes()
    with pytest.raises(FileExistsError):
        initialize_reviews(analysis, database)
    assert database.read_bytes() == before


def test_failed_initialization_cleans_only_new_file_and_can_retry(
    analysis: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "retry.sqlite"
    original_connect = sqlite3.connect

    def fail_new_database(path: object, *args: object, **kwargs: object) -> sqlite3.Connection:
        if path == database:
            raise sqlite3.OperationalError("disk write failure")
        return original_connect(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(reviews.sqlite3, "connect", fail_new_database)
        assert main(["review-init", str(analysis), "--db", str(database)]) == 2
    assert "Cannot initialize" in capsys.readouterr().err
    assert not database.exists()
    assert main(["review-init", str(analysis), "--db", str(database)]) == 0
    assert review_status(database)["selected_cases"] == 48


def test_same_request_racing_commits_once(workspace: tuple[Path, str]) -> None:
    database, identity = workspace

    def retry(_: int) -> str:
        return record_review(database, identity, "claim", "alice", 0, "same-request")["status"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(retry, (1, 2))) == ["accepted", "replayed"]
    assert review_status(database)["events"] == 1


def test_portable_workspace_does_not_need_original_analysis(
    workspace: tuple[Path, str],
    tmp_path: Path,
) -> None:
    database, identity = workspace
    record_review(database, identity, "claim", "alice", 0, "claim")
    copied = tmp_path / "moved" / "reviews.sqlite"
    copied.parent.mkdir()
    shutil.copyfile(database, copied)
    assert review_status(copied)["cases"][0]["owner"] == "alice"
