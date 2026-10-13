"""Delivery regressions: snapshot replay, corrupt evidence and trusted checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from creditvintage import review_bundle
from creditvintage.business import write_analysis
from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.review_bundle import export_reviews, verify_review_bundle
from creditvintage.reviews import initialize_reviews, record_review, review_status


@pytest.fixture(scope="module")
def analysis(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("bundle-analysis") / "verified"
    write_analysis(Path(__file__).resolve().parents[1] / "docs" / "demo", output)
    return output


@pytest.fixture
def delivery(analysis: Path, tmp_path: Path) -> tuple[Path, Path, str]:
    database = tmp_path / "reviews.sqlite"
    initialize_reviews(analysis, database)
    identity = review_status(database)["cases"][0]["application_id"]
    record_review(database, identity, "claim", "alice", 0, "claim")
    record_review(
        database,
        identity,
        "resolve",
        "alice",
        1,
        "resolve",
        note="Checked synthetic income evidence",
        evidence="synthetic-doc:001",
        outcome="escalated",
    )
    directory = tmp_path / "delivery"
    export_reviews(database, directory)
    return database, directory, identity


def _rewrite(directory: Path, name: str, value: Any) -> None:
    (directory / name).write_bytes(review_bundle._json(value))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest_path.write_bytes(review_bundle._json(manifest))


def test_portable_delivery_replays_without_database(delivery: tuple[Path, Path, str]) -> None:
    database, directory, _ = delivery
    expected = review_status(database)
    database.unlink()
    digest = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    result = verify_review_bundle(directory, expected_sha256=digest)
    assert {key: result[key] for key in expected} == expected
    assert result["checkpoint_matched"] is True
    assert result["events"] == 2
    assert result["counts"] == {"queued": 47, "in_review": 0, "resolved": 1}
    assert result["completion_rate"] == 1 / 48
    assert (
        "label" not in (directory / "review_queue.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert set(json.loads((directory / "source.json").read_bytes())) == {
        "id",
        "schema_version",
        "analysis_sha256",
        "queue_sha256",
        "data_kind",
    }


def test_empty_journal_is_deliverable(analysis: Path, tmp_path: Path) -> None:
    database, directory = tmp_path / "empty.sqlite", tmp_path / "empty"
    initialize_reviews(analysis, database)
    exported = export_reviews(database, directory)
    result = verify_review_bundle(directory)
    assert result["events"] == 0
    assert result["counts"]["queued"] == 48
    assert result["checkpoint_matched"] is False
    assert result["bundle_sha256"] == exported["bundle_sha256"]


def test_delivery_remains_one_snapshot_when_workspace_advances(
    delivery: tuple[Path, Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database, _, identity = delivery
    original_status = review_bundle._status

    def advance_after_capture(*arguments: Any) -> dict[str, Any]:
        record_review(database, identity, "reopen", "supervisor", 2, "reopen", note="New evidence")
        return original_status(*arguments)

    with monkeypatch.context() as patch:
        patch.setattr(review_bundle, "_status", advance_after_capture)
        export_reviews(database, tmp_path / "snapshot")
    assert review_status(database)["events"] == 3
    assert verify_review_bundle(tmp_path / "snapshot")["events"] == 2
    assert verify_review_bundle(tmp_path / "snapshot")["counts"]["resolved"] == 1


@pytest.mark.parametrize("name", review_bundle.FILES)
def test_digest_rejects_changed_payload(delivery: tuple[Path, Path, str], name: str) -> None:
    _, directory, _ = delivery
    with (directory / name).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(DataContractError, match="digest mismatch"):
        verify_review_bundle(directory)


def test_replay_rejects_forged_totals_even_with_updated_manifest(
    delivery: tuple[Path, Path, str],
) -> None:
    _, directory, _ = delivery
    status = json.loads((directory / "status.json").read_bytes())
    status["counts"]["resolved"] = 48
    status["completion_rate"] = 1.0
    _rewrite(directory, "status.json", status)
    with pytest.raises(DataContractError, match="event replay"):
        verify_review_bundle(directory)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"expected_version": 9}, "Stale"),
        ({"actor": "bob"}, "current owner"),
        ({"sequence": 4}, "sequence"),
        ({"occurred_at": "2026-10-13T10:00:00"}, "UTC"),
        ({"label": 1}, "fields"),
    ],
)
def test_replay_rejects_illegal_history_with_updated_manifest(
    delivery: tuple[Path, Path, str], changes: dict[str, Any], match: str
) -> None:
    _, directory, _ = delivery
    events = json.loads((directory / "events.json").read_bytes())
    events[1].update(changes)
    _rewrite(directory, "events.json", events)
    with pytest.raises(DataContractError, match=match):
        verify_review_bundle(directory)


def test_external_checkpoint_detects_consistent_history_rewrite(
    delivery: tuple[Path, Path, str],
) -> None:
    _, directory, _ = delivery
    saved = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    events = json.loads((directory / "events.json").read_bytes())
    events[1]["note"] = "A different valid narrative"
    _rewrite(directory, "events.json", events)
    # Notes do not alter projected status: internal checks cannot prove original authorship.
    assert verify_review_bundle(directory)["checkpoint_matched"] is False
    with pytest.raises(DataContractError, match="checkpoint mismatch"):
        verify_review_bundle(directory, expected_sha256=saved)


@pytest.mark.parametrize("value", [None, {}, [None]])
def test_bad_event_container_fails_as_contract_error(
    delivery: tuple[Path, Path, str], value: Any
) -> None:
    _, directory, _ = delivery
    _rewrite(directory, "events.json", value)
    with pytest.raises(DataContractError, match="objects"):
        verify_review_bundle(directory)


def test_extra_file_rejected_and_existing_delivery_never_overwritten(
    delivery: tuple[Path, Path, str],
) -> None:
    database, directory, _ = delivery
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(DataContractError, match="export"):
        export_reviews(database, directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    (directory / "future_labels.csv").write_text("label\n1\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="evidence files"):
        verify_review_bundle(directory)


def test_failed_write_cleans_only_new_export(
    delivery: tuple[Path, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, original, _ = delivery
    directory = tmp_path / "failed"
    original_write = Path.write_bytes

    def fail_events(path: Path, content: bytes) -> int:
        if path == directory / "events.json":
            raise OSError("disk full")
        return original_write(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail_events)
        with pytest.raises(DataContractError, match="disk full"):
            export_reviews(database, directory)
    assert not directory.exists()
    assert verify_review_bundle(original)["events"] == 2
    export_reviews(database, directory)
    assert verify_review_bundle(directory)["events"] == 2


def test_cli_checkpoint_and_missing_database(
    delivery: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, _, _ = delivery
    out = tmp_path / "cli"
    assert main(["review-export", str(database), "--out", str(out)]) == 0
    saved = json.loads(capsys.readouterr().out)["bundle_sha256"]
    assert main(["verify-reviews", str(out), "--expected-sha256", saved]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_matched"] is True
    assert main(["verify-reviews", str(out), "--expected-sha256", "0" * 64]) == 2
    assert "checkpoint mismatch" in capsys.readouterr().err
    missing = tmp_path / "missing.sqlite"
    assert main(["review-export", str(missing), "--out", str(tmp_path / "absent")]) == 2
    assert not missing.exists()
    assert not (tmp_path / "absent").exists()
