"""Delivery regressions: snapshot replay, corrupt evidence and trusted checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path
from shutil import copytree
from threading import Thread
from typing import Any
from zipfile import ZipFile

import pytest

from creditvintage import review_bundle, workbench
from creditvintage.business import write_analysis
from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.review_bundle import (
    compare_review_bundles,
    export_review_zip,
    export_reviews,
    verify_review_bundle,
)
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
    original, receipt = export_review_zip(database)
    identity = result["cases"][0]["application_id"]
    record_review(database, identity, "claim", "alice", 0, "claim")
    assert export_review_zip(
        database, through_event=0, expected_sha256=receipt["bundle_sha256"]
    ) == (original, receipt)


def test_cli_recreates_an_earlier_delivery_without_advancing_the_workspace(
    delivery: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, directory, identity = delivery
    original, receipt = export_review_zip(database)
    digest = receipt["bundle_sha256"]
    record_review(database, identity, "reopen", "supervisor", 2, "reopen", note="New evidence")
    replayed = tmp_path / "recreated"
    command = ["review-export", str(database), "--out", str(replayed)]

    assert main([*command, "--through-event", "4", "--expected-sha256", digest]) == 2
    assert "0 through 3" in capsys.readouterr().err
    assert main([*command, "--through-event", "1", "--expected-sha256", digest]) == 2
    assert "checkpoint mismatch" in capsys.readouterr().err
    assert not replayed.exists()

    assert main([*command, "--through-event", "2", "--expected-sha256", digest]) == 0
    assert json.loads(capsys.readouterr().out) == {"directory": str(replayed), **receipt}
    assert {file.name: file.read_bytes() for file in replayed.iterdir()} == {
        file.name: file.read_bytes() for file in directory.iterdir()
    }
    assert export_review_zip(database, through_event=2, expected_sha256=digest)[0] == original
    assert review_status(database)["events"] == 3


def test_downloaded_archive_uses_exact_directory_evidence(
    delivery: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, directory, _ = delivery
    raw, receipt = export_review_zip(database)
    assert export_review_zip(database)[0] == raw
    with ZipFile(BytesIO(raw)) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == {
            member.name: member.read_bytes() for member in directory.iterdir()
        }
    evidence = tmp_path / "download.zip"
    evidence.write_bytes(raw)
    database.unlink()
    digest = receipt["bundle_sha256"]
    assert main(["verify-reviews", str(evidence), "--expected-sha256", digest]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint_matched"] is True
    assert result["events"] == 2
    assert result["counts"] == {"queued": 47, "in_review": 0, "resolved": 1}
    assert main(["verify-reviews", str(evidence), "--expected-sha256", "0" * 64]) == 2
    assert "checkpoint mismatch" in capsys.readouterr().err
    assert evidence.read_bytes() == raw


def test_archive_rejects_ambiguous_or_extra_members(
    delivery: tuple[Path, Path, str], tmp_path: Path
) -> None:
    database, _, _ = delivery
    raw, _ = export_review_zip(database)
    evidence = tmp_path / "evidence.zip"
    evidence.write_bytes(raw)
    with ZipFile(evidence, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("status.json", b"{}")
    with pytest.raises(DataContractError, match="evidence files"):
        verify_review_bundle(evidence)

    evidence.write_bytes(raw)
    with ZipFile(evidence, "a") as archive:
        archive.writestr("extra.csv", b"")
    with pytest.raises(DataContractError, match="evidence files"):
        verify_review_bundle(evidence)

    evidence.write_bytes(b"incomplete download")
    with pytest.raises(DataContractError, match="Cannot verify review evidence"):
        verify_review_bundle(evidence)


@contextmanager
def _workbench_client(database: Path | None) -> Iterator[HTTPConnection]:
    with workbench.create_workbench_server(database, 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            yield client
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=5)


@pytest.fixture
def review_client(delivery: tuple[Path, Path, str]) -> Iterator[HTTPConnection]:
    database, _, _ = delivery
    with _workbench_client(database) as client:
        yield client


def test_reader_replays_a_delivery_without_the_review_database(
    delivery: tuple[Path, Path, str],
) -> None:
    database, directory, _ = delivery
    archive, receipt = export_review_zip(database)
    digest = receipt["bundle_sha256"]
    database.unlink()
    expected = verify_review_bundle(directory, expected_sha256=digest)
    expected["history"] = json.loads((directory / "events.json").read_bytes())

    with _workbench_client(None) as client:
        client.request("GET", "/")
        page = client.getresponse()
        assert page.status == 200
        assert b'data-mode="reader"' in page.read()
        origin = f"http://127.0.0.1:{client.port}"
        headers = {
            "Origin": origin,
            "Content-Type": "application/zip",
            "X-CreditVintage-Checkpoint": digest,
        }
        client.request("POST", "/api/replay", body=archive, headers=headers)
        replay = client.getresponse()
        assert replay.status == 200
        assert replay.getheader("Cache-Control") == "no-store"
        assert json.loads(replay.read()) == expected

        client.request("GET", "/api/state")
        unavailable = client.getresponse()
        assert unavailable.status == 404
        unavailable.read()
        client.request("POST", "/api/change", body=b"{}", headers={"Origin": origin})
        unavailable = client.getresponse()
        assert unavailable.status == 404
        unavailable.read()
    assert not database.exists()


def test_reader_requires_same_origin_and_checks_the_supplied_checkpoint(
    delivery: tuple[Path, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _, _ = delivery
    archive, _ = export_review_zip(database)
    with _workbench_client(None) as client:
        headers = {"Origin": "https://example.com", "Content-Type": "application/zip"}
        client.request("POST", "/api/replay", body=archive, headers=headers)
        forbidden = client.getresponse()
        assert forbidden.status == 403
        forbidden.read()

        headers["Origin"] = f"http://127.0.0.1:{client.port}"
        client.request("POST", "/api/replay", body=archive, headers=headers)
        success = client.getresponse()
        assert success.status == 200
        assert json.loads(success.read())["checkpoint_matched"] is False

        headers["X-CreditVintage-Checkpoint"] = "0" * 64
        client.request("POST", "/api/replay", body=archive, headers=headers)
        mismatch = client.getresponse()
        assert mismatch.status == 422
        assert "checkpoint mismatch" in json.loads(mismatch.read())["error"]

        with monkeypatch.context() as limits:
            limits.setattr(workbench, "MAX_REVIEW_ZIP_BYTES", len(archive) - 1)
            client.request("POST", "/api/replay", body=archive, headers=headers)
            oversized = client.getresponse()
            assert oversized.status == 413
            oversized.read()

        monkeypatch.setattr(workbench, "MAX_REVIEW_UNPACKED_BYTES", len(archive))
        client.request("POST", "/api/replay", body=archive, headers=headers)
        oversized_contents = client.getresponse()
        assert oversized_contents.status == 422
        assert "contents exceed" in json.loads(oversized_contents.read())["error"]


def test_review_web_can_start_with_only_a_port(monkeypatch: pytest.MonkeyPatch) -> None:
    starts: list[tuple[Path | None, int]] = []
    monkeypatch.setattr(workbench, "serve", lambda database, port: starts.append((database, port)))
    assert main(["review-web", "--port", "0"]) == 0
    assert main(["review-web", "reviews.sqlite", "--port", "9000"]) == 0
    assert starts == [(None, 0), (Path("reviews.sqlite"), 9000)]


def test_reader_compares_two_deliveries_without_a_database(
    delivery: tuple[Path, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, earlier_directory, identity = delivery
    earlier, baseline = export_review_zip(database)
    reopened = record_review(
        database, identity, "reopen", "supervisor", 2, "reopen", note="Check another document"
    )
    later, current = export_review_zip(database)
    later_zip = tmp_path / "later.zip"
    later_zip.write_bytes(later)
    database.unlink()
    baseline_digest, current_digest = baseline["bundle_sha256"], current["bundle_sha256"]

    with _workbench_client(None) as client:
        origin = f"http://127.0.0.1:{client.port}"
        boundary = "creditvintage-deliveries"
        headers = {
            "Origin": origin,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-CreditVintage-Baseline-Checkpoint": baseline_digest,
            "X-CreditVintage-Current-Checkpoint": current_digest,
        }

        def submit(parts: tuple[tuple[str, bytes], ...]) -> tuple[int, dict[str, Any]]:
            chunks = []
            for name, content in parts:
                chunks.append(
                    (
                        f"--{boundary}\r\n"
                        f'Content-Disposition: form-data; name="{name}"; '
                        'filename="delivery.zip"\r\n'
                        "Content-Type: application/zip\r\n\r\n"
                    ).encode("ascii")
                    + content
                    + b"\r\n"
                )
            chunks.append(f"--{boundary}--\r\n".encode("ascii"))
            client.request("POST", "/api/compare-deliveries", b"".join(chunks), headers)
            response = client.getresponse()
            return response.status, json.loads(response.read())

        pair = (("baseline", earlier), ("current", later))
        status, result = submit(pair)
        assert status == 200
        assert result == compare_review_bundles(
            earlier_directory,
            later_zip,
            baseline_sha256=baseline_digest,
            current_sha256=current_digest,
            include_actions=True,
        )
        assert result["new_actions"] == [reopened["event"]]
        assert result["changed_cases"][0]["after"]["status"] == "queued"

        headers["X-CreditVintage-Current-Checkpoint"] = "0" * 64
        status, result = submit(pair)
        assert status == 422 and "Current evidence" in result["error"]
        assert "checkpoint mismatch" in result["error"]
        headers["X-CreditVintage-Current-Checkpoint"] = current_digest
        headers["Origin"] = "https://example.com"
        assert submit(pair)[0] == 403
        headers["Origin"] = origin
        assert submit((pair[0], pair[0]))[0] == 422

        with monkeypatch.context() as limits:
            limits.setattr(workbench, "MAX_REVIEW_ZIP_BYTES", len(later) - 1)
            assert submit(pair)[0] == 413
        with ZipFile(BytesIO(earlier)) as archive:
            earlier_unpacked = sum(member.file_size for member in archive.infolist())
        monkeypatch.setattr(workbench, "MAX_REVIEW_UNPACKED_BYTES", earlier_unpacked)
        status, result = submit(pair)
        assert status == 422 and "Current evidence" in result["error"]
        assert "contents exceed" in result["error"]
    assert not database.exists()


def test_loopback_download_returns_verified_snapshot_and_checkpoint(
    review_client: HTTPConnection, tmp_path: Path
) -> None:
    review_client.request("GET", "/api/evidence", headers={"Sec-Fetch-Site": "cross-site"})
    forbidden = review_client.getresponse()
    assert forbidden.status == 403
    forbidden.read()
    review_client.request("GET", "/api/evidence", headers={"Sec-Fetch-Site": "same-origin"})
    response = review_client.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "application/zip"
    assert response.getheader("Cache-Control") == "no-store"
    digest = response.getheader("X-CreditVintage-Checkpoint")
    assert digest and response.getheader("Content-Disposition") == (
        f'attachment; filename="creditvintage-{digest[:12]}.zip"'
    )
    archive = tmp_path / "from-browser.zip"
    archive.write_bytes(response.read())
    result = verify_review_bundle(archive, expected_sha256=digest)
    assert result["checkpoint_matched"] is True
    assert result["counts"]["resolved"] == 1
    assert result["events"] == 2


def test_loopback_comparison_matches_the_two_cli_snapshots(
    delivery: tuple[Path, Path, str], review_client: HTTPConnection, tmp_path: Path
) -> None:
    database, baseline_directory, identity = delivery
    archive, baseline = export_review_zip(database)
    reopened = record_review(
        database, identity, "reopen", "supervisor", 2, "reopen", note="New evidence"
    )
    current_directory = tmp_path / "current"
    current = export_reviews(database, current_directory)

    headers = {
        "Origin": f"http://127.0.0.1:{review_client.port}",
        "Content-Type": "application/zip",
        "X-CreditVintage-Baseline-Checkpoint": baseline["bundle_sha256"],
    }
    review_client.request("POST", "/api/compare", body=archive, headers=headers)
    response = review_client.getresponse()
    assert response.status == 200
    assert response.getheader("Cache-Control") == "no-store"
    result = json.loads(response.read())
    assert result == compare_review_bundles(
        baseline_directory,
        current_directory,
        baseline_sha256=baseline["bundle_sha256"],
        include_actions=True,
    )
    assert result["added_events"] == 1
    assert result["new_actions"] == [reopened["event"]]
    assert result["changed_cases"][0]["after"]["status"] == "queued"
    assert result["current"]["bundle_sha256"] == current["bundle_sha256"]
    assert review_status(database)["events"] == 3

    headers["X-CreditVintage-Baseline-Checkpoint"] = "0" * 64
    review_client.request("POST", "/api/compare", body=archive, headers=headers)
    mismatch = review_client.getresponse()
    assert mismatch.status == 422
    assert "checkpoint mismatch" in json.loads(mismatch.read())["error"]

    record_review(database, identity, "claim", "alice", 3, "next-claim")
    digest = result["current"]["bundle_sha256"]
    route = f"/api/evidence?through_event=3&expected_sha256={digest}"
    review_client.request("GET", route, headers={"Sec-Fetch-Site": "cross-site"})
    blocked = review_client.getresponse()
    assert blocked.status == 403
    blocked.read()

    review_client.request("GET", route)
    download = review_client.getresponse()
    assert download.status == 200
    assert download.getheader("X-CreditVintage-Checkpoint") == digest
    path = tmp_path / "compared.zip"
    path.write_bytes(download.read())
    assert verify_review_bundle(path, expected_sha256=digest)["events"] == 3
    assert review_status(database)["events"] == 4

    review_client.request("GET", route.replace("through_event=3", "through_event=4"))
    wrong_snapshot = review_client.getresponse()
    assert wrong_snapshot.status == 422
    assert wrong_snapshot.getheader("X-CreditVintage-Checkpoint") is None
    assert "checkpoint mismatch" in json.loads(wrong_snapshot.read())["error"]
    review_client.request("GET", route + "&through_event=4")
    invalid = review_client.getresponse()
    assert invalid.status == 400
    invalid.read()


def test_loopback_comparison_checks_origin_and_archive_size(
    delivery: tuple[Path, Path, str], review_client: HTTPConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _, _ = delivery
    archive, _ = export_review_zip(database)
    headers = {"Origin": "https://example.com", "Content-Type": "application/zip"}
    review_client.request("POST", "/api/compare", body=archive, headers=headers)
    rejected = review_client.getresponse()
    assert rejected.status == 403
    rejected.read()

    headers["Origin"] = f"http://127.0.0.1:{review_client.port}"
    with monkeypatch.context() as limits:
        limits.setattr(workbench, "MAX_REVIEW_ZIP_BYTES", len(archive) - 1)
        review_client.request("POST", "/api/compare", body=archive, headers=headers)
        too_large = review_client.getresponse()
        assert too_large.status == 413
        too_large.read()

    monkeypatch.setattr(workbench, "MAX_REVIEW_UNPACKED_BYTES", len(archive))
    review_client.request("POST", "/api/compare", body=archive, headers=headers)
    too_large_unpacked = review_client.getresponse()
    assert too_large_unpacked.status == 422
    assert "contents exceed" in json.loads(too_large_unpacked.read())["error"]


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


def test_compare_deliveries_reports_appended_actions(
    delivery: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, baseline, identity = delivery
    before = {path.name: path.read_bytes() for path in baseline.iterdir()}
    baseline_digest = hashlib.sha256(before["manifest.json"]).hexdigest()
    second_case = review_status(database)["cases"][1]["application_id"]
    record_review(database, identity, "reopen", "supervisor", 2, "reopen", note="New evidence")
    record_review(database, second_case, "claim", "alice", 0, "another-claim")
    record_review(database, second_case, "release", "alice", 1, "release", note="Shift ended")
    archive, receipt = export_review_zip(database)
    current = tmp_path / "current.zip"
    current.write_bytes(archive)
    database.unlink()

    command = ["compare-reviews", str(baseline), str(current), "--baseline-sha256", baseline_digest]
    assert main([*command, "--current-sha256", receipt["bundle_sha256"]]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["baseline"]["checkpoint_matched"] is True
    assert result["current"]["checkpoint_matched"] is True
    assert result["baseline"]["counts"] == {"queued": 47, "in_review": 0, "resolved": 1}
    assert result["current"]["counts"] == {"queued": 48, "in_review": 0, "resolved": 0}
    assert result["added_events"] == 3
    assert "new_actions" not in result
    assert result["changed_cases"] == [
        {
            "application_id": identity,
            "before": {"version": 2, "status": "resolved", "outcome": "escalated"},
            "after": {"version": 3, "status": "queued", "outcome": None},
        },
        {
            "application_id": second_case,
            "before": {"version": 0, "status": "queued", "outcome": None},
            "after": {"version": 2, "status": "queued", "outcome": None},
        },
    ]
    assert main([*command, "--include-actions", "--current-sha256", receipt["bundle_sha256"]]) == 0
    detailed = json.loads(capsys.readouterr().out)
    with ZipFile(current) as evidence:
        assert detailed.pop("new_actions") == json.loads(evidence.read("events.json"))[2:]
    assert detailed == result
    assert main([*command, "--current-sha256", baseline_digest]) == 2
    error = capsys.readouterr()
    assert error.out == ""
    assert "Current evidence" in error.err and "checkpoint mismatch" in error.err
    with pytest.raises(DataContractError, match="history does not extend.*fewer events"):
        compare_review_bundles(current, baseline)
    assert current.read_bytes() == archive
    assert {path.name: path.read_bytes() for path in baseline.iterdir()} == before


def test_compare_identical_deliveries_reports_no_changes(delivery: tuple[Path, Path, str]) -> None:
    _, baseline, _ = delivery
    result = compare_review_bundles(baseline, baseline, include_actions=True)
    assert result["added_events"] == 0
    assert result["new_actions"] == []
    assert result["changed_cases"] == []
    assert result["baseline"] == result["current"]
    assert result["baseline"]["checkpoint_matched"] is False


def test_compare_rejects_a_valid_but_rewritten_history(
    delivery: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, baseline, _ = delivery
    current = copytree(baseline, tmp_path / "edited")
    events = json.loads((current / "events.json").read_bytes())
    events[1]["note"] = "A different valid narrative"
    _rewrite(current, "events.json", events)
    assert verify_review_bundle(current)["events"] == 2
    with pytest.raises(DataContractError, match="history does not extend.*event 2"):
        compare_review_bundles(baseline, current)


def test_compare_rejects_a_different_frozen_source(
    delivery: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, baseline, _ = delivery
    current = copytree(baseline, tmp_path / "other-source")
    status = json.loads((current / "status.json").read_bytes())
    status["source"]["analysis_sha256"] = "f" * 64
    _rewrite(current, "source.json", status["source"])
    _rewrite(current, "status.json", status)
    assert verify_review_bundle(current)["events"] == 2
    with pytest.raises(DataContractError, match="different frozen sources"):
        compare_review_bundles(baseline, current)


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


@pytest.mark.parametrize(
    "name, old, ambiguous",
    [
        ("source.json", '"id": 1', '"id": 0, "id": 1'),
        ("events.json", '"actor": "alice"', '"actor": "bob", "actor": "alice"'),
        (
            "manifest.json",
            '"format": "creditvintage.review-evidence.v1"',
            '"format": "unknown", "format": "creditvintage.review-evidence.v1"',
        ),
    ],
)
def test_duplicate_keys_rejected_even_with_matching_payload_hash(
    delivery: tuple[Path, Path, str], name: str, old: str, ambiguous: str
) -> None:
    _, directory, _ = delivery
    path = directory / name
    original = path.read_text(encoding="utf-8")
    assert old in original
    path.write_text(original.replace(old, ambiguous, 1), encoding="utf-8")
    if name != "manifest.json":
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_bytes(review_bundle._json(manifest))
    with pytest.raises(DataContractError, match="Duplicate JSON key"):
        verify_review_bundle(directory)
