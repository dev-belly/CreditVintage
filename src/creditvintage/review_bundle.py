"""Portable, label-free review evidence with replay and optional checkpoint anchoring."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo
from zlib import error as DeflateError

from creditvintage.core import DataContractError
from creditvintage.reviews import _connection, _load, _replay, _status
from creditvintage.validation import parse_json

FORMAT = "creditvintage.review-evidence.v1"
FILES = ("source.json", "review_queue.csv", "events.json", "status.json")


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _capture(database: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    with _connection(database) as connection:
        connection.execute("BEGIN")
        source, events, states = _load(connection)
        queue = connection.execute("SELECT queue_csv FROM source WHERE id=1").fetchone()[0]
    status = _status(source, events, states)
    captured: dict[str, bytes] = {
        "source.json": _json(source),
        "review_queue.csv": queue,
        "events.json": _json(events),
        "status.json": _json(status),
    }
    manifest = _json(
        {
            "format": FORMAT,
            "files": {
                name: hashlib.sha256(content).hexdigest() for name, content in captured.items()
            },
        }
    )
    captured["manifest.json"] = manifest
    return captured, {
        "bundle_sha256": hashlib.sha256(manifest).hexdigest(),
        "events": status["events"],
        "counts": status["counts"],
        "data_kind": source["data_kind"],
    }


def export_reviews(database: Path, directory: Path) -> dict[str, Any]:
    """Export one read-transaction snapshot without replacing an existing delivery."""
    try:
        captured, receipt = _capture(database)
        # mkdir is exclusive. Cleanup below can only remove this new export, never an old one.
        directory.mkdir(parents=True, exist_ok=False)
        try:
            for name, content in captured.items():
                (directory / name).write_bytes(content)
        except BaseException:
            for name in (*FILES, "manifest.json"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
            raise
        return {"directory": str(directory), **receipt}
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Cannot export reviews: {exc}") from exc


def export_review_zip(database: Path) -> tuple[bytes, dict[str, Any]]:
    """Package the same evidence files as the directory export."""
    try:
        captured, receipt = _capture(database)
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            for name, content in captured.items():
                item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(item, content, compress_type=ZIP_DEFLATED)
        return buffer.getvalue(), receipt
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Cannot export reviews: {exc}") from exc


def _read_evidence(path: Path) -> dict[str, bytes]:
    expected_names = {*FILES, "manifest.json"}
    if path.is_dir():
        names = {file.name for file in path.iterdir()}
        if names == expected_names:
            return {name: (path / name).read_bytes() for name in names}
    else:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                len(entries) == len(expected_names)
                and {x.filename for x in entries} == expected_names
            ):
                if any(x.compress_type not in (ZIP_STORED, ZIP_DEFLATED) for x in entries):
                    raise DataContractError("Unsupported review ZIP compression")
                return {entry.filename: archive.read(entry) for entry in entries}
    raise DataContractError("Unexpected or missing review evidence files")


def verify_review_bundle(directory: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Recompute cases from captured evidence; a trusted external digest detects rewrites."""
    result, _ = _verify_review_evidence(directory, expected_sha256)
    return result


def _verify_review_evidence(
    path: Path, expected_sha256: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        # Parse and replay captured bytes, never re-read mutable files after checking them.
        captured = _read_evidence(path)
        digest = hashlib.sha256(captured["manifest.json"]).hexdigest()
        if expected_sha256 is not None:
            if (
                len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
                or expected_sha256 != digest
            ):
                raise DataContractError("Review evidence checkpoint mismatch")
        manifest = parse_json(captured["manifest.json"], "manifest.json")
        expected = {
            "format": FORMAT,
            "files": {name: hashlib.sha256(captured[name]).hexdigest() for name in FILES},
        }
        if manifest != expected:
            raise DataContractError("Review evidence manifest or digest mismatch")
        source = parse_json(captured["source.json"], "source.json")
        events = parse_json(captured["events.json"], "events.json")
        if not isinstance(source, dict) or not isinstance(events, list):
            raise DataContractError("Invalid review evidence objects")
        if any(not isinstance(event, dict) for event in events):
            raise DataContractError("Invalid review event objects")
        states = _replay(source, captured["review_queue.csv"], events)
        status = _status(source, events, states)
        # Exact canonical bytes also reject extra fields and JSON's bool/int equality ambiguity.
        if captured["status.json"] != _json(status):
            raise DataContractError("Review status does not match event replay")
        return {
            **status,
            "bundle_sha256": digest,
            "checkpoint_matched": expected_sha256 is not None,
        }, events
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        BadZipFile,
        DeflateError,
        RuntimeError,
    ) as exc:
        raise DataContractError(f"Cannot verify review evidence: {exc}") from exc


def compare_review_bundles(
    baseline: Path,
    current: Path,
    *,
    baseline_sha256: str | None = None,
    current_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare two replayed deliveries from the same frozen source and journal."""
    try:
        before, baseline_events = _verify_review_evidence(baseline, baseline_sha256)
    except DataContractError as exc:
        raise DataContractError(f"Baseline evidence: {exc}") from exc
    try:
        after, current_events = _verify_review_evidence(current, current_sha256)
    except DataContractError as exc:
        raise DataContractError(f"Current evidence: {exc}") from exc

    if before["source"] != after["source"]:
        raise DataContractError("Review deliveries have different frozen sources")
    if current_events[: len(baseline_events)] != baseline_events:
        raise DataContractError("Current review history does not extend the baseline")

    # The verified source binds both snapshots to the same cases in the same order.
    case_fields = ("version", "status", "outcome")
    changed_cases = [
        {
            "application_id": earlier["application_id"],
            "before": {field: earlier[field] for field in case_fields},
            "after": {field: later[field] for field in case_fields},
        }
        for earlier, later in zip(before["cases"], after["cases"], strict=True)
        if earlier["version"] != later["version"]
    ]
    summary_fields = ("bundle_sha256", "checkpoint_matched", "events", "counts")
    return {
        "status": "verified",
        "source": before["source"],
        "baseline": {field: before[field] for field in summary_fields},
        "current": {field: after[field] for field in summary_fields},
        "added_events": len(current_events) - len(baseline_events),
        "changed_cases": changed_cases,
    }
