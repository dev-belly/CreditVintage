"""Portable, label-free review evidence with replay and optional checkpoint anchoring."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from creditvintage.core import DataContractError
from creditvintage.reviews import _connection, _load, _replay, _status

FORMAT = "creditvintage.review-evidence.v1"
FILES = ("source.json", "review_queue.csv", "events.json", "status.json")


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def export_reviews(database: Path, directory: Path) -> dict[str, Any]:
    """Export one read-transaction snapshot without replacing an existing delivery."""
    try:
        with _connection(database) as connection:
            connection.execute("BEGIN")
            source, events, states = _load(connection)
            queue = connection.execute("SELECT queue_csv FROM source WHERE id=1").fetchone()[0]
        status = _status(source, events, states)
        captured = {
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
        # mkdir is exclusive. Cleanup below can only remove this new export, never an old one.
        directory.mkdir(parents=True, exist_ok=False)
        try:
            for name, content in {**captured, "manifest.json": manifest}.items():
                (directory / name).write_bytes(content)
        except BaseException:
            for name in (*FILES, "manifest.json"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
            raise
        return {
            "directory": str(directory),
            "bundle_sha256": hashlib.sha256(manifest).hexdigest(),
            "events": status["events"],
            "counts": status["counts"],
            "data_kind": source["data_kind"],
        }
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Cannot export reviews: {exc}") from exc


def verify_review_bundle(directory: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Recompute cases from captured evidence; a trusted external digest detects rewrites."""
    try:
        names = {path.name for path in directory.iterdir()}
        if names != {*FILES, "manifest.json"}:
            raise DataContractError("Unexpected or missing review evidence files")
        # Parse and replay captured bytes, never re-read mutable files after checking them.
        captured = {name: (directory / name).read_bytes() for name in names}
        digest = hashlib.sha256(captured["manifest.json"]).hexdigest()
        if expected_sha256 is not None:
            if (
                len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
                or expected_sha256 != digest
            ):
                raise DataContractError("Review evidence checkpoint mismatch")
        manifest = json.loads(captured["manifest.json"])
        expected = {
            "format": FORMAT,
            "files": {name: hashlib.sha256(captured[name]).hexdigest() for name in FILES},
        }
        if manifest != expected:
            raise DataContractError("Review evidence manifest or digest mismatch")
        source = json.loads(captured["source.json"])
        events = json.loads(captured["events.json"])
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
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Cannot verify review evidence: {exc}") from exc
