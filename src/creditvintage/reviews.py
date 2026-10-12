"""Transactional, label-free review workspaces with an append-only event journal."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from creditvintage.business import QUEUE_COLUMNS, verify_analysis
from creditvintage.core import DataContractError
from creditvintage.validation import read_csv_rows

SCHEMA = """
CREATE TABLE source (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    analysis_sha256 TEXT NOT NULL,
    queue_sha256 TEXT NOT NULL,
    data_kind TEXT NOT NULL,
    queue_csv BLOB NOT NULL
);
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    application_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('claim', 'release', 'resolve', 'reopen')),
    actor TEXT NOT NULL,
    expected_version INTEGER NOT NULL CHECK (expected_version >= 0),
    note TEXT NOT NULL,
    evidence TEXT NOT NULL,
    outcome TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);
CREATE TRIGGER source_no_update BEFORE UPDATE ON source
BEGIN SELECT RAISE(ABORT, 'Frozen review source'); END;
CREATE TRIGGER source_no_delete BEFORE DELETE ON source
BEGIN SELECT RAISE(ABORT, 'Frozen review source'); END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'Append-only review journal'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'Append-only review journal'); END;
"""
EVENT_FIELDS = (
    "request_id",
    "application_id",
    "action",
    "actor",
    "expected_version",
    "note",
    "evidence",
    "outcome",
)


@contextmanager
def _connection(database: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    """Never let an operation silently create an empty workspace."""
    mode = "rw" if write else "ro"
    connection = sqlite3.connect(database.resolve().as_uri() + f"?mode={mode}", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if write:
            connection.execute("BEGIN IMMEDIATE")
        yield connection
        if write:
            connection.commit()
    except BaseException:
        if write:
            connection.rollback()
        raise
    finally:
        connection.close()


def initialize_reviews(analysis: Path, database: Path) -> dict[str, Any]:
    """Verify one captured analysis snapshot, then retain only its label-free queue."""
    # Parsing and verification use the same captured bytes even if files change on disk.
    captured = {
        path.relative_to(analysis).as_posix(): path.read_bytes()
        for path in analysis.rglob("*")
        if path.is_file()
    }
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary)
        for name, content in captured.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        verify_analysis(snapshot)
        queue = captured["review_queue.csv"]
        # Verify the expected queue contract explicitly at the operational boundary.
        rows = read_csv_rows(snapshot / "review_queue.csv", QUEUE_COLUMNS)
        data_kind = json.loads(captured["summary.json"])["data_kind"]
    anchor = hashlib.sha256(
        json.dumps(
            {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(captured.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    database.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation rejects an existing database, including concurrent initializers.
    with database.open("xb"):
        pass
    connection = None
    try:
        connection = sqlite3.connect(database)
        connection.executescript("BEGIN IMMEDIATE;\n" + SCHEMA)
        connection.execute(
            "INSERT INTO source VALUES (1, 1, ?, ?, ?, ?)",
            (
                anchor,
                hashlib.sha256(queue).hexdigest(),
                data_kind,
                queue,
            ),
        )
        connection.commit()
    except BaseException as exc:
        if connection is not None:
            connection.close()
            connection = None
        # The exclusive creation above means this is only our failed new file.
        database.unlink(missing_ok=True)
        if isinstance(exc, (sqlite3.Error, OSError)):
            raise DataContractError(f"Cannot initialize reviews: {exc}") from exc
        raise
    finally:
        if connection is not None:
            connection.close()
    return {
        "database": str(database),
        "cases": len(rows),
        "analysis_sha256": anchor,
        "data_kind": data_kind,
    }


def _text(value: Any, field: str, *, required: bool = True, maximum: int = 2000) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise DataContractError(f"Invalid review {field}")
    if value != value.strip() or (required and not value):
        raise DataContractError(f"Review {field} must be nonempty and trimmed")
    return value


def _validate_command(command: dict[str, Any]) -> None:
    for field in ("request_id", "application_id", "actor"):
        _text(command[field], field, maximum=200)
    for field in ("note", "evidence", "outcome"):
        _text(command[field], field, required=False)
    version = command["expected_version"]
    if type(version) is not int or version < 0:
        raise DataContractError("Review expected_version must be a nonnegative integer")
    action = command["action"]
    if action not in {"claim", "release", "resolve", "reopen"}:
        raise DataContractError("Unknown review action")
    if action == "claim":
        if any(command[field] for field in ("note", "evidence", "outcome")):
            raise DataContractError("Claim does not accept a note, evidence or outcome")
    elif action == "resolve":
        _text(command["note"], "note")
        _text(command["evidence"], "evidence")
        if command["outcome"] not in {"cleared", "escalated"}:
            raise DataContractError("Review outcome must be cleared or escalated")
    else:
        _text(command["note"], "note")
        if command["evidence"] or command["outcome"]:
            raise DataContractError("Release/reopen accepts a reason, not evidence or outcome")


def _apply(state: dict[str, Any], command: dict[str, Any]) -> None:
    """Reconstruct a case exclusively from its source and accepted journal events."""
    _validate_command(command)
    if command["expected_version"] != state["version"]:
        raise DataContractError(f"Stale review version; current version is {state['version']}")
    action, actor = command["action"], command["actor"]
    if action == "claim":
        if state["status"] != "queued":
            raise DataContractError("Only a queued case can be claimed")
        state.update(status="in_review", owner=actor)
    elif action in {"release", "resolve"}:
        if state["status"] != "in_review" or state["owner"] != actor:
            raise DataContractError("Only the current owner can release or resolve a case")
        if action == "release":
            state.update(status="queued", owner=None)
        else:
            state.update(status="resolved", outcome=command["outcome"])
    else:
        if state["status"] != "resolved":
            raise DataContractError("Only a resolved case can be reopened")
        state.update(status="queued", owner=None, outcome=None)
    state["version"] += 1


def _load(
    connection: sqlite3.Connection,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sources = connection.execute("SELECT * FROM source").fetchall()
    if len(sources) != 1 or sources[0]["schema_version"] != 1:
        raise DataContractError("Unsupported review workspace source")
    source = dict(sources[0])
    queue = source.pop("queue_csv")
    if not isinstance(queue, bytes) or hashlib.sha256(queue).hexdigest() != source["queue_sha256"]:
        raise DataContractError("Frozen review queue digest mismatch")
    for field in ("analysis_sha256", "queue_sha256"):
        digest = source[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DataContractError("Invalid review source fingerprint")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "review_queue.csv"
        path.write_bytes(queue)
        rows = read_csv_rows(path, QUEUE_COLUMNS)
    states = {}
    for rank, row in enumerate(rows, 1):
        identity = _text(row["application_id"], "application_id", maximum=200)
        if identity in states or row["review_rank"] != str(rank):
            raise DataContractError("Invalid review queue identities or rank")
        states[identity] = {**row, "status": "queued", "owner": None, "outcome": None, "version": 0}
    events = [dict(row) for row in connection.execute("SELECT * FROM events ORDER BY sequence")]
    seen = set()
    for sequence, event in enumerate(events, 1):
        if event["sequence"] != sequence or event["request_id"] in seen:
            raise DataContractError("Invalid review journal sequence or request identity")
        seen.add(event["request_id"])
        identity = event["application_id"]
        if identity not in states:
            raise DataContractError("Review event references an unknown case")
        # UTC timestamps identify receipt time, never source authenticity or label time.
        instant = datetime.fromisoformat(event["occurred_at"])
        if instant.tzinfo is None or instant.utcoffset() != UTC.utcoffset(instant):
            raise DataContractError("Review event time must be UTC")
        _apply(states[identity], event)
    return source, events, states


def record_review(
    database: Path,
    application_id: str,
    action: str,
    actor: str,
    expected_version: int,
    request_id: str,
    *,
    note: str = "",
    evidence: str = "",
    outcome: str = "",
) -> dict[str, Any]:
    """Accept at most one event per request, serialize writers and reject stale state."""
    command = dict(
        zip(
            EVENT_FIELDS,
            (
                request_id,
                application_id,
                action,
                actor,
                expected_version,
                note,
                evidence,
                outcome,
            ),
            strict=True,
        )
    )
    _validate_command(command)
    try:
        with _connection(database, write=True) as connection:
            _, events, states = _load(connection)
            for event in events:
                if event["request_id"] == request_id:
                    if any(event[field] != command[field] for field in EVENT_FIELDS):
                        raise DataContractError("Request ID was already used for different content")
                    return {"status": "replayed", "event": event}
            if application_id not in states:
                raise DataContractError("Unknown review case")
            _apply(states[application_id], command)
            occurred_at = datetime.now(UTC).isoformat(timespec="microseconds")
            cursor = connection.execute(
                "INSERT INTO events (" + ",".join(EVENT_FIELDS) + ",occurred_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (*command.values(), occurred_at),
            )
            event = {"sequence": cursor.lastrowid, **command, "occurred_at": occurred_at}
            return {"status": "accepted", "event": event}
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Cannot record review: {exc}") from exc


def review_status(database: Path) -> dict[str, Any]:
    """Return operational completion counts, never retrospective event-capture metrics."""
    try:
        with _connection(database) as connection:
            source, events, states = _load(connection)
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError(f"Cannot replay reviews: {exc}") from exc
    cases = list(states.values())
    counts = {
        status: sum(case["status"] == status for case in cases)
        for status in ("queued", "in_review", "resolved")
    }
    outcomes = {
        outcome: sum(case["outcome"] == outcome for case in cases)
        for outcome in ("cleared", "escalated")
    }
    return {
        "status": "verified",
        "source": source,
        "selected_cases": len(cases),
        "events": len(events),
        "counts": counts,
        "outcomes": outcomes,
        "completion_rate": counts["resolved"] / len(cases) if cases else None,
        "cases": cases,
    }
