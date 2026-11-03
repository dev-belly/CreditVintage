"""Independently replay SBA review budgets in SQLite after full evidence refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from creditvintage.core import DataContractError
from creditvintage.sba_credit import (
    METHODS,
    POLICY_COLUMNS,
    PREDICTION_COLUMNS,
    QUEUE_COLUMNS,
    SBA_FILES,
    verify_sba_benchmark,
)
from creditvintage.validation import finite_number, read_csv_rows

SQL = Path(__file__).with_name("sba_review_budget.sql")


def replay(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank scores without labels, then evaluate the three fixed budgets."""
    if not rows:
        raise DataContractError("Expected nonempty SBA predictions")
    scores, outcomes, identities = [], [], set()
    for row in rows:
        if set(row) != set(PREDICTION_COLUMNS):
            raise DataContractError("Unexpected SBA prediction columns")
        identity = row["record_id"]
        if not identity or identity in identities:
            raise DataContractError("SBA prediction IDs must be nonempty and unique")
        identities.add(identity)
        if row["chargeoff_24m"] not in {"0", "1"}:
            raise DataContractError("Expected a binary recorded chargeoff label")
        outcomes.append((identity, int(row["chargeoff_24m"])))
        for method in METHODS:
            value = finite_number(row[method], method)
            if not 0 <= value <= 1:
                raise DataContractError("Expected SBA probabilities within [0, 1]")
            scores.append((method, identity, value))
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE outcomes (
                record_id TEXT PRIMARY KEY,
                chargeoff_24m INTEGER NOT NULL CHECK(chargeoff_24m IN (0, 1))
            );
            CREATE TABLE scores (
                method TEXT NOT NULL, record_id TEXT NOT NULL, score REAL NOT NULL,
                PRIMARY KEY(method, record_id)
            );
        """)
        connection.executemany("INSERT INTO outcomes VALUES (?, ?)", outcomes)
        connection.executemany("INSERT INTO scores VALUES (?, ?, ?)", scores)
        connection.executescript(SQL.read_text(encoding="utf-8"))
        policies = [
            dict(row)
            for method in METHODS
            for row in connection.execute(
                "SELECT * FROM policy_results WHERE method = ? ORDER BY capacity", (method,)
            )
        ]
        queue = [
            dict(row)
            for method in METHODS
            for row in connection.execute(
                "SELECT * FROM queue_results WHERE method = ? ORDER BY rank", (method,)
            )
        ]
    return policies, queue


def compare(
    actual: list[dict[str, Any]], saved: list[dict[str, str]], columns: tuple[str, ...]
) -> None:
    if len(actual) != len(saved):
        raise DataContractError("SQL replay row count differs from saved evidence")
    for left, right in zip(actual, saved, strict=True):
        for column in columns:
            value = left[column]
            if isinstance(value, str):
                if value != right[column]:
                    raise DataContractError(f"SQL replay identity differs: {column}")
            elif value is None:
                if right[column] != "":
                    raise DataContractError(f"SQL replay missing denominator differs: {column}")
            elif not math.isclose(
                value, finite_number(right[column], column), abs_tol=1e-12, rel_tol=0
            ):
                raise DataContractError(f"SQL replay numeric value differs: {column}")


def audit(directory: Path) -> dict[str, Any]:
    """Capture a private bundle, refit it, and compare SQL against saved evidence."""
    with TemporaryDirectory(prefix="creditvintage-sba-sql-") as temporary:
        captured = Path(temporary)
        for name in (*SBA_FILES, "manifest.json", "evidence.zip"):
            shutil.copyfile(directory / name, captured / name)
        verification = verify_sba_benchmark(captured)
        predictions = read_csv_rows(captured / "predictions.csv", PREDICTION_COLUMNS)
        policies, queue = replay(predictions)
        compare(
            policies, read_csv_rows(captured / "policy_metrics.csv", POLICY_COLUMNS), POLICY_COLUMNS
        )
        compare(queue, read_csv_rows(captured / "review_queue.csv", QUEUE_COLUMNS), QUEUE_COLUMNS)
        hashes = {
            name: hashlib.sha256((captured / name).read_bytes()).hexdigest()
            for name in ("predictions.csv", "policy_metrics.csv", "review_queue.csv")
        }
    return {
        "schema": "creditvintage-sba-budget-sql-v1",
        "status": "matched_saved_evidence",
        "evidence_verification": verification,
        "sqlite_version": sqlite3.sqlite_version,
        "prediction_rows": len(predictions),
        "policy_rows": len(policies),
        "queue_rows": len(queue),
        "labels_used_for_ranking": False,
        "input_sha256": hashes,
        "policies": policies,
        "scope": (
            "Retrospective recorded-event concentration; "
            "not prevented losses or source authenticity"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Existing complete SBA evidence directory")
    args = parser.parse_args(argv)
    try:
        result = audit(args.directory)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"sba-review-sql: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
