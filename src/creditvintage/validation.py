"""Strict readers for portable report evidence, before arithmetic comparisons."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from creditvintage.core import DataContractError


def finite_number(value: Any, label: str) -> float:
    """NaN cannot be compared with a tolerance: reject it at the boundary."""
    if isinstance(value, bool):
        raise DataContractError(f"Expected a finite number, not a boolean: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Expected a finite number: {label}") from exc
    if not math.isfinite(number):
        raise DataContractError(f"Expected a finite number: {label}")
    return number


def parse_json(content: bytes | str, label: str) -> Any:
    """Parse captured evidence without ambiguous keys or non-finite JSON numbers."""

    def parse_number(value: str) -> float:
        return finite_number(value, f"{label} JSON number")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DataContractError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    return json.loads(
        content,
        parse_float=parse_number,
        parse_constant=parse_number,
        object_pairs_hook=unique_keys,
    )


def read_json_object(path: Path) -> dict[str, Any]:
    document = parse_json(path.read_text(encoding="utf-8"), path.name)
    if not isinstance(document, dict):
        raise DataContractError(f"Expected a JSON object: {path.name}")
    return document


def read_csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            if reader.fieldnames != list(columns):
                raise DataContractError(f"Unexpected CSV columns: {path.name}")
            rows = list(reader)
        except csv.Error as exc:
            raise DataContractError(
                f"Malformed CSV quoting in {path.name} at line {reader.reader.line_num}: {exc}"
            ) from exc
    if any(
        set(row) != set(columns) or any(value is None for value in row.values()) for row in rows
    ):
        raise DataContractError(f"Malformed CSV row: {path.name}")
    return rows
