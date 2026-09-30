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


def read_json_object(path: Path) -> dict[str, Any]:
    def parse_number(value: str) -> float:
        return finite_number(value, f"{path.name} JSON number")

    document = json.loads(
        path.read_text(encoding="utf-8"), parse_float=parse_number, parse_constant=parse_number
    )
    if not isinstance(document, dict):
        raise DataContractError(f"Expected a JSON object: {path.name}")
    return document


def read_csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise DataContractError(f"Unexpected CSV columns: {path.name}")
        rows = list(reader)
    if any(
        set(row) != set(columns) or any(value is None for value in row.values()) for row in rows
    ):
        raise DataContractError(f"Malformed CSV row: {path.name}")
    return rows
