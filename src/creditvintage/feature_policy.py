"""Explicit coverage policies, with exact counts and self-contained event replay."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import FEATURE_NAMES, DataContractError
from creditvintage.feature_inputs import decode_inputs
from creditvintage.feature_quality import (
    QUALITY_FILES,
    FeatureQualityResult,
    monitor_feature_quality,
    verify_feature_quality,
    write_feature_quality,
)
from creditvintage.feature_quality import (
    SCHEMA as QUALITY_SCHEMA,
)
from creditvintage.validation import read_json_object

POLICY_SCHEMA = "creditvintage-feature-policy-v1"
SCHEMA = "creditvintage-feature-check-v1"
RULE_KEYS = {"id", "population", "vintage", "feature", "min_applications", "min_coverage"}
CHECK_COLUMNS = (
    "id",
    "population",
    "vintage",
    "feature",
    "min_applications",
    "min_coverage",
    "applications",
    "available",
    "coverage_rate",
    "status",
    "reason",
)
SOURCE_FILES = (*QUALITY_FILES, "manifest.json")
CHECK_FILES = (
    "policy.json",
    "check.json",
    "checks.csv",
    "index.html",
    *(f"quality/{name}" for name in SOURCE_FILES),
)


@dataclass(frozen=True)
class FeatureCheckResult:
    quality: FeatureQualityResult
    policy: dict[str, Any]
    summary: dict[str, Any]


def _threshold(value: Any) -> Fraction:
    # Strings make decimal precision explicit. Never compare rounded report rates.
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:0(?:\.[0-9]{1,18})?|1(?:\.0{1,18})?)", value
    ):
        raise DataContractError("min_coverage must be a decimal string in [0, 1], up to 18 places")
    return Fraction(value)


def validate_policy(document: dict[str, Any]) -> dict[str, Any]:
    if set(document) != {"schema", "rules"} or document.get("schema") != POLICY_SCHEMA:
        raise DataContractError("Expected feature-policy schema and rules only")
    rules = document["rules"]
    if not isinstance(rules, list) or not rules:
        raise DataContractError("A feature policy requires at least one explicit rule")
    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != RULE_KEYS:
            raise DataContractError("Every feature rule must have exactly the six documented keys")
        identity = rule["id"]
        if not isinstance(identity, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identity
        ):
            raise DataContractError(
                "Rule IDs require 1-64 ASCII letters, digits, underscores or hyphens"
            )
        if identity in seen:
            raise DataContractError(f"Duplicate feature-policy rule ID: {identity}")
        seen.add(identity)
        if not isinstance(rule["population"], str) or rule["population"] not in {
            "reference",
            "current",
        }:
            raise DataContractError(f"Unknown policy population: {identity}")
        if not isinstance(rule["feature"], str) or rule["feature"] not in FEATURE_NAMES:
            raise DataContractError(f"Unknown policy feature: {identity}")
        vintage = rule["vintage"]
        if not isinstance(vintage, str):
            raise DataContractError(f"Policy vintage must be all or YYYY-MM: {identity}")
        if vintage != "all":
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", vintage):
                raise DataContractError(f"Policy vintage must be all or YYYY-MM: {identity}")
            try:
                date.fromisoformat(vintage + "-01")
            except ValueError as exc:
                raise DataContractError(f"Invalid policy month: {identity}") from exc
        if type(rule["min_applications"]) is not int or rule["min_applications"] < 1:
            raise DataContractError(f"min_applications must be a positive integer: {identity}")
        _threshold(rule["min_coverage"])
    return {"schema": POLICY_SCHEMA, "rules": [dict(rule) for rule in rules]}


def policy_digest(policy: dict[str, Any]) -> str:
    validated = validate_policy(policy)
    canonical = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def example_policy() -> dict[str, Any]:
    """An editable example, not a claim that these are appropriate business thresholds."""
    return {
        "schema": POLICY_SCHEMA,
        "rules": [
            {
                "id": f"current-{name}",
                "population": "current",
                "vintage": "all",
                "feature": name,
                "min_applications": 20,
                "min_coverage": "1.0",
            }
            for name in FEATURE_NAMES
        ],
    }


def initialize_policy(destination: Path) -> dict[str, Any]:
    policy = example_policy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return {
        "out": str(destination),
        "policy_sha256": policy_digest(policy),
        "rules": len(policy["rules"]),
    }


def evaluate_feature_policy(
    quality: FeatureQualityResult, document: dict[str, Any]
) -> FeatureCheckResult:
    policy = validate_policy(document)
    groups = {(row["population"], row["vintage"], row["feature"]): row for row in quality.coverage}
    checks = []
    for rule in policy["rules"]:
        vintage = rule["vintage"]
        start, end = quality.summary[f"{rule['population']}_window"]
        if vintage != "all" and not start[:7] <= vintage <= end[:7]:
            raise DataContractError(f"Rule month is outside its population window: {rule['id']}")
        group = groups.get((rule["population"], vintage, rule["feature"]))
        total = group["applications"] if group else 0
        available = group["available"] if group else 0
        threshold = _threshold(rule["min_coverage"])
        if total < rule["min_applications"]:
            status, reason = "insufficient_data", "below_minimum_applications"
        elif available * threshold.denominator < total * threshold.numerator:
            status, reason = "fail", "below_minimum_coverage"
        else:
            status, reason = "pass", "specified_conditions_met"
        checks.append(
            {
                **rule,
                "applications": total,
                "available": available,
                "coverage_rate": group["coverage_rate"] if group else None,
                "status": status,
                "reason": reason,
            }
        )
    counts = {
        status: sum(row["status"] == status for row in checks)
        for status in ("pass", "fail", "insufficient_data")
    }
    status = (
        "fail" if counts["fail"] else "insufficient_data" if counts["insufficient_data"] else "pass"
    )
    summary = {
        "schema": SCHEMA,
        "kind": "explicit_feature_coverage_policy",
        "status": status,
        "as_of": quality.summary["as_of"],
        "data_kind": quality.summary["data_kind"],
        "source_quality_schema": QUALITY_SCHEMA,
        "input_sha256": quality.summary["input_sha256"],
        "policy_sha256": policy_digest(policy),
        "counts": counts,
        "checks": checks,
        "model_fitted": False,
        "outcomes_required": False,
        "scope": (
            "Only specified coverage/sample rules; not scoring readiness or source authenticity"
        ),
    }
    return FeatureCheckResult(quality, policy, summary)


def _load_quality(directory: Path) -> FeatureQualityResult:
    # Validate one captured file set, never a changing external directory in mid-replay.
    with TemporaryDirectory(prefix="creditvintage-quality-capture-") as temporary:
        captured = Path(temporary)
        for name in SOURCE_FILES:
            (captured / name).write_bytes((directory / name).read_bytes())
        verify_feature_quality(captured)
        source = read_json_object(captured / "inputs.json")
    apps, features, as_of, config, kind = decode_inputs(source, schema=QUALITY_SCHEMA)
    return monitor_feature_quality(apps, features, as_of, config, data_kind=kind)


def _render(result: FeatureCheckResult) -> str:
    summary = result.summary
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                row["feature"],
                f"{row['available']}/{row['applications']}",
                row["status"],
                row["min_coverage"],
                row["min_applications"],
                row["population"],
                row["vintage"],
                row["id"],
                row["reason"],
            )
        )
        + "</tr>"
        for row in summary["checks"]
    )
    page = files("creditvintage").joinpath("feature_check.html").read_text(encoding="utf-8")
    values = {
        "@@STATUS@@": summary["status"],
        "@@CUTOFF@@": summary["as_of"],
        "@@KIND@@": summary["data_kind"],
        "@@POLICY@@": summary["policy_sha256"],
        "@@PASS@@": str(summary["counts"]["pass"]),
        "@@FAIL@@": str(summary["counts"]["fail"]),
        "@@INSUFFICIENT@@": str(summary["counts"]["insufficient_data"]),
    }
    for marker, value in values.items():
        page = page.replace(marker, html.escape(value))
    return page.replace("@@ROWS@@", rows)


def _write(result: FeatureCheckResult, destination: Path) -> dict[str, Any]:
    write_feature_quality(result.quality, destination / "quality")
    _json(destination / "policy.json", result.policy)
    _json(destination / "check.json", result.summary)
    _csv(destination / "checks.csv", result.summary["checks"], CHECK_COLUMNS)
    (destination / "index.html").write_text(_render(result), encoding="utf-8", newline="")
    manifest = {
        "schema": SCHEMA,
        "files_sha256": {name: _hash(destination / name) for name in CHECK_FILES},
    }
    _json(destination / "manifest.json", manifest)
    return manifest


def write_feature_check(result: FeatureCheckResult, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise DataContractError("Feature-check destination already exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".feature-check-") as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        manifest = _write(result, staged)
        if destination.exists():
            raise DataContractError("Feature-check destination already exists")
        staged.rename(destination)
    return manifest


def check_feature_quality(directory: Path, policy_path: Path, destination: Path) -> dict[str, Any]:
    policy = read_json_object(policy_path)
    result = evaluate_feature_policy(_load_quality(directory), policy)
    write_feature_check(result, destination)
    return result.summary


def verify_feature_check(
    directory: Path, *, expected_policy_sha256: str | None = None
) -> dict[str, Any]:
    try:
        policy = read_json_object(directory / "policy.json")
        result = evaluate_feature_policy(_load_quality(directory / "quality"), policy)
        if expected_policy_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_policy_sha256):
                raise DataContractError("Expected policy SHA-256 requires 64 lowercase hex digits")
            if result.summary["policy_sha256"] != expected_policy_sha256:
                raise DataContractError(
                    "Feature-check policy differs from the trusted expected digest"
                )
        with TemporaryDirectory(prefix="creditvintage-feature-check-replay-") as temporary:
            expected = Path(temporary)
            manifest = _write(result, expected)
            if read_json_object(directory / "manifest.json") != manifest:
                raise DataContractError("Feature-check manifest disagrees with policy/event replay")
            for name in CHECK_FILES:
                if (directory / name).read_bytes() != (expected / name).read_bytes():
                    raise DataContractError(f"Feature-check artifact disagrees with replay: {name}")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify feature check: {exc}") from exc
    return {
        "status": "verified",
        "policy_status": result.summary["status"],
        "policy_sha256": result.summary["policy_sha256"],
        "counts": result.summary["counts"],
        "input_sha256": result.summary["input_sha256"],
        "model_fitted": False,
    }
