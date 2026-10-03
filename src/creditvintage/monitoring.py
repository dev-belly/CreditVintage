"""Label-free, point-in-time score monitoring against a frozen calibration cohort."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import date, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import (
    _ID,
    HORIZON_DAYS,
    REPORT_LAG_DAYS,
    Application,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    snapshot,
)
from creditvintage.model import Config, _digest, _fit_stages, _matrix
from creditvintage.validation import finite_number, read_csv_rows, read_json_object

MONITOR_FILES = (
    "monitor.json",
    "reference_scores.csv",
    "monitor_scores.csv",
    "score_bins.csv",
    "index.html",
)
SCORE_COLUMNS = (
    "application_id",
    "decision_at",
    "vintage",
    "segment",
    "raw_pd",
    "calibrated_pd",
)
BIN_COLUMNS = (
    "bin",
    "lower",
    "upper",
    "reference_loans",
    "current_loans",
    "reference_share",
    "current_share",
    "psi_component",
)
SMOOTHING = 0.5


@dataclass(frozen=True)
class MonitorResult:
    summary: dict[str, Any]
    reference_scores: list[float]
    scores: list[dict[str, Any]]
    bins: list[dict[str, Any]]
    input_sha256: str


def score_psi(reference: np.ndarray, current: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    """Reference-quantile bins, with a stated half-count in every bin.

    PSI is a descriptive distribution distance. It is not a performance
    metric, statistical significance test, or automatic retraining rule.
    """
    if (
        reference.ndim != 1
        or current.ndim != 1
        or len(reference) < 20
        or len(current) < 20
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
        or not np.all((0 <= reference) & (reference <= 1))
        or not np.all((0 <= current) & (current <= 1))
    ):
        raise DataContractError("Score monitoring needs 20 finite probabilities per population")
    if float(reference.min()) == float(reference.max()):
        raise DataContractError("Reference scores have no distribution to compare")

    # Freeze boundaries on the reference only. Repeated quantiles collapse;
    # reporting the effective bin count avoids inventing empty deciles.
    candidates = [
        round(float(value), 10) for value in np.quantile(reference, np.arange(1, 10) / 10)
    ]
    edges = sorted({edge for edge in candidates if reference.min() < edge < reference.max()})
    if not edges:
        edges = [(float(reference.min()) + float(reference.max())) / 2]
    reference_counts = np.bincount(
        np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1
    )
    current_counts = np.bincount(
        np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1
    )
    count = len(edges) + 1
    reference_shares = (reference_counts + SMOOTHING) / (len(reference) + count * SMOOTHING)
    current_shares = (current_counts + SMOOTHING) / (len(current) + count * SMOOTHING)
    components = (current_shares - reference_shares) * np.log(current_shares / reference_shares)
    bins = [
        {
            "bin": i + 1,
            "lower": 0.0 if i == 0 else edges[i - 1],
            "upper": 1.0 if i == len(edges) else edges[i],
            "reference_loans": int(reference_counts[i]),
            "current_loans": int(current_counts[i]),
            "reference_share": round(float(reference_shares[i]), 8),
            "current_share": round(float(current_shares[i]), 8),
            "psi_component": round(float(components[i]), 8),
        }
        for i in range(count)
    ]
    return round(float(np.sum(components)), 6), bins


def monitor(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
    as_of: date,
    config: Config | None = None,
    *,
    data_kind: str = "user_supplied",
) -> MonitorResult:
    """Score applications visible at cutoff; current outcomes are never used."""
    config = config or Config()
    if type(as_of) is not date or as_of < config.test_start:
        raise DataContractError("Monitor cutoff must be on or after the test window starts")
    if data_kind not in {"synthetic", "user_supplied"}:
        raise DataContractError("Unknown data kind")
    # Only the train and calibration cohorts supply labels. A current-cohort
    # report must not alter whether a label-free monitoring run succeeds.
    stage_ids = {
        app.application_id
        for app in applications
        if (config.train_start <= app.decision_at <= config.train_end)
        or (config.calibration_start <= app.decision_at <= config.calibration_end)
    }
    stage_performance = [report for report in performance if report.application_id in stage_ids]
    model, calibrator, _, calibration, calibration_raw = _fit_stages(
        applications, features, stage_performance, config
    )
    current_rows = [
        row
        for row in snapshot(applications, features, [], as_of)
        if config.test_start <= row.application.decision_at <= config.test_end
    ]
    if len(current_rows) < 20:
        raise DataContractError("Monitor window needs at least 20 visible applications")
    current_raw = model.predict_proba(_matrix(current_rows))[:, 1]
    current_calibrated = np.asarray(calibrator.predict(current_raw), dtype=float)
    reference = [round(float(value), 10) for value in calibration_raw]
    current = [round(float(value), 10) for value in current_raw]
    psi, bins = score_psi(np.asarray(reference), np.asarray(current))
    scores = [
        {
            "application_id": row.application.application_id,
            "decision_at": row.application.decision_at.isoformat(),
            "vintage": row.application.decision_at.strftime("%Y-%m"),
            "segment": row.application.segment,
            "raw_pd": current[i],
            "calibrated_pd": round(float(current_calibrated[i]), 10),
        }
        for i, row in enumerate(current_rows)
    ]
    months = sorted({str(row["vintage"]) for row in scores})
    by_vintage = []
    for month in months:
        monthly = np.asarray([row["raw_pd"] for row in scores if row["vintage"] == month])
        by_vintage.append(
            {
                "vintage": month,
                "loans": len(monthly),
                "psi": score_psi(np.asarray(reference), monthly)[0] if len(monthly) >= 20 else None,
            }
        )
    summary = {
        "project": "CreditVintage",
        "kind": "label_free_score_monitor",
        "data_kind": data_kind,
        "as_of": as_of.isoformat(),
        "reference_window": [
            config.calibration_start.isoformat(),
            config.calibration_end.isoformat(),
        ],
        "current_window": [config.test_start.isoformat(), config.test_end.isoformat()],
        "reference_loans": len(calibration),
        "current_loans": len(scores),
        "score": "raw logistic probability; calibrator trained before test applications",
        "psi": psi,
        "requested_bins": 10,
        "effective_bins": len(bins),
        "smoothing_per_bin": SMOOTHING,
        "by_vintage": by_vintage,
    }
    return MonitorResult(
        summary, reference, scores, bins, _digest(applications, features, performance)
    )


def _render_monitor(result: MonitorResult) -> str:
    summary = result.summary
    months = "\n".join(
        f"<tr><td>{html.escape(row['vintage'])}</td><td>{row['loans']}</td>"
        f"<td>{row['psi']:.3f}</td></tr>"
        if row["psi"] is not None
        else f"<tr><td>{html.escape(row['vintage'])}</td><td>{row['loans']}</td>"
        "<td>Insufficient sample</td></tr>"
        for row in summary["by_vintage"]
    )
    template = files("creditvintage").joinpath("monitor.html").read_text(encoding="utf-8")
    replacements = {
        "@@AS_OF@@": summary["as_of"],
        "@@REFERENCE_LOANS@@": f"{summary['reference_loans']:,}",
        "@@CURRENT_LOANS@@": f"{summary['current_loans']:,}",
        "@@PSI@@": f"{summary['psi']:.3f}",
        "@@SOURCE@@": (
            "Synthetic example." if summary["data_kind"] == "synthetic" else "User-supplied data."
        ),
        "@@EFFECTIVE_BINS@@": str(summary["effective_bins"]),
        "@@MONTHS@@": months,
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def write_monitor(result: MonitorResult, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    _json(destination / "monitor.json", result.summary)
    _csv(
        destination / "reference_scores.csv",
        [{"raw_pd": value} for value in result.reference_scores],
        ("raw_pd",),
    )
    _csv(destination / "monitor_scores.csv", result.scores, SCORE_COLUMNS)
    _csv(destination / "score_bins.csv", result.bins, BIN_COLUMNS)
    (destination / "index.html").write_text(_render_monitor(result), encoding="utf-8")
    manifest = {
        "kind": "label_free_score_monitor",
        "input_sha256": result.input_sha256,
        "files_sha256": {name: _hash(destination / name) for name in MONITOR_FILES},
    }
    _json(destination / "manifest.json", manifest)
    return manifest


def verify_monitor(destination: Path) -> dict[str, Any]:
    """Recompute score drift from published rows, even if hashes were rewritten."""
    try:
        manifest = read_json_object(destination / "manifest.json")
        if manifest["kind"] != "label_free_score_monitor" or set(manifest["files_sha256"]) != set(
            MONITOR_FILES
        ):
            raise DataContractError("Unexpected monitor artifact inventory")
        for name, digest in manifest["files_sha256"].items():
            if _hash(destination / name) != digest:
                raise DataContractError(f"Artifact digest mismatch: {name}")
        summary = read_json_object(destination / "monitor.json")
        if summary["data_kind"] not in {"synthetic", "user_supplied"}:
            raise DataContractError("Unknown data kind")
        reference_rows = read_csv_rows(destination / "reference_scores.csv", ("raw_pd",))
        current_rows = read_csv_rows(destination / "monitor_scores.csv", SCORE_COLUMNS)
        bin_rows = read_csv_rows(destination / "score_bins.csv", BIN_COLUMNS)
        if (
            summary["kind"] != "label_free_score_monitor"
            or summary["reference_loans"] != len(reference_rows)
            or summary["current_loans"] != len(current_rows)
            or summary["requested_bins"] != 10
            or summary["smoothing_per_bin"] != SMOOTHING
            or len({row["application_id"] for row in current_rows}) != len(current_rows)
        ):
            raise DataContractError("Monitor counts or method disagree with score rows")
        cutoff = date.fromisoformat(summary["as_of"])
        start, end = (date.fromisoformat(value) for value in summary["current_window"])
        reference_start, reference_end = (
            date.fromisoformat(value) for value in summary["reference_window"]
        )
        if not (
            reference_start <= reference_end
            and reference_end + timedelta(days=HORIZON_DAYS + REPORT_LAG_DAYS) <= start <= end
            and start <= cutoff
        ):
            raise DataContractError(
                "Monitor windows must include the reference-label maturation gap"
            )
        for row in current_rows:
            if not _ID.fullmatch(row["application_id"]) or row["segment"] not in {
                "retail",
                "small_business",
            }:
                raise DataContractError("Monitor application identity or segment is invalid")
            decision = date.fromisoformat(row["decision_at"])
            if not start <= decision <= min(end, cutoff) or row["vintage"] != decision.strftime(
                "%Y-%m"
            ):
                raise DataContractError("Monitor decision dates disagree with cutoff")
            calibrated = float(row["calibrated_pd"])
            if not math.isfinite(calibrated) or not 0 <= calibrated <= 1:
                raise DataContractError("Invalid calibrated score")
        reference = np.asarray([float(row["raw_pd"]) for row in reference_rows])
        current = np.asarray([float(row["raw_pd"]) for row in current_rows])
        psi, bins = score_psi(reference, current)
        if (
            summary["psi"] != psi
            or summary["effective_bins"] != len(bins)
            or len(bin_rows) != len(bins)
        ):
            raise DataContractError("Monitor PSI or bin count disagrees with scores")
        for published, expected in zip(bin_rows, bins, strict=True):
            if any(
                abs(finite_number(published[key], f"monitor bin {key}") - float(value)) > 0.00000002
                for key, value in expected.items()
            ):
                raise DataContractError("Monitor bins disagree with scores")
        months = sorted({row["vintage"] for row in current_rows})
        expected_months = []
        for month in months:
            monthly = np.asarray(
                [float(row["raw_pd"]) for row in current_rows if row["vintage"] == month]
            )
            expected_months.append(
                {
                    "vintage": month,
                    "loans": len(monthly),
                    "psi": score_psi(reference, monthly)[0] if len(monthly) >= 20 else None,
                }
            )
        if summary["by_vintage"] != expected_months:
            raise DataContractError("Monitor vintages disagree with scores")
        result = MonitorResult(
            summary, reference.tolist(), current_rows, bins, manifest["input_sha256"]
        )
        if (destination / "index.html").read_bytes() != _render_monitor(result).encode("utf-8"):
            raise DataContractError("HTML monitor disagrees with verified evidence")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify monitor: {exc}") from exc
    return {
        "status": "verified",
        "loans": len(current_rows),
        "input_sha256": manifest["input_sha256"],
    }
