"""Retrospective, date-separated SBA charge-off evaluation and offline refitting."""

from __future__ import annotations

import html
import io
import math
import platform
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import DataContractError
from creditvintage.sba_source import (
    AUDIT_SHA256,
    DATA_SHA256,
    PROTOCOL_SHA256,
    SOURCE_URL,
    WINDOWS,
    SbaLoan,
    frozen_json,
    load_sba,
    resource,
    validate_cohorts,
)
from creditvintage.validation import read_csv_rows, read_json_object

METHODS = ("constant", "logistic", "isotonic")
METRICS = ("roc_auc", "average_precision", "brier")
SEED = 20261025
ITERATIONS = 300
NUMERICAL = {"numpy": "2.3.5", "scipy": "1.17.0", "scikit-learn": "1.8.0"}
PREDICTION_COLUMNS = ("record_id", "funded_at", "chargeoff_24m", *METHODS)
POLICY_COLUMNS = (
    "method",
    "capacity",
    "reviewed",
    "captured",
    "total_events",
    "precision",
    "capture",
)
QUEUE_COLUMNS = ("method", "record_id", "rank", "score")
YEAR_COLUMNS = ("year", "method", "records", "events", *METRICS)
BOOT_COLUMNS = ("replicate", "method", "records", "events", *METRICS)
SBA_FILES = (
    "source.csv.gz",
    "protocol.json",
    "audit.json",
    "SOURCE.md",
    "predictions.csv",
    "policy_metrics.csv",
    "review_queue.csv",
    "year_metrics.csv",
    "bootstrap.csv",
    "summary.json",
    "index.html",
)


def feature_matrix(loans: list[SbaLoan]) -> np.ndarray:
    """Only four declared approval inputs. Outcomes, identities and dates stay out."""
    return np.asarray(
        [
            [
                math.log1p(float(loan.gross_approval_usd)),
                float(loan.sba_guaranteed_usd) / float(loan.gross_approval_usd),
                float(loan.initial_interest_rate_pct) if loan.initial_interest_rate_pct else np.nan,
                loan.processing_method,
            ]
            for loan in loans
        ],
        dtype=object,
    )


def fit_scores(loans: list[SbaLoan]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validate_cohorts(loans)
    x = feature_matrix(loans)
    y = np.asarray([loan.chargeoff_24m for loan in loans], dtype=int)
    stages = {
        stage: np.asarray([i for i, loan in enumerate(loans) if loan.stage == stage])
        for stage in WINDOWS
    }
    train, calibration, test = (stages[key] for key in WINDOWS)
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    transform = ColumnTransformer(
        [
            ("numeric", numeric, [0, 1, 2]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [3]),
        ],
        sparse_threshold=0,
    )
    model = make_pipeline(transform, LogisticRegression(C=1, solver="lbfgs", max_iter=2000))
    with threadpool_limits(limits=1):
        model.fit(x[train], y[train])
        raw = model.predict_proba(x)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.round(raw[calibration], 6), y[calibration])
        scores = {
            "constant": np.full(len(test), np.mean(y[train])),
            "logistic": raw[test],
            "isotonic": calibrator.predict(np.round(raw[test], 6)),
        }
    learned_numeric = transform.named_transformers_["numeric"]
    audit = {
        "training_records": len(train),
        "calibration_records": len(calibration),
        "training_initial_rate_missing": sum(not loans[i].initial_interest_rate_pct for i in train),
        "numeric_medians": learned_numeric[0].statistics_.tolist(),
        "missing_indicator_columns": learned_numeric[0].indicator_.features_.tolist(),
        "scaler_mean": learned_numeric[1].mean_.tolist(),
        "scaler_scale": learned_numeric[1].scale_.tolist(),
        "approval_processing_categories": transform.named_transformers_["categorical"]
        .categories_[0]
        .tolist(),
        "logistic_coefficients": model[-1].coef_[0].tolist(),
        "logistic_intercept": model[-1].intercept_.tolist(),
        "isotonic_calibration_events": int(sum(y[calibration])),
        "isotonic_x_thresholds": calibrator.X_thresholds_.tolist(),
        "isotonic_y_thresholds": calibrator.y_thresholds_.tolist(),
        "test_used_to_fit": False,
    }
    return {method: np.round(values, 10) for method, values in scores.items()}, audit


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    both = len(set(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "average_precision": float(average_precision_score(y, scores)) if sum(y) else None,
        "brier": float(brier_score_loss(y, scores)),
    }


def month_bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    months: list[str],
    *,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired whole-month resampling; retain one-class replicates and valid counts."""
    groups = [np.flatnonzero(np.asarray(months) == month) for month in sorted(set(months))]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(1, iterations + 1):
        chosen = rng.choice(len(groups), size=len(groups), replace=True)
        indices = np.concatenate([groups[int(i)] for i in chosen])
        for method in METHODS:
            rows.append(
                {
                    "replicate": replicate,
                    "method": method,
                    "records": len(indices),
                    "events": int(sum(y[indices])),
                    **metrics(y[indices], scores[method][indices]),
                }
            )
    intervals: dict[str, Any] = {}
    for method in METHODS:
        intervals[method] = {}
        for metric in METRICS:
            values = [
                row[metric] for row in rows if row["method"] == method and row[metric] is not None
            ]
            intervals[method][metric] = {
                "valid_replicates": len(values),
                "interval_95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
            }
    paired = []
    for offset in range(0, len(rows), len(METHODS)):
        paired.append(rows[offset + 2]["brier"] - rows[offset + 1]["brier"])
    return rows, {
        "unit": "test_funding_month",
        "months": len(groups),
        "seed": seed,
        "requested_replicates": iterations,
        "methods": intervals,
        "paired_isotonic_minus_logistic_brier_95": np.quantile(paired, [0.025, 0.975]).tolist(),
        "scope": "Conditional on fitted models and frozen test sample; no refitting, "
        "source revision, sample-selection or parameter uncertainty.",
    }


def experiment(loans: list[SbaLoan]) -> dict[str, Any]:
    protocol = frozen_json("protocol.json", PROTOCOL_SHA256)
    conversion = frozen_json("audit.json", AUDIT_SHA256)
    scores, audit = fit_scores(loans)
    test = [loan for loan in loans if loan.stage == "test"]
    y = np.asarray([loan.chargeoff_24m for loan in test], dtype=int)
    policies = []
    queue: list[dict[str, Any]] = []
    for method in METHODS:
        ranking = sorted(
            range(len(test)), key=lambda i: (-float(scores[method][i]), test[i].record_id)
        )
        for capacity in (0.05, 0.10, 0.20):
            reviewed = math.ceil(len(test) * capacity)
            captured = int(sum(y[ranking[:reviewed]]))
            policies.append(
                {
                    "method": method,
                    "capacity": capacity,
                    "reviewed": reviewed,
                    "captured": captured,
                    "total_events": int(sum(y)),
                    "precision": captured / reviewed,
                    "capture": captured / int(sum(y)),
                }
            )
        queue.extend(
            {
                "method": method,
                "record_id": test[i].record_id,
                "rank": rank,
                "score": float(scores[method][i]),
            }
            for rank, i in enumerate(ranking[: math.ceil(len(test) * 0.1)], 1)
        )
    years: list[dict[str, Any]] = []
    for year in sorted({loan.funded_at.year for loan in test}):
        idx = np.asarray([i for i, loan in enumerate(test) if loan.funded_at.year == year])
        years.extend(
            {
                "year": year,
                "method": method,
                "records": len(idx),
                "events": int(sum(y[idx])),
                **metrics(y[idx], scores[method][idx]),
            }
            for method in METHODS
        )
    bootstrap, intervals = month_bootstrap(
        y, scores, [loan.funded_at.isoformat()[:7] for loan in test]
    )
    cohorts = {}
    for stage in WINDOWS:
        cohort = [loan for loan in loans if loan.stage == stage]
        cohorts[stage] = {
            "records": len(cohort),
            "events": sum(x.chargeoff_24m for x in cohort),
            "event_rate": sum(x.chargeoff_24m for x in cohort) / len(cohort),
            "funding_first": min(x.funded_at for x in cohort).isoformat(),
            "funding_last": max(x.funded_at for x in cohort).isoformat(),
            "horizon_last": max(x.horizon_end for x in cohort).isoformat(),
            "initial_rate_missing": sum(not x.initial_interest_rate_pct for x in cohort),
        }
    summary = {
        "schema": "creditvintage-sba-credit-v1",
        "data_kind": "public_observed_retrospective_funding_cohorts",
        "target": protocol["target"],
        "source": {
            "publisher": "U.S. Small Business Administration",
            "url": SOURCE_URL,
            "as_of": "2026-06-30",
            "records": len(loans),
            "csv_sha256": DATA_SHA256,
            "license": "LicenseRef-US-Government-Works",
        },
        "protocol": protocol,
        "conversion": conversion,
        "cohorts": cohorts,
        "borrower_group_overlap": 0,
        "fit_audit": audit,
        "test_metrics": {method: metrics(y, scores[method]) for method in METHODS},
        "bootstrap": intervals,
        "numerical_runtime": NUMERICAL,
    }
    predictions = [
        {
            "record_id": loan.record_id,
            "funded_at": loan.funded_at.isoformat(),
            "chargeoff_24m": loan.chargeoff_24m,
            **{method: float(scores[method][i]) for method in METHODS},
        }
        for i, loan in enumerate(test)
    ]
    return {
        "summary": summary,
        "predictions": predictions,
        "policies": policies,
        "queue": queue,
        "years": years,
        "bootstrap": bootstrap,
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]

    def cell(value: Any) -> str:
        return html.escape(str(value))

    def table(rows: list[list[Any]]) -> str:
        return "\n".join(
            "<tr>" + "".join(f"<td>{cell(x)}</td>" for x in row) + "</tr>" for row in rows
        )

    cohorts = summary["cohorts"]
    cohort_rows = [
        [
            stage,
            f"{cohorts[stage]['records']:,}",
            cohorts[stage]["events"],
            f"{cohorts[stage]['event_rate']:.3%}",
            cohorts[stage]["funding_first"] + " → " + cohorts[stage]["funding_last"],
            cohorts[stage]["horizon_last"],
            cohorts[stage]["initial_rate_missing"],
        ]
        for stage in WINDOWS
    ]
    metric_rows = []
    for method in METHODS:
        values = []
        for key in METRICS:
            point = summary["test_metrics"][method][key]
            low, high = summary["bootstrap"]["methods"][method][key]["interval_95"]
            values.append(f"{point:.6f}<small>95% {low:.6f} – {high:.6f}</small>")
        metric_rows.append(
            "<tr><th scope='row'>"
            + cell(method)
            + "</th>"
            + "".join(f"<td>{value}</td>" for value in values)
            + "</tr>"
        )
    policies = [
        [
            r["method"],
            f"{r['capacity']:.0%}",
            r["reviewed"],
            f"{r['captured']}/{r['total_events']}",
            f"{r['precision']:.2%}",
            f"{r['capture']:.2%}",
        ]
        for r in result["policies"]
    ]
    years = [
        [
            r["year"],
            r["method"],
            r["records"],
            r["events"],
            *(f"{r[k]:.6f}" if r[k] is not None else "unavailable" for k in METRICS),
        ]
        for r in result["years"]
    ]
    template = files("creditvintage").joinpath("sba_report.html").read_text(encoding="utf-8")
    for token, value in {
        "@@COHORTS@@": table(cohort_rows),
        "@@METRICS@@": "\n".join(metric_rows),
        "@@POLICIES@@": table(policies),
        "@@YEARS@@": table(years),
        "@@EVENTS@@": str(cohorts["test"]["events"]),
        "@@CSV_SHA@@": DATA_SHA256,
        "@@PROTOCOL_SHA@@": PROTOCOL_SHA256,
        "@@LIMITS@@": "".join(
            "<li>" + cell(x) + "</li>" for x in summary["protocol"]["limitations"]
        ),
    }.items():
        template = template.replace(token, value)
    return template


def _write(result: dict[str, Any], destination: Path) -> None:
    for name in ("source.csv.gz", "protocol.json", "audit.json", "SOURCE.md"):
        (destination / name).write_bytes(resource(name))
    for name, key, columns in (
        ("predictions.csv", "predictions", PREDICTION_COLUMNS),
        ("policy_metrics.csv", "policies", POLICY_COLUMNS),
        ("review_queue.csv", "queue", QUEUE_COLUMNS),
        ("year_metrics.csv", "years", YEAR_COLUMNS),
        ("bootstrap.csv", "bootstrap", BOOT_COLUMNS),
    ):
        _csv(destination / name, result[key], columns)
    _json(destination / "summary.json", result["summary"])
    (destination / "index.html").write_text(render_report(result), encoding="utf-8", newline="")
    _json(
        destination / "manifest.json",
        {
            "schema": "creditvintage-sba-manifest-v1",
            "files_sha256": {name: _hash(destination / name) for name in SBA_FILES},
        },
    )
    with zipfile.ZipFile(destination / "evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (*SBA_FILES, "manifest.json"):
            entry = zipfile.ZipInfo(name, (2026, 10, 25, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, (destination / name).read_bytes())


def write_sba_benchmark(destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Choose a new SBA destination: {destination}")
    result = experiment(load_sba())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sba-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        _write(result, staged)
        staged.rename(destination)
    return dict(result["summary"])


def _same(actual: Any, expected: Any, label: str, *, tolerance: float = 1e-6) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise DataContractError(f"SBA {label} fields differ")
        for key in expected:
            _same(actual[key], expected[key], label, tolerance=tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise DataContractError(f"SBA {label} length differs")
        for left, right in zip(actual, expected, strict=True):
            _same(left, right, label, tolerance=tolerance)
    elif isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or abs(actual - expected) > tolerance
        ):
            raise DataContractError(f"SBA {label} numeric value differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise DataContractError(f"SBA {label} identity or count differs")


def verify_sba_benchmark(directory: Path) -> dict[str, Any]:
    actual_runtime = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    if actual_runtime != NUMERICAL:
        raise DataContractError("SBA numerical runtime differs from the frozen versions")
    manifest = read_json_object(directory / "manifest.json")
    if (
        set(manifest) != {"schema", "files_sha256"}
        or manifest["schema"] != "creditvintage-sba-manifest-v1"
        or set(manifest["files_sha256"]) != set(SBA_FILES)
    ):
        raise DataContractError("SBA manifest inventory differs")
    for name in SBA_FILES:
        if _hash(directory / name) != manifest["files_sha256"][name]:
            raise DataContractError(f"SBA file digest differs: {name}")
    for name in ("protocol.json", "audit.json", "SOURCE.md"):
        if (directory / name).read_bytes() != resource(name):
            raise DataContractError(f"SBA frozen {name} differs")
    result = experiment(load_sba((directory / "source.csv.gz").read_bytes()))
    summary = read_json_object(directory / "summary.json")
    _same(summary, result["summary"], "summary")
    for name, key, columns in (
        ("predictions.csv", "predictions", PREDICTION_COLUMNS),
        ("policy_metrics.csv", "policies", POLICY_COLUMNS),
        ("review_queue.csv", "queue", QUEUE_COLUMNS),
        ("year_metrics.csv", "years", YEAR_COLUMNS),
        ("bootstrap.csv", "bootstrap", BOOT_COLUMNS),
    ):
        rows = read_csv_rows(directory / name, columns)
        if len(rows) != len(result[key]):
            raise DataContractError(f"SBA {name} row count differs")
        for row, expected in zip(rows, result[key], strict=True):
            for field, value in expected.items():
                if isinstance(value, float):
                    try:
                        number = float(row[field])
                    except ValueError as exc:
                        raise DataContractError(f"SBA {name} number invalid") from exc
                    _same(
                        number,
                        value,
                        name,
                        tolerance=1e-8 if key in {"predictions", "queue"} else 1e-6,
                    )
                elif row[field] != ("" if value is None else str(value)):
                    raise DataContractError(f"SBA {name} identity or count differs")
    # Re-render verified stated numbers: small cross-platform fit differences are allowed above.
    report_result = {**result, "summary": summary}
    for name, key, columns in (
        ("policy_metrics.csv", "policies", POLICY_COLUMNS),
        ("year_metrics.csv", "years", YEAR_COLUMNS),
    ):
        retained = read_csv_rows(directory / name, columns)
        report_result[key] = [
            {
                field: (float(row[field]) if isinstance(value, float) else value)
                for field, value in expected.items()
            }
            for row, expected in zip(retained, result[key], strict=True)
        ]
    if (directory / "index.html").read_text(encoding="utf-8") != render_report(report_result):
        raise DataContractError("SBA HTML differs from verified retained numbers")
    try:
        with zipfile.ZipFile(io.BytesIO((directory / "evidence.zip").read_bytes())) as archive:
            if archive.namelist() != [*SBA_FILES, "manifest.json"]:
                raise DataContractError("SBA ZIP inventory differs")
            for entry in archive.infolist():
                path = directory / entry.filename
                if (
                    entry.file_size != path.stat().st_size
                    or archive.read(entry) != path.read_bytes()
                ):
                    raise DataContractError("SBA ZIP content differs")
    except zipfile.BadZipFile as exc:
        raise DataContractError("Invalid SBA evidence ZIP") from exc
    return {
        "status": "verified",
        "refitted": True,
        "records": 35000,
        "test_records": 10000,
        "test_events": summary["cohorts"]["test"]["events"],
        "target": summary["target"],
        "methods": len(METHODS),
        "borrower_group_overlap": 0,
        "bootstrap_replicates": ITERATIONS,
        "python": platform.python_version(),
    }
