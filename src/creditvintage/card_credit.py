"""Frozen observed credit-card data, train-only transformations and complete split evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import math
import platform
import zipfile
import zlib
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import DataContractError
from creditvintage.model import _metrics
from creditvintage.public_credit import _compare_summary
from creditvintage.validation import finite_number, read_csv_rows, read_json_object

SOURCE_SHA256 = "4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb"
SEEDS = (7, 23, 41, 59, 83)
METHODS = ("constant", "logistic", "isotonic", "hist_gradient_boosting")
FEATURES = (
    "LIMIT_BAL",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *(f"BILL_AMT{i}" for i in range(1, 7)),
    *(f"PAY_AMT{i}" for i in range(1, 7)),
)
COLUMNS = (
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *FEATURES[1:],
    "default payment next month",
)
NUMERIC = (0, *range(7, 19))
CATEGORICAL = tuple(range(1, 7))
SPLIT_COLUMNS = ("seed", "record_id", "stage")
PREDICTION_COLUMNS = ("seed", "record_id", "native_default", *METHODS)
POLICY_COLUMNS = (
    "seed",
    "method",
    "capacity",
    "reviewed",
    "captured",
    "total_events",
    "precision",
    "capture",
)
METRIC_COLUMNS = ("seed", "method", "roc_auc", "average_precision", "brier")
QUEUE_COLUMNS = ("seed", "method", "record_id", "rank", "score")
TABLES = (
    ("splits.csv", "splits", SPLIT_COLUMNS),
    ("predictions.csv", "predictions", PREDICTION_COLUMNS),
    ("policy_metrics.csv", "policies", POLICY_COLUMNS),
    ("metrics.csv", "metrics", METRIC_COLUMNS),
    ("review_queue.csv", "queue", QUEUE_COLUMNS),
)
CARD_FILES = (
    "source.csv.gz",
    "SOURCE.md",
    *(item[0] for item in TABLES),
    "summary.json",
    "index.html",
)


def source_bytes() -> bytes:
    return files("creditvintage").joinpath("data/card_credit/credit_card.csv.gz").read_bytes()


def load_card_data(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
            content = source.read(2867208 + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise DataContractError("Invalid frozen card-credit gzip") from exc
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise DataContractError("Card-credit source digest differs from the frozen UCI conversion")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")), strict=True)
    if reader.fieldnames != list(COLUMNS):
        raise DataContractError("Unexpected card-credit columns")
    rows = list(reader)
    if len(rows) != 30000 or [row["ID"] for row in rows] != [str(i) for i in range(1, 30001)]:
        raise DataContractError("Card-credit original row identities differ")
    x = np.asarray([[int(row[name]) for name in FEATURES] for row in rows], dtype=float)
    y = np.asarray([int(row["default payment next month"]) for row in rows], dtype=int)
    if np.bincount(y).tolist() != [23364, 6636] or not np.isfinite(x).all():
        raise DataContractError("Card-credit labels or numeric values differ")
    return x, y


def card_splits(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    development, test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    train, calibration = train_test_split(
        development, test_size=0.25, stratify=y[development], random_state=seed + 1
    )
    return {
        name: np.sort(indices)
        for name, indices in (("train", train), ("calibration", calibration), ("test", test))
    }


def fit_card_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """The scoring API accepts no test outcomes; every transformation fits on train."""

    def transform() -> ColumnTransformer:
        return ColumnTransformer(
            [
                ("numeric", StandardScaler(), list(NUMERIC)),
                (
                    "status",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(CATEGORICAL),
                ),
            ],
            sparse_threshold=0,
        )

    with threadpool_limits(limits=1):
        logistic = make_pipeline(
            transform(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
        )
        logistic.fit(train_x, train_y)
        raw = logistic.predict_proba(test_x)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(logistic.predict_proba(calibration_x)[:, 1], calibration_y)
        tree = make_pipeline(
            transform(),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=15,
                max_depth=3,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
        )
        tree.fit(train_x, train_y)
        return {
            "constant": np.full(len(test_x), float(np.mean(train_y))),
            "logistic": raw,
            "isotonic": calibrator.predict(raw),
            "hist_gradient_boosting": tree.predict_proba(test_x)[:, 1],
        }


def card_experiment(raw: bytes) -> dict[str, Any]:
    x, y = load_card_data(raw)
    assignments: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for seed in SEEDS:
        parts = card_splits(y, seed)
        train, calibration, test = (parts[name] for name in ("train", "calibration", "test"))
        scores = fit_card_scores(x[train], y[train], x[calibration], y[calibration], x[test], seed)
        counts[str(seed)] = {
            name: {"records": len(indices), "events": int(sum(y[indices]))}
            for name, indices in parts.items()
        }
        assignments.extend(
            {"seed": seed, "record_id": int(i) + 1, "stage": name}
            for name, indices in parts.items()
            for i in indices
        )
        predictions.extend(
            {
                "seed": seed,
                "record_id": int(i) + 1,
                "native_default": int(y[i]),
                **{method: round(float(scores[method][j]), 12) for method in METHODS},
            }
            for j, i in enumerate(test)
        )
        for method in METHODS:
            metrics.append({"seed": seed, "method": method, **_metrics(y[test], scores[method])})
            ranks = np.lexsort((test, -scores[method]))
            for capacity in (0.05, 0.10, 0.20):
                selected = math.ceil(len(test) * capacity)
                captured = int(sum(y[test[ranks[:selected]]]))
                policies.append(
                    {
                        "seed": seed,
                        "method": method,
                        "capacity": capacity,
                        "reviewed": selected,
                        "captured": captured,
                        "total_events": int(sum(y[test])),
                        "precision": round(captured / selected, 6),
                        "capture": round(captured / int(sum(y[test])), 6),
                    }
                )
            queue.extend(
                {
                    "seed": seed,
                    "method": method,
                    "record_id": int(test[j]) + 1,
                    "rank": rank,
                    "score": round(float(scores[method][j]), 12),
                }
                for rank, j in enumerate(ranks[: math.ceil(len(test) * 0.1)], 1)
            )
    summary = {
        "schema": "creditvintage-card-credit-v1",
        "data_kind": "public_observed_static",
        "source": {
            "name": "Yeh (2009), Default of Credit Card Clients, UCI 350",
            "doi": "https://doi.org/10.24432/C55S3H",
            "license": "CC BY 4.0",
            "canonical_sha256": SOURCE_SHA256,
            "records": len(y),
            "events": int(sum(y)),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "split": "five frozen stratified 60/20/20 random splits",
            "features": list(FEATURES),
            "excluded_fields": ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"],
            "label": "Unchanged native 'default payment next month': 0/1",
            "history": "April–September 2005 repayment, bill and payment history in source records",
            "preprocessing": "train-only numeric scaling and repayment-status one-hot encoding",
            "logistic": "C=1; lbfgs; max_iter=2000; no class weighting",
            "tree": "learning_rate=.05; 100 iterations; depth=3; 15 leaves; min_leaf=20; L2=1",
            "calibration": "isotonic fitted only on the separate calibration partition",
            "tuning": "none; all frozen seeds retained; no model selected from test results",
            "review": "ceil(capacity*n_test), descending score then original source ID",
            "limits": [
                "One historical cohort; random customer splits do not validate future periods",
                "No publication/revision timestamps: information availability remains unvalidated",
                "Native subsequent-default target differs from the synthetic 180-day 90+ DPD label",
                "Overlapping repeated splits: min/max are descriptive, not confidence intervals",
                "ID is excluded from fitting; demographic exclusion does not demonstrate fairness",
                "No observed recoveries, intervention effects, live deployment or monetary savings",
            ],
        },
        "cohorts": counts,
        "results": {
            method: {
                metric: {
                    "mean": round(float(np.mean(values)), 6),
                    "min": min(values),
                    "max": max(values),
                }
                for metric in ("roc_auc", "average_precision", "brier")
                for values in [[float(row[metric]) for row in metrics if row["method"] == method]]
            }
            for method in METHODS
        },
        "metrics": metrics,
        "policies": policies,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return {
        "summary": summary,
        "splits": assignments,
        "predictions": predictions,
        "metrics": metrics,
        "policies": policies,
        "queue": queue,
    }


def render_card_report(summary: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(method.replace('_', ' '))}</th>"
        + "".join(
            f"<td>{summary['results'][method][metric]['mean']:.3f} "
            f"<small>({summary['results'][method][metric]['min']:.3f}–"
            f"{summary['results'][method][metric]['max']:.3f})</small></td>"
            for metric in ("roc_auc", "average_precision", "brier")
        )
        + "</tr>"
        for method in METHODS
    )
    limits = "".join(f"<li>{html.escape(s)}</li>" for s in summary["protocol"]["limits"])
    template = files("creditvintage").joinpath("card_report.html").read_text(encoding="utf-8")
    return template.replace("@@ROWS@@", rows).replace("@@LIMITS@@", limits)


def write_card_benchmark(destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise DataContractError("Choose a new card benchmark output directory")
    raw = source_bytes()
    result = card_experiment(raw)
    destination.mkdir(parents=True)
    (destination / "source.csv.gz").write_bytes(raw)
    (destination / "SOURCE.md").write_bytes(
        files("creditvintage").joinpath("data/card_credit/SOURCE.md").read_bytes()
    )
    for filename, key, columns in TABLES:
        _csv(destination / filename, result[key], columns)
    _json(destination / "summary.json", result["summary"])
    (destination / "index.html").write_text(
        render_card_report(result["summary"]), encoding="utf-8", newline=""
    )
    _json(destination / "manifest.json", {name: _hash(destination / name) for name in CARD_FILES})
    with zipfile.ZipFile(destination / "evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (*CARD_FILES, "manifest.json"):
            entry = zipfile.ZipInfo(name, (2026, 10, 15, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (destination / name).read_bytes())
    summary: dict[str, Any] = result["summary"]
    return summary


def verify_card_benchmark(directory: Path) -> dict[str, Any]:
    if read_json_object(directory / "manifest.json") != {
        name: _hash(directory / name) for name in CARD_FILES
    }:
        raise DataContractError("Card benchmark file digests differ")
    result = card_experiment((directory / "source.csv.gz").read_bytes())
    expected_notice = files("creditvintage").joinpath("data/card_credit/SOURCE.md").read_bytes()
    if (directory / "SOURCE.md").read_bytes() != expected_notice:
        raise DataContractError("Card benchmark source attribution differs")
    summary = read_json_object(directory / "summary.json")
    runtime = summary.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"python", "numpy", "scikit_learn"}
        or any(not isinstance(v, str) or not v for v in runtime.values())
    ):
        raise DataContractError("Invalid card benchmark runtime")
    result["summary"]["runtime"] = runtime
    _compare_summary(summary, result["summary"], "card summary")
    for filename, key, columns in TABLES:
        rows = read_csv_rows(directory / filename, columns)
        if len(rows) != len(result[key]):
            raise DataContractError(f"Card benchmark row count differs: {filename}")
        for actual, expected in zip(rows, result[key], strict=True):
            for name, value in expected.items():
                agrees = (
                    math.isclose(finite_number(actual[name], name), value, rel_tol=0, abs_tol=1e-8)
                    if type(value) is float
                    else actual[name] == str(value)
                )
                if not agrees:
                    raise DataContractError(f"Card model replay differs: {filename}:{name}")
    if (directory / "index.html").read_text(encoding="utf-8") != render_card_report(summary):
        raise DataContractError("Card report page differs from verified evidence")
    try:
        with zipfile.ZipFile(directory / "evidence.zip") as archive:
            if archive.namelist() != [*CARD_FILES, "manifest.json"] or any(
                archive.read(name) != (directory / name).read_bytes() for name in archive.namelist()
            ):
                raise DataContractError("Card evidence ZIP differs from verified files")
    except zipfile.BadZipFile as exc:
        raise DataContractError("Invalid card evidence ZIP") from exc
    return {
        "status": "verified",
        "records": 30000,
        "splits": len(SEEDS),
        "test_predictions": 30000,
        "model_scores": 120000,
        "refitted": True,
        "data_kind": "public_observed_static",
    }
