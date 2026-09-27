"""Prospective vintage splits, fixed baseline, independent PD calibration."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from creditvintage.core import (
    HORIZON_DAYS,
    REPORT_LAG_DAYS,
    Application,
    CohortRow,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    snapshot,
)


@dataclass(frozen=True)
class Config:
    train_start: date = date(2022, 1, 1)
    train_end: date = date(2022, 12, 31)
    calibration_start: date = date(2023, 7, 1)
    calibration_end: date = date(2023, 12, 31)
    test_start: date = date(2024, 7, 1)
    test_end: date = date(2024, 12, 31)
    evaluation_as_of: date = date(2025, 7, 1)
    scenario_lgd: float = 0.45
    review_fraction: float = 0.10
    seed: int = 20260927

    def __post_init__(self) -> None:
        lag = timedelta(days=HORIZON_DAYS + REPORT_LAG_DAYS)
        if not (
            self.train_start <= self.train_end
            and self.train_end + lag <= self.calibration_start <= self.calibration_end
            and self.calibration_end + lag <= self.test_start <= self.test_end
            and self.test_end + lag <= self.evaluation_as_of
        ):
            raise DataContractError("Vintage windows must include full outcome-maturation gaps")
        if not 0 < self.scenario_lgd <= 1 or not 0 < self.review_fraction <= 1:
            raise DataContractError("LGD and review fraction must be within (0, 1]")


@dataclass(frozen=True)
class Result:
    summary: dict[str, Any]
    predictions: list[dict[str, Any]]
    deciles: list[dict[str, Any]]
    vintages: list[dict[str, Any]]
    input_sha256: str


def _rows_between(rows: list[CohortRow], start: date, end: date, stage: str) -> list[CohortRow]:
    selected = [row for row in rows if start <= row.application.decision_at <= end]
    incomplete = [row for row in selected if row.label is None]
    if incomplete:
        counts: dict[str, int] = {}
        for row in incomplete:
            counts[row.status] = counts.get(row.status, 0) + 1
        raise DataContractError(f"{stage} has unlabelled loans: {counts}")
    if len(selected) < 20 or len({row.label for row in selected}) != 2:
        raise DataContractError(f"{stage} needs at least 20 loans and both outcomes")
    return selected


def _matrix(rows: list[CohortRow]) -> np.ndarray:
    return np.asarray(
        [
            [
                *row.features,
                math.log1p(row.application.principal_cents / 100),
                float(row.application.segment == "small_business"),
            ]
            for row in rows
        ],
        dtype=float,
    )


def _labels(rows: list[CohortRow]) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=int)


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(labels, probability)), 6),
        "average_precision": round(float(average_precision_score(labels, probability)), 6),
        "brier": round(float(brier_score_loss(labels, probability)), 6),
    }


def _vintage_bootstrap(
    rows: list[CohortRow], probability: np.ndarray, seed: int, iterations: int = 300
) -> dict[str, Any]:
    """Resample whole origination months; six months give wide, descriptive intervals."""
    month = np.asarray([row.application.decision_at.strftime("%Y-%m") for row in rows])
    labels = _labels(rows)
    groups = sorted(set(month))
    rng = np.random.default_rng(seed)
    auc: list[float] = []
    brier: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(month == group) for group in sampled])
        if len(np.unique(labels[indices])) != 2:
            continue
        auc.append(float(roc_auc_score(labels[indices], probability[indices])))
        brier.append(float(brier_score_loss(labels[indices], probability[indices])))
    if not auc:
        raise DataContractError("No valid vintage bootstrap replicates")
    return {
        "unit": "origination_month",
        "valid_replicates": len(auc),
        "requested_replicates": iterations,
        "roc_auc_95_interval": [round(float(v), 6) for v in np.quantile(auc, [0.025, 0.975])],
        "brier_95_interval": [round(float(v), 6) for v in np.quantile(brier, [0.025, 0.975])],
    }


def _digest(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
) -> str:
    payload = {
        "applications": sorted((asdict(item) for item in applications), key=str),
        "features": sorted((asdict(item) for item in features), key=str),
        "performance": sorted((asdict(item) for item in performance), key=str),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fit_stages(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
    config: Config,
) -> tuple[Pipeline, IsotonicRegression, list[CohortRow], list[CohortRow], np.ndarray]:
    """Freeze the model and calibrator before the first test application."""
    train = _rows_between(
        snapshot(applications, features, performance, config.calibration_start),
        config.train_start,
        config.train_end,
        "train",
    )
    calibration = _rows_between(
        snapshot(applications, features, performance, config.test_start),
        config.calibration_start,
        config.calibration_end,
        "calibration",
    )
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=config.seed)
    )
    model.fit(_matrix(train), _labels(train))
    calibration_raw = model.predict_proba(_matrix(calibration))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_raw, _labels(calibration))
    return model, calibrator, train, calibration, calibration_raw


def evaluate(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
    config: Config | None = None,
    *,
    data_kind: str = "user_supplied",
) -> Result:
    """Fit before calibration starts; calibrate before test starts; never fit on test."""
    config = config or Config()
    if data_kind not in {"synthetic", "user_supplied"}:
        raise DataContractError("Unknown data kind")
    model, calibrator, train, calibration, _ = _fit_stages(
        applications, features, performance, config
    )
    test = _rows_between(
        snapshot(applications, features, performance, config.evaluation_as_of),
        config.test_start,
        config.test_end,
        "test",
    )
    train_y = _labels(train)
    calibration_y = _labels(calibration)
    test_y = _labels(test)
    raw_pd = model.predict_proba(_matrix(test))[:, 1]
    calibrated_pd = np.asarray(calibrator.predict(raw_pd), dtype=float)
    if not np.all(np.isfinite(calibrated_pd)) or not np.all(
        (0 <= calibrated_pd) & (calibrated_pd <= 1)
    ):
        raise DataContractError("Invalid calibrated probabilities")

    # Rank on the published precision so the exported review policy can be
    # reconstructed exactly from predictions.csv, including score ties.
    published_pd = [round(float(value), 10) for value in calibrated_pd]
    review_count = math.ceil(len(test) * config.review_fraction)
    order = sorted(
        range(len(test)), key=lambda i: (-published_pd[i], test[i].application.application_id)
    )
    reviewed = set(order[:review_count])
    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(test):
        app = row.application
        predictions.append(
            {
                "application_id": app.application_id,
                "decision_at": app.decision_at.isoformat(),
                "vintage": app.decision_at.strftime("%Y-%m"),
                "segment": app.segment,
                "principal_cents": app.principal_cents,
                "label": int(test_y[i]),
                "raw_pd": round(float(raw_pd[i]), 10),
                "calibrated_pd": published_pd[i],
                "review_selected": int(i in reviewed),
            }
        )

    def aggregate(indices: list[int]) -> dict[str, Any]:
        values = calibrated_pd[indices]
        outcomes = test_y[indices]
        principal_usd = np.asarray(
            [test[i].application.principal_cents / 100 for i in indices], dtype=float
        )
        return {
            "loans": len(indices),
            "observed_default_rate": round(float(np.mean(outcomes)), 6),
            "mean_pd": round(float(np.mean(values)), 6),
            "principal_usd": round(float(np.sum(principal_usd)), 2),
            "scenario_expected_loss_usd": round(
                float(np.sum(values * principal_usd) * config.scenario_lgd), 2
            ),
            "scenario_observed_loss_proxy_usd": round(
                float(np.sum(outcomes * principal_usd) * config.scenario_lgd), 2
            ),
        }

    deciles = [
        {"risk_bucket": bucket + 1, **aggregate(indices.tolist())}
        for bucket, indices in enumerate(np.array_split(np.asarray(order), 10))
    ]
    months = sorted({row.application.decision_at.strftime("%Y-%m") for row in test})
    vintages = [
        {
            "vintage": month,
            **aggregate(
                [
                    i
                    for i, row in enumerate(test)
                    if row.application.decision_at.strftime("%Y-%m") == month
                ]
            ),
        }
        for month in months
    ]
    review_labels = test_y[list(reviewed)]
    summary = {
        "project": "CreditVintage",
        "data_kind": data_kind,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "label_rule": "90+ DPD within 180 days; negatives need all six reports",
        "model": "fixed logistic regression + separate isotonic calibration",
        "config": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in asdict(config).items()
        },
        "cohorts": {
            "train": {"loans": len(train), "default_rate": round(float(train_y.mean()), 6)},
            "calibration": {
                "loans": len(calibration),
                "default_rate": round(float(calibration_y.mean()), 6),
            },
            "test": {"loans": len(test), "default_rate": round(float(test_y.mean()), 6)},
        },
        "test_metrics": {
            "constant_train_rate": _metrics(test_y, np.full(len(test), train_y.mean())),
            "raw_logistic": _metrics(test_y, raw_pd),
            "calibrated": _metrics(test_y, calibrated_pd),
        },
        "vintage_bootstrap": _vintage_bootstrap(test, calibrated_pd, config.seed),
        "review_at_fixed_capacity": {
            "fraction": config.review_fraction,
            "loans": review_count,
            "precision": round(float(np.mean(review_labels)), 6),
            "default_capture": round(float(np.sum(review_labels) / np.sum(test_y)), 6),
        },
        "test_portfolio": aggregate(list(range(len(test)))),
    }
    return Result(
        summary, predictions, deciles, vintages, _digest(applications, features, performance)
    )
