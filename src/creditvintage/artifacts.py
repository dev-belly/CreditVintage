"""Portable result files, a static report, and independent artifact checks."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from creditvintage.core import _ID, DataContractError
from creditvintage.model import Config, Result
from creditvintage.validation import finite_number, read_csv_rows, read_json_object

PREDICTION_COLUMNS = (
    "application_id",
    "decision_at",
    "vintage",
    "segment",
    "principal_cents",
    "label",
    "raw_pd",
    "calibrated_pd",
    "review_selected",
)
AGGREGATE_COLUMNS = (
    "loans",
    "observed_default_rate",
    "mean_pd",
    "principal_usd",
    "scenario_expected_loss_usd",
    "scenario_observed_loss_proxy_usd",
)
RESULT_FILES = (
    "summary.json",
    "predictions.csv",
    "calibration_bins.csv",
    "vintage_metrics.csv",
    "index.html",
)


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_report(result: Result) -> str:
    summary = result.summary
    metrics = summary["test_metrics"]
    portfolio = summary["test_portfolio"]
    cohorts = summary["cohorts"]
    source = "SYNTHETIC EXAMPLE" if summary["data_kind"] == "synthetic" else "USER DATA"
    bars = "\n".join(
        f'<div class="vintage"><span>{html.escape(row["vintage"])}</span>'
        f'<div class="track"><i style="width:{row["observed_default_rate"] * 100:.2f}%"></i></div>'
        f"<strong>{row['observed_default_rate'] * 100:.1f}%</strong>"
        f"<small>mean PD {row['mean_pd'] * 100:.1f}% · n={row['loans']}</small></div>"
        for row in result.vintages
    )
    deciles = "\n".join(
        "<tr>"
        f"<td>{row['risk_bucket']}</td><td>{row['loans']}</td>"
        f"<td>{row['mean_pd'] * 100:.1f}%</td>"
        f"<td>{row['observed_default_rate'] * 100:.1f}%</td>"
        f"<td>${row['scenario_expected_loss_usd']:,.2f}</td>"
        "</tr>"
        for row in result.deciles
    )
    template = files("creditvintage").joinpath("report.html").read_text(encoding="utf-8")
    replacements = {
        "@@SOURCE@@": source,
        "@@FOOTER_NOTE@@": (
            "Public sample: synthetic only."
            if summary["data_kind"] == "synthetic"
            else "User-supplied data: review privacy before sharing."
        ),
        "@@TEST_LOANS@@": f"{cohorts['test']['loans']:,}",
        "@@DEFAULT_RATE@@": f"{cohorts['test']['default_rate'] * 100:.1f}",
        "@@BRIER@@": f"{metrics['calibrated']['brier']:.3f}",
        "@@RAW_BRIER@@": f"{metrics['raw_logistic']['brier']:.3f}",
        "@@CALIBRATION_DIRECTION@@": (
            "decreased"
            if metrics["calibrated"]["brier"] < metrics["raw_logistic"]["brier"]
            else "increased"
        ),
        "@@BASELINE_BRIER@@": f"{metrics['constant_train_rate']['brier']:.3f}",
        "@@CAPTURE@@": f"{summary['review_at_fixed_capacity']['default_capture'] * 100:.1f}",
        "@@CAPACITY@@": f"{summary['review_at_fixed_capacity']['fraction'] * 100:.0f}",
        "@@AUC@@": f"{metrics['calibrated']['roc_auc']:.3f}",
        "@@AP@@": f"{metrics['calibrated']['average_precision']:.3f}",
        "@@AUC_LOW@@": f"{summary['vintage_bootstrap']['roc_auc_95_interval'][0]:.3f}",
        "@@AUC_HIGH@@": f"{summary['vintage_bootstrap']['roc_auc_95_interval'][1]:.3f}",
        "@@SCENARIO_EL@@": f"{portfolio['scenario_expected_loss_usd']:,.2f}",
        "@@LGD@@": f"{summary['config']['scenario_lgd'] * 100:.0f}",
        "@@BARS@@": bars,
        "@@DECILES@@": deciles,
    }
    for token, content in replacements.items():
        template = template.replace(token, content)
    return template


def write_result(result: Result, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    _json(destination / "summary.json", result.summary)
    _csv(destination / "predictions.csv", result.predictions, PREDICTION_COLUMNS)
    _csv(destination / "calibration_bins.csv", result.deciles, ("risk_bucket", *AGGREGATE_COLUMNS))
    _csv(destination / "vintage_metrics.csv", result.vintages, ("vintage", *AGGREGATE_COLUMNS))
    (destination / "index.html").write_text(render_report(result), encoding="utf-8")
    manifest = {
        "input_sha256": result.input_sha256,
        "files_sha256": {name: _hash(destination / name) for name in RESULT_FILES},
    }
    _json(destination / "manifest.json", manifest)
    return manifest


def verify_result(destination: Path) -> dict[str, Any]:
    try:
        manifest = read_json_object(destination / "manifest.json")
        if set(manifest["files_sha256"]) != set(RESULT_FILES):
            raise DataContractError("Unexpected artifact inventory")
        for name, digest in manifest["files_sha256"].items():
            if _hash(destination / name) != digest:
                raise DataContractError(f"Artifact digest mismatch: {name}")
        summary = read_json_object(destination / "summary.json")
        predictions = read_csv_rows(destination / "predictions.csv", PREDICTION_COLUMNS)
        if len({row["application_id"] for row in predictions}) != len(predictions):
            raise DataContractError("Duplicate prediction ID")
        labels = np.asarray([int(row["label"]) for row in predictions])
        if len(predictions) != summary["cohorts"]["test"]["loans"] or set(labels) != {0, 1}:
            raise DataContractError("Prediction count or outcomes disagree with summary")
        config = summary["config"]
        checked_config = Config(
            train_start=date.fromisoformat(config["train_start"]),
            train_end=date.fromisoformat(config["train_end"]),
            calibration_start=date.fromisoformat(config["calibration_start"]),
            calibration_end=date.fromisoformat(config["calibration_end"]),
            test_start=date.fromisoformat(config["test_start"]),
            test_end=date.fromisoformat(config["test_end"]),
            evaluation_as_of=date.fromisoformat(config["evaluation_as_of"]),
            scenario_lgd=finite_number(config["scenario_lgd"], "scenario LGD"),
            review_fraction=finite_number(config["review_fraction"], "configured review fraction"),
            seed=config["seed"],
        )
        start, end = checked_config.test_start, checked_config.test_end
        if any(
            not _ID.fullmatch(row["application_id"])
            or row["segment"] not in {"retail", "small_business"}
            or int(row["principal_cents"]) <= 0
            or not start <= (decision := date.fromisoformat(row["decision_at"])) <= end
            or decision.strftime("%Y-%m") != row["vintage"]
            for row in predictions
        ):
            raise DataContractError("Prediction identity, date or segment disagrees with contract")
        if (
            abs(
                float(labels.mean())
                - finite_number(summary["cohorts"]["test"]["default_rate"], "test default rate")
            )
            > 0.000002
        ):
            raise DataContractError("Test default rate disagrees with predictions")
        train_rate = finite_number(
            summary["cohorts"]["train"]["default_rate"], "train default rate"
        )
        if not 0 <= train_rate <= 1:
            raise DataContractError("Invalid train default rate")
        baseline = summary["test_metrics"]["constant_train_rate"]
        baseline_brier = brier_score_loss(labels, np.full(len(labels), train_rate))
        if (
            abs(finite_number(baseline["roc_auc"], "baseline AUC") - 0.5) > 0.000002
            or abs(
                finite_number(baseline["average_precision"], "baseline AP") - float(labels.mean())
            )
            > 0.000002
            or abs(finite_number(baseline["brier"], "baseline Brier") - baseline_brier) > 0.000002
        ):
            raise DataContractError("Constant baseline disagrees with predictions")
        for key, title in (("raw_pd", "raw_logistic"), ("calibrated_pd", "calibrated")):
            p = np.asarray([float(row[key]) for row in predictions])
            if not np.all(np.isfinite(p)) or not np.all((0 <= p) & (p <= 1)):
                raise DataContractError("Invalid probability in predictions")
            computed = {
                "roc_auc": roc_auc_score(labels, p),
                "average_precision": average_precision_score(labels, p),
                "brier": brier_score_loss(labels, p),
            }
            if any(
                abs(
                    computed[name]
                    - finite_number(summary["test_metrics"][title][name], f"{title} {name}")
                )
                > 0.000002
                for name in computed
            ):
                raise DataContractError(f"Metrics disagree with predictions: {title}")
        review = np.asarray([int(row["review_selected"]) for row in predictions])
        policy = summary["review_at_fixed_capacity"]
        fraction = finite_number(policy["fraction"], "review fraction")
        review_count = math.ceil(len(predictions) * fraction)
        if (
            set(review) - {0, 1}
            or not 0 < fraction <= 1
            or fraction != checked_config.review_fraction
            or review_count != finite_number(policy["loans"], "review loans")
            or int(review.sum()) != review_count
        ):
            raise DataContractError("Review capacity disagrees with predictions")
        probability = np.asarray([float(row["calibrated_pd"]) for row in predictions])
        ranked = sorted(
            range(len(predictions)),
            key=lambda i: (-probability[i], predictions[i]["application_id"]),
        )
        if set(np.flatnonzero(review)) != set(ranked[:review_count]):
            raise DataContractError("Review selection is not the top-ranked fixed-capacity group")
        selected = labels[review == 1]
        if (
            abs(float(selected.mean()) - finite_number(policy["precision"], "review precision"))
            > 0.000002
            or abs(
                float(selected.sum() / labels.sum())
                - finite_number(policy["default_capture"], "review default capture")
            )
            > 0.000002
        ):
            raise DataContractError("Review outcomes disagree with predictions")

        principals = np.asarray([int(row["principal_cents"]) / 100 for row in predictions])
        lgd = checked_config.scenario_lgd

        def check_aggregate(indices: np.ndarray, stated: dict[str, Any]) -> None:
            if len(indices) != finite_number(stated["loans"], "cohort loans"):
                raise DataContractError("Cohort count disagrees with predictions")
            actual = {
                "observed_default_rate": float(labels[indices].mean()),
                "mean_pd": float(probability[indices].mean()),
                "principal_usd": float(principals[indices].sum()),
                "scenario_expected_loss_usd": float(
                    np.sum(probability[indices] * principals[indices]) * lgd
                ),
                "scenario_observed_loss_proxy_usd": float(
                    np.sum(labels[indices] * principals[indices]) * lgd
                ),
            }
            for name, value in actual.items():
                tolerance = 0.02 if name.endswith("usd") else 0.000002
                if abs(value - finite_number(stated[name], f"cohort {name}")) > tolerance:
                    raise DataContractError(f"Cohort {name} disagrees with predictions")

        check_aggregate(np.arange(len(predictions)), summary["test_portfolio"])
        bins = read_csv_rows(
            destination / "calibration_bins.csv", ("risk_bucket", *AGGREGATE_COLUMNS)
        )
        if len(bins) != 10:
            raise DataContractError("Risk bucket count disagrees with predictions")
        for rank, (indices, bucket) in enumerate(
            zip(np.array_split(ranked, 10), bins, strict=True), 1
        ):
            if int(bucket["risk_bucket"]) != rank:
                raise DataContractError("Risk bucket order is invalid")
            check_aggregate(indices, bucket)
        vintages = read_csv_rows(
            destination / "vintage_metrics.csv", ("vintage", *AGGREGATE_COLUMNS)
        )
        months = sorted({row["vintage"] for row in predictions})
        if [row["vintage"] for row in vintages] != months:
            raise DataContractError("Vintage coverage disagrees with predictions")
        for month, vintage in zip(months, vintages, strict=True):
            indices = np.asarray(
                [i for i, row in enumerate(predictions) if row["vintage"] == month]
            )
            check_aggregate(indices, vintage)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify result: {exc}") from exc
    return {
        "status": "verified",
        "loans": len(predictions),
        "input_sha256": manifest["input_sha256"],
    }
