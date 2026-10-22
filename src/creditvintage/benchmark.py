"""Frozen prospective models compared at the same global review capacity.

WOE cuts, supervised bin values and model parameters are fitted on training
applications only. Test outcomes are used only after scores have been frozen.
Explanations describe additive raw-logistic log odds, never causal effects.
"""

from __future__ import annotations

import html
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import (
    FEATURE_NAMES,
    Application,
    CohortRow,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    parse_calendar_date,
    snapshot,
)
from creditvintage.model import Config, _digest, _fit_stages, _labels, _matrix, _rows_between
from creditvintage.validation import finite_number, read_csv_rows, read_json_object

MODEL_FEATURES = (*FEATURE_NAMES, "log_principal_usd", "small_business")
METHODS = (
    "business_rule",
    "raw_logistic",
    "calibrated_logistic",
    "woe_logistic",
    "hist_gradient_boosting",
)
PREDICTION_COLUMNS = (
    "method",
    "application_id",
    "decision_at",
    "vintage",
    "segment",
    "principal_cents",
    "label",
    "score",
    "review_rank",
    "scorecard_points",
)
EXPLANATION_COLUMNS = (
    "application_id",
    "feature",
    "value",
    "training_mean",
    "coefficient_per_unit",
    "log_odds_contribution",
    "baseline_log_odds",
    "reason_rank",
)
POLICY_COLUMNS = (
    "method",
    "capacity",
    "group_type",
    "group",
    "applications",
    "events",
    "reviewed",
    "captured_events",
    "false_positives",
    "missed_events",
    "precision",
    "capture",
    "review_cost_usd",
    "captured_event_exposure_proxy_usd",
)
ARTIFACTS = (
    "summary.json",
    "predictions.csv",
    "policy_metrics.csv",
    "explanations.csv",
    "index.html",
)
POINTS_FACTOR = 20 / math.log(2)
POINTS_OFFSET = 600 - POINTS_FACTOR * math.log(50)


@dataclass(frozen=True)
class BenchmarkResult:
    summary: dict[str, Any]
    predictions: list[dict[str, Any]]
    policies: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    input_sha256: str


def _fit_woe(matrix: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    """Training quantile cuts with 0.5 event/non-event smoothing in every bin."""
    columns = []
    for j, name in enumerate(MODEL_FEATURES):
        values = matrix[:, j]
        unique = np.unique(values)
        cuts = (
            ((unique[:-1] + unique[1:]) / 2)
            if len(unique) <= 5
            else np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
        )
        bins = np.searchsorted(cuts, values, side="right")
        count = len(cuts) + 1
        events = np.bincount(bins, weights=labels, minlength=count)
        non_events = np.bincount(bins, weights=1 - labels, minlength=count)
        woe = np.log(
            ((events + 0.5) / (labels.sum() + 0.5 * count))
            / ((non_events + 0.5) / ((1 - labels).sum() + 0.5 * count))
        )
        columns.append(
            {
                "feature": name,
                "cuts": cuts.tolist(),
                "woe": woe.tolist(),
                "events": events.astype(int).tolist(),
                "non_events": non_events.astype(int).tolist(),
            }
        )
    return columns


def _woe_matrix(matrix: np.ndarray, columns: list[dict[str, Any]]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(column["woe"])[np.searchsorted(column["cuts"], matrix[:, j], side="right")]
            for j, column in enumerate(columns)
        ]
    )


def _rank(scores: np.ndarray, ids: list[str]) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))


def _policy(
    method: str,
    capacity: float,
    rows: list[CohortRow],
    scores: np.ndarray,
    review_cost: float,
    lgd: float,
) -> list[dict[str, Any]]:
    labels = _labels(rows)
    count = math.ceil(len(rows) * capacity)
    selected = set(_rank(scores, [row.application.application_id for row in rows])[:count])
    groups = [("portfolio", "all", list(range(len(rows))))]
    for kind in ("vintage", "segment"):
        values = [
            row.application.decision_at.strftime("%Y-%m")
            if kind == "vintage"
            else row.application.segment
            for row in rows
        ]
        groups.extend(
            (kind, group, [i for i, value in enumerate(values) if value == group])
            for group in sorted(set(values))
        )
    result = []
    for kind, group, indices in groups:
        reviewed = [i for i in indices if i in selected]
        caught = int(sum(labels[i] for i in reviewed))
        events = int(sum(labels[i] for i in indices))
        result.append(
            {
                "method": method,
                "capacity": capacity,
                "group_type": kind,
                "group": group,
                "applications": len(indices),
                "events": events,
                "reviewed": len(reviewed),
                "captured_events": caught,
                "false_positives": len(reviewed) - caught,
                "missed_events": events - caught,
                "precision": round(caught / len(reviewed), 8) if reviewed else None,
                "capture": round(caught / events, 8) if events else None,
                "review_cost_usd": round(len(reviewed) * review_cost, 2),
                "captured_event_exposure_proxy_usd": round(
                    sum(
                        rows[i].application.principal_cents / 100 * labels[i] * lgd
                        for i in reviewed
                    ),
                    2,
                ),
            }
        )
    return result


def _paired_intervals(
    rows: list[CohortRow],
    scores: dict[str, np.ndarray],
    capacities: tuple[float, ...],
    seed: int,
    replicates: int,
) -> list[dict[str, Any]]:
    labels = _labels(rows)
    ids = [row.application.application_id for row in rows]
    months = [row.application.decision_at.strftime("%Y-%m") for row in rows]
    groups = [np.flatnonzero(np.asarray(months) == month) for month in sorted(set(months))]
    rng = np.random.default_rng(seed)
    differences: dict[tuple[str, float], list[float]] = {
        (method, capacity): [] for method in METHODS[1:] for capacity in capacities
    }
    for _ in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.choice(len(groups), len(groups))])
        y = labels[indices]
        if not y.sum():
            continue
        sampled_ids = [ids[i] for i in indices]
        orders = {method: _rank(values[indices], sampled_ids) for method, values in scores.items()}
        for capacity in capacities:
            count = math.ceil(len(indices) * capacity)
            baseline = int(y[orders["business_rule"][:count]].sum())
            for method in METHODS[1:]:
                caught = int(y[orders[method][:count]].sum())
                differences[method, capacity].append((caught - baseline) / int(y.sum()))
    return [
        {
            "method": method,
            "capacity": capacity,
            "reference": "business_rule",
            "capture_difference_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "valid_replicates": len(values),
        }
        for (method, capacity), values in differences.items()
        if values
    ]


def benchmark(
    applications: list[Application],
    features: list[FeatureEvent],
    performance: list[PerformanceEvent],
    config: Config | None = None,
    *,
    data_kind: str = "user_supplied",
    capacities: tuple[float, ...] = (0.05, 0.10, 0.20),
    review_cost_usd: float = 5.0,
    bootstrap_replicates: int = 300,
) -> BenchmarkResult:
    config = config or Config()
    if data_kind not in {"synthetic", "user_supplied"}:
        raise DataContractError("Unknown data kind")
    if (
        not capacities
        or len(set(capacities)) != len(capacities)
        or any(not 0 < finite_number(c, "capacity") <= 1 for c in capacities)
    ):
        raise DataContractError("Capacities must be distinct fractions within (0, 1]")
    review_cost_usd = finite_number(review_cost_usd, "review cost")
    if review_cost_usd < 0 or type(bootstrap_replicates) is not int or bootstrap_replicates < 2:
        raise DataContractError(
            "Review cost must be nonnegative and bootstrap needs two replicates"
        )
    model, calibrator, train, calibration, _ = _fit_stages(
        applications, features, performance, config
    )
    test = sorted(
        _rows_between(
            snapshot(applications, features, performance, config.evaluation_as_of),
            config.test_start,
            config.test_end,
            "test",
        ),
        key=lambda row: row.application.application_id,
    )
    train_x, train_y = _matrix(train), _labels(train)
    test_x, test_y = _matrix(test), _labels(test)
    woe_columns = _fit_woe(train_x, train_y)
    woe_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=config.seed)
    )
    woe_model.fit(_woe_matrix(train_x, woe_columns), train_y)
    tree_parameters = {
        "max_iter": 100,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 30,
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": config.seed,
    }
    tree = HistGradientBoostingClassifier(**tree_parameters)
    tree.fit(train_x, train_y)
    raw = model.predict_proba(test_x)[:, 1]
    scores = {
        "business_rule": 0.5 * test_x[:, 0]
        + 0.3 * test_x[:, 1]
        + 0.2 * np.minimum(test_x[:, 2], 5) / 5,
        "raw_logistic": raw,
        "calibrated_logistic": calibrator.predict(raw),
        "woe_logistic": woe_model.predict_proba(_woe_matrix(test_x, woe_columns))[:, 1],
        "hist_gradient_boosting": tree.predict_proba(test_x)[:, 1],
    }
    # All rankings and reported metrics use the published precision, including ties.
    scores = {method: np.round(values, 10) for method, values in scores.items()}
    ids = [row.application.application_id for row in test]
    predictions, policies, metrics = [], [], {}
    for method, values in scores.items():
        if not np.all(np.isfinite(values)):
            raise DataContractError("Nonfinite benchmark score")
        ranks = {i: rank for rank, i in enumerate(_rank(values, ids), 1)}
        metrics[method] = {
            "roc_auc": float(roc_auc_score(test_y, values)),
            "average_precision": float(average_precision_score(test_y, values)),
            "brier": float(brier_score_loss(test_y, values)) if method != "business_rule" else None,
        }
        for i, row in enumerate(test):
            points = (
                (
                    POINTS_OFFSET
                    - POINTS_FACTOR
                    * float(
                        woe_model.decision_function(_woe_matrix(test_x[i : i + 1], woe_columns))[0]
                    )
                )
                if method == "woe_logistic"
                else None
            )
            predictions.append(
                {
                    "method": method,
                    "application_id": ids[i],
                    "decision_at": row.application.decision_at.isoformat(),
                    "vintage": row.application.decision_at.strftime("%Y-%m"),
                    "segment": row.application.segment,
                    "principal_cents": row.application.principal_cents,
                    "label": int(test_y[i]),
                    "score": float(values[i]),
                    "review_rank": ranks[i],
                    "scorecard_points": points,
                }
            )
        for capacity in capacities:
            policies.extend(
                _policy(method, capacity, test, values, review_cost_usd, config.scenario_lgd)
            )
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    coefficients = logistic.coef_[0] / scaler.scale_
    contributions = (test_x - scaler.mean_) * coefficients
    explanations = []
    for i, row in enumerate(test):
        ranks = {
            j: rank
            for rank, j in enumerate(
                sorted(range(len(MODEL_FEATURES)), key=lambda j: (-contributions[i, j], j)), 1
            )
        }
        for j, name in enumerate(MODEL_FEATURES):
            explanations.append(
                {
                    "application_id": row.application.application_id,
                    "feature": name,
                    "value": float(test_x[i, j]),
                    "training_mean": float(scaler.mean_[j]),
                    "coefficient_per_unit": float(coefficients[j]),
                    "log_odds_contribution": float(contributions[i, j]),
                    "baseline_log_odds": float(logistic.intercept_[0]),
                    "reason_rank": ranks[j],
                }
            )
    woe_scaler, woe_logistic = woe_model.steps[0][1], woe_model.steps[1][1]
    for j, column in enumerate(woe_columns):
        coefficient = float(woe_logistic.coef_[0, j] / woe_scaler.scale_[j])
        column["coefficient_per_woe"] = coefficient
        column["training_mean_woe"] = float(woe_scaler.mean_[j])
        column["points_contributions"] = [
            -POINTS_FACTOR * coefficient * (value - float(woe_scaler.mean_[j]))
            for value in column["woe"]
        ]
    summary = {
        "project": "CreditVintage capacity benchmark",
        "data_kind": data_kind,
        "config": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in asdict(config).items()
        },
        "cohorts": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "train_event_rate": float(train_y.mean()),
        "test_events": int(test_y.sum()),
        "capacities": list(capacities),
        "review_cost_usd": review_cost_usd,
        "test_metrics": metrics,
        "protocol": {
            "selection": "global top ceil(N * capacity); score descending, ID ascending",
            "model_selection": "none; fixed parameters; test never selects a winner",
            "business_rule": "0.5 utilization + 0.3 DTI + 0.2 min(prior, 5) / 5; not PD",
            "woe": "training-only quantile bins; log(event share / non-event share); smoothing 0.5",
            "tree_parameters": tree_parameters,
            "explanations": "raw-logistic log odds relative to training feature means; not causal",
            "loss_proxy": "captured event principal * assumed LGD; not prevented loss",
        },
        "woe_scorecard": {
            "bins": woe_columns,
            "baseline_points": POINTS_OFFSET - POINTS_FACTOR * float(woe_logistic.intercept_[0]),
            "reference_points": 600,
            "reference_good_bad_odds": 50,
            "points_to_double_odds": 20,
        },
        "paired_bootstrap": {
            "unit": "origination_month",
            "replicates": bootstrap_replicates,
            "scope": "fixed predictions; descriptive sampling uncertainty, not retraining",
            "intervals": _paired_intervals(
                test, scores, capacities, config.seed, bootstrap_replicates
            ),
        },
    }
    return BenchmarkResult(
        summary, predictions, policies, explanations, _digest(applications, features, performance)
    )


def render_benchmark(result: BenchmarkResult) -> str:
    source = (
        "SYNTHETIC RESEARCH" if result.summary["data_kind"] == "synthetic" else "USER-SUPPLIED DATA"
    )
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[key]))}</td>"
            for key in ("method", "capacity", "reviewed", "captured_events", "false_positives")
        )
        + f"<td>{100 * row['capture']:.1f}%</td></tr>"
        for row in result.policies
        if row["group_type"] == "portfolio"
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CreditVintage · Same budget, different policies</title>
<style>:root{{color-scheme:dark;font-family:system-ui;background:#111319;color:#f1f0e9}}
body{{margin:0;line-height:1.6}}main{{max-width:1050px;margin:auto;padding:3rem 1.5rem}}
h1{{font-size:clamp(2.4rem,6vw,4rem);line-height:1.1}}a{{color:#d7fa73}}p{{max-width:850px}}
.tag{{color:#d7fa73;font-family:monospace}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:.7rem;text-align:left;border-bottom:1px solid #353c45;white-space:nowrap}}
aside{{padding:1rem;border-left:3px solid #ffb994;background:#1b1e25}}code{{color:#d7fa73}}</style>
<main><a href="../">← CreditVintage</a><p class="tag">{source}</p>
<h1>Same review budget.<br>Different risk policies.</h1>
<p>Five frozen methods score the same {result.summary["cohorts"]["test"]} mature applications.
Training, calibration and test periods are separated by outcome-maturation gaps.
Each policy receives exactly the same global review capacity; ties use application ID.</p>
<aside>These are retrospective comparisons. Captured events are not prevented defaults.
Review cost and LGD are assumptions. A synthetic result demonstrates the experiment,
not performance on bank customers. No model is promoted based on this test set.</aside>
<h2>Capacity comparison</h2><div class="scroll"><table><thead><tr>
<th>Method</th><th>Budget fraction</th><th>Reviewed</th><th>Events captured</th>
<th>False positives</th><th>Event capture</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>Follow the evidence</h2><p><a href="predictions.csv">Scores and ranks</a> ·
<a href="policy_metrics.csv">Portfolio, month and segment metrics</a> ·
<a href="explanations.csv">Label-free raw-logistic explanations</a> ·
<a href="summary.json">Protocol, WOE scorecard and paired intervals</a> ·
<a href="manifest.json">Artifact hashes</a></p>
<p>The fixed business rule is a ranking score, not a default probability.
WOE boundaries and event rates are learned only from training applications.
Raw-logistic contributions add to its log odds relative to training means; they do not
explain calibrated probabilities or establish causality. Higher scorecard points imply
lower fitted risk; a 20-point increase doubles fitted good/bad odds.</p>
<p>Paired intervals resample complete origination months using the same draw for all methods.
Only six held-out months are available; inspect the intervals and group results before
interpreting a small difference. Run <code>creditvintage verify-benchmark DIRECTORY</code>
to independently recompute capacities, group outcomes and log-odds explanations.</p>
</main></html>"""


def write_benchmark(result: BenchmarkResult, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    _json(destination / "summary.json", result.summary)
    _csv(destination / "predictions.csv", result.predictions, PREDICTION_COLUMNS)
    _csv(destination / "policy_metrics.csv", result.policies, POLICY_COLUMNS)
    _csv(destination / "explanations.csv", result.explanations, EXPLANATION_COLUMNS)
    (destination / "index.html").write_text(render_benchmark(result), encoding="utf-8", newline="")
    manifest = {
        "input_sha256": result.input_sha256,
        "files_sha256": {name: _hash(destination / name) for name in ARTIFACTS},
    }
    _json(destination / "manifest.json", manifest)
    return manifest


def verify_benchmark(destination: Path) -> dict[str, Any]:
    """Recompute outcomes from portable rows without using fitted estimators."""
    try:
        manifest = read_json_object(destination / "manifest.json")
        if set(manifest["files_sha256"]) != set(ARTIFACTS):
            raise DataContractError("Unexpected benchmark artifact inventory")
        for name, digest in manifest["files_sha256"].items():
            if _hash(destination / name) != digest:
                raise DataContractError(f"Benchmark digest mismatch: {name}")
        summary = read_json_object(destination / "summary.json")
        predictions = read_csv_rows(destination / "predictions.csv", PREDICTION_COLUMNS)
        policies = read_csv_rows(destination / "policy_metrics.csv", POLICY_COLUMNS)
        explanations = read_csv_rows(destination / "explanations.csv", EXPLANATION_COLUMNS)
        by_method = {
            method: [row for row in predictions if row["method"] == method] for method in METHODS
        }
        if len(predictions) != len(METHODS) * summary["cohorts"]["test"]:
            raise DataContractError("Benchmark prediction coverage mismatch")
        reference = by_method["raw_logistic"]
        identities = {row["application_id"]: row for row in reference}
        if (
            len(identities) != len(reference)
            or len(reference) != summary["cohorts"]["test"]
            or {row["method"] for row in predictions} != set(METHODS)
        ):
            raise DataContractError("Duplicate or missing benchmark applications")
        if summary["data_kind"] not in {"synthetic", "user_supplied"}:
            raise DataContractError("Unknown benchmark data kind")
        config = summary["config"]
        config_arguments: dict[str, Any] = {
            key: parse_calendar_date(value, f"Config {key}")
            if key.endswith(("start", "end", "as_of"))
            else value
            for key, value in config.items()
        }
        Config(**config_arguments)
        capacities = summary["capacities"]
        if (
            not capacities
            or len(set(capacities)) != len(capacities)
            or any(not 0 < finite_number(value, "capacity") <= 1 for value in capacities)
        ):
            raise DataContractError("Invalid benchmark capacities")
        if finite_number(summary["review_cost_usd"], "review cost") < 0:
            raise DataContractError("Invalid review cost")
        expected_policy_keys = set()
        for method, rows in by_method.items():
            if {row["application_id"] for row in rows} != set(identities) or len(rows) != len(
                reference
            ):
                raise DataContractError("Methods do not score the same applications")
            for row in rows:
                origin = identities[row["application_id"]]
                if any(
                    row[key] != origin[key]
                    for key in ("decision_at", "vintage", "segment", "principal_cents", "label")
                ):
                    raise DataContractError("Benchmark outcomes differ across methods")
                decision = parse_calendar_date(row["decision_at"], "Benchmark decision_at")
                if (
                    not config["test_start"] <= decision.isoformat() <= config["test_end"]
                    or row["vintage"] != decision.strftime("%Y-%m")
                    or row["segment"] not in {"retail", "small_business"}
                    or int(row["principal_cents"]) <= 0
                    or row["label"] not in {"0", "1"}
                ):
                    raise DataContractError("Invalid benchmark application")
                score = finite_number(row["score"], "benchmark score")
                if method != "business_rule" and not 0 <= score <= 1:
                    raise DataContractError("Invalid benchmark probability")
            order = sorted(rows, key=lambda row: (-float(row["score"]), row["application_id"]))
            if any(int(row["review_rank"]) != rank for rank, row in enumerate(order, 1)):
                raise DataContractError("Benchmark ranks disagree with scores")
            labels = np.asarray([int(row["label"]) for row in rows])
            scores = np.asarray([float(row["score"]) for row in rows])
            actual_metrics = {
                "roc_auc": roc_auc_score(labels, scores),
                "average_precision": average_precision_score(labels, scores),
                "brier": brier_score_loss(labels, scores) if method != "business_rule" else None,
            }
            for key, actual in actual_metrics.items():
                stated = summary["test_metrics"][method][key]
                if (actual is None and stated is not None) or (
                    actual is not None and abs(actual - finite_number(stated, key)) > 1e-8
                ):
                    raise DataContractError("Benchmark metrics disagree with scores")
            for capacity in capacities:
                selected = {
                    row["application_id"] for row in order[: math.ceil(len(rows) * capacity)]
                }
                groups = [("portfolio", "all", rows)]
                for kind in ("vintage", "segment"):
                    groups.extend(
                        (kind, group, [row for row in rows if row[kind] == group])
                        for group in sorted({row[kind] for row in rows})
                    )
                for kind, group, members in groups:
                    policy_key = (method, float(capacity), kind, group)
                    expected_policy_keys.add(policy_key)
                    matches = [
                        row
                        for row in policies
                        if (row["method"], float(row["capacity"]), row["group_type"], row["group"])
                        == policy_key
                    ]
                    if len(matches) != 1:
                        raise DataContractError("Benchmark policy coverage mismatch")
                    stated = matches[0]
                    reviewed = [row for row in members if row["application_id"] in selected]
                    caught = sum(int(row["label"]) for row in reviewed)
                    events = sum(int(row["label"]) for row in members)
                    actual = {
                        "applications": len(members),
                        "events": events,
                        "reviewed": len(reviewed),
                        "captured_events": caught,
                        "false_positives": len(reviewed) - caught,
                        "missed_events": events - caught,
                        "precision": caught / len(reviewed) if reviewed else None,
                        "capture": caught / events if events else None,
                        "review_cost_usd": round(len(reviewed) * summary["review_cost_usd"], 2),
                        "captured_event_exposure_proxy_usd": round(
                            sum(
                                int(row["principal_cents"])
                                / 100
                                * int(row["label"])
                                * config["scenario_lgd"]
                                for row in reviewed
                            ),
                            2,
                        ),
                    }
                    for field, value in actual.items():
                        if (value is None and stated[field] != "") or (
                            value is not None
                            and abs(value - finite_number(stated[field], field)) > 1e-7
                        ):
                            raise DataContractError(
                                f"Benchmark policy {field} disagrees with predictions"
                            )
        if len(policies) != len(expected_policy_keys):
            raise DataContractError("Unexpected benchmark policies")
        if len(explanations) != len(reference) * len(MODEL_FEATURES):
            raise DataContractError("Benchmark explanation coverage mismatch")
        feature_values: dict[str, dict[str, float]] = {}
        model_parameters: dict[str, tuple[float, float, float]] = {}
        for app_id, prediction in identities.items():
            reasons = [row for row in explanations if row["application_id"] == app_id]
            if len(reasons) != len(MODEL_FEATURES) or {row["feature"] for row in reasons} != set(
                MODEL_FEATURES
            ):
                raise DataContractError("Benchmark explanation features mismatch")
            baseline = finite_number(reasons[0]["baseline_log_odds"], "baseline log odds")
            total = baseline
            feature_values[app_id] = {}
            for reason in reasons:
                contribution = finite_number(reason["log_odds_contribution"], "contribution")
                expected = (
                    finite_number(reason["value"], "value")
                    - finite_number(reason["training_mean"], "training mean")
                ) * finite_number(reason["coefficient_per_unit"], "coefficient")
                if (
                    abs(contribution - expected) > 1e-9
                    or float(reason["baseline_log_odds"]) != baseline
                ):
                    raise DataContractError("Benchmark explanation algebra mismatch")
                parameters = (
                    float(reason["training_mean"]),
                    float(reason["coefficient_per_unit"]),
                    baseline,
                )
                feature_name = reason["feature"]
                if (
                    feature_name in model_parameters
                    and model_parameters[feature_name] != parameters
                ):
                    raise DataContractError("Raw model parameters differ across explanations")
                model_parameters[feature_name] = parameters
                feature_values[app_id][feature_name] = finite_number(
                    reason["value"], "feature value"
                )
                total += contribution
            if abs(1 / (1 + math.exp(-total)) - float(prediction["score"])) > 1e-8:
                raise DataContractError(
                    "Benchmark explanation does not reconstruct raw probability"
                )
            ordered = sorted(
                reasons,
                key=lambda row: (
                    -float(row["log_odds_contribution"]),
                    MODEL_FEATURES.index(row["feature"]),
                ),
            )
            if any(int(row["reason_rank"]) != rank for rank, row in enumerate(ordered, 1)):
                raise DataContractError("Benchmark reason ranks disagree with contributions")
        scorecard = summary["woe_scorecard"]
        if [column["feature"] for column in scorecard["bins"]] != list(MODEL_FEATURES):
            raise DataContractError("Scorecard feature coverage mismatch")
        for column in scorecard["bins"]:
            cuts = [finite_number(value, "WOE cut") for value in column["cuts"]]
            if cuts != sorted(set(cuts)) or len(column["woe"]) != len(cuts) + 1:
                raise DataContractError("Invalid WOE bin boundaries")
            event_counts, non_event_counts = column["events"], column["non_events"]
            if (
                len(event_counts) != len(cuts) + 1
                or len(non_event_counts) != len(cuts) + 1
                or any(type(n) is not int or n < 0 for n in event_counts + non_event_counts)
                or sum(event_counts + non_event_counts) != summary["cohorts"]["train"]
            ):
                raise DataContractError("WOE training bin count mismatch")
            count = len(cuts) + 1
            for index, value in enumerate(column["woe"]):
                expected_woe = math.log(
                    ((event_counts[index] + 0.5) / (sum(event_counts) + 0.5 * count))
                    / ((non_event_counts[index] + 0.5) / (sum(non_event_counts) + 0.5 * count))
                )
                expected_points = (
                    -POINTS_FACTOR
                    * finite_number(column["coefficient_per_woe"], "WOE coefficient")
                    * (
                        finite_number(value, "WOE value")
                        - finite_number(column["training_mean_woe"], "WOE mean")
                    )
                )
                if (
                    abs(float(value) - expected_woe) > 1e-9
                    or abs(
                        finite_number(column["points_contributions"][index], "bin points")
                        - expected_points
                    )
                    > 1e-9
                ):
                    raise DataContractError("WOE bin values or points mismatch")
        for row in by_method["woe_logistic"]:
            values = feature_values[row["application_id"]]
            points = finite_number(scorecard["baseline_points"], "baseline points")
            for column in scorecard["bins"]:
                index = int(
                    np.searchsorted(column["cuts"], values[column["feature"]], side="right")
                )
                points += float(column["points_contributions"][index])
            if (
                abs(points - finite_number(row["scorecard_points"], "scorecard points")) > 1e-8
                or abs(
                    1 / (1 + math.exp(-(POINTS_OFFSET - points) / POINTS_FACTOR))
                    - float(row["score"])
                )
                > 1e-8
            ):
                raise DataContractError("Scorecard points do not reconstruct probability")
        for row in by_method["business_rule"]:
            values = feature_values[row["application_id"]]
            expected = (
                0.5 * values["utilization"]
                + 0.3 * values["debt_to_income"]
                + 0.2 * min(values["prior_delinquencies"], 5) / 5
            )
            if abs(expected - float(row["score"])) > 1e-8:
                raise DataContractError("Business rule score mismatch")
        cohorts = [
            CohortRow(
                Application(
                    row["application_id"],
                    parse_calendar_date(row["decision_at"], "Benchmark decision_at"),
                    int(row["principal_cents"]),
                    cast(Literal["retail", "small_business"], row["segment"]),
                ),
                (
                    feature_values[row["application_id"]]["utilization"],
                    feature_values[row["application_id"]]["debt_to_income"],
                    feature_values[row["application_id"]]["prior_delinquencies"],
                ),
                int(row["label"]),
                "mature",
            )
            for row in reference
        ]
        if sum(int(row["label"]) for row in reference) != summary["test_events"]:
            raise DataContractError("Benchmark event count mismatch")
        published_scores = {
            method: np.asarray([float(row["score"]) for row in rows])
            for method, rows in by_method.items()
        }
        # Every method was exported in the same application order.
        if any(
            [row["application_id"] for row in rows] != [row["application_id"] for row in reference]
            for rows in by_method.values()
        ):
            raise DataContractError("Benchmark method ordering mismatch")
        boot = summary["paired_bootstrap"]
        if (
            boot["unit"] != "origination_month"
            or type(boot["replicates"]) is not int
            or boot["replicates"] < 2
        ):
            raise DataContractError("Invalid paired bootstrap protocol")
        actual_intervals = _paired_intervals(
            cohorts, published_scores, tuple(capacities), config["seed"], boot["replicates"]
        )
        if len(actual_intervals) != len(boot["intervals"]):
            raise DataContractError("Paired interval coverage mismatch")
        for actual_interval, stated_interval in zip(
            actual_intervals, boot["intervals"], strict=True
        ):
            if any(
                actual_interval[key] != stated_interval[key]
                for key in ("method", "capacity", "reference", "valid_replicates")
            ) or not np.allclose(
                actual_interval["capture_difference_95_interval"],
                stated_interval["capture_difference_95_interval"],
                rtol=0,
                atol=1e-10,
            ):
                raise DataContractError("Paired intervals disagree with predictions")
        # CSV represents nullable numeric values as empty cells.
        rendered_policies: list[dict[str, Any]] = [
            {
                **row,
                "capacity": float(row["capacity"]),
                "capture": float(row["capture"]) if row["capture"] else None,
            }
            for row in policies
        ]
        result = BenchmarkResult(
            summary, predictions, rendered_policies, explanations, manifest["input_sha256"]
        )
        if (destination / "index.html").read_text(encoding="utf-8") != render_benchmark(result):
            raise DataContractError("Benchmark report disagrees with verified data")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"Cannot verify benchmark: {exc}") from exc
    return {
        "status": "verified",
        "applications": len(reference),
        "methods": len(METHODS),
        "input_sha256": manifest["input_sha256"],
    }
