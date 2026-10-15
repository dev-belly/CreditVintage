"""A separate, dated-data-free benchmark on attributed UCI credit observations."""

from __future__ import annotations

import hashlib
import html
import math
import platform
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditvintage.artifacts import _csv, _hash, _json
from creditvintage.core import DataContractError
from creditvintage.model import _metrics
from creditvintage.validation import finite_number, read_csv_rows, read_json_object

DATA_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"
SEED = 20261015
METHODS = ("constant_train_rate", "logistic", "isotonic")
NUMERIC = (1, 4, 7, 10, 15, 17)
CATEGORICAL = (0, 2, 3, 5, 6, 9, 11, 13, 14, 16, 18)
SPLIT_COLUMNS = ("record_id", "split")
PREDICTION_COLUMNS = ("record_id", "native_bad", *METHODS)
POLICY_COLUMNS = (
    "method",
    "capacity",
    "reviewed",
    "captured_bad",
    "total_bad",
    "precision",
    "capture",
)
QUEUE_COLUMNS = ("record_id", "rank", "score")
PUBLIC_FILES = (
    "source.data",
    "split.csv",
    "predictions.csv",
    "policy_metrics.csv",
    "review_queue.csv",
    "summary.json",
    "index.html",
)


def load_public_data(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise DataContractError("Public data digest differs from the frozen UCI original")
    rows = [line.split() for line in raw.decode("ascii").splitlines()]
    if len(rows) != 1000 or any(len(row) != 21 or row[-1] not in {"1", "2"} for row in rows):
        raise DataContractError("Unexpected UCI credit table")
    x = np.asarray([row[:-1] for row in rows], dtype=object)
    y = np.asarray([int(row[-1]) - 1 for row in rows], dtype=int)
    return x, y


def public_splits(y: np.ndarray) -> dict[str, np.ndarray]:
    development, test = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=SEED
    )
    train, calibration = train_test_split(
        development, test_size=0.25, stratify=y[development], random_state=SEED + 1
    )
    return {"train": np.sort(train), "calibration": np.sort(calibration), "test": np.sort(test)}


def fit_public_scores(
    x: np.ndarray, y: np.ndarray, splits: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Freeze preprocessing on train; reserve calibration exclusively for isotonic fitting."""
    train, calibration = splits["train"], splits["calibration"]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(NUMERIC)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ],
        remainder="drop",
    )
    model = make_pipeline(
        transform, LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
    )
    model.fit(x[train], y[train])
    raw = model.predict_proba(x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw[calibration], y[calibration])
    return {
        "constant_train_rate": np.full(len(y), float(np.mean(y[train]))),
        "logistic": raw,
        "isotonic": calibrator.predict(raw),
    }


def experiment(raw: bytes) -> dict[str, Any]:
    x, y = load_public_data(raw)
    splits = public_splits(y)
    scores = fit_public_scores(x, y, splits)
    test = splits["test"]
    policies: list[dict[str, Any]] = []
    rankings = {
        method: sorted(test, key=lambda i: (-float(scores[method][i]), int(i)))
        for method in METHODS
    }
    for method in METHODS:
        for capacity in (0.05, 0.10, 0.20):
            reviewed = math.ceil(len(test) * capacity)
            captured = int(sum(y[i] for i in rankings[method][:reviewed]))
            policies.append(
                {
                    "method": method,
                    "capacity": capacity,
                    "reviewed": reviewed,
                    "captured_bad": captured,
                    "total_bad": int(sum(y[test])),
                    "precision": round(captured / reviewed, 6),
                    "capture": round(captured / int(sum(y[test])), 6),
                }
            )
    summary = {
        "schema": "creditvintage-public-credit-v1",
        "data_kind": "public_observed_static",
        "source": {
            "name": "Hofmann (1994), Statlog (German Credit Data), UCI",
            "doi": "https://doi.org/10.24432/C5NC77",
            "license": "CC BY 4.0",
            "raw_sha256": DATA_SHA256,
            "records": len(y),
            "native_bad": int(sum(y)),
        },
        "protocol": {
            "seed": SEED,
            "split": "stratified random 60/20/20; frozen once; no search",
            "label": "UCI native good=1 -> 0; bad=2 -> 1; no invented default horizon",
            "numeric_feature_indices_zero_based": list(NUMERIC),
            "categorical_feature_indices_zero_based": list(CATEGORICAL),
            "excluded_fields": ["personal_status_and_sex", "age", "foreign_worker"],
            "preprocessing": (
                "train-only scaling and one-hot categories; unseen categories ignored"
            ),
            "logistic": "C=1; lbfgs; max_iter=1000; no class weighting or tuning",
            "review": "rank by score, then original UCI row; descriptive capacity only",
            "limitations": [
                "No decision/publication/revision dates: not point-in-time or out-of-time evidence",
                "Static historical German sample: not evidence about current Chinese credit",
                "200 test records: noisy single-split estimates; no selection or causal claim",
                "Excluding three fields does not demonstrate fairness or remove their proxies",
                "Native good/bad labels are not the synthetic experiment's 180-day 90+ DPD target",
                "No observed recoveries, monetary savings, approval policy or deployed service",
            ],
        },
        "splits": {
            stage: {"records": len(indices), "native_bad": int(sum(y[indices]))}
            for stage, indices in splits.items()
        },
        "test_metrics": {method: _metrics(y[test], scores[method][test]) for method in METHODS},
        "policies": policies,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    assignment = {int(i): stage for stage, indices in splits.items() for i in indices}
    return {
        "summary": summary,
        "split": [{"record_id": f"UCI{i + 1:04}", "split": assignment[i]} for i in range(len(y))],
        "predictions": [
            {
                "record_id": f"UCI{i + 1:04}",
                "native_bad": int(y[i]),
                **{method: round(float(scores[method][i]), 12) for method in METHODS},
            }
            for i in test
        ],
        "policies": policies,
        "queue": [
            {
                "record_id": f"UCI{i + 1:04}",
                "rank": rank,
                "score": round(float(scores["logistic"][i]), 12),
            }
            for rank, i in enumerate(rankings["logistic"][: math.ceil(len(test) * 0.10)], 1)
        ],
    }


def render_public_report(summary: dict[str, Any]) -> str:
    metrics = "".join(
        f"<tr><th>{html.escape(method)}</th><td>{summary['test_metrics'][method]['roc_auc']:.3f}</td>"
        f"<td>{summary['test_metrics'][method]['average_precision']:.3f}</td>"
        f"<td>{summary['test_metrics'][method]['brier']:.3f}</td></tr>"
        for method in METHODS
    )
    policies = "".join(
        f"<tr><th>{html.escape(row['method'])}</th><td>{row['reviewed']}</td>"
        f"<td>{row['captured_bad']}/{row['total_bad']}</td>"
        f"<td>{row['precision']:.1%}</td><td>{row['capture']:.1%}</td></tr>"
        for row in summary["policies"]
        if row["capacity"] == 0.10
    )
    limits = "".join(f"<li>{html.escape(item)}</li>" for item in summary["protocol"]["limitations"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CreditVintage | Observed public credit benchmark</title>
<style>body{{margin:0;background:#111319;color:#eee;font:17px/1.6 system-ui,sans-serif}}
main{{max-width:960px;margin:auto;padding:48px 24px}}
h1{{font-size:clamp(2rem,6vw,3.5rem);line-height:1.1}}
a,.tag{{color:#d7fa73}}p,li{{color:#bac3cd}}
table{{width:100%;border-collapse:collapse;margin:24px 0}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid #38404b}}
.scroll{{overflow-x:auto}}.note{{border-left:3px solid #eeb888;padding-left:18px}}
code{{overflow-wrap:anywhere}}a:focus-visible{{outline:2px solid #d7fa73}}</style></head>
<body><main><a href="../">CreditVintage</a>
<p class="tag">OBSERVED PUBLIC DATA / STATIC BENCHMARK</p>
<h1>German credit.<br>A fixed comparison.</h1>
<p>1,000 historical German credit records. Frozen stratified split: 600 train, 200 calibration,
200 test. The native good/bad target has 300 bad records, including 60 in the test split.
No hyperparameter search. Scaling and categories are learned only from training records.</p>
<p class="note">This dataset has no decision or publication dates. It cannot validate the
point-in-time controls or the 180-day outcome used in the separate synthetic experiment.</p>
<h2>Untouched test split</h2><div class="scroll"><table><thead><tr><th>Method</th><th>ROC-AUC</th>
<th>Average precision</th><th>Brier ↓</th></tr></thead><tbody>{metrics}</tbody></table></div>
<h2>The same 20-record review capacity</h2><div class="scroll"><table><thead><tr><th>Method</th>
<th>Reviewed</th><th>Captured bad</th><th>Precision</th><th>Capture</th></tr></thead>
<tbody>{policies}</tbody></table></div>
<p>A constant score is ordered by original row number to resolve ties. Review is a historical
ranking comparison, with no observed intervention, financial benefit or approval outcome.</p>
<h2>Reproduce and inspect</h2>
<p><code>creditvintage public-benchmark --out outputs/public-credit</code><br>
<code>creditvintage verify-public outputs/public-credit</code></p>
<p>Verification refits the frozen models from the original bytes and recorded split, then
checks scores, metrics, queue and page. It needs no network. Numerical tolerance is 1e-8 for
probabilities and 1e-6 for summary metrics; runtime versions are recorded.</p>
<p><a href="source.data">Original data</a> · <a href="split.csv">Every split assignment</a> ·
<a href="predictions.csv">Test labels and scores</a> ·
<a href="policy_metrics.csv">All review budgets</a> ·
<a href="review_queue.csv">Label-free logistic queue</a> · <a href="summary.json">Summary</a> ·
<a href="manifest.json">Digests</a></p><h2>What remains unproven</h2><ul>{limits}</ul>
<p>Source: Hans Hofmann (1994),
<a href="https://doi.org/10.24432/C5NC77">Statlog (German Credit Data)</a>,
UCI Machine Learning Repository. CC BY 4.0. Original data bytes preserved; native label recoded
and three model fields excluded as documented in <a href="summary.json">the protocol</a>.</p>
<p><code>SHA-256: {DATA_SHA256}</code></p>
</main></body></html>\n"""


def write_public_benchmark(destination: Path) -> dict[str, Any]:
    raw = files("creditvintage").joinpath("data/german_credit/german.data").read_bytes()
    result = experiment(raw)
    if destination.exists():
        raise DataContractError("Choose a new public benchmark output directory")
    destination.mkdir(parents=True)
    (destination / "source.data").write_bytes(raw)
    for name, key, columns in (
        ("split.csv", "split", SPLIT_COLUMNS),
        ("predictions.csv", "predictions", PREDICTION_COLUMNS),
        ("policy_metrics.csv", "policies", POLICY_COLUMNS),
        ("review_queue.csv", "queue", QUEUE_COLUMNS),
    ):
        _csv(destination / name, result[key], columns)
    _json(destination / "summary.json", result["summary"])
    (destination / "index.html").write_text(
        render_public_report(result["summary"]), encoding="utf-8", newline=""
    )
    _json(
        destination / "manifest.json",
        {
            "schema": "creditvintage-public-credit-v1",
            "files_sha256": {name: _hash(destination / name) for name in PUBLIC_FILES},
        },
    )
    summary: dict[str, Any] = result["summary"]
    return summary


def _compare_summary(actual: Any, expected: Any, label: str) -> None:
    if type(expected) is float:
        if type(actual) not in (int, float) or not math.isclose(
            finite_number(actual, label), expected, rel_tol=0, abs_tol=1e-6
        ):
            raise DataContractError(f"Public benchmark summary disagrees: {label}")
    elif type(actual) is not type(expected):
        raise DataContractError(f"Public benchmark summary type disagrees: {label}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise DataContractError(f"Public benchmark summary keys disagree: {label}")
        for key, value in expected.items():
            _compare_summary(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise DataContractError(f"Public benchmark summary length disagrees: {label}")
        for i, value in enumerate(expected):
            _compare_summary(actual[i], value, f"{label}[{i}]")
    elif actual != expected:
        raise DataContractError(f"Public benchmark summary disagrees: {label}")


def verify_public_benchmark(directory: Path) -> dict[str, Any]:
    manifest = read_json_object(directory / "manifest.json")
    if manifest.get("schema") != "creditvintage-public-credit-v1" or manifest.get(
        "files_sha256"
    ) != {name: _hash(directory / name) for name in PUBLIC_FILES}:
        raise DataContractError("Public benchmark manifest/digest mismatch")
    result = experiment((directory / "source.data").read_bytes())
    summary = read_json_object(directory / "summary.json")
    runtime = summary.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"python", "numpy", "scikit_learn"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise DataContractError("Invalid recorded public benchmark runtime")
    result["summary"]["runtime"] = (
        runtime  # Compare numerical outputs across recorded environments.
    )
    _compare_summary(summary, result["summary"], "summary")
    for name, key, columns in (
        ("split.csv", "split", SPLIT_COLUMNS),
        ("predictions.csv", "predictions", PREDICTION_COLUMNS),
        ("policy_metrics.csv", "policies", POLICY_COLUMNS),
        ("review_queue.csv", "queue", QUEUE_COLUMNS),
    ):
        actual_rows = read_csv_rows(directory / name, columns)
        if len(actual_rows) != len(result[key]):
            raise DataContractError(f"Public benchmark row count disagrees: {name}")
        for actual, expected in zip(actual_rows, result[key], strict=True):
            for column, value in expected.items():
                if type(value) is float:
                    agrees = math.isclose(
                        finite_number(actual[column], column), value, rel_tol=0, abs_tol=1e-8
                    )
                else:
                    agrees = actual[column] == str(value)
                if not agrees:
                    raise DataContractError(f"Public benchmark replay disagrees: {name}:{column}")
    if (directory / "index.html").read_text(encoding="utf-8") != render_public_report(summary):
        raise DataContractError("Public benchmark page disagrees with verified evidence")
    return {
        "status": "verified",
        "records": 1000,
        "test_records": 200,
        "refitted": True,
        "data_kind": "public_observed_static",
    }
