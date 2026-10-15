"""A separate, offline static-data benchmark; it does not invent event dates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/german-credit/german.data"
DATA_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"
SEEDS = (7, 23, 41, 59, 83)
MODELS = ("constant", "logistic", "hist_gradient_boosting")
NUMERIC = (1, 4, 7, 10, 12, 15, 17)  # Zero-based columns in the UCI codebook.
CATEGORICAL = tuple(i for i in range(20) if i not in NUMERIC)
METRICS = ("roc_auc", "average_precision", "brier", "precision_at_10pct", "capture_at_10pct")
FILES = ("predictions.csv", "splits.csv", "metrics.csv", "summary.json", "index.html")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path=DATA):
    if digest(path) != DATA_SHA256:
        raise ValueError("UCI source bytes differ from the attributed frozen dataset")
    rows = [line.split() for line in path.read_text(encoding="ascii").splitlines()]
    if len(rows) != 1000 or any(len(row) != 21 for row in rows):
        raise ValueError("Expected the original 1,000-row, 20-feature UCI dataset")
    features = np.asarray([row[:20] for row in rows], dtype=object)
    labels = np.asarray([int(row[20]) - 1 for row in rows], dtype=int)
    if np.bincount(labels).tolist() != [700, 300]:
        raise ValueError("Expected 1=good (700) and 2=bad (300), mapped to 0 and 1")
    return features, labels


def make_model(name, seed):
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(NUMERIC)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL),
            ),
        ],
        sparse_threshold=0,
    )
    if name == "constant":
        estimator = DummyClassifier(strategy="prior")
    elif name == "logistic":
        estimator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def fit_scores(name, seed, train_x, train_y, test_x):
    # The scoring function has no test-label argument. Both preprocessing steps
    # are fitted by the pipeline on train_x only; the target is never encoded.
    with threadpool_limits(limits=1):
        fitted = make_model(name, seed).fit(train_x, train_y)
        return fitted.predict_proba(test_x)[:, 1]


def split_indices(labels, seed):
    return train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )


def measure(labels, scores, row_ids):
    labels, scores, row_ids = np.asarray(labels), np.asarray(scores), np.asarray(row_ids)
    if not (labels.shape == scores.shape == row_ids.shape) or labels.ndim != 1:
        raise ValueError(
            "Labels, scores and source row IDs must have the same one-dimensional shape"
        )
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Metrics require both binary labels and finite scores")
    if ((scores < 0) | (scores > 1)).any() or len(set(row_ids)) != len(row_ids):
        raise ValueError("Scores must be in [0,1] and source row IDs unique")
    selected = np.zeros(len(labels), dtype=bool)
    selected[np.lexsort((row_ids, -scores))[: math.ceil(len(labels) * 0.1)]] = True
    captured = int(labels[selected].sum())
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "precision_at_10pct": captured / int(selected.sum()),
        "capture_at_10pct": captured / int(labels.sum()),
    }, selected


def aggregates(rows):
    return {
        model: {
            metric: {
                "mean": float(np.mean(values)),
                "min": min(values),
                "max": max(values),
            }
            for metric in METRICS
            for values in [[float(row[metric]) for row in rows if row["model"] == model]]
        }
        for model in MODELS
    }


def render(summary):
    body = ""
    for model in MODELS:
        cells = "".join(
            f"<td>{summary['results'][model][metric]['mean']:.3f} "
            f"<small>({summary['results'][model][metric]['min']:.3f}–"
            f"{summary['results'][model][metric]['max']:.3f})</small></td>"
            for metric in METRICS
        )
        body += f"<tr><th scope='row'>{model.replace('_', ' ')}</th>{cells}</tr>"
    return Path(__file__).with_suffix(".html").read_text(encoding="utf-8").replace("{{ROWS}}", body)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(out):
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    features, labels = load_data()
    predictions, assignments, results = [], [], []
    for seed in SEEDS:
        train, test = split_indices(labels, seed)
        for part, indices in [("train", train), ("test", test)]:
            assignments.extend(
                {"seed": seed, "row_id": int(i) + 1, "partition": part} for i in sorted(indices)
            )
        for model in MODELS:
            scores = fit_scores(model, seed, features[train], labels[train], features[test])
            metrics, selected = measure(labels[test], scores, test + 1)
            results.append({"seed": seed, "model": model, **metrics})
            predictions.extend(
                {
                    "seed": seed,
                    "model": model,
                    "row_id": int(i) + 1,
                    "label_bad_risk": int(label),
                    "score": float(score),
                    "review_top_10pct": int(chosen),
                }
                for i, label, score, chosen in zip(
                    test, labels[test], scores, selected, strict=True
                )
            )
    summary = {
        "dataset": "UCI Statlog German Credit (144)",
        "doi": "https://doi.org/10.24432/C5NC77",
        "data_sha256": DATA_SHA256,
        "data_kind": "public_historical_static",
        "positive_label": "2=bad credit risk, mapped to 1",
        "rows": 1000,
        "features": 20,
        "good_labels": 700,
        "bad_labels": 300,
        "split": "five repeated stratified 80/20 random splits; not temporal",
        "seeds": list(SEEDS),
        "train_rows_per_split": 800,
        "test_rows_per_split": 200,
        "bad_test_labels_per_split": 60,
        "inspection_rows_per_model_split": 20,
        "preprocessing": (
            "train-only StandardScaler on 7 numeric and one-hot on 13 categorical columns"
        ),
        "selection": "ceil(0.10 * n_test), descending score then ascending source row number",
        "model_parameters": {
            str(seed): {
                name: make_model(name, seed).named_steps["model"].get_params() for name in MODELS
            }
            for seed in SEEDS
        },
        "tuning": "none; fixed model settings and split seeds; all results retained",
        "uncertainty": "mean and observed min/max only; overlapping splits are not independent",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": aggregates(results),
    }
    out.mkdir(parents=True)
    for name, rows in [
        ("predictions.csv", predictions),
        ("splits.csv", assignments),
        ("metrics.csv", results),
    ]:
        write_csv(out / name, rows)
    write_json(out / "summary.json", summary)
    (out / "index.html").write_text(render(summary), encoding="utf-8", newline="\n")
    write_json(out / "manifest.json", {name: digest(out / name) for name in FILES})
    return summary


def verify(out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest != {name: digest(out / name) for name in FILES}:
        raise ValueError("Report file hashes differ")
    _, labels = load_data()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    expected_protocol = {
        "data_sha256": DATA_SHA256,
        "seeds": list(SEEDS),
        "rows": 1000,
        "features": 20,
        "train_rows_per_split": 800,
        "test_rows_per_split": 200,
        "bad_test_labels_per_split": 60,
        "inspection_rows_per_model_split": 20,
        "data_kind": "public_historical_static",
        "good_labels": 700,
        "bad_labels": 300,
    }
    if any(summary.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("Summary differs from the frozen benchmark protocol")
    predictions, assignments, saved_metrics = (read_csv(out / name) for name in FILES[:3])
    expected_assignments = []
    expected_groups, recomputed = set(), []
    for seed in SEEDS:
        train, test = split_indices(labels, seed)
        for part, indices in [("train", train), ("test", test)]:
            expected_assignments.extend(
                {"seed": str(seed), "row_id": str(int(i) + 1), "partition": part}
                for i in sorted(indices)
            )
        for model in MODELS:
            expected_groups.add((str(seed), model))
            rows = [r for r in predictions if r["seed"] == str(seed) and r["model"] == model]
            ids = np.asarray([int(r["row_id"]) for r in rows])
            if sorted(ids.tolist()) != sorted((test + 1).tolist()):
                raise ValueError("Prediction IDs differ from the held-out split")
            y = np.asarray([int(r["label_bad_risk"]) for r in rows])
            if not np.array_equal(y, labels[ids - 1]):
                raise ValueError("Saved labels differ from source rows")
            values, selected = measure(y, [float(r["score"]) for r in rows], ids)
            if not np.array_equal(
                [r["review_top_10pct"] for r in rows], selected.astype(int).astype(str)
            ):
                raise ValueError("Inspection flags differ from the fixed top-20 selection")
            if model == "constant" and not np.allclose(
                [float(r["score"]) for r in rows], 0.3, rtol=0, atol=1e-15
            ):
                raise ValueError("Constant baseline differs from training class frequency")
            recomputed.append({"seed": seed, "model": model, **values})
    if assignments != expected_assignments or len(predictions) != 3000:
        raise ValueError("Assignments or prediction count differ from the complete protocol")
    if (
        len(saved_metrics) != 15
        or {(r["seed"], r["model"]) for r in saved_metrics} != expected_groups
    ):
        raise ValueError("Expected all 15 model/split results")
    for expected in recomputed:
        actual = next(
            r
            for r in saved_metrics
            if r["seed"] == str(expected["seed"]) and r["model"] == expected["model"]
        )
        for metric in METRICS:
            if not math.isclose(float(actual[metric]), expected[metric], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Saved metric differs from test scores: {metric}")
    if summary["results"] != aggregates(recomputed) or (out / "index.html").read_text(
        encoding="utf-8"
    ) != render(summary):
        raise ValueError("Summary or HTML differs from recomputed results")
    return {
        "verified_splits": 5,
        "model_split_results": 15,
        "test_scores": 3000,
        "data_kind": "public_historical_static",
        "retrained": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="Create a new report directory")
    action.add_argument("--verify", type=Path, help="Recompute metrics from an existing report")
    args = parser.parse_args()
    try:
        result = run(args.out) if args.out else verify(args.verify)
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(1, f"Public credit benchmark: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
