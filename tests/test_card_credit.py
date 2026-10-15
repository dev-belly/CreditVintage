from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import zipfile

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from creditvintage.card_credit import (
    CATEGORICAL,
    FEATURES,
    METHODS,
    SEEDS,
    card_splits,
    fit_card_scores,
    load_card_data,
    source_bytes,
    verify_card_benchmark,
    write_card_benchmark,
)
from creditvintage.core import DataContractError


@pytest.fixture(scope="module")
def data():
    return load_card_data(source_bytes())


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    path = tmp_path_factory.mktemp("card-credit") / "report"
    write_card_benchmark(path)
    return path


def test_native_values_features_and_every_disjoint_partition(data):
    x, y = data
    assert x.shape == (30000, 19)
    assert np.bincount(y).tolist() == [23364, 6636]
    assert len(FEATURES) == 19
    assert not {"ID", "SEX", "AGE", "EDUCATION", "MARRIAGE"} & set(FEATURES)
    assert {-2, -1, 0, 1, 2}.issubset(set(x[:, CATEGORICAL[0]]))
    for seed in SEEDS:
        parts = card_splits(y, seed)
        assert [len(parts[name]) for name in ("train", "calibration", "test")] == [
            18000,
            6000,
            6000,
        ]
        assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(30000))
        assert int(sum(y[parts["test"]])) == 1327


def test_heldout_only_extremes_and_new_categories_cannot_change_other_test_scores(data):
    x, y = data
    train, calibration, test = x[:1200], x[1200:1600], x[1600:1630]
    base = fit_card_scores(train, y[:1200], calibration, y[1200:1600], test, 7)
    extreme = test[:1].copy()
    extreme[0, 0] = 1e12
    extreme[0, CATEGORICAL[0]] = 12345
    extended = fit_card_scores(
        train, y[:1200], calibration, y[1200:1600], np.concatenate([test, extreme]), 7
    )
    for method in METHODS:
        np.testing.assert_allclose(base[method], extended[method][:-1], rtol=0, atol=1e-12)
        assert np.isfinite(extended[method]).all()


def test_changed_source_is_rejected():
    raw = gzip.decompress(source_bytes()).replace(b"20000", b"20001", 1)
    with pytest.raises(DataContractError, match="source digest"):
        load_card_data(gzip.compress(raw, mtime=0))


def test_isotonic_predictions_resist_tiny_platform_score_drift(data, monkeypatch):
    x, y = data
    arguments = (x[:1200], y[:1200], x[1200:1600], y[1200:1600], x[1600:1630], 7)
    baseline = fit_card_scores(*arguments)
    original = LogisticRegression.predict_proba

    def drift(self, values):
        result = original(self, values)
        result[:, 1] += 1e-12
        result[:, 0] -= 1e-12
        return result

    monkeypatch.setattr(LogisticRegression, "predict_proba", drift)
    perturbed = fit_card_scores(*arguments)
    np.testing.assert_array_equal(baseline["isotonic"], perturbed["isotonic"])


def test_complete_output_refits_with_equal_budgets_and_label_free_queues(report):
    assert verify_card_benchmark(report) == {
        "status": "verified",
        "records": 30000,
        "splits": 5,
        "test_predictions": 30000,
        "model_scores": 120000,
        "refitted": True,
        "data_kind": "public_observed_static",
    }
    with (report / "policy_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    for capacity, count in (("0.05", 300), ("0.1", 600), ("0.2", 1200)):
        assert {int(row["reviewed"]) for row in rows if row["capacity"] == capacity} == {count}
    with (report / "review_queue.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "native_default" not in reader.fieldnames
        queue = list(reader)
    assert len(queue) == 12000
    _, y = load_card_data(source_bytes())
    for seed in SEEDS:
        wanted = (card_splits(y, seed)["test"][:600] + 1).tolist()
        actual = [
            int(row["record_id"])
            for row in queue
            if row["method"] == "constant" and int(row["seed"]) == seed
        ]
        assert actual == wanted
    assert (report / "evidence.zip").stat().st_size > 1000000
    with pytest.raises(DataContractError, match="new card benchmark"):
        write_card_benchmark(report)


def rehash(path, name):
    file = path / "manifest.json"
    manifest = json.loads(file.read_text(encoding="utf-8"))
    manifest[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    file.write_text(json.dumps(manifest), encoding="utf-8")


def test_different_numerical_runtime_fails_with_versions_before_fitting(
    report, tmp_path, monkeypatch
):
    path = tmp_path / "different-runtime"
    shutil.copytree(report, path)
    file = path / "summary.json"
    summary = json.loads(file.read_text(encoding="utf-8"))
    summary["runtime"]["scikit_learn"] = "999.0.0"
    file.write_text(json.dumps(summary), encoding="utf-8")
    rehash(path, "summary.json")

    def must_not_fit(*args):
        raise AssertionError("A mismatched runtime must be detected before fitting")

    monkeypatch.setattr("creditvintage.card_credit.card_experiment", must_not_fit)
    with pytest.raises(DataContractError, match=r"recorded versions:.*scikit-learn==999\.0\.0"):
        verify_card_benchmark(path)


@pytest.mark.parametrize("corruption", ["score", "metric", "queue", "zip"])
def test_rehashed_invented_results_or_changed_download_fail_refit(report, tmp_path, corruption):
    path = tmp_path / "copy"
    shutil.copytree(report, path)
    if corruption in {"score", "queue"}:
        filename = "predictions.csv" if corruption == "score" else "review_queue.csv"
        file = path / filename
        with file.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, list(reader)
        if corruption == "score":
            rows[0]["logistic"] = "0.99999"
        else:
            rows[0]["record_id"] = rows[1]["record_id"]
        with file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        rehash(path, filename)
    elif corruption == "metric":
        file = path / "summary.json"
        summary = json.loads(file.read_text(encoding="utf-8"))
        summary["results"]["logistic"]["roc_auc"]["mean"] = 0.999
        file.write_text(json.dumps(summary), encoding="utf-8")
        rehash(path, "summary.json")
    else:
        with zipfile.ZipFile(path / "evidence.zip", "a") as archive:
            archive.writestr("extra.txt", "invented evidence")
    with pytest.raises(DataContractError, match="disagrees|replay differs|ZIP differs"):
        verify_card_benchmark(path)
