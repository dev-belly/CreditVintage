from __future__ import annotations

import csv
import hashlib
import json
from importlib.resources import files

import numpy as np
import pytest

from creditvintage.cli import main
from creditvintage.core import DataContractError
from creditvintage.public_credit import (
    CATEGORICAL,
    NUMERIC,
    experiment,
    fit_public_scores,
    load_public_data,
    public_splits,
    verify_public_benchmark,
    write_public_benchmark,
)


@pytest.fixture(scope="module")
def raw():
    return files("creditvintage").joinpath("data/german_credit/german.data").read_bytes()


def test_real_table_has_native_labels_disjoint_splits_and_no_invented_dates(raw):
    x, y = load_public_data(raw)
    splits = public_splits(y)
    assert x.shape == (1000, 20)
    assert sum(y) == 300
    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 600,
        "calibration": 200,
        "test": 200,
    }
    assert len(set(np.concatenate(list(splits.values())))) == 1000
    assert {name: sum(y[indices]) for name, indices in splits.items()} == {
        "train": 180,
        "calibration": 60,
        "test": 60,
    }
    assert set(NUMERIC) | set(CATEGORICAL) == set(range(20)) - {8, 12, 19}
    result = experiment(raw)
    assert result["summary"]["data_kind"] == "public_observed_static"
    assert "decision_at" not in result["predictions"][0]
    assert "native_bad" not in result["queue"][0]
    assert len(result["queue"]) == 20
    for capacity in (0.05, 0.10, 0.20):
        policies = [p for p in result["policies"] if p["capacity"] == capacity]
        assert len(policies) == 3
        assert {p["reviewed"] for p in policies} == {round(200 * capacity)}
        assert {p["total_bad"] for p in policies} == {60}


def test_test_labels_do_not_change_scores_and_test_only_categories_do_not_change_training(raw):
    x, y = load_public_data(raw)
    splits = public_splits(y)
    original = fit_public_scores(x, y, splits)
    changed_labels = y.copy()
    changed_labels[splits["test"]] = 1 - changed_labels[splits["test"]]
    other = fit_public_scores(x, changed_labels, splits)
    for method in original:
        np.testing.assert_array_equal(original[method], other[method])
    changed_features = x.copy()
    changed_features[splits["test"], 0] = "TEST_ONLY_CATEGORY"
    changed_features[splits["test"], 1] = "10000000"
    other = fit_public_scores(changed_features, y, splits)
    for method in original:
        np.testing.assert_allclose(
            original[method][splits["train"]], other[method][splits["train"]], atol=1e-12, rtol=0
        )


def test_source_modification_fails_instead_of_becoming_a_new_public_source(raw):
    with pytest.raises(DataContractError, match="frozen UCI original"):
        load_public_data(raw.replace(b"A11", b"A12", 1))


@pytest.fixture
def bundle(tmp_path):
    destination = tmp_path / "public"
    write_public_benchmark(destination)
    return destination


def rehash(bundle, name):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files_sha256"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_public_bundle_refits_and_existing_output_is_preserved(bundle):
    assert verify_public_benchmark(bundle)["refitted"] is True
    before = (bundle / "summary.json").read_bytes()
    with pytest.raises(DataContractError, match="new public benchmark output"):
        write_public_benchmark(bundle)
    assert (bundle / "summary.json").read_bytes() == before


@pytest.mark.parametrize("filename", ["predictions.csv", "split.csv", "review_queue.csv"])
def test_rehashed_changed_scores_assignments_and_queue_fail_refit(bundle, filename):
    path = bundle / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    if filename == "predictions.csv":
        rows[0]["logistic"] = "0.99999"
    elif filename == "split.csv":
        rows[0]["split"] = "test" if rows[0]["split"] != "test" else "train"
    else:
        rows[0]["record_id"] = rows[1]["record_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    rehash(bundle, filename)
    with pytest.raises(DataContractError, match="replay disagrees"):
        verify_public_benchmark(bundle)


def test_rehashed_invented_metric_and_webpage_fail(bundle):
    path = bundle / "summary.json"
    original = path.read_bytes()
    summary = json.loads(original)
    summary["test_metrics"]["logistic"]["roc_auc"] = 0.999
    path.write_text(json.dumps(summary), encoding="utf-8")
    rehash(bundle, path.name)
    with pytest.raises(DataContractError, match="summary disagrees"):
        verify_public_benchmark(bundle)
    path.write_bytes(original)
    rehash(bundle, path.name)
    page = bundle / "index.html"
    page.write_text(page.read_text(encoding="utf-8") + "Invented result", encoding="utf-8")
    rehash(bundle, page.name)
    with pytest.raises(DataContractError, match="page disagrees"):
        verify_public_benchmark(bundle)


def test_installed_cli_public_commands(tmp_path, capsys):
    destination = tmp_path / "public"
    assert main(["public-benchmark", "--out", str(destination)]) == 0
    assert main(["verify-public", str(destination)]) == 0
    assert "public_observed_static" in capsys.readouterr().out
