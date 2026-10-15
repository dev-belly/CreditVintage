"""Regressions for a public-data example kept separate from dated loan events."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/german_credit.py"
SPEC = importlib.util.spec_from_file_location("public_credit_example", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.fixture(scope="module")
def public_report(tmp_path_factory):
    path = tmp_path_factory.mktemp("public-credit") / "report"
    example.run(path)
    return path


def rehash(path):
    example.write_json(
        path / "manifest.json", {name: example.digest(path / name) for name in example.FILES}
    )


def test_frozen_source_preserves_label_meaning_and_rejects_changed_bytes(tmp_path):
    _, labels = example.load_data()
    assert labels[:3].tolist() == [0, 1, 0]
    assert np.bincount(labels).tolist() == [700, 300]
    altered = tmp_path / "source.data"
    altered.write_bytes(example.DATA.read_bytes().replace(b"A11", b"A12", 1))
    with pytest.raises(ValueError, match="source bytes"):
        example.load_data(altered)


def test_preprocessing_does_not_learn_heldout_values_or_categories():
    features, labels = example.load_data()
    train, test = example.split_indices(labels, 7)
    heldout = features[test].copy()
    heldout[:, example.NUMERIC[0]] = 1e9
    heldout[:, example.CATEGORICAL[0]] = "HELDOUT_ONLY"
    fitted = example.make_model("logistic", 7).fit(features[train], labels[train])
    preprocess = fitted.named_steps["preprocess"]
    scaler = preprocess.named_transformers_["numeric"]
    np.testing.assert_allclose(
        scaler.mean_, features[train][:, example.NUMERIC].astype(float).mean(axis=0)
    )
    encoder = preprocess.named_transformers_["categorical"]
    assert all("HELDOUT_ONLY" not in values for values in encoder.categories_)
    # Unknown held-out categories remain scoreable without refitting the encoder.
    assert np.isfinite(fitted.predict_proba(heldout)).all()
    assert all("HELDOUT_ONLY" not in values for values in encoder.categories_)


def test_inspection_ties_depend_on_source_id_not_outcome_or_input_order():
    ids = np.arange(1, 21)
    labels = np.tile([0, 1], 10)
    _, selected = example.measure(labels, np.full(20, 0.3), ids)
    _, changed = example.measure(1 - labels, np.full(20, 0.3), ids)
    assert ids[selected].tolist() == [1, 2]
    np.testing.assert_array_equal(selected, changed)
    _, reversed_selection = example.measure(labels[::-1], np.full(20, 0.3), ids[::-1])
    assert sorted(ids[::-1][reversed_selection].tolist()) == [1, 2]


def test_complete_saved_run_verifies_without_retraining(public_report, monkeypatch):
    def must_not_train(*args, **kwargs):
        raise AssertionError("Verification must not retrain or overwrite the saved scores")

    monkeypatch.setattr(example, "fit_scores", must_not_train)
    checked = example.verify(public_report)
    assert checked == {
        "verified_splits": 5,
        "model_split_results": 15,
        "test_scores": 3000,
        "data_kind": "public_historical_static",
        "retrained": False,
    }
    with pytest.raises(FileExistsError):
        example.run(public_report)


@pytest.mark.parametrize("corruption", ["label", "train_row", "nan", "metric", "split", "page"])
def test_rehashed_inconsistent_evidence_is_rejected(public_report, tmp_path, corruption):
    out = tmp_path / "report"
    shutil.copytree(public_report, out)
    if corruption in {"label", "train_row", "nan"}:
        rows = example.read_csv(out / "predictions.csv")
        if corruption == "label":
            rows[0]["label_bad_risk"] = str(1 - int(rows[0]["label_bad_risk"]))
        elif corruption == "train_row":
            assignments = example.read_csv(out / "splits.csv")
            rows[0]["row_id"] = next(
                r["row_id"]
                for r in assignments
                if r["seed"] == rows[0]["seed"] and r["partition"] == "train"
            )
        else:
            rows[0]["score"] = "NaN"
        example.write_csv(out / "predictions.csv", rows)
    elif corruption == "metric":
        rows = example.read_csv(out / "metrics.csv")
        rows[0]["roc_auc"] = "0.99"
        example.write_csv(out / "metrics.csv", rows)
    elif corruption == "split":
        rows = example.read_csv(out / "splits.csv")
        rows[0]["partition"] = "test"
        example.write_csv(out / "splits.csv", rows)
    else:
        (out / "index.html").write_text("Invented best-model claim", encoding="utf-8")
    rehash(out)
    with pytest.raises(ValueError):
        example.verify(out)


def test_summary_cannot_change_the_data_kind_to_temporal(public_report, tmp_path):
    out = tmp_path / "report"
    shutil.copytree(public_report, out)
    summary = json.loads((out / "summary.json").read_text())
    summary["data_kind"] = "real_temporal_validation"
    example.write_json(out / "summary.json", summary)
    rehash(out)
    with pytest.raises(ValueError, match="protocol"):
        example.verify(out)
