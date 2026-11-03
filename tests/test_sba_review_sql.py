from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import zipfile
from pathlib import Path

import pytest

from creditvintage.core import DataContractError
from creditvintage.sba_credit import SBA_FILES, write_sba_benchmark

SCRIPT = Path(__file__).resolve().parents[1] / "examples/sba_review_sql.py"
SPEC = importlib.util.spec_from_file_location("sba_review_sql", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def predictions():
    return [
        {
            "record_id": f"loan-{i:02}",
            "funded_at": "2016-01-01",
            "chargeoff_24m": str(int(i in {0, 2, 20})),
            "constant": "0.1",
            "logistic": str(i / 20),
            "isotonic": "0.2",
        }
        for i in range(21)
    ]


def test_hand_calculated_ceilings_ties_and_independent_denominators():
    rows = predictions()
    random.Random(17).shuffle(rows)
    policies, queue = example.replay(rows)
    assert len(policies) == 9 and len(queue) == 9
    constant = [row for row in policies if row["method"] == "constant"]
    assert [row["reviewed"] for row in constant] == [2, 3, 5]
    assert [row["captured"] for row in constant] == [1, 2, 2]
    assert all(row["total_events"] == 3 for row in policies)
    assert constant[1]["precision"] == constant[1]["capture"] == 2 / 3
    assert constant[2]["precision"] == 2 / 5 and constant[2]["capture"] == 2 / 3
    assert [row["record_id"] for row in queue if row["method"] == "constant"] == [
        "loan-00",
        "loan-01",
        "loan-02",
    ]
    assert [row["record_id"] for row in queue if row["method"] == "logistic"] == [
        "loan-20",
        "loan-19",
        "loan-18",
    ]
    assert example.replay(list(reversed(rows))) == (policies, queue)


def test_changing_labels_cannot_change_selection_and_zero_events_are_null():
    rows = predictions()
    _, original = example.replay(rows)
    changed = [{**row, "chargeoff_24m": "0"} for row in rows]
    policies, queue = example.replay(changed)
    assert queue == original
    assert all(row["captured"] == row["total_events"] == 0 for row in policies)
    assert all(row["precision"] == 0 and row["capture"] is None for row in policies)


@pytest.mark.parametrize(
    "field,value",
    [
        ("logistic", "nan"),
        ("logistic", "inf"),
        ("logistic", "1.1"),
        ("logistic", "-0.1"),
        ("chargeoff_24m", "2"),
        ("chargeoff_24m", "0.0"),
    ],
)
def test_invalid_scores_or_labels_fail_before_sql(field, value):
    rows = predictions()
    rows[0][field] = value
    with pytest.raises(DataContractError):
        example.replay(rows)


def test_empty_duplicate_and_wrong_columns_are_rejected():
    with pytest.raises(DataContractError, match="nonempty"):
        example.replay([])
    rows = predictions()
    with pytest.raises(DataContractError, match="unique"):
        example.replay([*rows, rows[0]])
    rows[0].pop("funded_at")
    with pytest.raises(DataContractError, match="columns"):
        example.replay(rows)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sba-sql") / "report"
    write_sba_benchmark(directory)
    return directory


def test_actual_saved_sba_policies_and_all_queue_members_match_read_only(bundle, capsys):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
    assert example.main([str(bundle)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "matched_saved_evidence"
    assert result["evidence_verification"]["refitted"] is True
    assert result["prediction_rows"] == 10000 and result["queue_rows"] == 3000
    assert result["policy_rows"] == 9 and result["labels_used_for_ranking"] is False
    row = next(r for r in result["policies"] if r["method"] == "logistic" and r["capacity"] == 0.1)
    assert (row["reviewed"], row["captured"], row["total_events"]) == (1000, 26, 89)
    assert row["precision"] == 26 / 1000 and row["capture"] == 26 / 89
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}


def test_rehashed_policy_tampering_fails_before_sql(bundle, tmp_path, monkeypatch, capsys):
    copied = tmp_path / "report"
    shutil.copytree(bundle, copied)
    policy = copied / "policy_metrics.csv"
    lines = policy.read_text().splitlines()
    cells = lines[1].split(",")
    cells[3] = str(int(cells[3]) + 1)
    lines[1] = ",".join(cells)
    policy.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    manifest = json.loads((copied / "manifest.json").read_text())
    manifest["files_sha256"]["policy_metrics.csv"] = hashlib.sha256(policy.read_bytes()).hexdigest()
    (copied / "manifest.json").write_text(json.dumps(manifest))
    with zipfile.ZipFile(copied / "evidence.zip", "w") as archive:
        for name in (*SBA_FILES, "manifest.json"):
            archive.write(copied / name, name)

    def forbidden(*args):
        raise AssertionError("SQL must not run on unverified evidence")

    monkeypatch.setattr(example, "replay", forbidden)
    assert example.main([str(copied)]) == 2
    captured = capsys.readouterr()
    assert (
        captured.out == "" and "sba-review-sql:" in captured.err and "Traceback" not in captured.err
    )
