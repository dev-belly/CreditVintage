"""The reader entry must trace its displayed aggregates to held-out rows."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from creditvintage.core import DataContractError
from creditvintage.sba_credit import METHODS, write_sba_benchmark

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_credit_brief.py"
SPEC = importlib.util.spec_from_file_location("credit_brief_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
brief = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brief)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("brief-source") / "evidence"
    write_sba_benchmark(directory)
    return directory


def test_brief_counts_from_independent_prediction_ranking(source, tmp_path):
    destination = tmp_path / "brief"
    checked = brief.build_brief(source, destination)
    data = json.loads((destination / "data.json").read_text(encoding="utf-8"))
    page = (destination / "index.html").read_text(encoding="utf-8")
    embedded = re.search(
        r'<script type="application/json" id="study-data">(.*?)</script>', page, re.S
    )
    assert embedded is not None and json.loads(embedded.group(1)) == data
    assert checked["source_refitted"] is True and checked["fixed_comparisons"] == 9
    with (source / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    events = sum(int(row["chargeoff_24m"]) for row in predictions)
    assert len(predictions) == data["test_records"] == 10000
    assert events == data["test_events"] == 89
    assert len(data["policies"]) == 9
    for method in METHODS:
        ranking = sorted(predictions, key=lambda row: (-float(row[method]), row["record_id"]))
        for capacity in (5, 10, 20):
            reviewed = len(predictions) * capacity // 100
            captured = sum(int(row["chargeoff_24m"]) for row in ranking[:reviewed])
            assert next(
                row
                for row in data["policies"]
                if row["method"] == method and row["capacity_percent"] == capacity
            ) == {
                "method": method,
                "capacity_percent": capacity,
                "reviewed": reviewed,
                "captured": captured,
                "total_events": events,
            }
    for name, digest in data["evidence_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    assert "@@" not in page
    assert page.count('<article class="result-card ') == 3


def test_invalid_evidence_does_not_create_brief(source, tmp_path):
    changed = tmp_path / "changed"
    shutil.copytree(source, changed)
    (changed / "policy_metrics.csv").write_text("fabricated counts", encoding="utf-8")
    destination = tmp_path / "brief"
    with pytest.raises(DataContractError, match="digest differs"):
        brief.build_brief(changed, destination)
    assert not destination.exists()


def test_existing_brief_is_not_overwritten(source, tmp_path):
    destination = tmp_path / "brief"
    destination.mkdir()
    retained = destination / "index.html"
    retained.write_text("keep this reader's saved copy", encoding="utf-8")
    with pytest.raises(FileExistsError):
        brief.build_brief(source, destination)
    assert retained.read_text(encoding="utf-8") == "keep this reader's saved copy"
