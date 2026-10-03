"""Temporal integration and semantic forgery regressions, runnable with unittest."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from creditvintage.core import Application, DataContractError
from creditvintage.lineage import (
    LINEAGE_COLUMNS,
    SOURCE,
    adapt_bundle,
    cutoff,
    verify_lineage,
    write_demo,
)

try:
    from pitbridge.bundle import write_bundle
except ImportError:
    write_bundle = None


@unittest.skipIf(write_bundle is None, "PITBridge is an optional integration dependency")
class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pit"
        self.app = Application("loan-1", date(2024, 7, 1), 100000, "retail")
        self.data = {
            "data_kind": "synthetic",
            "decisions": [
                {
                    "decision_id": "loan-1",
                    "entity_id": "loan-1",
                    "decision_at": cutoff(self.app.decision_at),
                }
            ],
            "specs": [
                {"source": SOURCE, "feature": name, "max_age_days": 7}
                for name in ("utilization", "debt_to_income", "prior_delinquencies")
            ],
            "observations": [
                {
                    "record_id": name + "-v1",
                    "entity_id": "loan-1",
                    "source": SOURCE,
                    "feature": name,
                    "event_at": "2024-06-26T00:00:00Z",
                    "published_at": "2024-06-30T08:00:00Z",
                    "ingested_at": "2024-06-30T09:00:00Z",
                    "revision": 1,
                    "value": value,
                    "deleted": False,
                }
                for name, value in (
                    ("utilization", 0.2),
                    ("debt_to_income", 0.3),
                    ("prior_delinquencies", 1),
                )
            ],
        }

    def export(self):
        write_bundle(self.data, self.path)
        return adapt_bundle([self.app], self.path)

    def test_late_ingestion_does_not_revise_model_feature(self) -> None:
        original = self.data["observations"][0]
        self.data["observations"].append(
            {
                **original,
                "record_id": "utilization-v2",
                "revision": 2,
                "value": 0.99,
                "published_at": "2024-07-01T12:00:00Z",
                "ingested_at": "2024-07-02T00:00:00Z",
            }
        )
        features, lineage = self.export()
        self.assertEqual(next(f.value for f in features if f.name == "utilization"), 0.2)
        self.assertEqual(
            next(r["record_id"] for r in lineage if r["feature"] == "utilization"), "utilization-v1"
        )

    def test_cutoff_is_inclusive_to_the_microsecond(self) -> None:
        original = self.data["observations"][0]
        self.data["observations"].append(
            {
                **original,
                "record_id": "at-cutoff",
                "revision": 2,
                "value": 0.4,
                "published_at": cutoff(self.app.decision_at),
                "ingested_at": cutoff(self.app.decision_at),
            }
        )
        features, _ = self.export()
        self.assertEqual(next(f.value for f in features if f.name == "utilization"), 0.4)

    def test_age_boundary_is_inclusive(self) -> None:
        self.data["observations"][0]["event_at"] = "2024-06-24T23:59:59.999999Z"
        self.assertEqual(len(self.export()[0]), 3)
        self.data["observations"][0]["event_at"] = "2024-06-24T23:59:59.999998Z"
        with self.assertRaisesRegex(DataContractError, "stale"):
            self.export()

    def test_missing_and_deleted_features_fail(self) -> None:
        self.data["observations"].pop(0)
        with self.assertRaisesRegex(DataContractError, "no_history"):
            self.export()
        row = self.data["observations"][0]
        self.data["observations"].append(
            {**row, "record_id": "deleted", "revision": 2, "value": None, "deleted": True}
        )
        with self.assertRaisesRegex(DataContractError, "deleted"):
            self.export()

    def test_calendar_date_cannot_hide_an_intraday_decision(self) -> None:
        self.data["decisions"][0]["decision_at"] = "2024-07-01T12:00:00Z"
        with self.assertRaisesRegex(DataContractError, "cutoffs"):
            self.export()

    def test_wrong_entity_mapping_fails(self) -> None:
        self.data["decisions"][0]["entity_id"] = "another-borrower"
        with self.assertRaisesRegex(DataContractError, "entities"):
            self.export()

    def test_fraction_and_count_units_are_checked(self) -> None:
        self.data["observations"][0]["value"] = 20
        with self.assertRaisesRegex(DataContractError, "between zero and one"):
            self.export()
        self.data["observations"][0]["value"] = 0.2
        self.data["observations"][2]["value"] = 1.5
        with self.assertRaisesRegex(DataContractError, "integer"):
            self.export()

    def test_unknown_application_and_feature_contract_fail(self) -> None:
        self.data["observations"].append(
            {**self.data["observations"][0], "entity_id": "unknown", "record_id": "unknown"}
        )
        with self.assertRaisesRegex(DataContractError, "unknown application"):
            self.export()
        self.data["observations"].pop()
        self.data["specs"][0]["max_age_days"] = None
        with self.assertRaisesRegex(DataContractError, "contract"):
            self.export()

    def test_csv_formula_record_id_is_rejected(self) -> None:
        self.data["observations"][0]["record_id"] = "=HYPERLINK(1)"
        with self.assertRaisesRegex(DataContractError, "safe for CSV"):
            self.export()


@unittest.skipIf(write_bundle is None, "PITBridge is an optional integration dependency")
class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name) / "baseline"
        write_demo(cls.base, seed=7, per_vintage=20)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def setUp(self) -> None:
        self.temp_case = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_case.cleanup)
        self.path = Path(self.temp_case.name) / "case"
        shutil.copytree(self.base, self.path)

    def rehash(self, name):
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"][name] = hashlib.sha256((self.path / name).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def test_complete_saved_source_and_model_replay(self) -> None:
        checked = verify_lineage(self.path)
        self.assertEqual(checked["source_features"], 1440)
        self.assertTrue(checked["pit_replay"] and checked["model_replay"])

    def test_rehashed_lineage_forgery_is_rejected(self) -> None:
        with (self.path / "lineage.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["record_id"] = "forged-source"
        with (self.path / "lineage.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LINEAGE_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.rehash("lineage.csv")
        with self.assertRaisesRegex(DataContractError, "source replay"):
            verify_lineage(self.path)

    def test_rehashed_feature_forgery_is_rejected(self) -> None:
        path = self.path / "features.csv"
        content = path.read_text()
        first = content.splitlines()[1]
        fields = first.split(",")
        fields[2] = ".999999"
        path.write_text(content.replace(first, ",".join(fields), 1))
        self.rehash("features.csv")
        with self.assertRaisesRegex(DataContractError, "features disagree"):
            verify_lineage(self.path)

    def test_rehashed_input_digest_forgery_is_rejected(self) -> None:
        path = self.path / "evaluation/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["input_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        self.rehash("evaluation/manifest.json")
        with self.assertRaisesRegex(DataContractError, "input digest"):
            verify_lineage(self.path)

    def test_rehashed_explorer_forgery_is_rejected(self) -> None:
        path = self.path / "index.html"
        path.write_text(path.read_text().replace("Raw PD", "Invented PD"))
        self.rehash("index.html")
        with self.assertRaisesRegex(DataContractError, "explorer"):
            verify_lineage(self.path)

    def test_unsupported_manifest_inventory_fails(self) -> None:
        path = self.path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files_sha256"]["../outside"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DataContractError, "unexpected artifact paths"):
            verify_lineage(self.path)

    def test_malformed_manifest_is_a_contract_error(self) -> None:
        path = self.path / "manifest.json"
        original = json.loads(path.read_text())
        for malformed in (None, list(original["files_sha256"]), 7):
            with self.subTest(inventory=malformed):
                path.write_text(json.dumps({**original, "files_sha256": malformed}))
                with self.assertRaisesRegex(DataContractError, "unexpected artifact paths"):
                    verify_lineage(self.path)

    def test_rehashed_fractional_and_boolean_counts_are_rejected(self) -> None:
        path = self.path / "summary.json"
        original = json.loads(path.read_text())
        for value in (480.0, True):
            with self.subTest(value=value):
                path.write_text(json.dumps({**original, "applications": value}))
                self.rehash("summary.json")
                with self.assertRaisesRegex(DataContractError, "counts"):
                    verify_lineage(self.path)

    def test_rehashed_evaluation_config_must_match_the_model_replay(self) -> None:
        path = self.path / "evaluation/summary.json"
        summary = json.loads(path.read_text())
        summary["config"]["seed"] += 1
        path.write_text(json.dumps(summary))
        manifest_path = self.path / "evaluation/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["summary.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        self.rehash("evaluation/summary.json")
        self.rehash("evaluation/manifest.json")
        # The independent report verifier can now detect the changed seed while
        # replaying bootstrap intervals, before lineage refits the model.
        with self.assertRaisesRegex(DataContractError, "model replay: config|bootstrap"):
            verify_lineage(self.path)
