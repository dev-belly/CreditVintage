"""Replay user workflows from built wheels in a clean, source-free environment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import venv
from pathlib import Path
from textwrap import dedent
from typing import Any


def check(directory: Path, card_evidence: Path | None = None) -> dict[str, Any]:
    def wheel(prefix: str) -> Path:
        matches = list(directory.resolve().glob(f"{prefix}-*.whl"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {prefix} wheel; found {len(matches)}")
        return matches[0]

    credit_wheel = wheel("creditvintage_research")
    pit_wheel = wheel("pitbridge")
    expected_version = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(key, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    with tempfile.TemporaryDirectory(prefix="creditvintage-wheel-") as temporary:
        root = Path(temporary).resolve()
        installed = root / "environment"
        work = root / "work"
        work.mkdir()
        venv.EnvBuilder(with_pip=True).create(installed)
        executables = installed / ("Scripts" if os.name == "nt" else "bin")
        python = executables / ("python.exe" if os.name == "nt" else "python")
        cli = executables / ("creditvintage.exe" if os.name == "nt" else "creditvintage")

        def run(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(
                arguments,
                cwd=work,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            if result.returncode != expected:
                raise RuntimeError(
                    f"Command failed ({result.returncode}, expected {expected}): "
                    f"{arguments}\n{result.stdout}\n{result.stderr}"
                )
            return result

        def command(*arguments: str, expected: int = 0) -> dict[str, Any]:
            result = run(str(cli), *arguments, expected=expected)
            return json.loads(result.stdout) if expected == 0 else {"stderr": result.stderr}

        run(str(python), "-m", "pip", "install", str(credit_wheel))
        probe = run(
            str(python),
            "-I",
            "-c",
            "import importlib.util,json,creditvintage; "
            "from importlib.metadata import version; "
            "print(json.dumps({'origin':creditvintage.__file__, "
            "'version':version('creditvintage-research'), "
            "'pitbridge':importlib.util.find_spec('pitbridge') is not None}))",
        )
        location = json.loads(probe.stdout)
        if not Path(location["origin"]).resolve().is_relative_to(installed):
            raise ValueError("Imported CreditVintage from outside its isolated environment")
        if location["pitbridge"]:
            raise ValueError("The core wheel unexpectedly installed the optional integration")
        if location["version"] != expected_version:
            raise ValueError("Installed wheel version differs from the checked source version")
        calendar_probe = run(
            str(python),
            "-I",
            "-c",
            dedent("""
                import json
                from datetime import date, datetime, timedelta
                from creditvintage.core import (
                    REPORT_DAYS, Application, DataContractError, FeatureEvent,
                    PerformanceEvent, snapshot,
                )

                for age in (0, 181, 182):
                    decision = date.max - timedelta(days=age)
                    app = Application("calendar-end", decision, 100_000, "retail")
                    features = [
                        FeatureEvent(app.application_id, name, value, decision, decision)
                        for name, value in (
                            ("utilization", 0.2), ("debt_to_income", 0.3),
                            ("prior_delinquencies", 0),
                        )
                    ]
                    reports = [
                        PerformanceEvent(app.application_id,
                            decision + timedelta(days=day),
                            decision + timedelta(days=day + 2), 0)
                        for day in REPORT_DAYS
                    ] if age == 182 else []
                    row = snapshot([app], features, reports, date.max)[0]
                    expected = ((0, "observed_nondefault") if age == 182
                                else (None, "not_mature"))
                    if (row.label, row.status) != expected:
                        raise ValueError(f"Installed snapshot changed maturity at age {age}")
                for invalid in (None, "2025-01-01", datetime(2025, 1, 1)):
                    try:
                        snapshot([], [], [], invalid)
                    except DataContractError:
                        pass
                    else:
                        raise ValueError("Installed snapshot accepted a non-date cutoff")
                print(json.dumps({"status": "verified", "cases": 6}))
            """),
        )
        calendar_boundaries = json.loads(calendar_probe.stdout)
        run(str(python), "-I", "-m", "creditvintage", "--help")
        starter = command("init-inputs", "--out", "editable-inputs")
        diagnosed = command("check-inputs", "editable-inputs")
        if (
            starter["data_kind"] != "synthetic"
            or diagnosed["status"] != "ready_for_evaluation"
            or diagnosed["model_fitted"] is not False
            or diagnosed["preflight"]["input_rows"]["applications"] != 480
        ):
            raise ValueError("Installed CSV starter or model-free input check failed")
        command(
            "run",
            "--applications",
            "editable-inputs/applications.csv",
            "--features",
            "editable-inputs/features.csv",
            "--performance",
            "editable-inputs/performance.csv",
            "--config",
            "editable-inputs/cohorts.json",
            "--out",
            "editable-evaluation",
        )
        editable_result = command("verify", "editable-evaluation")
        if editable_result["loans"] != 120:
            raise ValueError("The actual CSV evaluation did not retain the expected holdout")
        csv_path = work / "editable-inputs" / "applications.csv"
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, input_rows = reader.fieldnames, list(reader)
        assert columns is not None
        original_principal = input_rows[0]["principal_cents"]
        input_rows[0]["principal_cents"] = "-1"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(input_rows)
        rejected_inputs = json.loads(
            run(str(cli), "check-inputs", "editable-inputs", expected=2).stdout
        )
        if (
            rejected_inputs["status"] != "invalid_inputs"
            or rejected_inputs["issues"][0]["file"] != "applications.csv"
            or rejected_inputs["issues"][0]["line"] != 2
        ):
            raise ValueError("Installed diagnostics failed to locate an invalid CSV row")
        input_rows[0]["principal_cents"] = original_principal
        input_rows[0]["decision_at"] = input_rows[0]["decision_at"].replace("-", "")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(input_rows)
        rejected_date = json.loads(
            run(str(cli), "check-inputs", "editable-inputs", expected=2).stdout
        )
        issue = rejected_date["issues"][0]
        if (
            rejected_date["issue_count"] != 1
            or issue["code"] != "value_type"
            or issue["column"] != "decision_at"
            or "YYYY-MM-DD" not in issue["message"]
        ):
            raise ValueError("Installed diagnostics accepted an alternate CSV date format")
        invalid_cutoff = run(
            str(cli), "monitor", "--as-of", "2024-W40-2", "--out", "invalid-cutoff", expected=2
        )
        if "YYYY-MM-DD" not in invalid_cutoff.stderr or (work / "invalid-cutoff").exists():
            raise ValueError("Installed monitor accepted an alternate cutoff date format")
        command("run", "--out", "evaluation")
        evaluation = command("verify", "evaluation")
        configuration = json.loads(
            (work / "evaluation" / "summary.json").read_text(encoding="utf-8")
        )["config"]
        configuration["train_start"] = "2022-02-01"
        (work / "cohorts.json").write_text(json.dumps(configuration), encoding="utf-8")
        preflight = command("preflight", "--config", "cohorts.json")
        if (
            preflight["model_fitted"] is not False
            or preflight["cohorts"]["train"]["applications"] != 880
            or preflight["applications_outside_windows"] != 80
        ):
            raise ValueError("Installed config/preflight did not apply custom calendar windows")
        command("run", "--config", "cohorts.json", "--out", "configured")
        command("verify", "configured")
        command("public-benchmark", "--out", "public-credit")
        public_credit = command("verify-public", "public-credit")
        if public_credit["test_records"] != 200 or public_credit["refitted"] is not True:
            raise ValueError(
                "Installed public data/experiment did not refit from bundled resources"
            )
        command("card-benchmark", "--out", "card-credit")
        card_credit = command("verify-card", "card-credit")
        if card_credit["records"] != 30000 or card_credit["model_scores"] != 120000:
            raise ValueError("Installed card-data workflow lost source rows or model results")
        if card_evidence is not None:
            card_evidence.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "card-credit" / "evidence.zip", card_evidence)
        command("benchmark", "--out", "benchmark")
        comparison = command("verify-benchmark", "benchmark")
        command("monitor", "--as-of", "2024-10-01", "--out", "monitor")
        command("verify-monitor", "monitor")
        command("init-inputs", "--out", "monitor-inputs")
        pending_root = work / "monitor-inputs"
        with (pending_root / "applications.csv").open(encoding="utf-8", newline="") as handle:
            current_ids = {
                row["application_id"]
                for row in csv.DictReader(handle)
                if "2024-07-01" <= row["decision_at"] <= "2024-12-31"
            }
        performance_path = pending_root / "performance.csv"
        with performance_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, pending_reports = reader.fieldnames, list(reader)
        assert columns is not None
        with performance_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                row for row in pending_reports if row["application_id"] not in current_ids
            )
        monitoring_inputs = (
            "--applications",
            "monitor-inputs/applications.csv",
            "--features",
            "monitor-inputs/features.csv",
            "--performance",
            "monitor-inputs/performance.csv",
            "--config",
            "monitor-inputs/cohorts.json",
        )
        input_bytes = {path.name: path.read_bytes() for path in pending_root.iterdir()}
        monitor_preflight = command("preflight", *monitoring_inputs, "--as-of", "2024-10-01")
        if (
            monitor_preflight["status"] != "ready_for_monitoring"
            or monitor_preflight["model_fitted"] is not False
            or [
                monitor_preflight["cohorts"][stage]["applications"]
                for stage in ("train", "calibration", "current")
            ]
            != [240, 120, 80]
            or monitor_preflight["current_applications_not_yet_visible"] != 40
            or "events" in monitor_preflight["cohorts"]["current"]
        ):
            raise ValueError("Installed monitoring preflight did not preserve label-free cohorts")
        incomplete_evaluation = command("preflight", *monitoring_inputs, expected=2)
        if "test has unlabelled loans" not in incomplete_evaluation["stderr"]:
            raise ValueError("Default preflight lost its mature evaluation gate")
        command(
            "monitor", *monitoring_inputs, "--as-of", "2024-10-01", "--out", "unlabelled-monitor"
        )
        if command("verify-monitor", "unlabelled-monitor")["loans"] != 80:
            raise ValueError("Installed monitor and preflight population counts disagree")
        if input_bytes != {path.name: path.read_bytes() for path in pending_root.iterdir()}:
            raise ValueError("Preflight or monitoring changed the source input files")
        command("analyze", "evaluation", "--out", "analysis")
        analysis = command("verify-analysis", "analysis")
        command("review-init", "analysis", "--db", "reviews.sqlite")
        initial = command("review-status", "reviews.sqlite")
        identity = initial["cases"][0]["application_id"]
        claim = (
            "review",
            "reviews.sqlite",
            identity,
            "claim",
            "--actor",
            "wheel-reviewer",
            "--version",
            "0",
            "--request-id",
            "claim-1",
        )
        accepted = command(*claim)
        retry = command(*claim)
        if accepted["status"] != "accepted" or retry != {
            "status": "replayed",
            "event": accepted["event"],
        }:
            raise ValueError("An identical installed-CLI retry did not replay the original event")
        stale = command(
            "review",
            "reviews.sqlite",
            identity,
            "claim",
            "--actor",
            "competing-reviewer",
            "--version",
            "0",
            "--request-id",
            "claim-2",
            expected=2,
        )
        if "Stale review version" not in stale["stderr"]:
            raise ValueError("A stale installed-CLI claim did not explain the version conflict")
        command(
            "review",
            "reviews.sqlite",
            identity,
            "resolve",
            "--actor",
            "wheel-reviewer",
            "--version",
            "1",
            "--request-id",
            "resolve-1",
            "--note",
            "Checked synthetic evidence",
            "--evidence",
            "synthetic-doc:wheel-001",
            "--outcome",
            "escalated",
        )
        final = command("review-status", "reviews.sqlite")
        selected = initial["selected_cases"]
        if (
            final["counts"] != {"queued": selected - 1, "in_review": 0, "resolved": 1}
            or final["events"] != 2
            or final["outcomes"] != {"cleared": 0, "escalated": 1}
            or final["completion_rate"] != 1 / selected
            or any("label" in case for case in final["cases"])
        ):
            raise ValueError("Installed review workflow violated counts, retry or label separation")
        delivery = command("review-export", "reviews.sqlite", "--out", "review-evidence")
        replayed = command(
            "verify-reviews", "review-evidence", "--expected-sha256", delivery["bundle_sha256"]
        )
        if (
            replayed["counts"] != final["counts"]
            or replayed["events"] != final["events"]
            or replayed["checkpoint_matched"] is not True
        ):
            raise ValueError("Installed review evidence did not replay its delivery snapshot")
        missing = command("lineage-demo", "--out", "missing-integration", expected=2)
        if (
            "Install PITBridge first" not in missing["stderr"]
            or (work / "missing-integration").exists()
        ):
            raise ValueError("Missing optional integration did not fail before creating output")
        run(str(python), "-m", "pip", "install", "--no-index", str(pit_wheel))
        probe = run(str(python), "-I", "-c", "import pitbridge; print(pitbridge.__file__)")
        if not Path(probe.stdout.strip()).resolve().is_relative_to(installed):
            raise ValueError("Imported PITBridge from outside its isolated environment")
        command("lineage-demo", "--out", "lineage")
        lineage = command("verify-lineage", "lineage")
        run(str(python), "-m", "pip", "check")
        return {
            "status": "verified",
            "version": location["version"],
            "installation": "non-editable wheels; clean venv; cwd outside both source trees",
            "core_without_pitbridge": True,
            "snapshot_calendar_boundaries": calendar_boundaries,
            "editable_csv_inputs": {
                "status": "verified",
                "applications": 480,
                "verified_test_loans": editable_result["loans"],
                "invalid_row_located": True,
                "strict_calendar_inputs": "verified",
                "model_fitted_in_check": False,
            },
            "evaluation": evaluation,
            "custom_config_and_preflight": "verified without model fitting in preflight",
            "monitor_preflight": monitor_preflight,
            "public_credit": public_credit,
            "card_credit": card_credit,
            "benchmark": comparison,
            "analysis": analysis,
            "selected_review_cases": selected,
            "accepted_review_events": final["events"],
            "retry_and_stale_version": "verified",
            "portable_review_evidence": "replayed with external checkpoint",
            "optional_lineage": lineage,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory with CreditVintage and PITBridge wheels"
    )
    parser.add_argument("--card-evidence", type=Path, help="Export the verified card evidence ZIP")
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.card_evidence), indent=2))
