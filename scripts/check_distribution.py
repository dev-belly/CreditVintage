"""Replay user workflows from built wheels in a clean, source-free environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def check(
    directory: Path, card_evidence: Path | None = None, sba_evidence: Path | None = None
) -> dict[str, Any]:
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
        command("sba-benchmark", "--out", "sba-credit")
        sba_credit = command("verify-sba", "sba-credit")
        if (
            sba_credit["records"] != 35000
            or sba_credit["test_records"] != 10000
            or sba_credit["test_events"] != 89
            or sba_credit["refitted"] is not True
            or sba_credit["target"] != "recorded_chargeoff_within_24_calendar_months"
        ):
            raise ValueError("Installed SBA study lost its dates, source or native target")
        if sba_evidence is not None:
            sba_evidence.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "sba-credit" / "evidence.zip", sba_evidence)
        sba_path = work / "sba-credit"
        original_summary = (sba_path / "summary.json").read_bytes()
        original_manifest = (sba_path / "manifest.json").read_bytes()
        for false_value, diagnostic in ((0.99, "SBA summary"), (10**400, "finite number")):
            sba_summary = json.loads(original_summary)
            sba_summary["test_metrics"]["logistic"]["roc_auc"] = false_value
            (sba_path / "summary.json").write_text(json.dumps(sba_summary), encoding="utf-8")
            sba_manifest = json.loads(original_manifest)
            sba_manifest["files_sha256"]["summary.json"] = hashlib.sha256(
                (sba_path / "summary.json").read_bytes()
            ).hexdigest()
            (sba_path / "manifest.json").write_text(json.dumps(sba_manifest), encoding="utf-8")
            rejected = command("verify-sba", "sba-credit", expected=2)
            if diagnostic not in rejected["stderr"] or "Traceback" in rejected["stderr"]:
                raise ValueError("Installed SBA verifier did not reject a false AUC cleanly")
        (sba_path / "summary.json").write_bytes(original_summary)
        (sba_path / "manifest.json").write_bytes(original_manifest)
        original_zip = (sba_path / "evidence.zip").read_bytes()
        damaged_zip = bytearray(original_zip)
        payload = (
            30
            + int.from_bytes(damaged_zip[26:28], "little")
            + int.from_bytes(damaged_zip[28:30], "little")
        )
        damaged_zip[payload] = 0x07
        (sba_path / "evidence.zip").write_bytes(damaged_zip)
        rejected = command("verify-sba", "sba-credit", expected=2)
        if (
            "Invalid SBA evidence ZIP" not in rejected["stderr"]
            or "Traceback" in rejected["stderr"]
        ):
            raise ValueError("Installed SBA verifier did not reject broken DEFLATE cleanly")
        (sba_path / "evidence.zip").write_bytes(original_zip)
        sba_credit["rehashed_false_metric_rejected"] = True
        sba_credit["oversized_metric_rejected"] = True
        sba_credit["damaged_zip_rejected"] = True
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
        # No outcome file exists during this installed two-table workflow.
        unavailable = pending_root / "outcomes-not-loaded.csv"
        performance_path.rename(unavailable)
        try:
            quality_flags = (
                "--applications",
                "monitor-inputs/applications.csv",
                "--features",
                "monitor-inputs/features.csv",
                "--config",
                "monitor-inputs/cohorts.json",
                "--as-of",
                "2024-10-01",
            )
            command(
                "monitor-features",
                "--applications",
                "monitor-inputs/applications.csv",
                "--features",
                "monitor-inputs/features.csv",
                "--config",
                "monitor-inputs/cohorts.json",
                "--as-of",
                "2024-10-01",
                "--out",
                "feature-monitor",
            )
            feature_monitor = command("verify-feature-monitor", "feature-monitor")
            command("monitor-feature-quality", *quality_flags, "--out", "healthy-quality")
            healthy_quality = command("verify-feature-quality", "healthy-quality")
            if healthy_quality["populations"]["current"]["complete_applications"] != 80:
                raise ValueError(
                    "Installed quality diagnostics changed healthy decision-time coverage"
                )
            with (pending_root / "features.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                feature_columns = reader.fieldnames
                gap_features = [
                    row
                    for row in reader
                    if (row["application_id"], row["name"]) != ("CV202407-0000", "utilization")
                ]
            assert feature_columns is not None
            with (work / "gap-features.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=feature_columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(gap_features)
            gap_flags = list(quality_flags)
            gap_flags[gap_flags.index("--features") + 1] = "gap-features.csv"
            command("monitor-feature-quality", *gap_flags, "--out", "gap-quality")
            feature_quality = command("verify-feature-quality", "gap-quality")
            current_quality = feature_quality["populations"]["current"]
            if (
                current_quality["applications"],
                current_quality["complete_applications"],
                current_quality["incomplete_applications"],
            ) != (80, 79, 1):
                raise ValueError(
                    "Installed diagnostics discarded an application with a missing input"
                )
            policy_init = command("init-feature-policy", "--out", "coverage-policy.json")
            trusted_policy = policy_init["policy_sha256"]
            passed_policy = command(
                "check-feature-quality",
                "healthy-quality",
                "--policy",
                "coverage-policy.json",
                "--out",
                "healthy-check",
            )
            if passed_policy["status"] != "pass":
                raise ValueError("Installed healthy coverage policy did not pass")
            command(
                "verify-feature-check", "healthy-check", "--expected-policy-sha256", trusted_policy
            )
            failed_policy = json.loads(
                run(
                    str(cli),
                    "check-feature-quality",
                    "gap-quality",
                    "--policy",
                    "coverage-policy.json",
                    "--out",
                    "gap-check",
                    expected=3,
                ).stdout
            )
            feature_check = command(
                "verify-feature-check", "gap-check", "--expected-policy-sha256", trusted_policy
            )
            if failed_policy["counts"] != {"pass": 2, "fail": 1, "insufficient_data": 0} or (
                feature_check["status"],
                feature_check["policy_status"],
            ) != ("verified", "fail"):
                raise ValueError(
                    "Installed evidence verification confused policy failure with integrity"
                )
            wrong_policy = command(
                "verify-feature-check",
                "gap-check",
                "--expected-policy-sha256",
                "0" * 64,
                expected=2,
            )
            if "trusted expected digest" not in wrong_policy["stderr"]:
                raise ValueError("Installed verifier did not enforce the external policy pin")
            undersized = json.loads((work / "coverage-policy.json").read_text(encoding="utf-8"))
            for rule in undersized["rules"]:
                rule["min_applications"] = 100
            (work / "small-policy.json").write_text(json.dumps(undersized), encoding="utf-8")
            small_policy = json.loads(
                run(
                    str(cli),
                    "check-feature-quality",
                    "gap-quality",
                    "--policy",
                    "small-policy.json",
                    "--out",
                    "small-check",
                    expected=3,
                ).stdout
            )
            if small_policy["status"] != "insufficient_data":
                raise ValueError("Installed small populations were not blocked by policy")
            check_path = work / "gap-check" / "check.json"
            false_check = json.loads(check_path.read_text(encoding="utf-8"))
            false_check["status"] = "pass"
            check_path.write_text(json.dumps(false_check), encoding="utf-8")
            check_manifest_path = work / "gap-check" / "manifest.json"
            check_manifest = json.loads(check_manifest_path.read_text(encoding="utf-8"))
            check_manifest["files_sha256"]["check.json"] = hashlib.sha256(
                check_path.read_bytes()
            ).hexdigest()
            check_manifest_path.write_text(json.dumps(check_manifest), encoding="utf-8")
            rejected_check = command("verify-feature-check", "gap-check", expected=2)
            if "replay" not in rejected_check["stderr"]:
                raise ValueError("Installed verifier accepted a rehashed false policy pass")
            strict_gap = command(
                "monitor-features", *gap_flags, "--out", "must-not-exist", expected=2
            )
            if (
                "Missing decision-time" not in strict_gap["stderr"]
                or (work / "must-not-exist").exists()
            ):
                raise ValueError("Coverage diagnostics weakened the strict distribution gate")
            quality_root = work / "gap-quality"
            quality_summary = quality_root / "quality.json"
            false_quality = json.loads(quality_summary.read_text(encoding="utf-8"))
            false_quality["populations"]["current"]["complete_applications"] = 80
            quality_summary.write_text(json.dumps(false_quality), encoding="utf-8")
            quality_manifest = quality_root / "manifest.json"
            false_quality_manifest = json.loads(quality_manifest.read_text(encoding="utf-8"))
            false_quality_manifest["files_sha256"]["quality.json"] = hashlib.sha256(
                quality_summary.read_bytes()
            ).hexdigest()
            quality_manifest.write_text(json.dumps(false_quality_manifest), encoding="utf-8")
            rejected_quality = command("verify-feature-quality", "gap-quality", expected=2)
            if "event replay" not in rejected_quality["stderr"]:
                raise ValueError("Installed quality replay accepted rehashed false coverage")
        finally:
            unavailable.rename(performance_path)
        if (
            feature_monitor["reference_applications"] != 120
            or feature_monitor["current_applications"] != 80
            or feature_monitor["monitored_inputs"] != 5
            or feature_monitor["model_fitted"] is not False
        ):
            raise ValueError("Installed feature monitoring lost label-free population counts")
        feature_root = work / "feature-monitor"
        feature_inputs = json.loads((feature_root / "inputs.json").read_text(encoding="utf-8"))
        if "performance" in feature_inputs:
            raise ValueError("Feature-monitor evidence unexpectedly contains outcomes")
        summary_path = feature_root / "monitor.json"
        false_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        false_summary["features"][0]["distance"] += 0.1
        summary_path.write_text(json.dumps(false_summary), encoding="utf-8")
        manifest_path = feature_root / "manifest.json"
        false_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        false_manifest["files_sha256"]["monitor.json"] = hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(false_manifest), encoding="utf-8")
        rejected_feature_report = command("verify-feature-monitor", "feature-monitor", expected=2)
        if "event replay" not in rejected_feature_report["stderr"]:
            raise ValueError("Installed feature replay accepted rehashed false distances")
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
            "feature_monitor": {**feature_monitor, "rehashed_false_distance_rejected": True},
            "feature_quality": {
                **feature_quality,
                "strict_missing_feature_gate": True,
                "rehashed_false_coverage_rejected": True,
            },
            "feature_policy": {
                **feature_check,
                "healthy_policy_passed": True,
                "insufficient_data_blocked": True,
                "external_policy_pin_enforced": True,
                "rehashed_false_pass_rejected": True,
            },
            "public_credit": public_credit,
            "card_credit": card_credit,
            "sba_credit": sba_credit,
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
    parser.add_argument("--sba-evidence", type=Path, help="Export the verified SBA evidence ZIP")
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.card_evidence, args.sba_evidence), indent=2))
