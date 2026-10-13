"""Replay user workflows from built wheels in a clean, source-free environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Any


def check(directory: Path) -> dict[str, Any]:
    def wheel(prefix: str) -> Path:
        matches = list(directory.resolve().glob(f"{prefix}-*.whl"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {prefix} wheel; found {len(matches)}")
        return matches[0]

    credit_wheel = wheel("creditvintage_research")
    pit_wheel = wheel("pitbridge")
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
            "print(json.dumps({'origin':creditvintage.__file__, "
            "'pitbridge':importlib.util.find_spec('pitbridge') is not None}))",
        )
        location = json.loads(probe.stdout)
        if not Path(location["origin"]).resolve().is_relative_to(installed):
            raise ValueError("Imported CreditVintage from outside its isolated environment")
        if location["pitbridge"]:
            raise ValueError("The core wheel unexpectedly installed the optional integration")
        run(str(python), "-I", "-m", "creditvintage", "--help")
        command("run", "--out", "evaluation")
        evaluation = command("verify", "evaluation")
        command("monitor", "--as-of", "2024-10-01", "--out", "monitor")
        command("verify-monitor", "monitor")
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
            "installation": "non-editable wheels; clean venv; cwd outside both source trees",
            "core_without_pitbridge": True,
            "evaluation": evaluation,
            "analysis": analysis,
            "selected_review_cases": selected,
            "accepted_review_events": final["events"],
            "retry_and_stale_version": "verified",
            "optional_lineage": lineage,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory with CreditVintage and PITBridge wheels"
    )
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))
