"""Probe interrupted initialization in the active installed package, without fitting."""

from __future__ import annotations

import contextlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from creditvintage import inputs
from creditvintage.cli import main

STAGES = (*inputs.INPUT_FILES, "case.json", "hash", "rename")


def check() -> dict[str, Any]:
    with TemporaryDirectory(prefix="creditvintage-installed-inputs-") as temporary:
        root = Path(temporary)
        baseline = root / "baseline"
        created = inputs.initialize_inputs(baseline)
        if created["data_kind"] != "synthetic" or created["model_fitted"] is not False:
            raise AssertionError("Input initialization changed its synthetic/model-free scope")
        original_bytes = {path.name: path.read_bytes() for path in baseline.iterdir()}
        if set(original_bytes) != {*inputs.INPUT_FILES, "case.json"}:
            raise AssertionError("Expected one complete five-file input case")
        original_records, original_json = inputs._write_records, inputs._json
        original_hash, original_rename = inputs._hash, Path.rename

        for index, stage in enumerate(STAGES):
            parent = root / str(index)
            parent.mkdir()
            destination = parent / "inputs"

            def write_records(
                directory: Path, filename: str, records: Any, *, phase: str = stage
            ) -> None:
                original_records(directory, filename, records)
                if filename == phase:
                    raise OSError("Interrupted installed input initialization")

            def write_json(path: Path, value: Any, *, phase: str = stage) -> None:
                original_json(path, value)
                if path.name == phase:
                    raise OSError("Interrupted installed input initialization")

            def file_hash(path: Path, *, phase: str = stage) -> str:
                if phase == "hash":
                    raise OSError("Interrupted installed input initialization")
                return original_hash(path)

            def rename(path: Path, target: Path, *, phase: str = stage) -> Path:
                if phase == "rename":
                    raise OSError("Interrupted installed input initialization")
                return original_rename(path, target)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.object(inputs, "_write_records", write_records),
                patch.object(inputs, "_json", write_json),
                patch.object(inputs, "_hash", file_hash),
                patch.object(Path, "rename", rename),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = main(["init-inputs", "--out", str(destination)])
            if (
                status != 2
                or stdout.getvalue()
                or "Interrupted installed input initialization" not in stderr.getvalue()
                or destination.exists()
                or list(parent.iterdir())
            ):
                raise AssertionError(f"Installed initialization left a partial case: {stage}")
            with contextlib.redirect_stdout(io.StringIO()):
                if main(["init-inputs", "--out", str(destination)]) != 0:
                    raise AssertionError(f"Installed initialization cannot retry: {stage}")
            diagnosis = inputs.check_inputs(destination)
            if (
                diagnosis["status"] != "ready_for_evaluation"
                or diagnosis["model_fitted"] is not False
            ):
                raise AssertionError(f"Retried installed inputs are not ready: {stage}")
            if {p.name: p.read_bytes() for p in destination.iterdir()} != original_bytes:
                raise AssertionError(f"Retry changed successful input bytes: {stage}")
            if list(parent.iterdir()) != [destination]:
                raise AssertionError("Temporary input files survived a successful retry")

        destination = root / "existing"
        original_json = inputs._json

        def create_existing(path: Path, value: Any) -> None:
            original_json(path, value)
            if path.name == "case.json":
                destination.mkdir()
                (destination / "sentinel.txt").write_bytes(b"preserve existing inputs")

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(inputs, "_json", create_existing),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = main(["init-inputs", "--out", str(destination)])
        if (
            status != 2
            or stdout.getvalue()
            or "not overwritten" not in stderr.getvalue()
            or list(destination.iterdir()) != [destination / "sentinel.txt"]
            or (destination / "sentinel.txt").read_bytes() != b"preserve existing inputs"
            or set(p.name for p in root.iterdir())
            != {"baseline", "existing", *(str(i) for i in range(len(STAGES)))}
        ):
            raise AssertionError("Installed publication replaced an existing input case")
    return {
        "status": "verified",
        "version": version("creditvintage-research"),
        "interrupted_stages": list(STAGES),
        "clean_retries": len(STAGES),
        "successful_files_identical": True,
        "existing_target_preserved": True,
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
