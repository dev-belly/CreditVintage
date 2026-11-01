from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("release_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
VERSION = tomllib.loads((SCRIPT.parents[1] / "pyproject.toml").read_text())["project"]["version"]
COMMIT = "a" * 40


@pytest.fixture
def inputs(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / f"creditvintage_research-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"creditvintage_research-{VERSION}.dist-info/METADATA",
            f"Name: creditvintage-research\nVersion: {VERSION}\n",
        )
    with tarfile.open(dist / f"creditvintage_research-{VERSION}.tar.gz", "w:gz") as archive:
        content = f'[project]\nversion = "{VERSION}"\n'.encode()
        info = tarfile.TarInfo(f"creditvintage_research-{VERSION}/pyproject.toml")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    check = tmp_path / "checked.json"
    check.write_text(
        json.dumps(
            {
                "status": "verified",
                "version": VERSION,
                "input_initialization": {
                    "status": "verified",
                    "version": VERSION,
                    "interrupted_stages": [
                        "applications.csv",
                        "features.csv",
                        "performance.csv",
                        "cohorts.json",
                        "case.json",
                        "hash",
                        "rename",
                    ],
                    "clean_retries": 7,
                    "successful_files_identical": True,
                    "existing_target_preserved": True,
                },
                "card_credit": {"refitted": True},
                "sba_credit": {
                    "refitted": True,
                    "target": "recorded_chargeoff_within_24_calendar_months",
                    "rehashed_false_metric_rejected": True,
                },
            }
        )
    )
    return dist, check, tmp_path / "release"


def test_exact_version_commit_artifacts_and_preserved_existing_directory(inputs):
    dist, check, output = inputs
    (dist / "pitbridge-0.1.0-py3-none-any.whl").write_bytes(b"unrelated package")
    result = release.prepare(dist, check, output, COMMIT)
    assert result == {"status": "verified", "version": VERSION, "commit": COMMIT, "assets": 5}
    assert not (output / "pitbridge-0.1.0-py3-none-any.whl").exists()
    with pytest.raises(FileExistsError):
        release.prepare(dist, check, output, COMMIT)
    with pytest.raises(ValueError, match="commit"):
        release.verify(output, "b" * 40)
    manifest = output / "release-manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        release.verify(output, COMMIT)


def test_unverified_installed_wheel_cannot_become_a_release(inputs):
    dist, check, output = inputs
    check.write_text(json.dumps({"status": "failed", "card_credit": {"refitted": True}}))
    with pytest.raises(ValueError, match="must pass"):
        release.prepare(dist, check, output, COMMIT)
    assert not output.exists()


def test_changed_distribution_and_missing_exact_version_are_rejected(inputs):
    dist, check, output = inputs
    release.prepare(dist, check, output, COMMIT)
    wheel = next(output.glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest"):
        release.verify(output, COMMIT)
    next(dist.glob("*.whl")).unlink()
    with pytest.raises(ValueError, match="exactly one"):
        release.prepare(dist, check, output.parent / "second", COMMIT)


@pytest.mark.parametrize(
    "field,value",
    [("refitted", False), ("target", "180_day_default"), ("rehashed_false_metric_rejected", False)],
)
def test_release_requires_native_sba_refit_and_tamper_check(inputs, field, value):
    dist, check, output = inputs
    result = json.loads(check.read_text())
    result["sba_credit"][field] = value
    check.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="must pass"):
        release.prepare(dist, check, output, COMMIT)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("version", "0.8.1"),
        ("interrupted_stages", []),
        ("clean_retries", 0),
        ("successful_files_identical", False),
        ("existing_target_preserved", False),
    ],
)
def test_recovery_evidence_is_required_even_after_rehashing(inputs, field, value):
    dist, check, output = inputs
    valid = check.read_text()
    result = json.loads(valid)
    result["input_initialization"][field] = value
    check.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="input-initialization recovery"):
        release.prepare(dist, check, output, COMMIT)
    assert not output.exists()
    check.write_text(valid)
    release.prepare(dist, check, output, COMMIT)
    (output / "installed-check.json").write_text(json.dumps(result))
    manifest = json.loads((output / "release-manifest.json").read_text())
    manifest["files_sha256"]["installed-check.json"] = release.digest(
        output / "installed-check.json"
    )
    (output / "release-manifest.json").write_text(json.dumps(manifest))
    (output / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{release.digest(path)}  {path.name}\n"
            for path in sorted(output.iterdir())
            if path.name != "CHECKSUMS.sha256"
        )
    )
    with pytest.raises(ValueError, match="input-initialization recovery"):
        release.verify(output, COMMIT)


@pytest.mark.parametrize("field", ["version", "input_initialization"])
def test_release_rejects_missing_installed_version_or_recovery(inputs, field):
    dist, check, output = inputs
    result = json.loads(check.read_text())
    result.pop(field)
    check.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        release.prepare(dist, check, output, COMMIT)
    assert not output.exists()
