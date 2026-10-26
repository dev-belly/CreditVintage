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
