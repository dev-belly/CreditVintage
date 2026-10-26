"""Prepare checked distribution files and verify their exact commit/version/digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    distributions: Path, check_result: Path, destination: Path, commit: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Release requires an exact lowercase 40-character commit SHA")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected a stable three-part project version")
    result = json.loads(check_result.read_text(encoding="utf-8"))
    if (
        result.get("status") != "verified"
        or result.get("card_credit", {}).get("refitted") is not True
        or result.get("sba_credit", {}).get("refitted") is not True
        or result.get("sba_credit", {}).get("target")
        != "recorded_chargeoff_within_24_calendar_months"
        or result.get("sba_credit", {}).get("rehashed_false_metric_rejected") is not True
    ):
        raise ValueError(
            "Installed distribution and card-data refitting must pass before packaging"
        )
    wheels = list(distributions.glob(f"creditvintage_research-{version}-*.whl"))
    archive = distributions / f"creditvintage_research-{version}.tar.gz"
    if len(wheels) != 1 or not archive.is_file():
        raise ValueError("Expected exactly one current-version wheel and source distribution")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as contents:
        metadata = [name for name in contents.namelist() if name.endswith(".dist-info/METADATA")]
        if (
            len(metadata) != 1
            or BytesParser().parsebytes(contents.read(metadata[0]))["Version"] != version
        ):
            raise ValueError("Wheel metadata version differs from the source version")
    with tarfile.open(archive) as contents:
        member = contents.extractfile(f"creditvintage_research-{version}/pyproject.toml")
        if member is None or tomllib.loads(member.read().decode())["project"]["version"] != version:
            raise ValueError("Source-distribution version differs from the source version")
    if destination.exists():
        raise FileExistsError(f"Choose a new release directory: {destination}")
    destination.mkdir(parents=True)
    for path in (wheel, archive):
        shutil.copyfile(path, destination / path.name)
    shutil.copyfile(check_result, destination / "installed-check.json")
    manifest = {
        "schema": "creditvintage-checked-release-v1",
        "version": version,
        "commit": commit,
        "files_sha256": {
            name: digest(destination / name)
            for name in (wheel.name, archive.name, "installed-check.json")
        },
    }
    (destination / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )
    (destination / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in sorted(destination.iterdir())),
        encoding="utf-8",
        newline="",
    )
    return verify(destination, commit)


def verify(directory: Path, commit: str) -> dict[str, Any]:
    manifest = json.loads((directory / "release-manifest.json").read_text(encoding="utf-8"))
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    expected = {
        f"creditvintage_research-{version}-py3-none-any.whl",
        f"creditvintage_research-{version}.tar.gz",
        "installed-check.json",
    }
    if (
        manifest.get("schema") != "creditvintage-checked-release-v1"
        or manifest.get("commit") != commit
        or manifest.get("version") != version
        or set(manifest.get("files_sha256", {})) != expected
    ):
        raise ValueError("Release commit, version or distribution set differs")
    if manifest["files_sha256"] != {name: digest(directory / name) for name in expected}:
        raise ValueError("Release file digest differs")
    if {p.name for p in directory.iterdir()} != expected | {
        "release-manifest.json",
        "CHECKSUMS.sha256",
    }:
        raise ValueError("Unexpected release file set")
    wanted = "".join(
        f"{digest(directory / name)}  {name}\n"
        for name in sorted(expected | {"release-manifest.json"})
    )
    if (directory / "CHECKSUMS.sha256").read_text(encoding="utf-8") != wanted:
        raise ValueError("Release checksum list differs")
    check = json.loads((directory / "installed-check.json").read_text(encoding="utf-8"))
    if (
        check.get("status") != "verified"
        or check.get("card_credit", {}).get("refitted") is not True
        or check.get("sba_credit", {}).get("refitted") is not True
        or check.get("sba_credit", {}).get("target")
        != "recorded_chargeoff_within_24_calendar_months"
        or check.get("sba_credit", {}).get("rehashed_false_metric_rejected") is not True
    ):
        raise ValueError("Release lacks installed-distribution refit evidence")
    return {"status": "verified", "version": version, "commit": commit, "assets": 5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("distributions", type=Path)
    build.add_argument("check_result", type=Path)
    build.add_argument("destination", type=Path)
    build.add_argument("--commit", required=True)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = (
        prepare(args.distributions, args.check_result, args.destination, args.commit)
        if args.command == "prepare"
        else verify(args.directory, args.commit)
    )
    print(json.dumps(result, indent=2))
