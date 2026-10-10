"""The standalone explorer must provide a complete, reproducible download."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZipFile

from creditvintage.core import DataContractError
from creditvintage.lineage import BUNDLE_FILES, verify_lineage, write_demo

try:
    import pitbridge.bundle
except ImportError:
    pitbridge = None


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.links.append(dict(attrs))


@unittest.skipIf(pitbridge is None, "PITBridge is an optional integration dependency")
class EvidenceDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name) / "generated"
        cls.summary = write_demo(cls.base, seed=7)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_download_link_targets_the_complete_verified_inventory(self) -> None:
        self.assertEqual(self.summary["explorer_version"], 2)
        parser = Links()
        parser.feed((self.base / "index.html").read_text(encoding="utf-8"))
        links = [link for link in parser.links if link.get("href") == "evidence.zip"]
        self.assertEqual(len(links), 1)
        self.assertIn("download", links[0])
        manifest = json.loads((self.base / "manifest.json").read_text(encoding="utf-8"))
        with ZipFile(self.base / "evidence.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 19)
            self.assertEqual(
                set(archive.namelist()),
                {f"lineage/{name}" for name in (*BUNDLE_FILES, "manifest.json")},
            )
            for name in (*BUNDLE_FILES, "manifest.json"):
                with self.subTest(file=name):
                    content = archive.read(f"lineage/{name}")
                    # The pinned PIT bundle keeps its original source bytes.
                    if not name.startswith("pit/"):
                        self.assertNotIn(b"\r\n", content)
                    self.assertEqual(content, (self.base / name).read_bytes())
                    if name != "manifest.json":
                        self.assertEqual(
                            hashlib.sha256(content).hexdigest(), manifest["files_sha256"][name]
                        )

    def test_extracted_download_replays_in_a_separate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            with ZipFile(self.base / "evidence.zip") as archive:
                archive.extractall(destination)
            checked = verify_lineage(destination / "lineage")
        self.assertTrue(checked["verified"] and checked["pit_replay"] and checked["model_replay"])
        self.assertEqual(checked["applications"], 480)
        self.assertEqual(checked["source_features"], 1440)

    def test_repeated_generation_preserves_bytes_and_excludes_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "repeated"
            destination.mkdir()
            (destination / "private-notes.txt").write_text("Not part of the evidence inventory.")
            self.assertEqual(write_demo(destination, seed=7), self.summary)
            self.assertEqual(
                (destination / "evidence.zip").read_bytes(),
                (self.base / "evidence.zip").read_bytes(),
            )
            with ZipFile(destination / "evidence.zip") as archive:
                for item in archive.infolist():
                    self.assertEqual(item.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(item.external_attr >> 16, 0o100644)

    def test_legacy_explorer_evidence_still_replays_without_the_new_download_link(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "legacy"
            shutil.copytree(self.base, destination)
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            del summary["explorer_version"]
            (destination / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            page = (destination / "index.html").read_text(encoding="utf-8")
            page = page.replace('<a href="evidence.zip" download>Complete evidence ZIP</a>', "")
            (destination / "index.html").write_text(page, encoding="utf-8", newline="")
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            for name in ("summary.json", "index.html"):
                manifest["files_sha256"][name] = hashlib.sha256(
                    (destination / name).read_bytes()
                ).hexdigest()
            (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (destination / "evidence.zip").unlink()
            self.assertTrue(verify_lineage(destination)["model_replay"])
            summary["explorer_version"] = 2
            (destination / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            manifest["files_sha256"]["summary.json"] = hashlib.sha256(
                (destination / "summary.json").read_bytes()
            ).hexdigest()
            (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataContractError, "explorer"):
                verify_lineage(destination)

    def test_invalid_explorer_versions_are_not_coerced(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "invalid"
            shutil.copytree(self.base, destination)
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            for version in (True, "2", 2.0, 3):
                with self.subTest(version=version):
                    summary["explorer_version"] = version
                    (destination / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                    manifest["files_sha256"]["summary.json"] = hashlib.sha256(
                        (destination / "summary.json").read_bytes()
                    ).hexdigest()
                    (destination / "manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(DataContractError, "Unsupported lineage"):
                        verify_lineage(destination)


if __name__ == "__main__":
    unittest.main()
