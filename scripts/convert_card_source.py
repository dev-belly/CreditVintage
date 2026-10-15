"""Rebuild the bundled integer CSV from a separately downloaded original UCI ZIP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import zipfile
from pathlib import Path

from creditvintage.card_credit import COLUMNS, SOURCE_SHA256

ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
EXCEL_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"


def convert(archive: Path, output: Path) -> None:
    import pandas as pd

    if output.exists():
        raise FileExistsError(f"Choose a new output file: {output}")
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Original UCI ZIP digest differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        excel = source.read("default of credit card clients.xls")
    if hashlib.sha256(excel).hexdigest() != EXCEL_SHA256:
        raise ValueError("Original UCI Excel digest differs")
    table = pd.read_excel(io.BytesIO(excel), header=1, engine="xlrd")
    if table.shape != (30000, 25) or tuple(table.columns) != COLUMNS or table.isna().any().any():
        raise ValueError("Unexpected UCI source table")
    integers = table.astype("int64")
    if not (table == integers).all().all():
        raise ValueError("Conversion would discard fractional values")
    content = integers.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Canonical CSV differs; review the conversion before updating the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(content, mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.archive, args.output)
