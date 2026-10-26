"""Rebuild the de-identified 35,000-row sample from two pinned official SBA files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from creditvintage.core import DataContractError
from creditvintage.sba_source import (
    DOWNLOAD_ROOT,
    SOURCES,
    WINDOWS,
    SbaLoan,
    canonical_csv,
    convert_record,
    resource,
)


def prepare(source_dir: Path, destination: Path, *, download: bool = False) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Choose a new destination: {destination}")
    source_dir.mkdir(parents=True, exist_ok=True)
    for _, name, digest, _ in SOURCES:
        path = source_dir / name
        if not path.exists() and download:
            urllib.request.urlretrieve(DOWNLOAD_ROOT + name, path)
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Official source digest differs: {name}")
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    earliest: dict[str, tuple[str, str]] = {}
    source_counts: dict[str, int] = {}
    # No outcome is used in deduplication or sampling priority.
    for source, name, _, expected_rows in SOURCES:
        count = 0
        with (source_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            for ordinal, row in enumerate(csv.DictReader(handle), 1):
                count += 1
                try:
                    loan = convert_record(row, source, ordinal)
                except DataContractError:
                    raise
                except ValueError as exc:
                    excluded[str(exc)] += 1
                    continue
                counts[loan.stage] += 1
                key = (loan.funded_at.isoformat(), loan.record_id)
                group = loan.borrower_group_sha256
                if group not in earliest or key < earliest[group]:
                    earliest[group] = key
        if count != expected_rows:
            raise ValueError(f"Official logical row count differs: {name}")
        source_counts[name] = count
    heaps: dict[str, list[tuple[int, str, SbaLoan]]] = {stage: [] for stage in WINDOWS}
    group_counts: Counter[str] = Counter()
    for source, name, digest, _ in SOURCES:
        with (source_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            for ordinal, row in enumerate(csv.DictReader(handle), 1):
                try:
                    loan = convert_record(row, source, ordinal)
                except DataContractError:
                    raise
                except ValueError:
                    continue
                if earliest[loan.borrower_group_sha256] != (
                    loan.funded_at.isoformat(),
                    loan.record_id,
                ):
                    continue
                group_counts[loan.stage] += 1
                priority = int(hashlib.sha256(f"{digest}:{ordinal}".encode()).hexdigest(), 16)
                heap = heaps[loan.stage]
                item = (-priority, loan.record_id, loan)
                size = WINDOWS[loan.stage][2]
                if len(heap) < size:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    loans = []
    for stage, (_, _, size) in WINDOWS.items():
        if len(heaps[stage]) != size:
            raise ValueError("Insufficient eligible SBA loans")
        loans.extend(
            sorted(
                (entry[2] for entry in heaps[stage]),
                key=lambda loan: (loan.funded_at, loan.record_id),
            )
        )
    raw = canonical_csv(loans)
    audit = {
        "schema": "creditvintage-sba-conversion-v1",
        "source_logical_rows": source_counts,
        "excluded_by_reason": dict(sorted(excluded.items())),
        "eligible_by_stage_before_groups": dict(counts),
        "earliest_groups_by_stage": dict(group_counts),
        "selected_by_stage": {stage: WINDOWS[stage][2] for stage in WINDOWS},
        "converted_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "converted_csv_bytes": len(raw),
        "borrower_group_overlap": len(loans) - len({x.borrower_group_sha256 for x in loans}),
    }
    destination.mkdir()
    (destination / "source.csv.gz").write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    (destination / "protocol.json").write_bytes(resource("protocol.json"))
    (destination / "audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline=""
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-frozen", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source_dir, args.out, download=args.download)
    if args.verify_frozen:
        for name in ("source.csv.gz", "protocol.json", "audit.json"):
            actual = (args.out / name).read_bytes()
            expected = resource(name)
            if name.endswith(".gz"):
                actual, expected = gzip.decompress(actual), gzip.decompress(expected)
            if actual != expected:
                raise ValueError(f"Rebuilt SBA {name} differs from the frozen conversion")
    print(json.dumps({"status": "verified" if args.verify_frozen else "converted", **result}))
