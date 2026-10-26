"""Frozen, observed SBA loans with explicit dates and native charge-off outcomes."""

from __future__ import annotations

import calendar
import csv
import gzip
import hashlib
import io
import json
import math
import re
import zlib
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from typing import Any

from creditvintage.core import DataContractError, parse_calendar_date

SOURCE_URL = "https://data.sba.gov/dataset/7a-504-foia"
DOWNLOAD_ROOT = "https://data.sba.gov/sites/default/files/uploaded_resources/"
SOURCES = (
    (
        "00",
        "FOIA_7a_FY2000_FY2009_asof_260630.csv",
        "66674e18a700fbba0378c25118c291ac2759784a550c643cab5747eced763d6a",
        690333,
    ),
    (
        "10",
        "FOIA_7a_FY2010_FY2019_asof_260630.csv",
        "01a3e2c7988a6f4052e53f218a309feb2ec2fe42887bebdc0fa94ac8b1024ade",
        545751,
    ),
)
AS_OF = date(2026, 6, 30)
WINDOWS = {
    "train": (date(2006, 1, 1), date(2010, 12, 31), 20000),
    "calibration": (date(2013, 1, 1), date(2013, 12, 31), 5000),
    "test": (date(2016, 1, 1), date(2017, 12, 31), 10000),
}
# Populated only by the reviewed, hash-checked source conversion.
DATA_SHA256 = "08c60deb116345624e75cdbf22ea305be33ebeba17d2d2c11292a5e3ba1da8f1"
DATA_SIZE = 6637900
PROTOCOL_SHA256 = "99194b3f83ec07b94ef06b619c0e0230e7658b7afdd8d31c35da63275b39e829"
AUDIT_SHA256 = "51ff0a447d0b489f8a25de82379042e9345a243fee2f84a99b28403fbf67a6f6"


@dataclass(frozen=True)
class SbaLoan:
    record_id: str
    borrower_group_sha256: str
    source_file: str
    source_record: int
    stage: str
    approval_date: date
    funded_at: date
    horizon_end: date
    gross_approval_usd: str
    sba_guaranteed_usd: str
    initial_interest_rate_pct: str
    processing_method: str
    source_loan_status: str
    paid_in_full_at: date | None
    charged_off_at: date | None
    chargeoff_24m: int


COLUMNS = tuple(SbaLoan.__dataclass_fields__)


def anniversary(value: date) -> date:
    """The inclusive 24-calendar-month boundary, with month-end clamping."""
    return value.replace(
        year=value.year + 2, day=min(value.day, calendar.monthrange(value.year + 2, value.month)[1])
    )


def borrower_key(name: str, state: str, zipcode: str) -> str:
    """An imperfect recorded-identity heuristic; no claim of legal entity matching."""
    normalized = [re.sub(r"[^A-Z0-9]", "", value.upper()) for value in (name, state, zipcode)]
    if not all(normalized):
        raise ValueError("missing_borrower_group")
    return hashlib.sha256("|".join(normalized).encode("ascii")).hexdigest()


def _money(value: str, *, positive: bool = False) -> str:
    try:
        number = Decimal(value.strip())
        if not number.is_finite() or number < 0 or (positive and number == 0):
            raise ValueError("invalid_money")
        if number != number.quantize(Decimal("0.01")):
            raise ValueError("invalid_money")
        return f"{number:.2f}"
    except (InvalidOperation, ArithmeticError) as exc:
        raise ValueError("invalid_money") from exc


def _date(value: str) -> date | None:
    return parse_calendar_date(value.strip(), "SBA date") if value.strip() else None


def convert_record(row: dict[str, str], source: str, ordinal: int) -> SbaLoan:
    """Convert one original logical CSV row; exclusions are explicit ValueErrors."""
    if row["AsOfDate"].strip() != AS_OF.isoformat() or row["Program"].strip() != "7A":
        raise DataContractError("SBA source program or snapshot version differs")
    try:
        funded, approved = _date(row["FirstDisbursementDate"]), _date(row["ApprovalDate"])
        paid, charged = _date(row["PaidInFullDate"]), _date(row["ChargeOffDate"])
    except DataContractError as exc:
        raise ValueError("invalid_date") from exc
    if funded is None or approved is None:
        raise ValueError("missing_funding_or_approval")
    if funded > AS_OF or approved > funded:
        raise ValueError("invalid_date_order")
    stage = next((key for key, (low, high, _) in WINDOWS.items() if low <= funded <= high), None)
    if stage is None:
        raise ValueError("outside_frozen_windows")
    end = anniversary(funded)
    if end > AS_OF:
        raise ValueError("immature_horizon")
    status = "".join(row["LoanStatus"].split())
    if status not in {"PIF", "CHGOFF", "EXEMPT"}:
        raise ValueError("unusable_loan_status")
    if status == "CHGOFF" and charged is None:
        raise ValueError("chargeoff_date_missing")
    if any(d is not None and (d < funded or d > AS_OF) for d in (paid, charged)):
        raise ValueError("invalid_outcome_date")
    gross = _money(row["GrossApproval"], positive=True)
    guarantee = _money(row["SBAGuaranteedApproval"])
    if Decimal(guarantee) > Decimal(gross):
        raise ValueError("guarantee_exceeds_gross")
    rate = row["InitialInterestRate"].strip()
    if rate:
        try:
            number = float(rate)
        except ValueError as exc:
            raise ValueError("invalid_initial_rate") from exc
        if not math.isfinite(number) or not 0 <= number <= 100:
            raise ValueError("invalid_initial_rate")
        rate = format(number, ".12g")
    method = row["ProcessingMethod"].strip()
    if not method:
        raise ValueError("missing_processing_method")
    group = borrower_key(row["BorrName"], row["BorrState"], row["BorrZip"])
    return SbaLoan(
        f"SBA{source}-{ordinal:07d}",
        group,
        source,
        ordinal,
        stage,
        approved,
        funded,
        end,
        gross,
        guarantee,
        rate,
        method,
        status,
        paid,
        charged,
        int(charged is not None and charged <= end),
    )


def portable_row(loan: SbaLoan) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, date) else "" if value is None else value
        for key, value in asdict(loan).items()
    }


def canonical_csv(loans: list[SbaLoan]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(portable_row(loan) for loan in loans)
    return stream.getvalue().encode("utf-8")


def resource(name: str) -> bytes:
    return files("creditvintage").joinpath("data/sba_credit", name).read_bytes()


def load_sba(compressed: bytes | None = None) -> list[SbaLoan]:
    try:
        with gzip.GzipFile(
            fileobj=io.BytesIO(resource("source.csv.gz") if compressed is None else compressed)
        ) as handle:
            raw = handle.read(DATA_SIZE + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise DataContractError("Invalid frozen SBA gzip") from exc
    if len(raw) != DATA_SIZE or hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise DataContractError("Frozen SBA source digest differs")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise DataContractError("SBA source columns differ")
    loans = []
    for row in reader:
        loans.append(
            SbaLoan(
                row["record_id"],
                row["borrower_group_sha256"],
                row["source_file"],
                int(row["source_record"]),
                row["stage"],
                parse_calendar_date(row["approval_date"], "approval"),
                parse_calendar_date(row["funded_at"], "funding"),
                parse_calendar_date(row["horizon_end"], "horizon"),
                row["gross_approval_usd"],
                row["sba_guaranteed_usd"],
                row["initial_interest_rate_pct"],
                row["processing_method"],
                row["source_loan_status"],
                _date(row["paid_in_full_at"]),
                _date(row["charged_off_at"]),
                int(row["chargeoff_24m"]),
            )
        )
    validate_cohorts(loans)
    return loans


def validate_cohorts(loans: list[SbaLoan]) -> None:
    if len({loan.record_id for loan in loans}) != len(loans):
        raise DataContractError("Duplicate SBA record")
    if len({loan.borrower_group_sha256 for loan in loans}) != len(loans):
        raise DataContractError("SBA borrower group overlaps")
    for stage, (low, high, size) in WINDOWS.items():
        cohort = [loan for loan in loans if loan.stage == stage]
        if len(cohort) != size or {loan.chargeoff_24m for loan in cohort} != {0, 1}:
            raise DataContractError("SBA cohort size or outcomes differ")
        for loan in cohort:
            if (
                not low <= loan.funded_at <= high
                or loan.approval_date > loan.funded_at
                or loan.horizon_end != anniversary(loan.funded_at)
                or loan.horizon_end > AS_OF
                or loan.chargeoff_24m
                != int(loan.charged_off_at is not None and loan.charged_off_at <= loan.horizon_end)
            ):
                raise DataContractError("SBA cohort date or target violates protocol")
    if any(loan.stage not in WINDOWS for loan in loans):
        raise DataContractError("Unknown SBA cohort")
    if (
        max(loan.horizon_end for loan in loans if loan.stage == "train")
        >= WINDOWS["calibration"][0]
        or max(loan.horizon_end for loan in loans if loan.stage == "calibration")
        >= WINDOWS["test"][0]
    ):
        raise DataContractError("SBA outcome horizons overlap the next stage")


def frozen_json(name: str, expected: str) -> dict[str, Any]:
    raw = resource(name)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise DataContractError(f"Frozen SBA {name} differs")
    result: dict[str, Any] = json.loads(raw)
    return result
