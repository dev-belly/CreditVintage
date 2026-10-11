from pathlib import Path

import pytest

from creditvintage.core import DataContractError, load_csv_bundle
from creditvintage.validation import read_csv_rows

TABLES = {
    "applications": (
        "application_id,decision_at,segment,principal_cents\n",
        'loan-1,2022-01-01,retail,"100000',
    ),
    "features": (
        "application_id,name,observed_at,available_at,value\n",
        'loan-1,utilization,2021-12-31,2021-12-31,"0.2',
    ),
    "performance": (
        "application_id,as_of,available_at,days_past_due\n",
        'loan-1,2022-07-01,2022-07-02,"0',
    ),
}


@pytest.mark.parametrize("malformed", list(TABLES))
def test_input_tables_reject_unterminated_quotes(tmp_path, malformed) -> None:
    paths = []
    for name, (header, row) in TABLES.items():
        path = tmp_path / f"{name}.csv"
        path.write_text(
            header + row + ("\n" if name == malformed else '"\n'), encoding="utf-8", newline=""
        )
        paths.append(path)
    with pytest.raises(DataContractError, match=rf"Malformed CSV quoting.*{malformed}.csv.*line 2"):
        load_csv_bundle(*paths)


def test_evidence_reader_rejects_unterminated_quote(tmp_path) -> None:
    path = tmp_path / "evidence.csv"
    path.write_text('id,value\nx,"0.2\n', encoding="utf-8", newline="")
    with pytest.raises(DataContractError, match="Malformed CSV quoting.*evidence.csv.*line 2"):
        read_csv_rows(path, ("id", "value"))


def test_valid_quoted_crlf_input_and_evidence_remain_supported(tmp_path) -> None:
    paths: list[Path] = []
    for name, (header, row) in TABLES.items():
        path = tmp_path / f"{name}.csv"
        path.write_bytes((header + row + '"\n').replace("\n", "\r\n").encode("utf-8"))
        paths.append(path)
    applications, features, performance = load_csv_bundle(*paths)
    assert applications[0].principal_cents == 100000
    assert features[0].value == 0.2
    assert performance[0].days_past_due == 0
    evidence = tmp_path / "evidence.csv"
    evidence.write_bytes(b'"id","value"\r\n"x","0.2"\r\n')
    assert read_csv_rows(evidence, ("id", "value")) == [{"id": "x", "value": "0.2"}]
