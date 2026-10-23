from datetime import date, datetime, timedelta

import pytest

from creditvintage.core import (
    REPORT_DAYS,
    Application,
    DataContractError,
    FeatureEvent,
    PerformanceEvent,
    snapshot,
)


@pytest.mark.parametrize("age", [0, 181, 182])
def test_snapshot_near_calendar_end_preserves_maturation_policy(age):
    decision = date.max - timedelta(days=age)
    application = Application("calendar-end", decision, 100_000, "retail")
    features = [
        FeatureEvent(application.application_id, name, value, decision, decision)
        for name, value in (
            ("utilization", 0.2),
            ("debt_to_income", 0.3),
            ("prior_delinquencies", 0),
        )
    ]
    reports = (
        [
            PerformanceEvent(
                application.application_id,
                decision + timedelta(days=day),
                decision + timedelta(days=day + 2),
                0,
            )
            for day in REPORT_DAYS
        ]
        if age == 182
        else []
    )
    row = snapshot([application], features, reports, date.max)[0]
    assert row.features == (0.2, 0.3, 0.0)
    assert (row.label, row.status) == (
        (0, "observed_nondefault") if age == 182 else (None, "not_mature")
    )


@pytest.mark.parametrize("cutoff", [None, "2025-01-01", datetime(2025, 1, 1)])
def test_snapshot_requires_calendar_date_even_for_empty_population(cutoff):
    with pytest.raises(DataContractError, match="cutoff must be a calendar date"):
        snapshot([], [], [], cutoff)
