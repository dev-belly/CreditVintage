"""Deterministic synthetic event stream; never represented as bank data."""

from __future__ import annotations

import math
from datetime import date, timedelta
from random import Random
from typing import Literal

from creditvintage.core import REPORT_DAYS, Application, FeatureEvent, PerformanceEvent


def generate(
    seed: int = 20260927, per_vintage: int = 80
) -> tuple[list[Application], list[FeatureEvent], list[PerformanceEvent]]:
    if per_vintage < 10:
        raise ValueError("At least ten loans per vintage are required")
    rng = Random(seed)
    months = [(2022, m) for m in range(1, 13)]
    months += [(2023, m) for m in range(7, 13)]
    months += [(2024, m) for m in range(7, 13)]
    applications: list[Application] = []
    features: list[FeatureEvent] = []
    reports: list[PerformanceEvent] = []
    for year, month in months:
        decision = date(year, month, 1)
        for index in range(per_vintage):
            app_id = f"CV{year}{month:02d}-{index:04d}"
            segment: Literal["retail", "small_business"] = (
                "small_business" if rng.random() < 0.28 else "retail"
            )
            principal_cents = rng.randint(200_000, 3_000_000)
            utilization = round(rng.uniform(0.03, 0.97), 6)
            dti = round(rng.uniform(0.08, 0.8), 6)
            prior = rng.choices([0, 1, 2, 3], weights=[65, 24, 8, 3])[0]
            applications.append(Application(app_id, decision, principal_cents, segment))
            for name, value in (
                ("utilization", utilization),
                ("debt_to_income", dti),
                ("prior_delinquencies", float(prior)),
            ):
                features.append(
                    FeatureEvent(
                        app_id,
                        name,
                        value,
                        decision - timedelta(days=5),
                        decision - timedelta(days=1),
                    )
                )
                if index % 10 == 0:
                    # A later revision must never alter the decision-time feature.
                    features.append(
                        FeatureEvent(
                            app_id,
                            name,
                            value * 0.5 if name != "prior_delinquencies" else 0.0,
                            decision,
                            decision + timedelta(days=10),
                        )
                    )
            logit = (
                -3.45
                + 1.9 * utilization
                + 1.3 * dti
                + 0.6 * prior
                + (0.32 if segment == "small_business" else 0)
                + 0.16 * (year - 2022)
            )
            probability = 1 / (1 + math.exp(-logit))
            default_month = rng.randint(3, 6) if rng.random() < probability else None
            for report_month, day in enumerate(REPORT_DAYS, 1):
                if default_month is None:
                    dpd = 30 if rng.random() < 0.06 else 0
                elif report_month >= default_month:
                    dpd = 90
                elif report_month == default_month - 1:
                    dpd = 60
                elif report_month == default_month - 2:
                    dpd = 30
                else:
                    dpd = 0
                report_day = decision + timedelta(days=day)
                reports.append(
                    PerformanceEvent(app_id, report_day, report_day + timedelta(days=2), dpd)
                )
    return applications, features, reports
