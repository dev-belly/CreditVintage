# Run on your own calendar

The event schema stays deliberately narrow. You need application-time versions
of `utilization`, `debt_to_income` and `prior_delinquencies`, principal in integer
USD cents, one of the two documented segments, and actual dated DPD reports.
Changing cohort dates does not turn an unrelated table into this event contract.
Do not invent dates, segments or DPD anchors to force a public dataset into it.

## 1. Make all dates explicit

Copy [examples/cohort-config.json](../examples/cohort-config.json) and edit all
seven dates. This complete synthetic example configuration is:

```json
{
  "train_start": "2022-01-01",
  "train_end": "2022-12-31",
  "calibration_start": "2023-07-01",
  "calibration_end": "2023-12-31",
  "test_start": "2024-07-01",
  "test_end": "2024-12-31",
  "evaluation_as_of": "2025-07-01",
  "scenario_lgd": 0.45,
  "review_fraction": 0.1,
  "seed": 20260927
}
```

Dates must be canonical `YYYY-MM-DD`. All seven are required in a supplied
configuration so a forgotten date cannot silently come from the demonstration.
`scenario_lgd`, `review_fraction` and `seed` are optional and retain their shown
defaults. JSON duplicate/unknown keys, NaN/infinity, boolean fractions and
invalid seed types fail. An explicit CLI `--seed` takes precedence over JSON.
Without `--config`, existing commands retain their original defaults.

The same 180-day outcome plus two-day reporting lag applies. The final training
application must mature before calibration starts; calibration must mature
before test; the final test application must mature by `evaluation_as_of`.
These checks cannot be disabled with configuration. This is still an
operational 90+ DPD research target, not an audited regulatory default definition.

## 2. Check a mature evaluation before fitting

Use one line on Linux/macOS or Windows PowerShell:

```text
creditvintage preflight --applications applications.csv --features features.csv --performance performance.csv --config cohorts.json
```

On Windows an unactivated environment can use
`.\.venv\Scripts\creditvintage.exe` in place of `creditvintage`.

Successful JSON contains `status: ready_for_evaluation`, `model_fitted: false`,
the actual resolved configuration, an event digest, row counts and the three
cohort counts. `applications_outside_windows` includes gap-period and any other
excluded applications. For example, moving the default synthetic training start
to `2022-02-01` reports **880 training applications and 80 excluded**, instead
of silently making those January applications disappear.

Each evaluated stage needs at least 20 applications and both outcomes. Missing
decision-time features, incomplete non-event coverage, ambiguous revisions or
unmatured labels fail with a nonzero exit status. No model is fitted and no
output directory is created. This check validates the supplied event contract;
it cannot authenticate source timestamps, establish feature correctness or
predict model quality. It stops at the first contract failure rather than
pretending to provide a complete data-quality audit.

## 3. Use the same file for the actual experiment

```text
creditvintage run --applications applications.csv --features features.csv --performance performance.csv --config cohorts.json --out outputs/private
creditvintage verify outputs/private
creditvintage benchmark --applications applications.csv --features features.csv --performance performance.csv --config cohorts.json --out outputs/private-comparison
creditvintage verify-benchmark outputs/private-comparison
```

The resolved config is recorded in the resulting evidence. The synthetic
generator retains its original dates; changing JSON does not relocate synthetic
events. Regression tests additionally shift complete event data four years
earlier and exercise evaluation, comparison and monitoring on that calendar.

For current applications without mature outcomes, use `monitor` with the same
configuration and an explicit `--as-of`:

```text
creditvintage monitor --applications applications.csv --features features.csv --performance performance.csv --config cohorts.json --as-of 2024-10-01 --out outputs/early-private
```

Train/calibration still need their mature outcomes. The current monitoring
cohort does not. `preflight` intentionally checks a complete mature evaluation;
it should not be used to reject an intentionally unlabelled monitoring cohort.
Keep application-level inputs, preflight output and exported private reports
within the appropriate access boundary.
