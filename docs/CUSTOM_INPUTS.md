# Run on your own calendar

The event schema stays deliberately narrow. You need application-time versions
of `utilization`, `debt_to_income` and `prior_delinquencies`, principal in integer
USD cents, one of the two documented segments, and actual dated DPD reports.
Changing cohort dates does not turn an unrelated table into this event contract.
Do not invent dates, segments or DPD anchors to force a public dataset into it.

## 0. Start from complete CSVs

Version 0.3.0 creates a complete, editable example from an installed package:

```text
creditvintage init-inputs --out inputs/example
creditvintage check-inputs inputs/example
```

Choose a new directory: an existing file or folder is never overwritten. The
example has 480 synthetic applications, 1,584 feature records and 2,880
performance records, with 240/120/120 training/calibration/test applications.
It writes `applications.csv`, `features.csv`, `performance.csv`, `cohorts.json`
and `case.json`. The last file records the original synthetic example and file
hashes; it does not authenticate replacement inputs. Keep its synthetic origin
clear when sharing the generated case. The evaluator describes CSV ingestion as
`user_supplied`; that describes how inputs arrived, not whether they are observed.

Replace all three CSVs and edit the seven dates for your own compatible event
records. The checker reads these four named input files from the supplied
directory and reuses the actual record types, snapshot logic and evaluation
gates. It first collects row/header/type failures across all files. When rows
are valid, it checks duplicate IDs/revisions, orphan references, report dates
and missing decision-time features. When those checks and configuration pass,
it locates incomplete outcomes across all three cohorts and checks minimum
size and both outcomes. Later checks are marked `blocked` when earlier failures
prevent a trustworthy evaluation; they are not silently reported as passed.

The commands were introduced in `v0.3.0`. The malformed-header fix is available
from `v0.3.1`: an unterminated quote in a CSV header returns `csv_quoting` JSON and exit
code 2, and the checker continues inspecting the other files. The older
`v0.3.0` raises a parsing error for that case; its assets are preserved. See the
[patch and upgrade notes](releases/v0.3.1.md) for installation and scope.

The diagnostic JSON includes `model_fitted: false`, input row counts, completed/blocked
checks, issue codes, file names and physical CSV line numbers. A line identifies
the end of a CSV record, including multi-line records; configuration and cohort
errors and unreadable files/encodings can have a null line. Valid input returns exit code 0 and the actual
`preflight` result. Invalid data returns exit code 2 and JSON on stdout. Argument
syntax errors still use the normal CLI error path.

`--max-issues 20` limits displayed issues, not the reported total; the default
is 100 and the allowed range is 1–1,000. The checker does not repair, fill,
drop or rewrite input records. It performs no fitting or filesystem writes.
Domain validation can report the first violated rule within an individual row;
the tool is not a complete data-quality or source-authenticity audit.

Run and verify the same files after a successful check:

```text
creditvintage run --applications inputs/example/applications.csv --features inputs/example/features.csv --performance inputs/example/performance.csv --config inputs/example/cohorts.json --out outputs/csv-example
creditvintage verify outputs/csv-example
```

For intentionally unlabelled current cohorts, use `preflight --as-of YYYY-MM-DD`
from v0.4.0 and the monitoring workflow below. `check-inputs` continues to diagnose
a complete mature evaluation; its maturity failures do not reject monitoring use.

## 1. Make all dates explicit

On current `main`, each `principal_cents` is an integer in `[1, 2^63-1]`, matching
the SQLite storage used by the business workflow. Cents remain exact integers;
values above `2^53` are not rounded through a float at ingestion. Portfolio
total cents still have a separate int64 check before SQL aggregation.
Unsupported amounts are diagnosed as `record_contract` at their CSV physical
line before readiness or fitting. Enormous integer feature values and config
fractions also raise ordinary contract errors rather than binary64 overflows.
The released v0.8.0 assets predate these validation changes; install current
source to use them. Existing supported inputs and saved report values are
unchanged.

Use exact `YYYY-MM-DD` calendar dates throughout the CSVs, the JSON configuration
and the monitoring `--as-of` argument. For example, use `2024-10-01`, not
`20241001` or the ISO week date `2024-W40-2`. Timestamps, surrounding spaces and
impossible dates are also invalid. Version `v0.3.2` and later apply
the same strict parser to these inputs and to saved report evidence.
`check-inputs` reports alternate date forms as `value_type`, including their
file, physical line and column; it does not silently normalize them. These are
row errors, so relationship and evaluation checks remain `blocked` until they
are fixed. Published `v0.3.1` wheels predate this change and still accept compact
and week-date forms in CSVs and `--as-of`; their JSON configuration was already
strict. [Upgrade to v0.4.0](releases/v0.4.0.md) for consistent checks in the installed
package. Existing reports generated by the package use `YYYY-MM-DD` and remain
compatible. Correct alternate forms explicitly in source inputs before rerunning
evaluation; do not rewrite saved evidence or its hashes to make verification pass.

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

## 4. Before outcomes mature

For current applications without mature outcomes, use `monitor` with the same
configuration and an explicit `--as-of`:

```text
creditvintage preflight --applications applications.csv --features features.csv --performance performance.csv --config cohorts.json --as-of 2024-10-01
creditvintage monitor --applications applications.csv --features features.csv --performance performance.csv --config cohorts.json --as-of 2024-10-01 --out outputs/early-private
```

Train/calibration still need their mature outcomes. The current monitoring
cohort does not. With `--as-of`, the v0.4.0 preflight shares the actual monitor's
cutoff, label scope and population gates, and the fitting path's historical
cohort selection. It fits no model and writes no files. Success returns JSON
with schema `creditvintage-monitor-preflight-v1`, status `ready_for_monitoring`
and `model_fitted: false`; failure returns exit code 2 and a contract message on
stderr. It stops at the first failure, unlike the evaluation-oriented row
diagnostics in `check-inputs`.

The cutoff must be on or after `test_start`. Historical training and calibration
still require at least 20 records, both outcomes, complete mature labels and
decision-time features. The current cohort requires at least 20 applications
visible by the cutoff, using the configured test window; it has no outcome-count
requirement. Current outcome rows cannot change readiness, although all supplied
rows remain represented in the source digest and raw input counts. CSV syntax
and record types are checked on ingestion, including rows unused by monitoring.

For the 480-application synthetic starter at `2024-10-01`:

| Population | Applications | Outcomes required? |
| --- | ---: | --- |
| Training | 240 | Yes, visible by calibration start |
| Calibration / reference | 120 | Yes, visible by test start |
| Current and visible | 80 | No |
| Current but not yet visible | 40 | Excluded until their decision dates |

The JSON reports `current_applications_not_yet_visible` separately from
`applications_outside_windows`. Its current cohort has no event/non-event counts,
scores or PSI. Readiness checks cannot establish source authenticity, guarantee
successful numerical fitting or prove that reference scores will have enough
variation for PSI. The numerical monitor performs those later checks.

To reproduce the missing-current-label case from the current installed source
checkout (the example was added after the v0.4.0 distributions were built):

```text
python examples/monitoring_without_outcomes.py --out outputs/unlabelled-case
creditvintage preflight --applications outputs/unlabelled-case/inputs/applications.csv --features outputs/unlabelled-case/inputs/features.csv --performance outputs/unlabelled-case/inputs/performance.csv --config outputs/unlabelled-case/inputs/cohorts.json --as-of 2024-10-01
creditvintage verify-monitor outputs/unlabelled-case/report
```

The script creates a new synthetic starter, removes only the current cohort's
performance rows using the configured decision dates, saves `preflight.json`,
then fits and verifies the actual monitor. Expect 240 training, 120 calibration,
80 visible current and 40 future current applications, with zero current outcome
rows. Open `outputs/unlabelled-case/report/index.html`. The script refuses an
existing output directory. Only the preflight step is model-free; the subsequent
monitor intentionally fits the historical model. This is a local demonstration,
not evidence of an independent user's trial.

Omit `--as-of` to retain the original complete mature-evaluation preflight.

From v0.5.0, [model-free feature monitoring](FEATURE_MONITORING.md) is another
option when the task is to inspect input distributions. It reads only
`applications.csv` and `features.csv`, so neither historical nor current outcome
records are needed. It keeps the same strict decision-time feature contract;
missing-feature rates are not estimated by dropping incomplete applications.

```text
creditvintage monitor-features --applications applications.csv --features features.csv --config cohorts.json --as-of 2024-10-01 --out outputs/input-distributions
creditvintage verify-feature-monitor outputs/input-distributions
```

Keep application-level inputs, preflight output and exported private reports
within the appropriate access boundary.

For valid event records with absent or late decision-time features, v0.6.0 adds
[feature-coverage diagnostics](FEATURE_QUALITY.md). It accepts only the application
and feature CSV paths and retains each expected slot instead of discarding the
incomplete application. It diagnoses coverage and visible timing; it does not
relax the existing scoring/distribution gates or accept malformed feature values.

