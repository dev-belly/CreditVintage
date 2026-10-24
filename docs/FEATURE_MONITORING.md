# Inspect input distributions without outcome labels

Available in **v0.5.0**. `monitor-features` compares the inputs that were available
at each application's decision date. Unlike `monitor`, it does not fit a model
and does not open a performance/outcome file, including for the reference cohort.

## Run the synthetic example

Install the [checked v0.5.0 wheel](releases/v0.5.0.md) or the current source in a
Python 3.12+ virtual environment, then run:

```text
creditvintage monitor-features --as-of 2024-10-01 --out outputs/feature-monitor
creditvintage verify-feature-monitor outputs/feature-monitor
```

Open `outputs/feature-monitor/index.html`. The default generated example has
480 reference applications, 320 visible current applications and 160 current
applications after the cutoff. These are synthetic records. The report compares
the three named features, principal in cents and the two customer segments.

## Supply only application and feature CSVs

Use the same [application/feature columns and units](CUSTOM_INPUTS.md) as the
existing event workflow and the same seven-date cohort configuration:

```text
creditvintage init-inputs --out inputs/example
creditvintage monitor-features --applications inputs/example/applications.csv --features inputs/example/features.csv --config inputs/example/cohorts.json --as-of 2024-10-01 --out outputs/custom-feature-monitor
creditvintage verify-feature-monitor outputs/custom-feature-monitor
```

The starter is a separate 480-application synthetic case: expect 120 reference,
80 visible current and 40 future current applications at this cutoff. The command
does not use the starter's `performance.csv`; that file can be absent. Both CSV
paths must be supplied together. Input files are read without modification.
Supplied tables are marked `user_supplied`, including copies of the starter;
the command does not authenticate their provenance.

For a virtual environment named `.venv`, substitute `.venv/bin/python -m
creditvintage` on Linux/macOS or `.\.venv\Scripts\python.exe -m creditvintage`
on Windows when the command is not on your PATH.

## Population and time contract

- Reference decisions fall in `calibration_start` through `calibration_end`.
  This reuses the existing calendar but requires no calibration outcomes.
- Current decisions fall in `test_start` through `test_end`, restricted to dates
  on or before `--as-of`. The cutoff must be on or after `test_start`.
- Both populations need at least 20 applications. Every monthly current slice
  uses the same full reference population. Slices below 20 retain their counts
  with `status=insufficient_sample` and `distances=null`.
- `snapshot` selects the latest feature revision available by the application's
  own decision date, rather than by the report cutoff. Later corrections do not
  rewrite historical decision inputs. No outcome labels are passed to snapshot.
- The existing strict snapshot validation applies to the complete supplied
  application/feature dataset. Duplicate identities, ambiguous revisions,
  unknown application references and missing decision-time features fail.
  Visible applications outside the two windows still need valid features.
  Future applications do not need their features before they become visible.

Missing values are not filled with zero, and incomplete applications are not
silently dropped. This report therefore does **not** measure missing-rate or
late-feature trends. Use input diagnostics to resolve invalid input before
comparing distributions. The fixed three-feature and two-segment event contract
is not a general-purpose arbitrary-column drift API.

## Measures and arithmetic

For a numeric input, let `F_ref(x)` and `F_cur(x)` be the empirical fractions of
the two populations whose values are less than or equal to `x`. The reported
KS distance is `max_x |F_ref(x) - F_cur(x)|`, evaluated at every distinct value
in the union of both samples. It ranges from 0 to 1. The comparison uses exact
integer cumulative counts before the final division; principal values remain
integer cents, including distinctions above `2^53`.

For example, 10 zeroes and 10 ones in reference versus 5 zeroes and 15 ones in
current have distance `|10/20 - 5/20| = 0.25`. Constant populations are supported:
all `0.2` versus all `0.8` gives distance 1; identical constants give 0.
No quantile bins, jitter or smoothing are introduced. The summary retains the
smallest value achieving the largest difference and both CDFs at that value.

For `segment`, total variation is half the sum of the absolute differences in
the `retail` and `small_business` proportions. All-retail reference versus a
50/50 current population gives `(0.5 + 0.5)/2 = 0.5`.

Distances are rounded to six decimals. Means and exported fractions are rounded
to eight decimals. Numeric minima/maxima and saved input values retain their
source types. Means are descriptive binary64 values, not exact monetary sums.

These are **descriptive marginal distances**. They are not KS p-values,
statistical significance claims, credit deterioration, estimated default rates
or calibrated action thresholds. Different measures and sample sizes should not
be ranked as an automatic importance score. Unchanged marginals can hide a
joint-distribution change; seasonality and composition can change marginals
without establishing an error or a cause. This workflow adds no automatic
approval, alert, retraining or model-selection behavior.

## Retained evidence and replay

| File | Contents |
| --- | --- |
| `inputs.json` | Normalized application and feature events, exact config, cutoff and data-kind marker; no performance records |
| `reference_features.csv`, `current_features.csv` | Decision-time rows used for all five input comparisons; no outcomes or predicted scores |
| `distributions.csv` | Numeric cumulative counts at every union value; category counts per segment, with `count_kind` distinguishing them |
| `monitor.json` | Cohort counts, numeric summaries, distances and monthly results |
| `index.html` | The report rendered from those results |
| `manifest.json` | Schema, normalized event-input digest and file digests |

`verify-feature-monitor` validates and replays the retained event inputs through
the snapshot contract, rebuilds the results and HTML, and compares every artifact.
Editing a distance, snapshot row, distribution count or page and then refreshing
its file hash does not bypass replay. Tests also compare numeric distances with
SciPy's independent two-sample KS implementation and hand-calculated examples.
The verifier performs no model fitting and needs no original external CSVs.

Replay establishes internal consistency with the retained events. It cannot
authenticate their source, recover omitted records, verify truthful availability
dates or detect a fully regenerated replacement bundle without an independently
retained trusted checkpoint. The report schema is
`creditvintage-feature-monitor-v1`; unsupported or malformed schemas fail.

Report creation stages a complete bundle and refuses an existing destination,
including an empty directory. Use a new output directory for a new run. Invalid
inputs fail before report creation. Existing score-monitor reports retain their
original schema and verification command.

## Python entry points

```python
from datetime import date
from pathlib import Path
from creditvintage.core import load_feature_csvs
from creditvintage.feature_monitoring import (
    monitor_features, write_feature_monitor, verify_feature_monitor,
)

applications, features = load_feature_csvs(Path("applications.csv"), Path("features.csv"))
result = monitor_features(applications, features, date(2024, 10, 1))
write_feature_monitor(result, Path("outputs/feature-monitor"))
verify_feature_monitor(Path("outputs/feature-monitor"))
```

Pass a `Config` or the CLI's `--config` for another calendar. The existing
`preflight --as-of` and `monitor` serve a different task: historical model readiness
and model-score distributions, with mature historical outcomes still required.
