# Diagnose decision-time feature coverage and visible delay

Available in **v0.6.0**. This diagnostic keeps every expected application/feature
slot, including unavailable inputs. It complements the strict distribution and
scoring commands without filling or dropping missing features. It reads only
application and feature events, with no outcome file or model fitting.

```text
creditvintage monitor-feature-quality --as-of 2024-10-01 --out outputs/feature-quality
creditvintage verify-feature-quality outputs/feature-quality
```

Open `outputs/feature-quality/index.html`, or inspect the
[published synthetic example](https://dev-belly.github.io/CreditVintage/feature-quality/).
Choose a new output directory for every run; existing destinations are refused.

## The deliberately incomplete example

The built-in synthetic example has 480 reference applications, 320 visible
current applications and 160 future current applications. Four different July
applications each have one deliberately unavailable feature:

| Application / input | Injected situation | Status at 2024-10-01 |
| --- | --- | --- |
| `CV202407-0000` / utilization | All events for this feature are removed | `no_visible_event` |
| `CV202407-0001` / debt-to-income | Observed June 26, available July 4, after the July 1 decision | `late_for_decision`, 3 days late |
| `CV202407-0002` / prior delinquencies | Observed July 2 and available July 3, after the decision | `observed_after_decision` |
| `CV202407-0003` / utilization | Available October 2, beyond the report cutoff | `no_visible_event`; no future reason or delay is reported |

All 320 current applications remain in every feature's denominator: utilization
is available for 318/320, and the other two inputs for 319/320 each. Exactly
316 current applications have all three inputs; four are incomplete. These
injected counts illustrate the method, not a real data-source failure rate.
At `--per-vintage 20`, the corresponding populations are 120/80/40, with 76
complete and four incomplete visible current applications.

## Use your two CSVs

Install the [checked release](releases/v0.6.0.md), then use the original
[application/feature column contract and units](CUSTOM_INPUTS.md):

```text
creditvintage monitor-feature-quality --applications applications.csv --features features.csv --config cohorts.json --as-of 2024-10-01 --out outputs/private-feature-quality
creditvintage verify-feature-quality outputs/private-feature-quality
```

Both CSV paths must be supplied together. The performance file can be absent.
The complete seven-date configuration is shared with the existing event workflow;
only its calibration and test windows define these two diagnostic populations.
No labels or maturation checks are needed for coverage. Supplied inputs are marked
`user_supplied`, without automatically authenticating their origin.

If a virtual environment is not activated, replace `creditvintage` with
`.venv/bin/python -m creditvintage` on Linux/macOS or
`.\.venv\Scripts\python.exe -m creditvintage` on Windows.

## What each status means

Each visible application in a configured population contributes exactly three
slots, one per required feature. Classification uses only feature events with
`available_at <= report cutoff`, in this order:

1. **`available`:** an event was observed and available by the decision. Select
   the latest eligible availability date, using the same selector as `snapshot`.
2. **`late_for_decision`:** no eligible event exists, but a pre-decision
   observation arrived after the decision and by the report cutoff. Select its
   earliest availability, so a later correction cannot inflate the first delay.
3. **`observed_after_decision`:** only observations made after the decision are
   visible. Select the first visible arrival. A later observation is not evidence
   that the feature existed at decision time.
4. **`no_visible_event`:** no event is visible by the report cutoff. This includes
   events supplied in the file whose availability is still in the future.

A later report can learn why an input was unavailable, changing `no_visible_event`
to `late_for_decision`. It still cannot make that late event available at the
historical decision. A feature with an on-time event remains `available` even if
late corrections also exist.

The cutoff must be a calendar date on or after the configured test start. Future
applications are counted separately and have no slots until they become visible.
Applications outside the two windows are excluded from coverage, but the complete
input still undergoes identity and revision validation. Unknown references,
duplicate applications, ambiguous same-day revisions and malformed values fail
before report creation. The feature types/units remain strict: this diagnostic
does not count raw blank or `NaN` CSV values as valid feature events.

## Denominators and timing

- Feature coverage is `available slots / visible applications` for that feature,
  population and month. Incomplete applications stay in all denominators.
- Complete-application counts require all three slots. Two unavailable inputs
  in one application count as one incomplete application, not two applications.
- Overall coverage uses the overall counts, not an unweighted average of monthly
  percentages. Small months retain their actual counts and descriptive rates.
- An empty population has zero observed counts, `coverage_rate=null` and status
  `no_visible_applications`. It is not treated as either 0% missing or fully ready.
- **Observation age:** decision minus observation date, for selected on-time events.
- **Availability lag:** availability minus observation date. Coverage summaries
  report its median/maximum on selected on-time events, with their sample count.
- **Lateness:** first visible availability minus decision date, only for
  `late_for_decision`. A missing or post-decision-only slot has no estimated
  lateness; an unknown delay is not replaced with zero.

All timing uses the existing daily date contract, not intraday latency. No
automatic threshold, alert, imputation, model retraining or credit decision is
introduced. Event absence differs from raw-column nullness, and these descriptive
diagnostics do not establish credit risk, model deterioration or causality.

## Diagnostic completion and scoring readiness are different

A successfully generated diagnostic returns exit code 0, including when its
report contains incomplete coverage. `verify-feature-quality` returning
`status=verified` means that its evidence recomputes; it does not mean the inputs
are complete or ready for scoring. Inspect the population status and counts.
Invalid identity/schema/value input or a failed replay returns exit code 2.

The existing `snapshot`, `monitor-features`, model readiness and scoring paths
continue to reject missing decision-time features. Repair or explicitly obtain
compatible data before using those paths; the diagnostic does not bypass them.

## Evidence and replay

| File | Contents |
| --- | --- |
| `inputs.json` | Canonical application/feature events, config, cutoff and data-kind marker; no outcomes |
| `quality.json` | Population counts, complete/incomplete counts, coverage and monthly results |
| `availability.csv` | Every expected slot, reason, selected visible event and applicable timing; missing cells remain empty |
| `coverage.csv` | Reference/current totals and every visible month; exact counts, coverage rates and timing summaries |
| `index.html` | Report, including the first 50 unavailable slots; the CSV retains all slots |
| `manifest.json` | Schema, event-input digest and exact file hashes |

The verifier decodes and validates the retained input events, rebuilds every
slot/status and aggregate, regenerates the page and compares every artifact.
Refreshing hashes after changing coverage or rows does not bypass replay.
Replay does not authenticate event sources, verify truthful availability dates,
recover omitted records or detect a fully regenerated replacement without an
independently retained trusted checkpoint. Schema:
`creditvintage-feature-quality-v1`.

Existing v0.5.0 feature-distribution reports retain their original bytes and
verifier. A compatibility fixture generated by the public v0.5.0 wheel pins all
six file hashes while the event codec and selector are shared with this route.

## Python API

```python
from datetime import date
from pathlib import Path
from creditvintage.core import load_feature_csvs
from creditvintage.feature_quality import (
    monitor_feature_quality, write_feature_quality, verify_feature_quality,
)

applications, features = load_feature_csvs(Path("applications.csv"), Path("features.csv"))
result = monitor_feature_quality(applications, features, date(2024, 10, 1))
write_feature_quality(result, Path("outputs/feature-quality"))
verify_feature_quality(Path("outputs/feature-quality"))
```

Pass `Config` as the fourth argument for another valid calendar.

## Check explicit acceptance conditions

From v0.7.0, [coverage policies](FEATURE_POLICY.md) check a verified diagnostic
against user-specified overall/monthly coverage and minimum sample conditions.
They retain failed/insufficient verdicts and source events. A diagnostic that
verifies can fail its policy; a partial or tolerant policy pass still does not
establish scoring readiness. Existing quality artifacts retain their v1 format.
