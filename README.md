# CreditVintage

**A point-in-time credit cohort evaluation, from application records to a
reviewable risk report.** It is designed to demonstrate the data controls a
banking or risk analytics team should ask about before trusting a score.

[Live synthetic report](https://dev-belly.github.io/CreditVintage/demo/) ·
[Early score monitor](https://dev-belly.github.io/CreditVintage/monitor/) ·
[Published report files](docs/demo/index.html) ·
[Test predictions](docs/demo/predictions.csv) ·
[Run summary](docs/demo/summary.json) ·
[Artifact hashes](docs/demo/manifest.json)

> **Data status:** The published experiment is entirely synthetic. Its ROC-AUC,
> default rates, and loss scenario do not describe borrowers, a bank portfolio,
> or a deployable credit policy.

## What is actually evaluated

Each loan has a decision date, a principal amount, an application-time feature
history, and dated performance reports. The model predicts an **operational
180-day 90+ days-past-due event**. This is an explicit research label, not a
claim of regulatory default, IFRS 9 expected credit loss, or credit approval
fitness. Regulatory default definitions can include other criteria; see the
[Basel Framework](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/36/inforce/2023-01-01/published/2022-12-08).

| Control | Enforced behavior |
| --- | --- |
| Feature availability | Only the latest value **published by the decision date** is used. Later revisions cannot change historical applications. |
| Label maturity | Every loan waits 180 days plus two reporting days. A nondefault needs all six scheduled reports at days 30, 60, 90, 120, 150 and 180. Missing coverage is flagged, never treated as good. |
| Outcome revisions | A performance report is visible only after its `available_at` date; later corrections do not alter an earlier snapshot. |
| Prospective stages | Training outcomes mature before the first calibration application. Calibration outcomes mature before the first test application. |
| Probability quality | Fixed logistic baseline and a separate isotonic calibration sample. ROC-AUC, average precision and Brier are all shown on untouched test vintages. |
| Decision capacity | The top 10% review capacity is fixed in advance; precision and default capture are measured on test. |
| Monetary scenario | `predicted probability × origination principal × assumed 45% LGD`. This is a scenario proxy, not observed recovery loss or an accounting provision. |
| Early monitoring | A frozen model scores visible test applications before their labels mature. Raw-score distribution is compared with the calibration cohort without using current outcomes. |

Dates are calendar dates with **end-of-day availability semantics**. Real
intraday decisions need timestamps and a source-system availability audit.
Loan identity, date types, principal cents, segment, feature ranges, duplicate
updates, and report coverage are validated. Incomplete evaluation cohorts fail
explicitly rather than shrinking the denominator silently.

## Published run

Seed `20260927`, 80 generated loans per origination month. Training: Jan–Dec
2022 (960 loans); calibration: Jul–Dec 2023 (480); untouched test: Jul–Dec 2024
(480). The two six-month gaps let outcomes mature before the next stage; the
test observation cutoff is 2025-07-01.

| Test method | ROC-AUC | Average precision | Brier ↓ |
| --- | ---: | ---: | ---: |
| Constant training event rate | 0.500 | 0.223 | 0.174 |
| Fixed logistic model | **0.747** | **0.434** | **0.155** |
| Logistic + independent isotonic calibration | 0.735 | 0.407 | 0.158 |

Calibration **worsened** Brier here. The result is retained. The test event
rate was 22.3% versus a mean calibrated estimate of 18.8%, illustrating the
value of inspecting drift instead of reporting AUC alone. A 300-resample
vintage-block bootstrap gives a descriptive ROC-AUC interval of 0.653–0.801.
Only six test months exist; that interval is not a guarantee under future
distribution shift. The fixed 10% review group contains 48 loans and captures
23.4% of test defaults. All rounded headline numbers come from committed
[summary.json](docs/demo/summary.json); [predictions.csv](docs/demo/predictions.csv)
contains every test label, raw score, calibrated score, and review flag.

## Before outcomes mature: score monitoring

The separate `monitor` command freezes the same train and calibration stages,
then scores only applications visible by a chosen cutoff. It needs **no
performance reports from the current cohort**. Its raw-score population
stability index (PSI) compares current applications with the calibration cohort;
it is a distribution diagnostic, not a default-rate, quality, or significance
test. It never triggers automatic approval, retraining, or alerts.

The monitoring path uses performance reports only for the train and calibration
cohorts. It builds the current application snapshot without performance reports,
so current outcomes cannot change the score distribution or whether the monitor
passes its cohort checks. The input digest still covers all supplied event rows;
it identifies the source bundle, including rows the monitor did not use.

```bash
creditvintage monitor --as-of 2024-10-01 --out outputs/early
creditvintage verify-monitor outputs/early
```

The [published synthetic monitoring example](https://dev-belly.github.io/CreditVintage/monitor/)
has 480 reference and 320 visible current applications. Its PSI is **0.041**
overall, with a 0.220 September vintage diagnostic on 80 loans. These are
descriptive values on generated data, not validated decision thresholds. Ten
quantile bins are learned **only from reference raw scores**, repeated edges
are collapsed, and each bin receives a stated 0.5-count smoothing term. A
monthly PSI is omitted below 20 loans. The [score rows](docs/monitor/monitor_scores.csv),
[reference scores](docs/monitor/reference_scores.csv), [bin contributions](docs/monitor/score_bins.csv),
and [manifest](docs/monitor/manifest.json) let `verify-monitor` independently
recompute the result. Current labels are absent from the monitor output.

With supplied CSVs, pass the same three input paths shown below and an
appropriate `--as-of` date. The train and calibration outcome gaps still need
complete reports; the current window may have none. Keep borrower-level
monitor files private.

## Run and verify

Python 3.12+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
creditvintage run --out outputs/latest
creditvintage verify outputs/latest
creditvintage monitor --as-of 2024-10-01 --out outputs/early
creditvintage verify-monitor outputs/early
pytest -q
ruff check src tests
mypy src
```

The command creates a standalone `index.html`, CSV tables, `summary.json`,
and a manifest of SHA-256 output digests plus an input-event digest. `verify`
checks file digests and independently recomputes test AUC, average precision,
Brier, the test default rate, and the constant baseline from the prediction
rows. It also replays all 300 whole-month bootstrap draws from saved predictions
and the recorded seed, checking interval endpoints and valid replicate counts.
It enforces the full training, calibration and test outcome-maturation
gaps, then checks prediction dates against the stated test window and their
vintage labels. It also checks that the published review
flags select exactly the highest-ranked loans at the stated capacity, with
application ID breaking score ties and the recorded configuration defining
the same capacity. Both verifiers reject non-finite numeric claims (`NaN`,
infinity and overflow), duplicate JSON keys, duplicate or malformed CSV columns, and fractional
cohort counts. `verify-monitor` also validates current application identities,
segments and the reference-label maturation gap. Both rebuild the HTML page from
the verified evidence, rejecting a rehashed webpage with invented claims. These checks run even when
the output hashes have been updated to match edited files; the regression
suite covers those inconsistent reports alongside the original published
examples. The summary records Python, NumPy and
scikit-learn versions; exact floating-point output across environments is not
promised. The hashes detect accidental edits, not an
adversary who can rewrite files and manifest together.

To use privately held event data, supply **all three** UTF-8 CSV paths:

```bash
creditvintage run --applications applications.csv \
  --features features.csv --performance performance.csv --out outputs/private
```

| CSV | Required columns |
| --- | --- |
| `applications.csv` | `application_id,decision_at,principal_cents,segment` |
| `features.csv` | `application_id,name,value,observed_at,available_at` |
| `performance.csv` | `application_id,as_of,available_at,days_past_due` |

Dates are `YYYY-MM-DD`. Segments are `retail` or `small_business`; feature
names are `utilization`, `debt_to_income`, `prior_delinquencies`.
`principal_cents` is a positive integer USD-cent amount. The fixed vintage
windows above remain in force for supplied data; the Python `Config` API can
define other mature, nonoverlapping windows. Keep borrower-level files private.

## Design limits

- The public labels are generated from the same synthetic risk drivers used as
  model inputs. Stronger-than-chance metrics validate the **pipeline**, not
  real-world default prediction.
- Six exact 30-day reporting anchors simplify coverage. Actual servicing feeds
  need calendar-aware schedules, partial repayment, prepayment, restructuring,
  recoveries, exposure at default, and audited source timestamps.
- Isotonic calibration is measured against a fixed logistic baseline; no
  hyperparameter search or test-driven calibration choice is performed. A
  larger independent calibration set may be required in practice. See the
  [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
- No protected attributes are used, but their absence does not prove fairness.
  Selection bias, approvals-only labels, population drift, and policy impacts
  need separate study before any real use.
- Score PSI alone cannot tell whether calibration, realized loss, or fairness
  has changed. The published early example includes four current vintages;
  binning, smoothing, sample size, and shifting acceptance policy all affect
  the diagnostic.

## Related open-source work

[scorecardpy](https://github.com/ShichenXie/scorecardpy) provides traditional
scorecard development and PSI evaluation; [Evidently](https://github.com/evidentlyai/evidently)
provides general data and model monitoring. Credit risk case studies such as
[Lending Club Credit Risk](https://github.com/tubolyroli/lending-club-credit-risk)
and [RiskLens](https://github.com/Tussar98/risklens) show vintage maturity,
calibration, and drift in broader pipelines. CreditVintage focuses its small,
testable contract on application-time availability, explicit outcome maturity,
and the transition from label-free monitoring to out-of-time evaluation.

The package has a standard-library event contract, NumPy/scikit-learn model
evaluation, a dependency-free static HTML report, strict tests for timing and
corruption, and CI on Python 3.12/3.13. This is research software, not a credit
decision service.

## Source-to-prediction lineage

[Open the PITBridge × CreditVintage explorer](https://dev-belly.github.io/CreditVintage/lineage/) · [Download the complete synthetic evidence bundle](https://dev-belly.github.io/CreditVintage/lineage/evidence.zip). The `lineage-demo` command writes a standalone `index.html`: select a held-out application to trace its three model features to the source record, revision, publication, ingestion and decision cutoff. Its default generated synthetic cohort has **480 applications, 1,440 selected features and 120 held-out predictions**. The original evaluation above remains a separate 480-test-application example. The public example is generated and verified during the Pages build. Its source events, input CSVs, predictions and manifests are available together in the downloadable bundle; generated outputs are kept out of Git history.

The adapter is a real code path, not a diagram connecting two independent demos. It verifies PITBridge's saved bundle through SQL and its separate Python oracle, enforces application/entity mapping and feature units, exports model inputs, and runs CreditVintage on those exported CSVs. The source fixture includes **192 observations unavailable at their decision**, including corrections published on the decision date but ingested the next day.

Install the pinned reference implementation before using this optional integration:

```bash
git clone https://github.com/dev-belly/PITBridge.git ../PITBridge
git -C ../PITBridge checkout 2f2438aa9b62502c7b1c9773b572d09f6b04f9ea
python -m pip install -e ../PITBridge
creditvintage lineage-demo --out outputs/lineage
creditvintage verify-lineage outputs/lineage
```

Open `outputs/lineage/index.html` in your browser after local generation. Alternatively, download and extract the online evidence ZIP, then run `creditvintage verify-lineage lineage` against the extracted folder. `verify-lineage` checks file hashes, replays the original PIT sources, reconstructs every feature and lineage row, independently verifies the evaluation artifacts, and refits the model from saved applications/features/performance. It compares saved predictions using a numerical tolerance. [The integration contract](docs/LINEAGE.md) explains the units, UTC convention, dependency pin and boundaries.

The original `run`, `monitor`, `verify` and `verify-monitor` commands do not need PITBridge installed. CI exercises the integration with a pinned PITBridge checkout on Python 3.12 and 3.13.
