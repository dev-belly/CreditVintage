# CreditVintage

**A point-in-time credit cohort evaluation, from application records to a
reviewable risk report.** It is designed to demonstrate the data controls a
banking or risk analytics team should ask about before trusting a score.

[Live synthetic report](https://dev-belly.github.io/CreditVintage/demo/) ·
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

## Run and verify

Python 3.12+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
creditvintage run --out outputs/latest
creditvintage verify outputs/latest
pytest -q
ruff check src tests
mypy src
```

The command creates a standalone `index.html`, CSV tables, `summary.json`,
and a manifest of SHA-256 output digests plus an input-event digest. `verify`
checks file digests and independently recomputes test AUC, average precision,
and Brier from the prediction rows. The summary records Python, NumPy and
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

The package has a standard-library event contract, NumPy/scikit-learn model
evaluation, a dependency-free static HTML report, strict tests for timing and
corruption, and CI on Python 3.12/3.13. This is research software, not a credit
decision service.
