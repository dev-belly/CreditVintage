# CreditVintage

[![CI](https://github.com/dev-belly/CreditVintage/actions/workflows/ci.yml/badge.svg)](https://github.com/dev-belly/CreditVintage/actions/workflows/ci.yml)

**A point-in-time credit cohort evaluation, from application records to a
reviewable risk report.** It is designed to demonstrate the data controls a
banking or risk analytics team should ask about before trusting a score.

- **Use your own CSVs:** [Create and validate an editable input case](#start-with-editable-csv-inputs).
- **Compare models on public credit data:** [Start with the public benchmarks](#try-observed-public-data).
- **Inspect decision-time data and human review:** [Trace source availability](#source-to-prediction-lineage) and [inspect the review workflow](#a-durable-human-review-workflow).

> **Data status:** The point-in-time event experiment is synthetic. Separate
> [card](docs/CARD_CREDIT.md) and [German credit](docs/PUBLIC_CREDIT.md) experiments
> use 30,000 and 1,000 historical observed records. Their native targets and
> random customer splits cannot validate the event pipeline's time controls
> or a deployable credit policy.

[Comparison with established tools and remaining gaps / 同方向项目对照](docs/COMPARISON.zh-CN.md).

## Start with editable CSV inputs

Follow [Run and verify](#run-and-verify) to install the
[checked release](https://github.com/dev-belly/CreditVintage/releases).
The examples use the installed `creditvintage` command. If the virtual environment
is not activated, replace `creditvintage` with `.venv/bin/python -m creditvintage`
on Linux/macOS or `.\.venv\Scripts\python.exe -m creditvintage` on Windows.
Then create a complete input case and check it before training:

```text
creditvintage init-inputs --out inputs/example
creditvintage check-inputs inputs/example
creditvintage run --applications inputs/example/applications.csv --features inputs/example/features.csv --performance inputs/example/performance.csv --config inputs/example/cohorts.json --out outputs/csv-example
creditvintage verify outputs/csv-example
```

The starter contains **480 explicitly synthetic applications**, all feature and
performance records, and a complete calendar configuration. Replace the CSVs and
calendar with data that meets the [documented contract](docs/CUSTOM_INPUTS.md).
The check collects errors with file names and physical line numbers, including
duplicates, missing decision-time features and incomplete mature outcomes.
It neither fits models nor changes the input files. Invalid input returns JSON
and exit code 2; readiness means the input/evaluation contract passed, without
establishing source authenticity or model quality.

Since `v0.3.1`, the checker also reports an unterminated quote in a CSV header as
`csv_quoting` JSON and exit code 2, while continuing to check the other files.
The older `v0.3.0` raises a parsing error for that case. See the
[patch and upgrade notes](docs/releases/v0.3.1.md); existing `v0.3.0` assets are preserved.

## Try observed public data

The larger [UCI credit-card experiment](docs/CARD_CREDIT.md) uses **30,000
observed records**, five frozen 18,000/6,000/6,000 train/calibration/test splits,
and four fixed methods. It retains every split, all 120,000 model scores and
60 equal-capacity policy comparisons. Numeric scaling and categorical encoding
fit on train; isotonic calibration fits only on its separate partition.
[Open the card benchmark report](https://dev-belly.github.io/CreditVintage/card-credit/).

```bash
creditvintage card-benchmark --out outputs/card-credit
creditvintage verify-card outputs/card-credit
```

Its native subsequent-default label and single historical cohort do not
establish future-period performance. The independent German credit experiment
below uses a different source and native good/bad target.

```bash
creditvintage public-benchmark --out outputs/public-credit
creditvintage verify-public outputs/public-credit
```

After installing the package as below, these commands work offline. The frozen
600/200/200 stratified split compares a constant baseline, logistic regression
and independent isotonic calibration. Preprocessing fits only on train. On the
200 held-out records, logistic ROC-AUC is **0.774**, Brier **0.162**; calibration
worsens Brier to **0.169**, and that result stays visible. Test predictions,
split membership, original attributed bytes and a label-free 20-record queue
are exported. Verification refits from those bytes instead of trusting hashes
alone. [Protocol, provenance and limits](docs/PUBLIC_CREDIT.md).
[Open the German credit report](https://dev-belly.github.io/CreditVintage/public-credit/).

## Compare fixed models across five public-data splits

The separate [repeated comparison](https://dev-belly.github.io/CreditVintage/public-credit-comparison/)
reuses the packaged UCI source, native labels and the same 17-feature contract.
A constant baseline, logistic regression and histogram gradient boosting share
five fixed stratified 800/200 train/test splits. All 15 results, 3,000 test scores
and every split assignment are saved. Preprocessing is fitted on training rows
only; model settings are fixed. [Protocol and interpretation](docs/PUBLIC_CREDIT_COMPARISON.md).

| Method | Mean ROC-AUC | Observed AUC range | Mean Brier ↓ | Mean top-10% precision |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.500 | 0.500–0.500 | 0.2100 | 0.230 |
| Logistic | 0.767 | 0.732–0.822 | 0.1714 | **0.740** |
| Histogram gradient boosting | **0.771** | 0.728–0.844 | **0.1692** | 0.730 |

Each test set has 60 bad labels and each method receives 20 inspection slots.
The constant method's top-20 subset is an arbitrary source-row tie break.
The tree does not win every metric. These five overlapping splits provide
descriptive means and observed ranges, not independent trials or confidence
intervals. This comparison uses different splits from the calibration experiment
above; their metrics must not be directly combined.

From the source checkout after installing the project:

```bash
python examples/german_credit.py --out outputs/public-credit-comparison
python examples/german_credit.py --verify outputs/public-credit-comparison
```

Data loads from the installed package, so there is no second dataset copy or
network fetch. Saved-score verification checks source labels, held-out membership,
metrics and inspection flags without refitting. The package's separate
`verify-public` command refits its original calibration experiment.

## What is actually evaluated

New: [same-budget policy comparison](https://dev-belly.github.io/CreditVintage/benchmark/)
compares a fixed business rule, raw/calibrated logistic regression, a training-only
WOE scorecard and histogram gradient boosting at 5%, 10% and 20% review capacity.
It exports monthly/segment outcomes, paired month-bootstrap intervals and
label-free raw-model explanations. [Protocol and limitations](docs/BENCHMARK.md).
Run `creditvintage benchmark --out outputs/benchmark`, then
`creditvintage verify-benchmark outputs/benchmark`. This dated-event policy experiment remains synthetic.

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

[Open the live synthetic report](https://dev-belly.github.io/CreditVintage/demo/) or
inspect the [published report files](docs/demo/index.html) and
[artifact hashes](docs/demo/manifest.json).

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

## From model output to business analysis

The `analyze` command verifies an existing mature evaluation report before
running SQLite cohort queries. It exports portfolio, customer-segment,
origination-vintage and vintage × segment views, plus the original fixed-capacity
review queue **without future outcome labels**. Every group is checked against
independent Python counts and sums. No database server or new dependency is needed.

```bash
creditvintage analyze docs/demo --out outputs/business-case
creditvintage verify-analysis outputs/business-case
```

Choose an output directory that does not yet exist. Open its `index.html`; the
folder contains 21 group rows, 48 queue rows, the actual SQL, a complete frozen
source-report copy and a digest manifest. Verification reconstructs the CSV,
summary, queue, SQL and page from the embedded source, so moving the folder does
not break replay. Invalid source evidence fails before the output is created.
Pages generates and verifies the [public analysis](https://dev-belly.github.io/CreditVintage/analysis/)
from the original committed report during publication.

The synthetic source has 107 research events in 480 mature applications. The
48-application review group contains 25 events: precision is **25/48 = 52.08%**,
while event capture is **25/107 = 23.36%**. Overall calibrated mean PD is 18.78%;
principal-weighted PD is 18.49%. The preset 10% capacity is global, not a separate
10% quota for every month or segment. Zero denominators remain null; comparisons
use the previous observed vintage in the same segment. These are descriptive
historical measures, not evidence that an intervention reduced losses. See the
[Chinese business walkthrough](docs/BUSINESS_CASE.md) for denominator checks,
customer-segment interpretation, a minimal SQL query and interview follow-ups.

## A durable human review workflow

Move the verified, label-free queue into a SQLite workspace with claim, release,
resolve and reopen actions. Each conclusion requires a reason and evidence reference.
Concurrent claims are serialized in one transaction and checked against a case version;
identical request retries return the original event. Review history is append-only,
and current state is reconstructed from the frozen queue and events.

```bash
creditvintage review-init outputs/business-case --db outputs/reviews.sqlite
creditvintage review-status outputs/reviews.sqlite
```

The [five-minute walkthrough](docs/REVIEW_WORKFLOW.md) includes complete PowerShell
commands, competing-reviewer and retry examples, design choices and interview prompts.
After review, `creditvintage review-export outputs/reviews.sqlite --out outputs/review-evidence`
delivers a portable CSV/JSON evidence bundle. `creditvintage verify-reviews outputs/review-evidence`
replays all cases without the database. The [delivery walkthrough](docs/REVIEW_DELIVERY.md)
shows how a separately saved `--expected-sha256` checkpoint also detects a consistent rewrite;
internal hashes alone do not authenticate the author or evidence.
The workspace stores no future performance labels. Its completion rate counts resolved
work, not captured credit events or prevented loss. This is a trusted local CLI workspace:
actor names are declarations, evidence references are not authenticated documents,
and a hosted multi-user service would need identity, authorization and attachment controls.

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
creditvintage preflight --as-of 2024-10-01
creditvintage monitor --as-of 2024-10-01 --out outputs/early
creditvintage verify-monitor outputs/early
```

Since v0.4.0, `preflight --as-of` checks the monitoring cohorts without fitting
or writing files. It reports `ready_for_monitoring`, mature train/calibration
counts, visible current applications and current applications not yet visible.
Current outcomes are not required or counted. This checks input/cohort readiness;
it produces no scores or PSI and cannot guarantee successful fitting or useful
score variation. Without `--as-of`, `preflight` retains its mature evaluation checks.

For an end-to-end synthetic example with **all current outcomes removed**, run
`python examples/monitoring_without_outcomes.py --out outputs/unlabelled-case`
from an installed source checkout. It saves the editable inputs and preflight
JSON, then fits the monitor on historical labels and verifies its report.
The [walkthrough](docs/CUSTOM_INPUTS.md#4-before-outcomes-mature) shows how to repeat
the checks directly with the CLI.

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

Version **0.4.0** distributions are published under [GitHub Releases](https://github.com/dev-belly/CreditVintage/releases/tag/v0.4.0)
only after all main CI jobs pass. The downloadable wheel is the same artifact
that passed clean installation and complete offline refitting. Each release
includes its source archive, installed-check evidence, commit/version manifest
and SHA-256 checksums. Existing version tags are preserved; changes require a
version bump. Install the wheel into a Python 3.12+ virtual environment, then
use the commands above without cloning this repository. The
[upgrade notes](docs/releases/v0.4.0.md) give the exact wheel URL and a short
installation check. Existing commands, inputs and model protocols remain
compatible; `preflight --as-of` adds the optional monitoring check.

For the current `main` source, use Python 3.12+ and Git. Start in a new parent
directory and obtain the repository:

```text
git clone --branch main https://github.com/dev-belly/CreditVintage.git
cd CreditVintage
```

Then, from that repository root, on Linux or macOS:

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

On Windows PowerShell, run the environment's executables directly; activation
and execution-policy changes are unnecessary:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\creditvintage.exe run --out outputs/latest
.\.venv\Scripts\creditvintage.exe verify outputs/latest
.\.venv\Scripts\creditvintage.exe monitor --as-of 2024-10-01 --out outputs/early
.\.venv\Scripts\creditvintage.exe verify-monitor outputs/early
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe src
```

For later examples on Windows, replace `creditvintage` with
`.\.venv\Scripts\creditvintage.exe` and `python` with `.\.venv\Scripts\python.exe`.
CreditVintage writes its HTML and JSON with UTF-8 and LF endings, and `.gitattributes` preserves
committed evidence bytes even with Git's `core.autocrlf=true`. Regenerate older
Windows exports with the updated package before verifying them. CI checks
generated and committed reports, the lineage download and its replay on Linux
Python 3.12/3.13 and Windows Python 3.12.

### Check the package a new user actually installs

The separate **Installed distributions** CI jobs build a source archive, build
its wheel, and install that wheel in a clean virtual environment. All commands
run outside both repositories, so an editable checkout cannot hide a missing
SQL file, HTML template or command entry point. They exercise evaluation,
monitoring, SQL analysis and a 48-case review workspace, including an identical
retry and a competing stale-version claim. The core paths run before PITBridge
is installed; missing integration fails cleanly without creating an output.
The pinned PITBridge wheel is then installed and the full lineage bundle replayed.

To run the same check locally, first clone the pinned PITBridge reference as
shown in [Source-to-prediction lineage](#source-to-prediction-lineage), then:

```bash
python -m pip install build
python -m build --outdir dist
python -m build --wheel --outdir dist ../PITBridge
python scripts/check_distribution.py dist
```

Use the virtual environment's `python.exe` on Windows, as above. The replay
script creates and removes its own isolated environment and temporary synthetic
reports. Runtime dependencies may be downloaded during wheel installation;
it does not publish reports or change an existing review workspace. Its explicit
summary distinguishes package installation from numerical or business validity.

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
scikit-learn versions; exact floating-point output across arbitrary environments is not
promised. The card experiment's v0.2.1 wheel pins NumPy, SciPy and scikit-learn;
CI refits the same evidence across Linux/Python 3.12–3.13 and Windows/Python 3.12.
Its verifier reports a mismatched numerical runtime before fitting. The hashes detect accidental edits, not an
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

Dates are exactly `YYYY-MM-DD`, including `monitor --as-of`; compact dates,
ISO week dates and timestamps are rejected in v0.3.2 and later.
Earlier wheels predate this stricter CSV/cutoff parsing; see the
[date input and compatibility notes](docs/CUSTOM_INPUTS.md#1-make-all-dates-explicit).
Segments are `retail` or `small_business`; feature
names are `utilization`, `debt_to_income`, `prior_delinquencies`.
`principal_cents` is a positive integer USD-cent amount. The default vintage
windows above apply when `--config` is omitted. Supply a strict JSON
configuration to `run`, `benchmark`, `monitor` or `preflight` to use your own
dates, LGD scenario and review fraction. The file must give all seven calendar
dates; unknown keys, duplicate keys, non-finite numbers and shortened maturity
gaps fail explicitly. An explicit `--seed` overrides the JSON seed.

```bash
creditvintage preflight --applications applications.csv --features features.csv \
  --performance performance.csv --config examples/cohort-config.json
creditvintage run --applications applications.csv --features features.csv \
  --performance performance.csv --config examples/cohort-config.json --out outputs/private
```

Without `--as-of`, `preflight` checks the mature train/calibration/test cohorts without fitting or
writing a report. It counts excluded applications so an incorrect date range
cannot look like complete coverage. It checks input consistency, not source
authenticity or model quality. For immature current cohorts use
`preflight --as-of YYYY-MM-DD` followed by `monitor` at the same cutoff. The synthetic
generator still uses its documented fixed dates. [Custom-input walkthrough](docs/CUSTOM_INPUTS.md).
Keep borrower-level files private.

## Design limits

- The event-demo labels are generated from the same synthetic risk drivers used as
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
git -C ../PITBridge checkout ed19dc698534f45a2b646fb4976ff6b01966b0cc
python -m pip install -e ../PITBridge
creditvintage lineage-demo --out outputs/lineage
creditvintage verify-lineage outputs/lineage
```

Open `outputs/lineage/index.html` in your browser after local generation. Its download section includes `evidence.zip`, generated alongside the page, containing the 18 evidence artifacts and the root manifest. Alternatively, download and extract the online evidence ZIP, then run `creditvintage verify-lineage lineage` against the extracted folder. `verify-lineage` checks file hashes, replays the original PIT sources, reconstructs every feature and lineage row, independently verifies the evaluation artifacts, and refits the model from saved applications/features/performance. It compares saved predictions using a numerical tolerance. [The integration contract](docs/LINEAGE.md) explains the units, UTC convention, dependency pin and boundaries.

The original `run`, `monitor`, `verify` and `verify-monitor` commands do not need PITBridge installed. CI exercises the integration with a pinned PITBridge checkout on Python 3.12 and 3.13.

## License

Project code is available under the [MIT License](LICENSE).
The bundled UCI datasets retain their **CC BY 4.0** terms and attribution;
see the [German credit source notice](src/creditvintage/data/german_credit/SOURCE.md)
and [credit-card source notice](src/creditvintage/data/card_credit/SOURCE.md).
The distribution metadata uses `MIT AND CC-BY-4.0` because the package contains
both project code and those datasets. This does not relicense the datasets.

