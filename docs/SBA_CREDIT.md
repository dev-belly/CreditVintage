# Observed SBA loans across funding cohorts

This is **retrospective funding-cohort validation**, using real reported dates
and a native outcome. It complements the synthetic event-time pipeline and
static UCI experiments. It cannot establish historical public availability:
the final SBA snapshot lacks first-publication times and historical revisions
of the approval features. No `available_at` fields are invented.

[Public report](https://dev-belly.github.io/CreditVintage/sba-credit/) ·
[Frozen source and license](../src/creditvintage/data/sba_credit/SOURCE.md) ·
[Protocol](../src/creditvintage/data/sba_credit/protocol.json) ·
[Conversion audit](../src/creditvintage/data/sba_credit/audit.json).

## Run the installed release

Python 3.12+ in a virtual environment; installation needs internet. Evaluation
and refitting then work offline, without PITBridge or a source checkout.

```text
python -m pip install https://github.com/dev-belly/CreditVintage/releases/download/v0.8.0/creditvintage_research-0.8.0-py3-none-any.whl
python -m creditvintage sba-benchmark --out outputs/sba-credit
python -m creditvintage verify-sba outputs/sba-credit
```

Open `outputs/sba-credit/index.html`. Verification returns `status: verified`,
`refitted: true`, 35,000 source / 10,000 test records, **89 recorded test events**
and zero repeated borrower group keys. Exit 0 means execution/verification
succeeded; exit 2 means invalid input, evidence or output. Choose a new output
directory on repeat; prior evidence is never overwritten.

## Source conversion before modeling

The official [SBA catalogue](https://data.sba.gov/dataset/7a-504-foia) lists
quarterly FOIA files and U.S. Government Works licensing. This study pins the
2026-06-30 files for FY2000–2009 and FY2010–2019, totaling **1,236,084 logical
records**. Original file and dictionary SHA256 values are in the source notice.

Eligibility requires valid approval/funding/outcome dates, usable recorded
status, a mature 24-calendar-month horizon, finite approval amounts and usable
recorded grouping fields. Unknown CHGOFF dates are excluded, not assigned zero.
Missing initial rates remain missing. The audit counts every exclusion and
retains eligible sizes before and after recorded-identity grouping.

Normalize recorded borrower name/state/ZIP, then hash the combined key. Retain
the earliest eligible loan per key across all three windows; break date ties
by original record ID. Per stage, take the lowest SHA256 of original-file digest
and logical row ordinal. Labels never enter sampling priority. This fixed
protocol retains 35,000 rows without outcome balancing. Grouping is heuristic:
name/address changes can evade matching; normalization can merge distinct firms.
The exported key is pseudonymous, linkable to the public original, not anonymous.

| Stage | Funding dates | Loans | 24-month events | Missing initial rate | Latest horizon |
| --- | --- | ---: | ---: | ---: | --- |
| Train | 2006-01-05–2010-12-30 | 20,000 | 816 (4.08%) | 15,250 | 2012-12-30 |
| Calibration | 2013-01-01–2013-12-31 | 5,000 | 42 (0.84%) | 15 | 2015-12-31 |
| Test | 2016-01-01–2017-12-31 | 10,000 | 89 (0.89%) | 0 | 2019-12-31 |

Training outcomes mature before calibration funding, and calibration outcomes
mature before test funding. Gaps are **calendar periods**, not a fixed number
of rows as in `TimeSeriesSplit(gap=...)`. Both event rates and missing-rate
coverage change substantially; do not describe the cohorts as exchangeable.

### Rebuild the exact sample from originals

From the source checkout after installing the project:

```text
python scripts/prepare_sba.py --source-dir sba-raw --download --out outputs/sba-rebuilt --verify-frozen
```

The optional download is about **574 MB**. Existing originals are reused only
when their hashes match. The converter scans both complete CSVs twice, checks
logical record counts, derives group membership and hash priorities, and
compares canonical decompressed CSV bytes, protocol and audit to the frozen
resources. Wrong snapshots fail before output creation. A changed official
file requires a reviewed dataset/protocol revision; never silently update pins.

## Target and model protocol

The target is **recorded SBA charge-off within 24 calendar months of first
disbursement**, inclusive of the anniversary, with leap-day month-end clamping.
A later lifetime charge-off is zero for this horizon. EXEMPT is the source's
disclosure status; absence of a recorded date is not independent adjudication
that no default occurred. This is neither 180-day delinquency nor UCI's native
classification. No monetary savings or recoveries are estimated.

Only four approval inputs enter modeling:

- `log1p(gross approval USD)`;
- SBA guaranteed share of approval;
- initial interest rate percent;
- approval processing method.

Training-only median imputation with missing indicators, standard scaling and
one-hot encoding handle these fields. Unseen categories are ignored. The final
status, current term, charge-off/paid dates and amounts, borrower keys, dates,
stage IDs, business age, jobs and collateral are excluded from predictors.

Logistic regression uses C=1, lbfgs, max_iter=2,000, no class weighting and no
hyperparameter search. The constant baseline uses training event rate. Isotonic
calibration fits **only the 2013 calibration labels**, on raw probabilities
rounded to six decimals; published probabilities have ten decimals. Missingness
and transformation parameters, coefficients and isotonic thresholds remain in
`summary.json`. Test outcomes never fit or select transformations or models.

## Retain all results

| Method | Test AUC ↑ | AP ↑ | Brier ↓ | Events in 1,000 review slots |
| --- | ---: | ---: | ---: | ---: |
| Constant train rate | 0.500000 | 0.008900 | 0.009838 | 7 / 89 |
| Logistic | 0.672341 | 0.018470 | 0.009011 | 26 / 89 |
| Isotonic | 0.658399 | 0.016661 | 0.008819 | 24 / 89 |

Calibration improves Brier while reducing AUC/AP and fixed-capacity capture.
The logistic 10% queue captures **29.21%** of recorded events at **2.60%**
precision. This is descriptive historical concentration, not a prevented-loss
estimate or an operational lending rule. Constant-score ties use record ID,
so that method's queue is an arbitrary ordering.

There are only **89 held-out events**. AUC's paired funding-month bootstrap
95% interval is **0.6246–0.7175** for logistic regression. All three methods use
the same 300 resampled sets of the 24 test months; all 900 rows are retained.
One-class replicates keep counts and Brier; unavailable AUC/AP have explicit
nulls and per-metric valid counts. Intervals condition on this sample and
these fitted models; they do not cover source revisions, training uncertainty,
hyperparameter selection or broader populations. Both test years remain visible.

## Portable evidence and verifier

The complete bundle contains frozen source/protocol/audit/license notice,
10,000 test predictions, nine same-budget policy rows, 3,000 outcome-free
queue rows, six year/method rows, 900 bootstrap rows, summary and HTML.
`manifest.json` binds every file; `evidence.zip` contains the same inventory.

`verify-sba` checks frozen bytes and numerical versions, refits the models,
recomputes outputs and checks score/order/count contracts, HTML and ZIP bytes.
Predictions use absolute tolerance 1e-8; summary/derived metrics use 1e-6 for
small OS/library fit differences. Identities, counts, source and protocol are
exact. This verifies internal consistency, not source truth or independent use.
The release gate also rejects a false AUC after its file hash is recomputed.
Linux-built evidence is replayed on Windows/Python 3.12 and Ubuntu/Python 3.13.

An [actual-run feedback form](https://github.com/dev-belly/CreditVintage/issues/new?template=sba_reproduction.yml)
asks for version, environment, exact commands, observed result and obstacles.
Providing the form and running CI do not establish independent adoption.

## Relationship to established tools

The [scikit-learn leakage guide](https://scikit-learn.org/1.8/common_pitfalls.html#data-leakage)
recommends splitting before learning preprocessing and using pipelines.
The [TimeSeriesSplit documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
describes time-ordered splits and a sample-count gap. This implementation uses
real calendar maturity gaps and separate calibration, preserving these
principles without pretending a row gap is a 24-month waiting period.

This is a fixed research case, not a general experiment-management or feature
platform. The original point-in-time pipeline still uses synthetic publication
and revision events. Historical revisions and first-publication records would
be required to combine that contract with independently observed availability.
