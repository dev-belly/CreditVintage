# Observed credit data, with a different validation question

[Open the public experiment](https://dev-belly.github.io/CreditVintage/public-credit/).

This experiment adds **observed historical credit records** to the repository.
It does not replace the synthetic event experiment: each supports a different
claim, and their targets and splits must not be merged.

| Evidence | What it supports | What it cannot establish |
| --- | --- | --- |
| Synthetic dated applications and revisions | Executable tests of availability, outcome maturity, lineage and cohort accounting | Predictive quality on actual borrowers |
| UCI German credit observations | A reproducible native good/bad classification comparison on a fixed held-out split | Decision-time availability, future-period performance, modern Chinese lending or 180-day 90+ DPD prediction |

## Source and frozen protocol

Hans Hofmann (1994), *Statlog (German Credit Data)*, UCI Machine Learning
Repository, [DOI 10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
The [official dataset page](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
licenses it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [source notice](../src/creditvintage/data/german_credit/SOURCE.md) records
the download, retrieval date, license and original file digests.

The package contains the unchanged 1,000-row `german.data` and original codebook.
SHA-256: `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871`.
Commands and CI never fetch a changing dataset from the network.

- Native class 1 (good) becomes 0; native class 2 (bad) becomes 1. There are
  700 good and 300 bad records. No DPD horizon or decision date is invented.
- Seed **20261015** fixes a stratified 600/200/200 train/calibration/test split.
  The two split calls use that seed and seed + 1; every original row assignment
  is exported. The stages contain 180/60/60 bad records respectively.
- Six numeric fields use training-only standardization; 11 categorical fields
  use training-only one-hot encoding with unknown categories ignored. Personal
  status/sex, age and foreign-worker status are omitted from the model. This
  omission does not establish fairness or eliminate proxies.
- Logistic regression uses fixed `C=1`, `lbfgs`, `max_iter=1000`, no class
  weighting and no hyperparameter search. A separate isotonic calibrator uses
  only the 200 calibration records. Test labels never fit either stage.
- The constant training class rate is the reference. Policies get the same
  5%, 10% and 20% test review capacity. Score ties use original row order.
  A constant-score policy's row-order result is not an optimized business rule.

## Retained results

The initial run used Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.8.0.

| Method | Test ROC-AUC | Average precision | Brier ↓ | Bad labels in 20 reviewed records |
| --- | ---: | ---: | ---: | ---: |
| Constant train class rate | 0.500000 | 0.300000 | 0.210000 | 6 |
| Logistic | 0.774286 | 0.677000 | 0.162263 | 17 |
| Logistic + isotonic | 0.759643 | 0.605599 | 0.169376 | 17 |

Calibration worsened both Brier and ranking metrics on this split. It remains
in the comparison. At 10% capacity the logistic queue has **17/20 = 85%** native
bad-label precision and captures **17/60 = 28.33%** of test bad labels. These
are small-sample historical ranking results, not prevented defaults or losses.
The 200-record holdout gives noisy single-split estimates; no confidence or
stability across future periods is established. Once public results are viewed,
this holdout is not a fresh validation set for later model selection.

## Reproduce offline

After installing CreditVintage (Python 3.12+):

```text
creditvintage public-benchmark --out outputs/public-credit
creditvintage verify-public outputs/public-credit
```

Use a new output directory. On Windows an unactivated environment can use
`.\.venv\Scripts\creditvintage.exe`. Open the resulting `index.html`.

The bundle includes original source bytes, `split.csv`, all 200 test labels and
scores, all nine policy/budget rows, a label-free 20-record logistic review
queue, `summary.json`, HTML and a digest manifest. Verification validates the
frozen original source digest, recomputes the split, refits models, compares
probabilities, metrics and queue, and rebuilds the page from verified claims.
It detects rehashed changes to scores, assignments, metrics or page text;
internal consistency still does not authenticate a publisher.

Recorded package versions identify the generating environment. Probability
comparisons allow absolute error 1e-8; six-decimal summary metrics allow 1e-6.
A changed numerical implementation outside those tolerances requires a fresh
report and an explicit comparison; version ranges do not promise identical
floating-point calculations forever. The installed-distribution CI checks this
path outside the source tree on Linux and Windows.

The original [source codebook](../src/creditvintage/data/german_credit/german.doc)
also describes a 5:1 classification cost matrix. This experiment reports
capacity ranking and probability quality; it does not optimize that matrix,
infer a monetary cost, estimate recovery losses or propose an approval rule.
A meaningful next validation needs larger appropriate cohorts with documented
dates, outcome definitions and observation coverage.
