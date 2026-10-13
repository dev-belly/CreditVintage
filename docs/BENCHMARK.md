# Same-budget credit policy comparison

Question: **if only 5%, 10% or 20% of mature applications can be reviewed,
what does each frozen ranking method add over a declared simple rule?**

[Published synthetic experiment](https://dev-belly.github.io/CreditVintage/benchmark/)

```bash
creditvintage benchmark --out outputs/benchmark
creditvintage verify-benchmark outputs/benchmark
```

Supply `--applications`, `--features` and `--performance` together to evaluate
your own event bundle under the existing data contract. The fixed 2022 training,
2023 calibration and 2024 test windows retain 182-day maturity gaps. Real data
must have genuine decision, availability and performance dates; invented dates
cannot turn a static public classification dataset into a prospective study.
The public site always generates synthetic data. No supplied customer data is
read or uploaded by its build script.

| Method | Frozen training and score definition |
| --- | --- |
| Business rule | `0.5 * utilization + 0.3 * DTI + 0.2 * min(prior_delinquencies, 5) / 5`; ranking score, not PD |
| Raw logistic | Existing training-only scaler and logistic regression, `C=1` |
| Calibrated logistic | Existing logistic model with isotonic calibration fitted on the separate calibration cohort |
| WOE logistic | Training quantile bins, 0.5 event/non-event smoothing, training-only scaler and `C=1` logistic regression |
| Histogram gradient boosting | 100 iterations, learning rate 0.05, 7 leaves, minimum 30 samples per leaf, L2=1, no random early-stopping split |

All fitted models use the same five inputs: utilization, DTI, prior
delinquencies, log principal in USD and the small-business indicator. WOE cuts
and **supervised** bin values use training outcomes only, including for test
values outside the training range. Parameters are declared before comparison;
test outcomes neither tune a model nor choose a deployed winner.

Scores are exported to ten decimal places. Each method ranks all test
applications by descending exported score, then ascending application ID, and
receives `ceil(N * capacity)` review places. Monthly and segment results are
subsets of that same global queue; they do not receive separate capacity.
Calibrated score ties can change review composition even when raw probabilities
have finer ordering. A method's count, precision, capture and false positives
can be reconstructed from `predictions.csv` without refitting.

## Explanations and scorecard

`explanations.csv` contains no performance labels. For the **raw logistic**
model it records the five application-time values, training feature means and
coefficients in original units. Each contribution is
`(value - training_mean) * coefficient`. The sum plus baseline log odds exactly
reconstructs the raw probability. Positive contributions increase fitted risk
relative to the training mean; these are model explanations, not causal effects,
and do not decompose the calibrated probability or the tree model.

The WOE bin counts, boundaries and point contributions appear in `summary.json`.
The scorecard has 600 points at fitted good/bad odds of 50 and 20 points to
double those odds. Higher points mean lower fitted risk. This is a minimal
research scorecard, without regulatory approval or credit-granting claims.

## Robustness and financial assumptions

Paired 95% intervals resample all six held-out origination months 300 times.
Every method uses the same sampled applications and its same fixed capacity.
The interval estimates descriptive uncertainty in **event-capture differences
against the business rule**, conditional on the frozen models; it does not
include model retraining or represent a treatment effect. Six groups are too
few for strong generalization claims.

The review cost defaults to an assumed USD 5 per case and can be changed with
`--review-cost-usd`. The captured event exposure proxy is event principal times
the existing assumed 45% LGD. Neither is an estimate of avoided loss. Capturing
a retrospectively observed event is not evidence that review prevented it.

The generator creates features and outcomes using a declared synthetic
mechanism. Public comparisons are demonstrations of this protocol. External
validation on licensed, time-stamped data remains necessary for real borrower
performance claims. Keep the original evaluation's worse calibrated Brier
result; adding another model does not erase that outcome.
