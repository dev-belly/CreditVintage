# 30,000 observed card records: fixed protocol and complete replay

[Open the report](https://dev-belly.github.io/CreditVintage/card-credit/) ·
[Download its complete evidence](https://dev-belly.github.io/CreditVintage/card-credit/evidence.zip).

This larger experiment extends the observed-data validation beyond the 1,000-row
German table. It uses UCI dataset 350, *Default of Credit Card Clients*, Yeh
(2009), [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H), under CC BY 4.0.
The [source notice](../src/creditvintage/data/card_credit/SOURCE.md) records the
official download, original ZIP/Excel hashes and canonical conversion hash.

## Reproduce from an installed wheel

Python 3.12+, using the wheel from [Releases](https://github.com/dev-belly/CreditVintage/releases)
or a local installation of this repository:

```text
creditvintage card-benchmark --out outputs/card-credit
creditvintage verify-card outputs/card-credit
```

On Windows use `.\.venv\Scripts\creditvintage.exe` for an unactivated virtual
environment. Choose a new output directory. Both commands run offline; Excel,
pandas and xlrd are not runtime requirements. Open the resulting `index.html`.

The original 30,000 rows retain 23,364 native non-events and 6,636 native events.
The unchanged target column is `default payment next month`. No event dates,
DPD anchor reports or new labels are manufactured to fit the separate event API.

## Five splits and four methods

Fixed seeds `7, 23, 41, 59, 83` each produce 18,000 training, 6,000 calibration
and 6,000 test rows. The test stage is drawn first; the remaining development
stage is divided 75/25 with seed + 1. Both draws stratify the native target.
All row assignments are saved. Repeated test sets overlap, so 30,000 saved test
occurrences are not 30,000 distinct independently held-out customers.

| Method | Fitting |
| --- | --- |
| Constant | Training event frequency |
| Logistic | Fixed C=1, lbfgs, max_iter=2000, no class weighting |
| Logistic + isotonic | Same training model; calibration labels used only by isotonic |
| Histogram gradient boosting | Fixed 100 iterations, rate .05, depth 3, 15 leaves, min leaf 20, L2=1; no early stopping |

The 19 input fields are credit limit, six repayment statuses, six bill amounts
and six payment amounts. Numeric scaling and one-hot categories fit only on
training rows. Original ID and four demographic columns are excluded; this does
not establish fairness or remove proxies. Repayment codes -2 and 0 remain
separate native categories without invented interpretations. Negative bill
amounts are retained. There is no test-driven parameter or seed selection.

Every method receives 300/600/1,200 review records at 5%/10%/20% capacity in each
split. Scores descend, with original ID resolving ties. Review queues omit
outcomes. All 60 method/seed/budget rows are available; no intervention, loss
reduction or deployment is inferred from captured historical events.

The five-split mean ROC-AUC is 0.770626 for logistic and 0.780984 for the tree;
mean Brier is 0.136172 and 0.134666 respectively. Isotonic mean Brier is
0.136054, with its complete per-split results retained. These are descriptive
fixed-protocol results, not independent confidence estimates or a model-selection
claim. The report includes observed ranges and all 20 model/split rows.

## Source and model replay

The package includes a 980 KB compressed integer CSV. Its uncompressed SHA-256
is `4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb`.
Gzip packaging bytes can differ across Python/zlib versions; original values,
row order and canonical CSV bytes determine identity. A bounded decompression
read rejects unrelated source bytes before parsing them.

Verification checks every file digest, validates that frozen source, recomputes
the five splits, refits all models and compares scores, labels, review ranking,
counts, aggregate metrics and page text. It checks that the downloadable ZIP
contains exactly the same verified files. Recomputed internal hashes cannot
hide inconsistent scores or an altered download. Hashes and internal replay
still do not authenticate the publisher or the underlying source records.

To independently rebuild the source conversion, download the exact archive in
the source notice, install `python -m pip install ".[source]"`, then run:

```text
python scripts/convert_card_source.py original.zip rebuilt.csv.gz
```

The script verifies original ZIP and Excel digests, preserves every integer and
checks the canonical CSV digest. It performs no network request itself.

The source is a single historical Taiwan cohort with April–September 2005
repayment/bill/payment history. It has no per-record publication or revision
history. Random customer splits cannot validate original information availability,
future periods, current Chinese credit, recoveries or the event experiment's
180-day 90+ DPD research target.
