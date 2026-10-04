# Source-to-prediction integration

This integration uses PITBridge's scalar decision-time snapshots as CreditVintage model inputs. It does not consume PITBridge's rolling cash-flow statistics or export CreditVintage probabilities into StressAtlas.

## Explicit boundary contract

| Field | Contract |
| :--- | :--- |
| Identity | Each application ID must equal the PIT decision ID and entity ID. No implicit borrower join. |
| Decision | CreditVintage's calendar date means `23:59:59.999999` on that UTC day. Intraday decisions are rejected. |
| Availability | `max(published_at, ingested_at)` must be at or before the decision. |
| Source | Exactly `credit_application`; three explicit feature specs, each with a seven-day maximum age. |
| Features | `utilization` and `debt_to_income` are fractions in `[0,1]`; `prior_delinquencies` is an integer count in `[0,20]`. |
| Missingness | Missing, stale or deleted features fail the adapter. No zero imputation. |
| Money | Application principal is integer **USD cents**. No currency conversion. |
| Outcome | CreditVintage's existing 180-day 90+ DPD research label, full reporting coverage and maturation lag. |

A calendar-day export is safe only under the enforced UTC end-of-day convention. The source timestamps and actual cutoff remain in `lineage.csv`; they are not reconstructed from the shortened feature dates.

## Preserved evidence

`pit/` contains the original observations, decisions and specs, snapshot rows, an explicitly unsafe comparison, the original PIT report and its manifest. `applications.csv`, `features.csv` and `performance.csv` are the actual model inputs. `lineage.csv` ties each input feature to a selected source record. `evaluation/` contains the report and held-out predictions produced from those inputs. The root manifest names an exact fixed inventory; relative-path additions fail verification.

The default generated case contains 480 synthetic applications over the 2022 training, 2023 H2 calibration and 2024 H2 held-out vintages. Of 1,632 source observations, 192 are not available by their decision. The 1,440 exported values come from eligible source rows. There are 120 held-out predictions. These are sample counts, not a population leakage estimate. The repository versions the generator and adapter. GitHub Pages builds the synthetic case from that code, verifies the original reports and the complete lineage replay, then publishes the interactive explorer and a ZIP of its fixed evidence inventory. Pull requests exercise the build but cannot deploy. The generated bundle is not committed into Git history.

For example, a utilization correction is published at noon on a decision date but ingested at midnight the next day. Its availability is one microsecond after the UTC day-end cutoff, so the original revision remains the model feature.

## Replay and numerical tolerance

`verify-lineage` calls PITBridge's hash and semantic verification, which checks the SQL snapshots against independent Python enumeration. It reruns the adapter and compares the exported features and lineage. It checks the evaluation artifacts with CreditVintage's independent metric verifier, then refits from the three preserved input CSVs and compares each held-out probability with `rtol=1e-8, atol=1e-10`. Prediction identities and metadata must agree exactly. The root model-input digest must match the replay.

The dependency reference is PITBridge commit `2f2438aa9b62502c7b1c9773b572d09f6b04f9ea`. CI checks out that exact commit. The adapter accepts its `pitbridge/0.1.0` bundle schema; recording a reference commit in a manifest is documentation, not a signature attesting the installed package. File hashes and semantic replay do not establish real-world source authenticity.

## Tests

Integration regressions cover late ingestion, inclusive cutoffs to the microsecond, freshness boundaries, missing and deleted values, mismatched identities, intraday decisions, fraction/count units, unsafe exported record IDs and forged files with updated hashes. The semantic checks reject changed feature values, lineage, model-input digests, explorer content, malformed inventories, fractional or boolean counts and evaluation configurations that disagree with the full model replay. Tests can be run with `pytest` or the standard library's `unittest` discovery after both packages are installed.

The generator uses only seeded `random.Random` draws and fixed calendar dates; it does not read external borrower data. Model calibration remains measured rather than assumed to improve the result. There is no production ingestion, lending-policy deployment or regulatory validation claim.

## Published case

[Interactive source-to-prediction explorer](https://dev-belly.github.io/CreditVintage/lineage/) · [Complete evidence ZIP](https://dev-belly.github.io/CreditVintage/lineage/evidence.zip). The ZIP extracts into a `lineage/` folder. With both packages installed, run `creditvintage verify-lineage lineage`; its prediction replay uses the same numerical tolerance described above. Archive members have fixed timestamps and preserve the verified source bytes.
