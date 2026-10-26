# SBA observed funding-cohort data

Publisher: **U.S. Small Business Administration**. Official catalogue:
<https://data.sba.gov/dataset/7a-504-foia>, checked 2026-10-26.
Snapshot date: **2026-06-30**. The catalogue designates **U.S. Government Works**;
see <https://www.usa.gov/government-copyright>. No SBA endorsement is implied.

License identifier: **LicenseRef-US-Government-Works**. This designation applies
to these SBA source data; it does not relicense the project's MIT software or
the separately attributed CC BY 4.0 UCI data. The catalogue's designation is
not a claim that all material hosted by government websites is unrestricted.

## Pinned originals

Files are under <https://data.sba.gov/sites/default/files/uploaded_resources/>.
SHA256 values bind the exact snapshot, including its historical corrections.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `FOIA_7a_FY2000_FY2009_asof_260630.csv` | 318425094 | `66674e18a700fbba0378c25118c291ac2759784a550c643cab5747eced763d6a` |
| `FOIA_7a_FY2010_FY2019_asof_260630.csv` | 255101999 | `01a3e2c7988a6f4052e53f218a309feb2ec2fe42887bebdc0fa94ac8b1024ade` |
| `7a_504_foia_data_dictionary.xlsx` | 24529 | `777ba97a6f92e325f79905610816cb59d41f1971cd964c54dc30ef02730a3941` |

The CSVs contain 690,333 and 545,751 logical records, respectively. The data
dictionary defines initial interest rate and processing method at approval;
current term and final outcome information are deliberately excluded from
predictors. EXEMPT is a disclosure status for disbursed loans not reported as
cancelled, paid or charged off. It is not an independently verified no-default
status. Outcomes here mean **recorded charge-offs in this final snapshot**.

## Frozen conversion

`protocol.json` records the fixed target, windows, grouping, sampling, features
and numerical protocol. `audit.json` records all source row counts, exclusions,
eligible/grouped cohort sizes and the canonical CSV digest. The pinned converter
is `scripts/prepare_sba.py`; no network is needed for installed-wheel evaluation.

1. Verify both original file digests before creating output. Parse logical CSV
   rows with one-based ordinals, not physical line numbers.
2. Retain approval/funding and outcome dates that satisfy the declared checks;
   retain only fixed 2006–2010, 2013 and 2016–2017 funding windows with mature
   24-calendar-month horizons. Invalid dates, nonfinite/negative money, missing
   group fields and CHGOFF without a charge-off date are counted as exclusions.
3. Normalize recorded borrower name/state/ZIP by removing non-ASCII
   alphanumeric characters after uppercasing. Hash the pipe-joined values.
   Retain the earliest eligible funding date per key, ties by original record ID.
   This is an imperfect matching heuristic, not a verified legal entity ID.
4. For each stage, select the lowest SHA256 of `original_file_sha:ordinal`.
   Fix 20,000 train, 5,000 calibration, 10,000 test records. Outcomes never enter
   sampling priority. Sort by stage, funding date and original record ID.
5. Export the declared 16 columns as UTF-8 CSV with LF endings and deterministic
   gzip metadata. Borrower names, addresses and ZIPs are not exported. Hashed
   keys and source row IDs are pseudonymous and linkable to the public source;
   do not describe the resulting dataset as anonymous.

The source CSV is bundled for offline refitting and audited against its exact
digest. Raw data is about 574 MB and remains outside the repository. Original
source reconstruction is optional and separately documented in `docs/SBA_CREDIT.md`.

## Target and limitations

`chargeoff_24m=1` iff the recorded charge-off date falls between first funding
and its inclusive 24-month anniversary. Leap-day month ends are clamped.
Later lifetime charge-offs are 0 for this horizon; missing CHGOFF dates are
excluded. Paid-in-full dates are retained as source evidence, not predictors.

The final snapshot has no first-publication times or historical feature
revisions. These are retrospective funding cohorts; historical public
availability cannot be reconstructed and `available_at` is not invented.
This target differs from UCI native labels and the synthetic 180-day
delinquency study. Results do not establish fairness, prevented losses,
production readiness, current Chinese credit performance or independent use.
