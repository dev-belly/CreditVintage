# Default of Credit Card Clients — frozen conversion

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C55S3H.

- Official page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Archive: https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip
- Retrieved: 2026-10-15 UTC.
- License: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/.
- Original ZIP SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
- Original `default of credit card clients.xls` SHA-256:
  `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
- Uncompressed canonical CSV SHA-256:
  `4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb`.

Changes: the original sheet's second row supplies headers. All 30,000 rows,
25 columns, integer values and source order are retained. The table is encoded
as UTF-8 CSV with LF line endings, then gzip-compressed. There are 23,364 native
zero labels and 6,636 native one labels. No source field, label, date or missing
value is synthesized. The source-rebuild script verifies the original archive
and Excel hashes before conversion and verifies the canonical CSV hash after it.
Gzip packaging bytes may vary across Python/zlib versions; the uncompressed
canonical SHA-256 defines the frozen table identity.

The experiment uses 19 financial-history fields. ID, sex, education, marital
status and age are not model inputs. Source repayment-status codes such as -2
and 0 remain distinct categories; they are not assigned invented meanings.
Demographic exclusion does not establish fairness or remove proxy effects.

This is a single historical Taiwan cohort. It supplies repayment/bill/payment
history for April–September 2005 and the native `default payment next month`
label, without per-record publication or revision histories. Random partitions
are not future-period or point-in-time validation. No 180-day DPD target,
recovery loss, intervention result or source-system availability is inferred.

The dataset retains its CC BY 4.0 terms independently of project code. Credit
is retained in the package, report and downloadable evidence. No endorsement
by Yeh or UCI is implied. A digest identifies bytes and cannot validate the
underlying source records or sampling process.
