# Statlog (German Credit Data)

Hans Hofmann (1994). *Statlog (German Credit Data)*. UCI Machine Learning
Repository. DOI: https://doi.org/10.24432/C5NC77.

- Dataset and current license statement: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Download: https://archive.ics.uci.edu/static/public/144/statlog%2Bgerman%2Bcredit%2Bdata.zip
- Retrieved: 2026-10-15 UTC.
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**:
  https://creativecommons.org/licenses/by/4.0/
- `german.data`: unchanged original categorical/numeric 1,000-row table.
  SHA-256: `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871`.
- `german.doc`: unchanged original codebook. SHA-256:
  `69b92cb551111bc37098698288849c96f0fd87ce81a2e54451ff39aa1e53fe49`.

Redistribution retains attribution and this license notice. The project MIT
license does not replace the dataset's CC BY 4.0 terms. No endorsement by
Hofmann or UCI is implied. The frozen SHA-256 identifies the original bytes;
it does not establish source-record correctness or the absence of sampling bias.

The separate `public-benchmark` transforms native class 1 (good) to 0 and
class 2 (bad) to 1; excludes personal status/sex, age and foreign-worker status
from model inputs; learns numeric scaling and category encoding only on train;
and saves a fixed 600/200/200 stratified random split. Excluding these fields
does not prove fairness or eliminate proxies. Raw files remain unchanged.

No decision dates, historical publication dates, revisions, 180-day DPD labels,
recovery losses or modern Chinese lending population are supplied. They must
not be fabricated to make this static table fit the point-in-time event API.
