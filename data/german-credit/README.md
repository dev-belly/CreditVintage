# UCI Statlog German Credit — source and attribution

Hans Hofmann (1994), **Statlog (German Credit Data)**, UCI Machine Learning
Repository. DOI: <https://doi.org/10.24432/C5NC77>.

- Official dataset: <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
- Download: <https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip>
- Retrieved: 2026-10-15.
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
  The dataset and original codebook retain their own license, separate from the repository's code.

`german.data` and `german.doc` are unchanged files extracted from that archive.
The source is 1,000 rows with 20 attributes and one class: **1 = good, 2 = bad
credit risk**. The example maps those classes to 0 and 1; it does not change the
source bytes. Original row numbers identify rows, not verified unique borrowers.
Attribute meanings, including the seven numeric versus thirteen categorical
fields, are in the original [codebook](german.doc).

| File | SHA-256 |
| --- | --- |
| Downloaded ZIP | `e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b` |
| `german.data` | `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871` |

No application, publication, ingestion or outcome dates are supplied. The data
cannot be converted into the project's dated-event input format without inventing
information. The [separate public-data protocol](../../docs/PUBLIC_CREDIT.md)
therefore uses stratified random splits and calls the positive class a **bad
credit-risk label**, not a measured 180-day 90+ DPD event. All 20 original
attributes are used for this historical comparison, including age, personal
status/sex and foreign-worker status; feature availability, sampling provenance,
fairness and suitability for lending decisions are not established by this example.
