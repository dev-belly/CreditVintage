# Try a business case, then contribute a reproducible example

Start with the [SQL walkthrough](docs/BUSINESS_CASE.md) or the
[five-minute review workflow](docs/REVIEW_WORKFLOW.md). You do not need a model
server, a database service or private borrower records.

Install `python -m pip install -e ".[dev]"` in a Python 3.12+ environment.
Run `pytest tests/test_business.py tests/test_reviews.py -q` for those paths,
then `ruff check src tests`, `ruff format --check src tests` and `mypy src`.
The optional lineage path additionally requires the PITBridge revision pinned
in README and CI. CI exercises the complete suite on Linux and Windows.

Useful contributions preserve an explicit business contract: same-ID retry,
competing review requests, source availability, label maturity, missing data,
or a denominator with no qualifying observations. Include a small regression
with an independently calculated expected result and describe the old/new behavior.

For a failed trial, use the [trial feedback template](https://github.com/dev-belly/CreditVintage/issues/new?template=trial_feedback.yml).
Include the command, OS/Python, expected versus actual behavior and a minimal
synthetic sample. Keep names, borrower identifiers, credentials and private
documents out of public issues. An evidence reference in a review is a label
for a material; do not upload its private contents as part of a report.

Proposals for real decision policies need separate validated data and objectives.
The existing synthetic evaluation must remain identified as synthetic, and
completion counts must not be described as prevented credit losses.
