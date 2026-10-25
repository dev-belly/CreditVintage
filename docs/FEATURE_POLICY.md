# Explicit feature-coverage policies

Available in **v0.7.0**. The existing [quality diagnostic](FEATURE_QUALITY.md)
explains missing inputs but succeeds even when coverage is incomplete. This
separate workflow checks it against conditions you explicitly choose. It does
not train, read outcomes, fill gaps or relax scoring/distribution requirements.

## Run the example

After installing the [checked wheel](releases/v0.7.0.md), use new output paths:

```text
python -m creditvintage monitor-feature-quality --as-of 2024-10-01 --out outputs/quality
python -m creditvintage init-feature-policy --out coverage-policy.json
python -m creditvintage check-feature-quality outputs/quality --policy coverage-policy.json --out outputs/check
python -m creditvintage verify-feature-check outputs/check
```

The third command deliberately returns **3** for this example; keep its report
and run the fourth separately. Verification returns **0**, with `status: verified`
and `policy_status: fail`. Automated scripts must handle the expected example
failure instead of treating it as a successful gate.

Open `outputs/check/index.html` or the [published synthetic check](https://dev-belly.github.io/CreditVintage/feature-check/).
Of 320 current applications, utilization is available for 318 and the other two
inputs for 319 each. The generated example requires at least 20 applications and
full coverage per current feature. All three rules fail. Four applications are
incomplete; three failed feature rules and four incomplete applications are
different counts.

Review and edit the policy for your task. Example limits are illustrative, not
calibrated business recommendations. Your diagnostic may come from your own two
CSV tables and date configuration; the checker validates its retained events and
all diagnostic artifacts before producing a result.

## Exact policy contract

```json
{
  "schema": "creditvintage-feature-policy-v1",
  "rules": [
    {
      "id": "july-utilization",
      "population": "current",
      "vintage": "2024-07",
      "feature": "utilization",
      "min_applications": 20,
      "min_coverage": "1.0"
    }
  ]
}
```

Every rule requires exactly these six fields, with no hidden defaults.

| Field | Accepted meaning |
| --- | --- |
| `id` | Unique 1–64 ASCII letters, digits, underscores or hyphens; start with letter/digit |
| `population` | `reference` (calibration window) or `current` (test window) |
| `vintage` | `all`, or valid `YYYY-MM` intersecting that population's configured window |
| `feature` | `utilization`, `debt_to_income` or `prior_delinquencies` |
| `min_applications` | Positive JSON integer; booleans, floats and zero rejected |
| `min_coverage` | Decimal **string** in `[0, 1]`, up to 18 fractional places; no exponent, whitespace or nonfinite values |

Rules form a nonempty ordered list. Duplicate IDs/JSON keys, missing/unknown fields
and unsupported populations/features are errors. Rule order is retained and part
of policy identity. Overlapping predicates with different IDs are allowed, such
as an overall and a July check.

All expected slots stay in the denominator. Coverage counts inputs available
**at the decision**, for applications visible at the report cutoff. Known late
and post-decision events are not on-time availability. Future events cannot
satisfy earlier cutoffs.

Conditions use integer cross-multiplication against exact decimal fractions,
never rounded `coverage_rate`. Exact 2/3 fails required `"0.66666667"`, even though
the display is 0.66666667. Exact 78/80 passes `"0.975"` but fails
`"0.975000000000000001"`.

## Results and exit codes

| Condition | Rule result |
| --- | --- |
| Count below `min_applications` | `insufficient_data` |
| Enough applications, exact coverage below threshold | `fail` |
| Enough applications, coverage at or above threshold | `pass` |

Empty populations/months are insufficient even with required coverage `"0"`;
their display rate stays null. A future month within the configured window has
no visible evidence yet and is not dropped. A month outside the window is an
invalid policy, preventing a typo from silently checking another period.

All rules remain visible. Overall precedence: any `fail` makes the result `fail`;
otherwise any `insufficient_data` makes it insufficient; otherwise `pass`. Counts
expose mixed failures and insufficient groups separately.

| Command | Exit 0 | Exit 3 | Exit 2 |
| --- | --- | --- | --- |
| `init-feature-policy` | Example created | — | Existing path or write error |
| `check-feature-quality` | Every explicit rule passes | Failed/insufficient; evidence still written | Invalid policy, inconsistent source, existing destination or I/O error |
| `verify-feature-check` | Evidence replays; inspect `policy_status` | — | Evidence mismatch, invalid pin or different expected policy |

A tolerant or partial policy may pass with incomplete data. This does **not**
permit `run`, `monitor` or `monitor-features` to score missing inputs. Policy pass,
evidence verification, scoring readiness and source authenticity differ.

## Retained evidence and trusted policy

Each check contains `policy.json`, `check.json`, `checks.csv`, `index.html` and an
outer `manifest.json`, plus the complete six-file diagnostic under `quality/`.
It is self-contained: removing the original report does not prevent replay.
Existing destinations are refused; source reports and policy files are unchanged.

Verification rebuilds the diagnostic from events, reruns the rules and compares
every expected artifact byte. Changing a verdict to pass and recomputing its
manifest hash still fails.

Retain the `policy_sha256` printed by `init-feature-policy` **outside** the bundle
before sending it elsewhere. After editing a policy, review it and retain its
current digest from the check result separately. Supply that trusted digest:

```text
python -m creditvintage verify-feature-check outputs/check --expected-policy-sha256 YOUR_PREVIOUSLY_SAVED_DIGEST
```

Replace the placeholder with 64 lowercase hexadecimal digits. The digest hashes
validated policy JSON with sorted object keys and compact separators, encoded as
UTF-8; it is not the hash of the pretty-printed policy file.

Without a trusted pin, a wholly regenerated bundle with a different policy can
be internally consistent and verify. With the original pin, it is rejected.
The pin binds rules; it does not authenticate events or detect fully regenerated
replacement events under the same policy.

## Scope and validation

This is a fixed three-feature event-coverage policy, not arbitrary-column nullness,
a general Expectation framework, automatic thresholds, a monitoring service or
credit approval. No email/Slack actions are sent.

Tests cover hand-counted fractions, inclusive limits, display rounding,
small/empty/future months, mixed outcomes, strict parsing, trusted pins,
self-contained replay, source immutability and rehashed false verdicts. Installed
checks run healthy, missing-input and insufficient cases with the performance
file absent. Old quality/distribution formats and the public-v0.5.0 compatibility
fixture remain unchanged. [Mature-project comparison](COMPARISON.zh-CN.md).
