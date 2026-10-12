-- One verified held-out application contributes once to each aggregation scope.
-- :lgd is the recorded scenario assumption, never fitted to observed outcomes.
WITH expanded AS (
    SELECT 'portfolio' AS scope, 'all' AS cohort, 'all' AS customer_segment, *
    FROM predictions
    UNION ALL
    SELECT 'vintage', vintage, 'all', * FROM predictions
    UNION ALL
    SELECT 'segment', 'all', segment, * FROM predictions
    UNION ALL
    SELECT 'vintage_segment', vintage, segment, * FROM predictions
), grouped AS (
    SELECT
        scope, cohort AS vintage, customer_segment AS segment,
        COUNT(*) AS loans,
        SUM(label) AS events,
        1.0 * SUM(label) / COUNT(*) AS event_rate,
        AVG(calibrated_pd) AS mean_pd,
        SUM(principal_cents) AS principal_cents,
        SUM(principal_cents * calibrated_pd) / SUM(principal_cents) AS exposure_weighted_pd,
        SUM(review_selected) AS reviewed,
        SUM(CASE WHEN review_selected = 1 THEN label ELSE 0 END) AS reviewed_events,
        SUM(principal_cents * calibrated_pd) * :lgd / 100.0 AS scenario_expected_loss_usd,
        SUM(CASE WHEN label = 1 THEN principal_cents ELSE 0 END) * :lgd / 100.0
            AS scenario_event_loss_proxy_usd
    FROM expanded
    GROUP BY scope, cohort, customer_segment
), changes AS (
    SELECT *, CASE WHEN scope IN ('vintage', 'vintage_segment') THEN
        LAG(event_rate) OVER (PARTITION BY scope, segment ORDER BY vintage)
    END AS previous_event_rate
    FROM grouped
)
SELECT
    scope, vintage, segment, loans, events,
    ROUND(event_rate, 8) AS event_rate,
    ROUND(mean_pd, 8) AS mean_pd,
    ROUND(event_rate - mean_pd, 8) AS event_pd_gap,
    principal_cents,
    ROUND(exposure_weighted_pd, 8) AS exposure_weighted_pd,
    reviewed, reviewed_events,
    ROUND(1.0 * reviewed / loans, 8) AS review_share,
    ROUND(1.0 * reviewed_events / NULLIF(reviewed, 0), 8) AS review_precision,
    ROUND(1.0 * reviewed_events / NULLIF(events, 0), 8) AS event_capture,
    ROUND(previous_event_rate, 8) AS previous_event_rate,
    ROUND(event_rate - previous_event_rate, 8) AS event_rate_change,
    ROUND(scenario_expected_loss_usd, 2) AS scenario_expected_loss_usd,
    ROUND(scenario_event_loss_proxy_usd, 2) AS scenario_event_loss_proxy_usd
FROM changes
ORDER BY scope, vintage, segment;
