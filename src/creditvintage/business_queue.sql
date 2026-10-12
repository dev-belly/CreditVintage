-- Historical fixed-capacity selection. Labels are deliberately absent.
-- Do not use future outcomes as an input to prioritization.
SELECT
    ROW_NUMBER() OVER (ORDER BY calibrated_pd DESC, application_id) AS review_rank,
    application_id, decision_at, vintage, segment, principal_cents, raw_pd, calibrated_pd
FROM predictions
WHERE review_selected = 1
ORDER BY calibrated_pd DESC, application_id;
