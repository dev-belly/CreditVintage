CREATE TABLE predictions (
    application_id TEXT PRIMARY KEY,
    decision_at TEXT NOT NULL,
    vintage TEXT NOT NULL,
    segment TEXT NOT NULL CHECK (segment IN ('retail', 'small_business')),
    principal_cents INTEGER NOT NULL CHECK (principal_cents > 0),
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    raw_pd REAL NOT NULL CHECK (raw_pd BETWEEN 0 AND 1),
    calibrated_pd REAL NOT NULL CHECK (calibrated_pd BETWEEN 0 AND 1),
    review_selected INTEGER NOT NULL CHECK (review_selected IN (0, 1))
);
