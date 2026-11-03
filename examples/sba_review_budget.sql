-- Scores choose the queue. Recorded outcomes are joined only for evaluation.
-- Equal scores use the same explicit record-ID ordering as the frozen study.
CREATE TEMP VIEW ranked_scores AS
SELECT method, record_id, score,
       ROW_NUMBER() OVER (
           PARTITION BY method ORDER BY score DESC, record_id COLLATE BINARY
       ) AS selection_rank
FROM scores;

CREATE TEMP VIEW policy_results AS
WITH capacities(percent) AS (VALUES (5), (10), (20)),
     population AS (
         SELECT COUNT(*) AS loans, SUM(chargeoff_24m) AS events FROM outcomes
     )
SELECT r.method,
       c.percent / 100.0 AS capacity,
       COUNT(*) AS reviewed,
       SUM(o.chargeoff_24m) AS captured,
       p.events AS total_events,
       1.0 * SUM(o.chargeoff_24m) / NULLIF(COUNT(*), 0) AS precision,
       1.0 * SUM(o.chargeoff_24m) / NULLIF(p.events, 0) AS capture
FROM ranked_scores AS r
CROSS JOIN capacities AS c
CROSS JOIN population AS p
JOIN outcomes AS o ON o.record_id = r.record_id
-- Integer ceiling: the frozen budgets are exactly 5%, 10% and 20%.
WHERE r.selection_rank <= (p.loans * c.percent + 99) / 100
GROUP BY r.method, c.percent;

-- The retained queue is the fixed 10% queue, with no future outcome column.
CREATE TEMP VIEW queue_results AS
SELECT method, record_id, selection_rank AS rank, score
FROM ranked_scores
WHERE selection_rank <= ((SELECT COUNT(*) FROM outcomes) * 10 + 99) / 100;
