
-- ================================================================
-- STARTUP FAILURE PATTERN ANALYSIS — SQL REFERENCE
-- All queries designed for SQLite / standard ANSI SQL
-- ================================================================

-- 1. Yearly ecosystem pulse with median
SELECT year,
       COUNT(*) AS deals,
       COUNT(DISTINCT startup) AS unique_startups,
       ROUND(SUM(amount)/1e6, 1) AS total_funding_M,
       ROUND(AVG(amount)/1e6, 3) AS avg_deal_M,
       ROUND(MEDIAN(amount)/1e6, 3) AS median_deal_M,
       SUM(CASE WHEN amount >= 10000000 THEN 1 ELSE 0 END) AS mega_deals
FROM startups
GROUP BY year ORDER BY year;

-- 2. Industry risk matrix
SELECT industry,
       COUNT(*) AS total_deals,
       ROUND(100.0*SUM(CASE WHEN amount=0 THEN 1 ELSE 0 END)/COUNT(*),1) AS unfunded_pct,
       ROUND(AVG(CASE WHEN amount>0 THEN amount END)/1e6, 3) AS avg_funded_M
FROM startups
WHERE industry NOT IN ('Unknown','')
GROUP BY industry HAVING total_deals >= 10
ORDER BY unfunded_pct DESC;

-- 3. Multi-round survival proxy
SELECT startup,
       COUNT(*) AS rounds,
       SUM(amount) AS total_raised,
       MAX(year)-MIN(year) AS lifespan_years,
       CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS survived
FROM startups
GROUP BY startup
ORDER BY total_raised DESC;

-- 4. City quality with statistical view
SELECT city,
       COUNT(*) AS deals,
       COUNT(DISTINCT industry) AS diversification,
       ROUND(SUM(amount)/1e6,1) AS total_M,
       ROUND(100.0*SUM(CASE WHEN amount>0 THEN 1 ELSE 0 END)/COUNT(*),1) AS funded_pct
FROM startups
WHERE city NOT IN ('Unknown','')
GROUP BY city HAVING deals >= 10
ORDER BY total_M DESC;

-- 5. Pareto analysis — funding concentration
WITH ranked AS (
    SELECT startup,
           SUM(amount) AS total_raised,
           RANK() OVER (ORDER BY SUM(amount) DESC) AS rnk
    FROM startups
    WHERE amount > 0
    GROUP BY startup
),
totals AS (SELECT COUNT(*) AS n, SUM(total_raised) AS grand_total FROM ranked)
SELECT r.startup, r.total_raised, r.rnk,
       ROUND(100.0*r.rnk/t.n, 2) AS rank_pct,
       ROUND(100.0*SUM(r2.total_raised)/t.grand_total, 2) AS cum_funding_pct
FROM ranked r
CROSS JOIN totals t
JOIN ranked r2 ON r2.rnk <= r.rnk
GROUP BY r.rnk ORDER BY r.rnk;

-- 6. Quarterly deal flow (seasonal pattern)
SELECT year, quarter, COUNT(*) AS deals,
       ROUND(SUM(amount)/1e6,1) AS total_M
FROM startups
GROUP BY year, quarter ORDER BY year, quarter;

-- 7. Industry market share over time (window function)
WITH base AS (
    SELECT year, industry, COUNT(*) AS deals
    FROM startups
    WHERE industry NOT IN ('Unknown','')
    GROUP BY year, industry
),
yr_totals AS (SELECT year, SUM(deals) AS yr_total FROM base GROUP BY year)
SELECT b.year, b.industry,
       ROUND(100.0*b.deals/y.yr_total, 1) AS market_share_pct
FROM base b JOIN yr_totals y ON b.year=y.year
ORDER BY b.year, b.deals DESC;

-- 8. Top investors with portfolio diversity
SELECT investor,
       COUNT(DISTINCT startup) AS portfolio_size,
       COUNT(DISTINCT industry) AS industries_covered,
       COUNT(DISTINCT city) AS cities_active,
       ROUND(SUM(amount)/1e6,2) AS total_deployed_M
FROM startups
WHERE investor NOT IN ('Unknown','') AND LENGTH(investor) < 60
GROUP BY investor HAVING portfolio_size >= 3
ORDER BY portfolio_size DESC LIMIT 15;

-- 9. Funding band survival analysis
SELECT
    CASE
        WHEN total_raised = 0              THEN 'None'
        WHEN total_raised < 100000         THEN 'Under $100K'
        WHEN total_raised < 1000000        THEN '$100K to $1M'
        WHEN total_raised < 5000000        THEN '$1M to $5M'
        WHEN total_raised < 20000000       THEN '$5M to $20M'
        WHEN total_raised < 100000000      THEN '$20M to $100M'
        ELSE '$100M+'
    END AS funding_band,
    COUNT(*) AS startups,
    SUM(survived) AS multi_round,
    ROUND(100.0*SUM(survived)/COUNT(*),1) AS survival_rate_pct
FROM (
    SELECT startup,
           SUM(amount) AS total_raised,
           CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS survived
    FROM startups GROUP BY startup
)
GROUP BY funding_band;
