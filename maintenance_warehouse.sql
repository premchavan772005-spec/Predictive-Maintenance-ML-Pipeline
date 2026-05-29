USE predictive_maintenance_db;

-- 1. Check the distribution of normal operations vs. equipment failures
SELECT 
    Target,
    COUNT(*) AS total_records,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM fact_sensor_logs), 2) AS percentage
FROM fact_sensor_logs
GROUP BY Target;

-- 2. Breakdown of the exact engineering failure modes
SELECT 
    Failure_Type,
    COUNT(*) AS total_failures
FROM fact_sensor_logs
WHERE Target = 1
GROUP BY Failure_Type
ORDER BY total_failures DESC;