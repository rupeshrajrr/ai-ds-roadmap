# Day 5 — SQL & DuckDB: NYC Taxi Analytics

Five production-grade analytical SQL queries on NYC Yellow Taxi Parquet files.
Covers window functions (RANK, DENSE_RANK, rolling moving averages), CTEs,
aggregations, and fare anomaly detection via z-scores.

## Reproduce
```powershell
.venv\Scripts\Activate.ps1
cd day5_duckdb
python run_queries.py
```

## Queries

| # | Description | Technique |
|---|-------------|-----------|
| 1 | Daily trip volume & moving boundaries | CTE + 7-Day Moving Window Frame |
| 2 | Peak hourly demand & weekday cycles | Conditional Aggregation (CASE WHEN) |
| 3 | Top 5 revenue routes per zone | DENSE_RANK() partition by location |
| 4 | Passenger count tipping efficiencies | Analytical Grouping + Descriptive ratios |
| 5 | Transaction anomaly detection | Z-score isolation via 3x Standard Deviation |


