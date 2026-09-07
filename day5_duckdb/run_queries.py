import os
import duckdb

def run_pipeline():
    os.makedirs("results", exist_ok=True)
    conn = duckdb.connect()
    print("🦆 Connected to DuckDB successfully.")
    
    parquet_path = "data/yellow_tripdata_2024-01.parquet"
    
    queries = {
        "query1_daily_volume": f"""
            WITH daily_stats AS (
                SELECT 
                    tpep_pickup_datetime::DATE AS trip_date,
                    COUNT(*) AS daily_trips,
                    ROUND(SUM(total_amount), 2) AS daily_revenue,
                    ROUND(AVG(trip_distance), 2) AS avg_distance
                FROM read_parquet('{parquet_path}')
                WHERE tpep_pickup_datetime >= '2024-01-01' 
                  AND tpep_pickup_datetime < '2024-02-01'
                GROUP BY trip_date
            )
            SELECT trip_date, daily_trips, daily_revenue, avg_distance,
                   ROUND(AVG(daily_trips) OVER (ORDER BY trip_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS rolling_7d_avg_trips
            FROM daily_stats ORDER BY trip_date
        """,
        "query2_hourly_demand": f"""
            SELECT 
                EXTRACT(hour FROM tpep_pickup_datetime) AS pickup_hour,
                COUNT(CASE WHEN EXTRACT(dow FROM tpep_pickup_datetime) IN (1,2,3,4,5) THEN 1 END) AS weekday_trips,
                COUNT(CASE WHEN EXTRACT(dow FROM tpep_pickup_datetime) IN (0,6) THEN 1 END) AS weekend_trips,
                ROUND(AVG(fare_amount), 2) AS avg_hourly_fare
            FROM read_parquet('{parquet_path}')
            WHERE tpep_pickup_datetime >= '2024-01-01' AND tpep_pickup_datetime < '2024-02-01'
            GROUP BY pickup_hour ORDER BY pickup_hour
        """,
        "query3_top_routes": f"""
            WITH route_revenue AS (
                SELECT 
                    PULocationID, DOLocationID,
                    COUNT(*) AS total_trips,
                    ROUND(SUM(total_amount), 2) AS gross_revenue,
                    DENSE_RANK() OVER (PARTITION BY PULocationID ORDER BY SUM(total_amount) DESC) AS revenue_rank
                FROM read_parquet('{parquet_path}')
                WHERE tpep_pickup_datetime >= '2024-01-01' AND tpep_pickup_datetime < '2024-02-01'
                GROUP BY PULocationID, DOLocationID
            )
            SELECT PULocationID, DOLocationID, total_trips, gross_revenue, revenue_rank
            FROM route_revenue WHERE revenue_rank <= 5 ORDER BY PULocationID, revenue_rank
        """,
        "query4_tipping_efficiency": f"""
            SELECT 
                passenger_count, COUNT(*) AS total_trips,
                ROUND(AVG(tip_amount), 2) AS avg_tip,
                ROUND(AVG(fare_amount), 2) AS avg_fare,
                ROUND(AVG(tip_amount / NULLIF(fare_amount, 0)) * 100, 2) AS avg_tip_percentage
            FROM read_parquet('{parquet_path}')
            WHERE tpep_pickup_datetime >= '2024-01-01' AND tpep_pickup_datetime < '2024-02-01'
              AND fare_amount > 0 AND passenger_count IS NOT NULL
            GROUP BY passenger_count HAVING total_trips > 100 ORDER BY passenger_count
        """,
        "query5_anomaly_detection": f"""
            WITH stats AS (
                SELECT AVG(fare_amount) AS avg_f, STDDEV(fare_amount) AS std_f,
                       AVG(trip_distance) AS avg_d, STDDEV(trip_distance) AS std_d
                FROM read_parquet('{parquet_path}')
                WHERE tpep_pickup_datetime >= '2024-01-01' AND tpep_pickup_datetime < '2024-02-01'
                  AND fare_amount > 0 AND trip_distance > 0
            )
            SELECT tpep_pickup_datetime, PULocationID, DOLocationID, trip_distance, fare_amount,
                   ROUND((fare_amount - stats.avg_f) / stats.std_f, 2) AS fare_z_score,
                   ROUND((trip_distance - stats.avg_d) / stats.std_d, 2) AS dist_z_score
            FROM read_parquet('{parquet_path}'), stats
            WHERE tpep_pickup_datetime >= '2024-01-01' AND tpep_pickup_datetime < '2024-02-01'
              AND (fare_amount > (stats.avg_f + 3 * stats.std_f) OR trip_distance > (stats.avg_d + 3 * stats.std_d))
            ORDER BY fare_amount DESC LIMIT 100
        """
    }

    for name, sql in queries.items():
        csv_output_path = f"results/{name}.csv"
        print(f"🚀 Processing: {name}...")
        copy_query = f"COPY ({sql}) TO '{csv_output_path}' (HEADER, DELIMITER ',')"
        conn.execute(copy_query)
        rows_written = conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{csv_output_path}')").fetchone()[0]
        print(f"✅ Saved results to {csv_output_path} ({rows_written:,} rows extracted).\n")

if __name__ == "__main__":
    run_pipeline()
