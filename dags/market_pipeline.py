# dags/market_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="market_prediction_pipeline",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    ingest   = PythonOperator(task_id="ingest_data",       python_callable=run_ingestion)
    sentiment= PythonOperator(task_id="label_sentiment",   python_callable=run_sentiment)
    build_ts = PythonOperator(task_id="build_timeseries",  python_callable=run_timeseries)
    train    = PythonOperator(task_id="train_models",      python_callable=run_training)
    evaluate = PythonOperator(task_id="evaluate_models",   python_callable=run_evaluation)

    ingest >> sentiment >> build_ts >> train >> evaluate