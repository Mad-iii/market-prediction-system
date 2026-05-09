# dags/market_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess

def run_script(script_path):
    result = subprocess.run(["python", script_path], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    if result.returncode != 0:
        raise Exception(f"Script {script_path} failed with error: {result.stderr}")
    print(result.stdout)

def run_ingestion():
    run_script("src/ingestion/run_all.py")

def run_sentiment():
    run_script("src/sentiment/run.py")

def run_timeseries():
    run_script("src/timeseries/run.py")

def run_training():
    run_script("src/models/run_train.py")

def run_evaluation():
    run_script("src/models/run_eval.py")

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