from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.insert(0, '/home/jagac/airflow/dags/current-business-climate')

from extract_youtube import youtube_main
from extract_reddit import reddit_main
from utils import append_tables

with DAG(
    dag_id = "text_pipeline",
    schedule_interval = "@weekly",
    start_date = datetime(year = 2023, month =5, day = 12),
    catchup = False
) as dag:
    
    task_start = BashOperator(
        task_id = "start",
        bash_command = "date"
    )
    
    task_reddit_etl = PythonOperator(
        task_id = "reddit_etl",
        python_callable = reddit_main
    )
    
    task_youtube_etl = PythonOperator(
        task_id = "youtube_etl",
        python_callable = youtube_main
    )

    
    combine_tables = PythonOperator(
        task_id = "combine_tables",
        python_callable = append_tables
    )
    
    
task_start >> task_reddit_etl
task_start >> task_youtube_etl
[task_reddit_etl, task_youtube_etl] >> combine_tables
