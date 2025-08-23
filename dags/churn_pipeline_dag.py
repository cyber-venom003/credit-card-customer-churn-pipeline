"""
Apache Airflow DAG for Customer Churn Prediction Pipeline

This DAG orchestrates the entire customer churn prediction pipeline with the following flow:
ingestion >> validation >> transformation >> model_training

The pipeline includes:
1. Data ingestion from PostgreSQL and REST API
2. Data validation and quality checks
3. Feature engineering and transformation
4. Model building and training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

# Default arguments for the DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create the DAG
dag = DAG(
    'customer_churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end customer churn prediction pipeline',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    max_active_runs=1,
    tags=['churn', 'prediction', 'ml', 'customer_analytics']
)

# Define the tasks

# Start task
start = DummyOperator(
    task_id='start',
    dag=dag
)

# Data Ingestion Task
ingestion = BashOperator(
    task_id='ingestion',
    bash_command='cd /Users/priyansh/Desktop/DMML_Assignment/credit-card-customer-churn-pipeline && python src/2_ingestion.py',
    dag=dag
)

# Data Validation Task
validation = BashOperator(
    task_id='validation',
    bash_command='cd /Users/priyansh/Desktop/DMML_Assignment/credit-card-customer-churn-pipeline && python src/4_validation.py',
    dag=dag
)

# Feature Engineering/Transformation Task
transformation = BashOperator(
    task_id='transformation',
    bash_command='cd /Users/priyansh/Desktop/DMML_Assignment/credit-card-customer-churn-pipeline && python src/6_transformation.py',
    dag=dag
)

# Model Training Task
model_training = BashOperator(
    task_id='model_training',
    bash_command='cd /Users/priyansh/Desktop/DMML_Assignment/credit-card-customer-churn-pipeline && python src/9_model_building.py',
    dag=dag
)

# End task
end = DummyOperator(
    task_id='end',
    dag=dag
)

# Define the task dependencies
start >> ingestion >> validation >> transformation >> model_training >> end

# Task documentation
ingestion.doc_md = """
## Data Ingestion Task

This task fetches data from multiple sources:
- PostgreSQL database (customer_demographics and transactions tables)
- REST API (customer_behavior endpoint)

**Output**: Raw data files saved in data/raw/ directory, partitioned by date.
"""

validation.doc_md = """
## Data Validation Task

This task performs data quality checks:
- Automatically finds the most recently ingested data
- Checks for null values and duplicate records
- Generates validation reports in docs/ directory

**Output**: Validation reports (CSV and Markdown formats).
"""

transformation.doc_md = """
## Feature Engineering Task

This task creates predictive features:
- Merges data from different sources using customer_id
- Engineers new features (total_spend, transaction_frequency, customer_tenure, etc.)
- Saves final dataset to SQLite database

**Output**: Feature-rich dataset in data/features/features.db
"""

model_training.doc_md = """
## Model Training Task

This task builds the churn prediction model:
- Uses MLflow for experiment tracking and reproducibility
- Trains RandomForestClassifier with engineered features
- Logs hyperparameters, metrics, and model artifacts
- Saves trained model locally

**Output**: Trained model, evaluation metrics, and MLflow artifacts.
"""

# DAG documentation
dag.doc_md = """
# Customer Churn Prediction Pipeline

## Overview
This DAG implements a complete end-to-end pipeline for predicting customer churn using machine learning.

## Pipeline Flow
1. **Data Ingestion** → Fetches data from PostgreSQL and REST API
2. **Data Validation** → Performs quality checks and generates reports
3. **Feature Engineering** → Creates predictive features and merges data sources
4. **Model Training** → Trains RandomForest model with MLflow tracking

## Schedule
- Runs daily at 2:00 AM
- Maximum 1 active run at a time
- Includes retry logic for failed tasks

## Dependencies
- Each step depends on the successful completion of the previous step
- Data flows from raw ingestion through to final model artifacts

## Outputs
- Raw data files (CSV/JSON)
- Validation reports
- Feature database (SQLite)
- Trained model and MLflow artifacts
"""
