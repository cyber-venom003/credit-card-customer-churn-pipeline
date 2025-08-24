import pandas as pd
import numpy as np
import os
import glob
import sqlite3
import logging

# --- Dynamic Path Configuration (THE FIX) ---
# This makes the script runnable from anywhere in your project
# Get the absolute path of the directory where this script is located (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Get the parent directory of SCRIPT_DIR, which is your project's root folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define paths relative to the project root
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'transformation.log')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
FEATURE_STORE_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'features.db')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

# --- Logging Configuration ---
# Ensure the logs directory exists before setting up logging
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_latest_data_path(source_name: str) -> str:
    """Finds the most recent data folder for a given source."""
    search_path = os.path.join(RAW_DATA_PATH, source_name, '*')
    all_subdirs = glob.glob(search_path)
    if not all_subdirs:
        raise FileNotFoundError(f"No data found for source: {source_name}")
    return max(all_subdirs, key=os.path.getmtime)

def process_data():
    """
    Main function to load, process, and engineer features from raw data.
    """
    print("--- Starting Data Transformation and Feature Engineering ---")
    logging.info("--- Starting Data Transformation and Feature Engineering ---")

    try:
        # --- 1. Load the Latest Raw Data ---
        logging.info("Loading latest raw data...")
        
        demographics_path = find_latest_data_path('customer_demographics')
        transactions_path = find_latest_data_path('transactions')
        behavior_path = find_latest_data_path('customer_behavior')

        df_demo = pd.read_csv(os.path.join(demographics_path, 'customer_demographics.csv'))
        df_trans = pd.read_csv(os.path.join(transactions_path, 'transactions.csv'))
        # The behavior file is json, so we load it accordingly
        df_behav = pd.read_json(os.path.join(behavior_path, 'customer_behavior.json'))
        
        logging.info("Successfully loaded raw data.")

        # --- 2. Initial Cleaning and Preprocessing ---
        logging.info("Performing initial data cleaning...")
        
        df_demo['registration_date'] = pd.to_datetime(df_demo['registration_date'])
        df_demo['churn_date'] = pd.to_datetime(df_demo['churn_date'])
        df_trans['transaction_date'] = pd.to_datetime(df_trans['transaction_date'])
        # The behavior data comes in a nested structure, extract the actual data
        df_behav = pd.json_normalize(df_behav['data'])
        
        df_demo['churn'] = (df_demo['account_status'] == 'Churned').astype(int)

        # --- 3. Feature Engineering ---
        logging.info("Starting feature engineering...")
        
        trans_agg = df_trans.groupby('customer_id').agg(
            total_transactions=('transaction_id', 'count'),
            total_spend=('transaction_amount', 'sum'),
            avg_transaction_amount=('transaction_amount', 'mean'),
            std_transaction_amount=('transaction_amount', 'std'),
            total_transaction_fees=('transaction_fee', 'sum'),
            first_transaction_date=('transaction_date', 'min'),
            last_transaction_date=('transaction_date', 'max')
        ).reset_index()

        behav_agg = df_behav.groupby('customer_id').agg(
            total_logins=('total_logins', 'sum'),
            avg_session_duration=('avg_session_minutes', 'mean'),
            total_pages_viewed=('total_app_crashes', 'sum'),  # Using app crashes as proxy for engagement
            total_support_tickets=('total_support_tickets', 'sum'),
            avg_satisfaction_score=('avg_satisfaction_score', 'mean')
        ).reset_index()

        # --- 4. Merge DataFrames into a Single Feature Set ---
        logging.info("Merging data sources into a final feature set...")
        
        df_features = df_demo.copy()
        df_features = pd.merge(df_features, trans_agg, on='customer_id', how='left')
        df_features = pd.merge(df_features, behav_agg, on='customer_id', how='left')

        # --- 5. Create More Advanced Features ---
        logging.info("Creating advanced time-based and interaction features...")

        end_date = df_features['last_transaction_date'].max()
        df_features['tenure_days'] = (df_features['churn_date'].fillna(end_date) - df_features['registration_date']).dt.days

        df_features['days_since_last_transaction'] = (end_date - df_features['last_transaction_date']).dt.days
        
        df_features['transactions_per_day'] = df_features['total_transactions'] / df_features['tenure_days']
        
        df_features.replace([np.inf, -np.inf], 0, inplace=True)

        # --- 6. Final Cleaning and Preparation for Model ---
        logging.info("Performing final data cleaning...")

        num_cols_to_fill = [
            'total_transactions', 'total_spend', 'avg_transaction_amount', 'std_transaction_amount',
            'total_transaction_fees', 'total_logins', 'avg_session_duration', 'total_pages_viewed',
            'total_support_tickets', 'avg_satisfaction_score', 'days_since_last_transaction', 'tenure_days'
        ]
        for col in num_cols_to_fill:
            df_features[col] = df_features[col].fillna(0)
        
        categorical_cols = ['gender', 'income_bracket', 'education_level', 'subscription_plan', 'customer_segment']
        df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
        
        cols_to_drop = [
            'registration_date', 'churn_date', 'churn_type', 'location_city', 'location_state', 
            'marketing_channel', 'account_status', 'first_transaction_date', 'last_transaction_date'
        ]
        # Some columns might not exist if all values were NaN, so drop safely
        df_features = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])

        cols = ['customer_id'] + [col for col in df_features if col != 'customer_id']
        df_features = df_features[cols]
        
        # --- 7. Save to Processed Layer and Feature Store ---
        logging.info("Saving final feature set...")
        
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        df_features.to_csv(os.path.join(PROCESSED_DATA_PATH, 'final_features.csv'), index=False)
        
        os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
        conn = sqlite3.connect(FEATURE_STORE_PATH)
        df_features.to_sql('customer_features', conn, if_exists='replace', index=False)
        conn.close()

        print(f"SUCCESS: Successfully transformed data and saved {len(df_features)} records to the feature store.")
        logging.info(f"Successfully transformed data and saved {len(df_features)} records to the feature store.")

    except FileNotFoundError as e:
        logging.error(f"Data processing failed: {e}")
        print(f"ERROR: Data processing failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during data transformation: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred during data transformation: {e}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == "__main__":
    process_data()