#!/usr/bin/env python3
"""
Data Ingestion Script for Customer Churn Prediction Pipeline

This script fetches data from two sources:
1. PostgreSQL Database: customer_demographics and transactions tables
2. REST API: customer_behavior JSON endpoint

Outputs are saved as CSV and JSON files in data/raw/ directory, partitioned by date.
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from pathlib import Path
import psycopg2
from psycopg2 import OperationalError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Data ingestion class for fetching data from multiple sources."""
    
    def __init__(self, config):
        """Initialize the DataIngestion class with configuration."""
        self.config = config
        self.base_path = Path("data/raw")
        self.today = datetime.now().strftime("%Y-%m-%d")
        
        # Create directories if they don't exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            self.base_path / "customer_demographics" / self.today,
            self.base_path / "transactions" / self.today,
            self.base_path / "customer_behavior" / self.today
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _get_postgres_connection(self):
        """Establish PostgreSQL database connection."""
        try:
            engine = create_engine(
                f"postgresql://{self.config['db_user']}:{self.config['db_password']}@"
                f"{self.config['db_host']}:{self.config['db_port']}/{self.config['db_name']}"
            )
            logger.info("Successfully connected to PostgreSQL database")
            return engine
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
    
    def fetch_customer_demographics(self):
        """Fetch customer demographics data from PostgreSQL."""
        try:
            engine = self._get_postgres_connection()
            query = "SELECT * FROM customer_demographics"
            
            logger.info("Fetching customer demographics data...")
            df = pd.read_sql(query, engine)
            
            # Save to CSV
            output_path = self.base_path / "customer_demographics" / self.today / "customer_demographics.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Successfully saved customer demographics data: {len(df)} records to {output_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching customer demographics: {str(e)}")
            raise
    
    def fetch_transactions(self):
        """Fetch transactions data from PostgreSQL."""
        try:
            engine = self._get_postgres_connection()
            query = "SELECT * FROM transactions"
            
            logger.info("Fetching transactions data...")
            df = pd.read_sql(query, engine)
            
            # Save to CSV
            output_path = self.base_path / "transactions" / self.today / "transactions.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Successfully saved transactions data: {len(df)} records to {output_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching transactions: {str(e)}")
            raise
    
    def fetch_customer_behavior(self):
        """Fetch customer behavior data from REST API."""
        try:
            url = self.config['api_url']
            headers = self.config.get('api_headers', {})
            
            logger.info(f"Fetching customer behavior data from: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save to JSON
            output_path = self.base_path / "customer_behavior" / self.today / "customer_behavior.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Successfully saved customer behavior data to {output_path}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching customer behavior from API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in customer behavior fetch: {str(e)}")
            raise
    
    def run_ingestion(self):
        """Run the complete data ingestion process."""
        logger.info("Starting data ingestion process...")
        
        try:
            # Fetch data from all sources
            demographics_df = self.fetch_customer_demographics()
            transactions_df = self.fetch_transactions()
            behavior_data = self.fetch_customer_behavior()
            
            logger.info("Data ingestion completed successfully!")
            
            return {
                'demographics': demographics_df,
                'transactions': transactions_df,
                'behavior': behavior_data
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise

def main():
    """Main function to run the data ingestion."""
    # Configuration - in production, this would come from environment variables or config files
    config = {
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_port': os.getenv('DB_PORT', '5432'),
        'db_name': os.getenv('DB_NAME', 'customer_db'),
        'db_user': os.getenv('DB_USER', 'username'),
        'db_password': os.getenv('DB_PASSWORD', 'password'),
        'api_url': os.getenv('API_URL', 'https://api.example.com/customer_behavior'),
        'api_headers': {
            'Authorization': os.getenv('API_TOKEN', 'your_token_here'),
            'Content-Type': 'application/json'
        }
    }
    
    try:
        ingestion = DataIngestion(config)
        results = ingestion.run_ingestion()
        logger.info("Data ingestion pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
