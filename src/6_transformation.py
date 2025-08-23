#!/usr/bin/env python3
"""
Feature Engineering Script for Customer Churn Prediction Pipeline

This script will:
1. Load the cleaned, processed data
2. Merge the different data sources using customer_id
3. Engineer new, meaningful features
4. Save the final, feature-rich dataset into a SQLite database
"""

import os
import sys
import logging
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/transformation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering class for creating predictive features."""
    
    def __init__(self):
        """Initialize the FeatureEngineer class."""
        self.processed_path = Path("data/processed")
        self.features_path = Path("data/features")
        self.features_db_path = self.features_path / "features.db"
        
        # Create features directory if it doesn't exist
        self.features_path.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.demographics_df = None
        self.transactions_df = None
        self.behavior_df = None
        self.features_df = None
    
    def load_processed_data(self):
        """Load the cleaned and processed data from all sources."""
        logger.info("Loading processed data...")
        
        try:
            # Load customer demographics
            demographics_file = self.processed_path / "customer_demographics_processed.csv"
            if demographics_file.exists():
                self.demographics_df = pd.read_csv(demographics_file)
                logger.info(f"✅ Loaded demographics data: {self.demographics_df.shape}")
            else:
                logger.warning(f"⚠️  Demographics file not found: {demographics_file}")
            
            # Load transactions data
            transactions_file = self.processed_path / "transactions_processed.csv"
            if transactions_file.exists():
                self.transactions_df = pd.read_csv(transactions_file)
                logger.info(f"✅ Loaded transactions data: {self.transactions_df.shape}")
            else:
                logger.warning(f"⚠️  Transactions file not found: {transactions_file}")
            
            # Load customer behavior data
            behavior_file = self.processed_path / "customer_behavior_processed.csv"
            if behavior_file.exists():
                self.behavior_df = pd.read_csv(behavior_file)
                logger.info(f"✅ Loaded behavior data: {self.behavior_df.shape}")
            else:
                logger.warning(f"⚠️  Behavior file not found: {behavior_file}")
            
            # Check if we have at least demographics and transactions data
            if self.demographics_df is None or self.transactions_df is None:
                raise ValueError("Essential data files (demographics and transactions) not found")
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def merge_data_sources(self):
        """Merge different data sources using customer_id."""
        logger.info("Merging data sources...")
        
        try:
            # Start with demographics as the base
            if self.demographics_df is not None:
                merged_df = self.demographics_df.copy()
                logger.info(f"Starting with demographics data: {merged_df.shape}")
                
                # Merge with transactions data
                if self.transactions_df is not None:
                    # Aggregate transaction-level data to customer level
                    transaction_features = self._create_transaction_features()
                    merged_df = merged_df.merge(transaction_features, on='customer_id', how='left')
                    logger.info(f"✅ Merged with transaction features: {merged_df.shape}")
                
                # Merge with behavior data
                if self.behavior_df is not None:
                    merged_df = merged_df.merge(self.behavior_df, on='customer_id', how='left')
                    logger.info(f"✅ Merged with behavior data: {merged_df.shape}")
                
                self.features_df = merged_df
                logger.info(f"✅ Data merging completed. Final shape: {self.features_df.shape}")
                
            else:
                raise ValueError("Demographics data not available for merging")
                
        except Exception as e:
            logger.error(f"Error merging data sources: {str(e)}")
            raise
    
    def _create_transaction_features(self):
        """Create customer-level features from transaction data."""
        logger.info("Creating transaction-based features...")
        
        if self.transactions_df is None:
            return pd.DataFrame()
        
        try:
            # Group by customer_id and calculate aggregate features
            transaction_features = self.transactions_df.groupby('customer_id').agg({
                'transaction_amount': ['sum', 'mean', 'std', 'count'],
                'transaction_date': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            transaction_features.columns = [
                'customer_id',
                'total_spend',
                'average_transaction_amount',
                'transaction_amount_std',
                'transaction_frequency',
                'first_transaction_date',
                'last_transaction_date'
            ]
            
            # Calculate additional features
            transaction_features['transaction_amount_range'] = (
                transaction_features['transaction_amount_std'] / 
                transaction_features['average_transaction_amount']
            ).fillna(0)
            
            # Calculate customer tenure (days between first and last transaction)
            transaction_features['first_transaction_date'] = pd.to_datetime(
                transaction_features['first_transaction_date']
            )
            transaction_features['last_transaction_date'] = pd.to_datetime(
                transaction_features['last_transaction_date']
            )
            
            transaction_features['customer_tenure_days'] = (
                transaction_features['last_transaction_date'] - 
                transaction_features['first_transaction_date']
            ).dt.days
            
            # Calculate average days between transactions
            transaction_features['avg_days_between_transactions'] = (
                transaction_features['customer_tenure_days'] / 
                transaction_features['transaction_frequency']
            ).fillna(0)
            
            # Drop date columns as they're not needed for modeling
            transaction_features = transaction_features.drop([
                'first_transaction_date', 'last_transaction_date'
            ], axis=1)
            
            logger.info(f"✅ Created {len(transaction_features.columns)-1} transaction features")
            return transaction_features
            
        except Exception as e:
            logger.error(f"Error creating transaction features: {str(e)}")
            return pd.DataFrame()
    
    def engineer_additional_features(self):
        """Engineer additional meaningful features."""
        logger.info("Engineering additional features...")
        
        if self.features_df is None:
            logger.error("No features dataframe available for engineering")
            return
        
        try:
            # Create interaction features
            if 'age' in self.features_df.columns and 'income' in self.features_df.columns:
                self.features_df['age_income_ratio'] = (
                    self.features_df['age'] / self.features_df['income']
                ).fillna(0)
                logger.info("✅ Created age_income_ratio feature")
            
            # Create categorical encoding for high-cardinality variables
            categorical_columns = self.features_df.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if self.features_df[col].nunique() > 2:  # Only encode if more than 2 unique values
                    # Create dummy variables for categorical columns
                    dummies = pd.get_dummies(self.features_df[col], prefix=col, drop_first=True)
                    self.features_df = pd.concat([self.features_df, dummies], axis=1)
                    # Drop original categorical column
                    self.features_df = self.features_df.drop(col, axis=1)
                    logger.info(f"✅ Encoded categorical column: {col}")
            
            # Create polynomial features for important numerical variables
            if 'total_spend' in self.features_df.columns:
                self.features_df['total_spend_squared'] = self.features_df['total_spend'] ** 2
                logger.info("✅ Created total_spend_squared feature")
            
            if 'transaction_frequency' in self.features_df.columns:
                self.features_df['transaction_frequency_squared'] = self.features_df['transaction_frequency'] ** 2
                logger.info("✅ Created transaction_frequency_squared feature")
            
            # Create ratio features
            if 'total_spend' in self.features_df.columns and 'transaction_frequency' in self.features_df.columns:
                self.features_df['spend_per_transaction'] = (
                    self.features_df['total_spend'] / self.features_df['transaction_frequency']
                ).fillna(0)
                logger.info("✅ Created spend_per_transaction feature")
            
            # Handle infinite values
            self.features_df = self.features_df.replace([np.inf, -np.inf], 0)
            
            # Fill remaining null values
            numeric_columns = self.features_df.select_dtypes(include=[np.number]).columns
            self.features_df[numeric_columns] = self.features_df[numeric_columns].fillna(0)
            
            logger.info(f"✅ Feature engineering completed. Final shape: {self.features_df.shape}")
            
        except Exception as e:
            logger.error(f"Error engineering additional features: {str(e)}")
            raise
    
    def save_to_sqlite(self):
        """Save the feature-rich dataset to SQLite database."""
        logger.info("Saving features to SQLite database...")
        
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.features_db_path)
            
            # Save features to database
            self.features_df.to_sql('customer_features', conn, if_exists='replace', index=False)
            
            # Create indexes for better query performance
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_features(customer_id)")
            
            # Get table info
            cursor.execute("PRAGMA table_info(customer_features)")
            columns_info = cursor.fetchall()
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Features saved to SQLite database: {self.features_db_path}")
            logger.info(f"✅ Table 'customer_features' created with {len(columns_info)} columns")
            
            # Save a CSV copy for easy inspection
            csv_path = self.features_path / "customer_features.csv"
            self.features_df.to_csv(csv_path, index=False)
            logger.info(f"✅ CSV copy saved to: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving to SQLite: {str(e)}")
            raise
    
    def run_transformation(self):
        """Run the complete feature engineering process."""
        logger.info("Starting feature engineering process...")
        
        try:
            # Step 1: Load processed data
            self.load_processed_data()
            
            # Step 2: Merge data sources
            self.merge_data_sources()
            
            # Step 3: Engineer additional features
            self.engineer_additional_features()
            
            # Step 4: Save to SQLite database
            self.save_to_sqlite()
            
            logger.info("Feature engineering completed successfully!")
            
            return self.features_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

def main():
    """Main function to run the feature engineering."""
    try:
        engineer = FeatureEngineer()
        features_df = engineer.run_transformation()
        logger.info("Feature engineering pipeline completed successfully!")
        
        # Display summary
        if features_df is not None:
            print(f"\n📊 Feature Engineering Summary:")
            print(f"   • Total customers: {len(features_df)}")
            print(f"   • Total features: {len(features_df.columns)}")
            print(f"   • Feature types: {features_df.dtypes.value_counts().to_dict()}")
            print(f"   • Database location: {engineer.features_db_path}")
        
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
