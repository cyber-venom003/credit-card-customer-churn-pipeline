#!/usr/bin/env python3
"""
Utility Functions for Customer Churn Prediction Pipeline

This module contains reusable utility functions for the pipeline.
"""

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root directory
    """
    # Get the current file's directory (src/)
    current_dir = Path(__file__).parent
    # Go up one level to get the project root
    project_root = current_dir.parent
    return project_root

def get_features() -> pd.DataFrame:
    """
    Connect to the SQLite database and retrieve the entire customer features table.
    
    Returns:
        pd.DataFrame: DataFrame containing all customer features
        
    Raises:
        FileNotFoundError: If the features database doesn't exist
        sqlite3.Error: If there's an error connecting to or querying the database
        Exception: For any other unexpected errors
    """
    try:
        # Define the path to the features database
        project_root = get_project_root()
        features_db_path = project_root / "data/features/features.db"
        
        # Check if the database file exists
        if not features_db_path.exists():
            raise FileNotFoundError(
                f"Features database not found at {features_db_path}. "
                "Please run the feature engineering step first."
            )
        
        # Connect to the SQLite database
        logger.info(f"Connecting to features database: {features_db_path}")
        conn = sqlite3.connect(features_db_path)
        
        # Query to get all features
        query = "SELECT * FROM customer_features"
        logger.info("Executing query: SELECT * FROM customer_features")
        
        # Read the data into a DataFrame
        features_df = pd.read_sql_query(query, conn)
        
        # Close the database connection
        conn.close()
        
        logger.info(f"Successfully retrieved {len(features_df)} customer records with {len(features_df.columns)} features")
        
        return features_df
        
    except FileNotFoundError as e:
        logger.error(f"Database file not found: {str(e)}")
        raise
    except sqlite3.Error as e:
        logger.error(f"SQLite database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_features: {str(e)}")
        raise

def get_features_with_filter(customer_ids: Optional[list] = None, 
                           limit: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieve customer features with optional filtering and limiting.
    
    Args:
        customer_ids (list, optional): List of specific customer IDs to retrieve
        limit (int, optional): Maximum number of records to return
        
    Returns:
        pd.DataFrame: DataFrame containing filtered customer features
    """
    try:
        # Define the path to the features database
        project_root = get_project_root()
        features_db_path = project_root / "data/features/features.db"
        
        # Check if the database file exists
        if not features_db_path.exists():
            raise FileNotFoundError(
                f"Features database not found at {features_db_path}. "
                "Please run the feature engineering step first."
            )
        
        # Connect to the SQLite database
        logger.info(f"Connecting to features database: {features_db_path}")
        conn = sqlite3.connect(features_db_path)
        
        # Build the query based on parameters
        query = "SELECT * FROM customer_features"
        params = []
        
        if customer_ids:
            placeholders = ','.join(['?' for _ in customer_ids])
            query += f" WHERE customer_id IN ({placeholders})"
            params.extend(customer_ids)
            logger.info(f"Filtering by {len(customer_ids)} customer IDs")
        
        if limit:
            query += f" LIMIT {limit}"
            logger.info(f"Limiting results to {limit} records")
        
        # Execute the query
        logger.info(f"Executing query: {query}")
        features_df = pd.read_sql_query(query, conn, params=params)
        
        # Close the database connection
        conn.close()
        
        logger.info(f"Successfully retrieved {len(features_df)} customer records")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error in get_features_with_filter: {str(e)}")
        raise

def get_feature_columns() -> list:
    """
    Get the list of feature column names (excluding customer_id).
    
    Returns:
        list: List of feature column names
    """
    try:
        features_df = get_features()
        # Exclude customer_id from features
        feature_columns = [col for col in features_df.columns if col != 'customer_id']
        return feature_columns
    except Exception as e:
        logger.error(f"Error getting feature columns: {str(e)}")
        raise

def get_database_info() -> dict:
    """
    Get information about the features database.
    
    Returns:
        dict: Dictionary containing database information
    """
    try:
        features_db_path = Path("data/features/features.db")
        
        if not features_db_path.exists():
            return {"error": "Database not found"}
        
        conn = sqlite3.connect(features_db_path)
        cursor = conn.cursor()
        
        # Get table information
        cursor.execute("PRAGMA table_info(customer_features)")
        columns_info = cursor.fetchall()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM customer_features")
        row_count = cursor.fetchone()[0]
        
        # Get database size
        db_size = features_db_path.stat().st_size
        
        conn.close()
        
        info = {
            "database_path": str(features_db_path),
            "table_name": "customer_features",
            "column_count": len(columns_info),
            "row_count": row_count,
            "database_size_bytes": db_size,
            "database_size_mb": round(db_size / (1024 * 1024), 2),
            "columns": [col[1] for col in columns_info]  # Column names
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    try:
        print("🔍 Testing utility functions...")
        
        # Test database info
        print("\n📊 Database Information:")
        db_info = get_database_info()
        for key, value in db_info.items():
            if key != "columns":  # Skip columns list for cleaner output
                print(f"   {key}: {value}")
        
        # Test getting features
        print("\n📈 Testing get_features() function:")
        features = get_features()
        print(f"   ✅ Retrieved {len(features)} records with {len(features.columns)} columns")
        print(f"   📋 First few columns: {list(features.columns[:5])}")
        
        # Test getting feature columns
        print("\n🔧 Testing get_feature_columns() function:")
        feature_cols = get_feature_columns()
        print(f"   ✅ Found {len(feature_cols)} feature columns")
        print(f"   📋 First few features: {feature_cols[:5]}")
        
        print("\n✅ All utility functions tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing utility functions: {str(e)}")
        print("   This is expected if the feature engineering step hasn't been run yet.")
