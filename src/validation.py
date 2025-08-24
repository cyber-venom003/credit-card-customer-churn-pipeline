#!/usr/bin/env python3
"""
Data Validation Script for Customer Churn Prediction Pipeline

This script automatically finds and reads the most recently ingested raw data from data/raw/ subdirectories
and performs data quality checks including null values and duplicate records.
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import glob
from utils import get_project_root

# Get project root for absolute paths
project_root = get_project_root()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs/validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation class for checking data quality."""
    
    def __init__(self):
        """Initialize the DataValidator class."""
        self.base_path = project_root / "data/raw"
        self.docs_path = project_root / "docs"
        self.validation_results = {}
        
        # Create docs directory if it doesn't exist
        self.docs_path.mkdir(exist_ok=True)
    
    def find_latest_data(self):
        """Find the most recently ingested data from all subdirectories."""
        latest_data = {}
        
        # Look for customer_demographics data
        demographics_pattern = str(self.base_path / "customer_demographics" / "*" / "*.csv")
        demographics_files = glob.glob(demographics_pattern)
        if demographics_files:
            latest_demographics = max(demographics_files, key=os.path.getctime)
            latest_data['demographics'] = latest_demographics
            logger.info(f"Found latest demographics data: {latest_demographics}")
        
        # Look for transactions data
        transactions_pattern = str(self.base_path / "transactions" / "*" / "*.csv")
        transactions_files = glob.glob(transactions_pattern)
        if transactions_files:
            latest_transactions = max(transactions_files, key=os.path.getctime)
            latest_data['transactions'] = latest_transactions
            logger.info(f"Found latest transactions data: {latest_transactions}")
        
        # Look for customer_behavior data
        behavior_pattern = str(self.base_path / "customer_behavior" / "*" / "*.json")
        behavior_files = glob.glob(behavior_pattern)
        if behavior_files:
            latest_behavior = max(behavior_files, key=os.path.getctime)
            latest_data['behavior'] = latest_behavior
            logger.info(f"Found latest behavior data: {latest_behavior}")
        
        return latest_data
    
    def validate_csv_data(self, file_path, data_type):
        """Validate CSV data for quality issues."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Validating {data_type} data: {len(df)} records, {len(df.columns)} columns")
            
            validation_result = {
                'file_path': str(file_path),
                'data_type': data_type,
                'total_records': len(df),
                'total_columns': len(df.columns),
                'null_counts': df.isnull().sum().to_dict(),
                'duplicate_records': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Check for critical issues
            critical_issues = []
            if df.isnull().sum().sum() > len(df) * 0.5:  # More than 50% nulls
                critical_issues.append("High percentage of null values")
            
            if df.duplicated().sum() > len(df) * 0.1:  # More than 10% duplicates
                critical_issues.append("High percentage of duplicate records")
            
            validation_result['critical_issues'] = critical_issues
            validation_result['validation_status'] = 'PASS' if not critical_issues else 'FAIL'
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating {data_type} data: {str(e)}")
            return {
                'file_path': str(file_path),
                'data_type': data_type,
                'error': str(e),
                'validation_status': 'ERROR'
            }
    
    def validate_json_data(self, file_path, data_type):
        """Validate JSON data for quality issues."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Validating {data_type} data: {len(data)} records")
            
            # Convert to DataFrame for easier validation if it's a list of records
            if isinstance(data, list):
                df = pd.DataFrame(data)
                validation_result = {
                    'file_path': str(file_path),
                    'data_type': data_type,
                    'total_records': len(data),
                    'total_columns': len(df.columns) if len(data) > 0 else 0,
                    'null_counts': df.isnull().sum().to_dict() if len(data) > 0 else {},
                    'duplicate_records': df.duplicated().sum() if len(data) > 0 else 0,
                    'data_types': df.dtypes.to_dict() if len(data) > 0 else {},
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024 if len(data) > 0 else 0
                }
            else:
                validation_result = {
                    'file_path': str(file_path),
                    'data_type': data_type,
                    'total_records': 1,
                    'total_columns': len(data.keys()) if isinstance(data, dict) else 0,
                    'null_counts': {},
                    'duplicate_records': 0,
                    'data_types': {},
                    'memory_usage_mb': 0
                }
            
            validation_result['validation_status'] = 'PASS'
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating {data_type} data: {str(e)}")
            return {
                'file_path': str(file_path),
                'data_type': data_type,
                'error': str(e),
                'validation_status': 'ERROR'
            }
    
    def run_validation(self):
        """Run the complete data validation process."""
        logger.info("Starting data validation process...")
        
        # Find latest data
        latest_data = self.find_latest_data()
        
        if not latest_data:
            logger.warning("No data files found for validation")
            return
        
        # Validate each data source
        for data_type, file_path in latest_data.items():
            logger.info(f"Validating {data_type} data...")
            
            if file_path.endswith('.csv'):
                result = self.validate_csv_data(file_path, data_type)
            elif file_path.endswith('.json'):
                result = self.validate_json_data(file_path, data_type)
            else:
                logger.warning(f"Unsupported file type for {data_type}: {file_path}")
                continue
            
            self.validation_results[data_type] = result
        
        # Generate validation report
        self.generate_report()
        
        logger.info("Data validation completed successfully!")
    
    def generate_report(self):
        """Generate a comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Generate CSV report
        csv_report_path = self.docs_path / f"validation_report_{timestamp}.csv"
        report_data = []
        
        for data_type, result in self.validation_results.items():
            if 'error' not in result:
                report_data.append({
                    'Data Type': data_type,
                    'File Path': result['file_path'],
                    'Total Records': result['total_records'],
                    'Total Columns': result['total_columns'],
                    'Null Records': sum(result['null_counts'].values()),
                    'Duplicate Records': result['duplicate_records'],
                    'Memory Usage (MB)': round(result['memory_usage_mb'], 2),
                    'Validation Status': result['validation_status'],
                    'Critical Issues': '; '.join(result.get('critical_issues', []))
                })
            else:
                report_data.append({
                    'Data Type': data_type,
                    'File Path': result['file_path'],
                    'Total Records': 'ERROR',
                    'Total Columns': 'ERROR',
                    'Null Records': 'ERROR',
                    'Duplicate Records': 'ERROR',
                    'Memory Usage (MB)': 'ERROR',
                    'Validation Status': 'ERROR',
                    'Critical Issues': result['error']
                })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(csv_report_path, index=False)
        logger.info(f"CSV validation report saved to: {csv_report_path}")
        
        # Generate Markdown report
        md_report_path = self.docs_path / f"validation_report_{timestamp}.md"
        self._generate_markdown_report(md_report_path, report_data)
        logger.info(f"Markdown validation report saved to: {md_report_path}")
    
    def _generate_markdown_report(self, file_path, report_data):
        """Generate a markdown format validation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(file_path, 'w') as f:
            f.write(f"# Data Validation Report\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")
            f.write(f"## Summary\n\n")
            
            total_datasets = len(report_data)
            passed_datasets = sum(1 for row in report_data if row['Validation Status'] == 'PASS')
            failed_datasets = sum(1 for row in report_data if row['Validation Status'] == 'FAIL')
            error_datasets = sum(1 for row in report_data if row['Validation Status'] == 'ERROR')
            
            f.write(f"- **Total Datasets:** {total_datasets}\n")
            f.write(f"- **Passed Validation:** {passed_datasets}\n")
            f.write(f"- **Failed Validation:** {failed_datasets}\n")
            f.write(f"- **Errors:** {error_datasets}\n\n")
            
            f.write(f"## Detailed Results\n\n")
            f.write(f"| Data Type | Records | Columns | Nulls | Duplicates | Status | Issues |\n")
            f.write(f"|-----------|---------|---------|-------|------------|--------|--------|\n")
            
            for row in report_data:
                f.write(f"| {row['Data Type']} | {row['Total Records']} | {row['Total Columns']} | "
                       f"{row['Null Records']} | {row['Duplicate Records']} | "
                       f"{row['Validation Status']} | {row['Critical Issues']} |\n")
            
            f.write(f"\n## Recommendations\n\n")
            
            if failed_datasets > 0:
                f.write(f"- **Data Quality Issues Detected:** Review datasets with FAIL status\n")
                f.write(f"- **Null Values:** Consider imputation strategies for missing data\n")
                f.write(f"- **Duplicates:** Investigate and remove duplicate records\n")
            
            if error_datasets > 0:
                f.write(f"- **Processing Errors:** Check file formats and data structure\n")
            
            if passed_datasets == total_datasets:
                f.write(f"- **All datasets passed validation successfully!**\n")

def main():
    """Main function to run the data validation."""
    try:
        validator = DataValidator()
        validator.run_validation()
        logger.info("Data validation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Data validation pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
