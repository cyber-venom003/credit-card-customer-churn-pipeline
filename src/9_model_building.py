#!/usr/bin/env python3
"""
Model Building Script for Customer Churn Prediction Pipeline

This script uses MLflow to ensure experiment reproducibility and performs the following actions:
1. Use the get_features() function from src/utils.py to load the feature data
2. Split the data into training and testing sets
3. Train a RandomForestClassifier model
4. Log the model's hyperparameters and key evaluation metrics to MLflow
5. Log the final trained model object using mlflow.sklearn.log_model()
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# MLflow imports
import mlflow
import mlflow.sklearn

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# Import utility function
from utils import get_features, get_project_root

# Get project root for absolute paths
project_root = get_project_root()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs/model_building.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChurnModelBuilder:
    """Model building class for customer churn prediction."""
    
    def __init__(self):
        """Initialize the ChurnModelBuilder class."""
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
        # MLflow experiment name
        self.experiment_name = "customer_churn_prediction"
        
        # Set MLflow tracking URI (local filesystem)
        mlflow.set_tracking_uri(f"file:{project_root}/mlruns")
        
        # Create logs directory if it doesn't exist
        (project_root / "logs").mkdir(exist_ok=True)
    
    def load_features(self):
        """Load features using the get_features() utility function."""
        logger.info("Loading features from SQLite database...")
        
        try:
            self.features_df = get_features()
            logger.info(f"✅ Successfully loaded {len(self.features_df)} records with {len(self.features_df.columns)} features")
            
            # Display basic information about the dataset
            logger.info(f"Dataset shape: {self.features_df.shape}")
            logger.info(f"Feature columns: {list(self.features_df.columns)}")
            logger.info(f"Data types: {self.features_df.dtypes.value_counts().to_dict()}")
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
    
    def prepare_data(self):
        """Prepare data for modeling by identifying target variable and features."""
        logger.info("Preparing data for modeling...")
        
        try:
            # Look for churn column (case-insensitive)
            churn_columns = [col for col in self.features_df.columns if 'churn' in col.lower()]
            
            if not churn_columns:
                raise ValueError("No churn column found in the dataset. Please ensure the target variable exists.")
            
            target_column = churn_columns[0]
            logger.info(f"✅ Found target column: {target_column}")
            
            # Check target variable distribution
            target_counts = self.features_df[target_column].value_counts()
            logger.info(f"Target variable distribution: {target_counts.to_dict()}")
            
            # Separate features and target
            X = self.features_df.drop([target_column, 'customer_id'], axis=1, errors='ignore')
            y = self.features_df[target_column]
            
            # Remove any remaining non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            logger.info(f"✅ Prepared {len(X.columns)} features for modeling")
            logger.info(f"✅ Target variable: {target_column} with {len(y)} samples")
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"✅ Data split completed:")
            logger.info(f"   Training set: {self.X_train.shape}")
            logger.info(f"   Testing set: {self.X_test.shape}")
            
            # Scale the features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            logger.info("✅ Feature scaling completed")
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_model(self):
        """Train the RandomForestClassifier model."""
        logger.info("Training RandomForestClassifier model...")
        
        try:
            # Set MLflow experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Define hyperparameters
                hyperparameters = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                # Log hyperparameters
                mlflow.log_params(hyperparameters)
                logger.info(f"✅ Logged hyperparameters: {hyperparameters}")
                
                # Initialize and train the model
                self.model = RandomForestClassifier(**hyperparameters)
                
                # Train the model
                self.model.fit(self.X_train_scaled, self.y_train)
                logger.info("✅ Model training completed")
                
                # Make predictions
                y_pred = self.model.predict(self.X_test_scaled)
                y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted'),
                    'recall': recall_score(self.y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                logger.info("✅ Logged evaluation metrics:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"   {metric_name}: {metric_value:.4f}")
                
                # Log feature importance
                feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log top 10 features
                top_features = feature_importance.head(10)
                for idx, row in top_features.iterrows():
                    mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
                
                logger.info("✅ Logged feature importance")
                logger.info("Top 5 features:")
                for idx, row in top_features.head().iterrows():
                    logger.info(f"   {row['feature']}: {row['importance']:.4f}")
                
                # Log the model
                mlflow.sklearn.log_model(
                    self.model, 
                    "random_forest_churn_model",
                    registered_model_name="churn_prediction_model"
                )
                logger.info("✅ Model logged to MLflow")
                
                # Log confusion matrix as artifact
                cm = confusion_matrix(self.y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm, 
                    index=['Actual No Churn', 'Actual Churn'],
                    columns=['Predicted No Churn', 'Predicted Churn']
                )
                
                cm_path = project_root / "logs" / "confusion_matrix.csv"
                cm_df.to_csv(cm_path, index=True)
                mlflow.log_artifact(str(cm_path))
                logger.info("✅ Confusion matrix logged as artifact")
                
                # Log classification report as artifact
                report = classification_report(self.y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                report_path = project_root / "logs" / "classification_report.csv"
                report_df.to_csv(report_path, index=True)
                mlflow.log_artifact(str(report_path))
                logger.info("✅ Classification report logged as artifact")
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    self.model, 
                    self.X_train_scaled, 
                    self.y_train, 
                    cv=5, 
                    scoring='f1_weighted'
                )
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                mlflow.log_metric("cv_f1_mean", cv_mean)
                mlflow.log_metric("cv_f1_std", cv_std)
                
                logger.info(f"✅ Cross-validation F1 score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
                
                # Log run ID for reference
                run_id = mlflow.active_run().info.run_id
                logger.info(f"✅ MLflow run completed. Run ID: {run_id}")
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model_locally(self):
        """Save the trained model locally for deployment."""
        logger.info("Saving model locally...")
        
        try:
            # Create models directory if it doesn't exist
            models_dir = project_root / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Save model using joblib
            import joblib
            model_path = models_dir / "churn_prediction_model.joblib"
            scaler_path = models_dir / "feature_scaler.joblib"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"✅ Model saved to: {model_path}")
            logger.info(f"✅ Scaler saved to: {scaler_path}")
            
            # Save feature names
            feature_names_path = models_dir / "feature_names.txt"
            with open(feature_names_path, 'w') as f:
                for feature in self.X_train.columns:
                    f.write(f"{feature}\n")
            
            logger.info(f"✅ Feature names saved to: {feature_names_path}")
            
        except Exception as e:
            logger.error(f"Error saving model locally: {str(e)}")
            raise
    
    def run_model_building(self):
        """Run the complete model building process."""
        logger.info("Starting model building process...")
        
        try:
            # Step 1: Load features
            self.load_features()
            
            # Step 2: Prepare data
            self.prepare_data()
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Save model locally
            self.save_model_locally()
            
            logger.info("Model building completed successfully!")
            
        except Exception as e:
            logger.error(f"Model building failed: {str(e)}")
            raise

def main():
    """Main function to run the model building."""
    try:
        model_builder = ChurnModelBuilder()
        model_builder.run_model_building()
        logger.info("Model building pipeline completed successfully!")
        
        # Display summary
        print(f"\n🎯 Model Building Summary:")
        print(f"   • Model type: RandomForestClassifier")
        print(f"   • Training samples: {model_builder.X_train.shape[0]}")
        print(f"   • Testing samples: {model_builder.X_test.shape[0]}")
        print(f"   • Features used: {model_builder.X_train.shape[1]}")
        print(f"   • MLflow experiment: {model_builder.experiment_name}")
        print(f"   • Local model saved to: models/")
        
    except Exception as e:
        logger.error(f"Model building pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
