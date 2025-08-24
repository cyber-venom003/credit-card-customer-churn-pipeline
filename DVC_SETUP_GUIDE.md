# DVC Setup Guide for Feature Tracking

This guide explains how to use DVC (Data Version Control) to track features and data artifacts in the credit card customer churn prediction pipeline.

## What is DVC?

DVC (Data Version Control) is a tool that helps you version control large data files, ML models, and data pipelines. It integrates seamlessly with Git to provide a complete version control solution for data science projects.

## Current DVC Setup

### Tracked Files

The following files are currently tracked by DVC:

#### Raw Data
- `data/raw/customer_demographics/2025-08-24/customer_demographics.csv` - Customer demographic information
- `data/raw/transactions/2025-08-24/transactions.csv` - Transaction history data
- `data/raw/customer_behavior/2025-08-24/customer_behavior.json` - Customer behavior metrics

#### Processed Features
- `data/processed/final_features.csv` - Final engineered features for model training
- `data/features/features.db` - SQLite feature store

#### Models
- `models/churn_prediction_model.joblib` - Trained churn prediction model
- `models/feature_scaler.joblib` - Feature scaling parameters
- `models/feature_names.txt` - List of feature names (tracked by Git)

### Pipeline

A DVC pipeline has been created in `dvc.yaml` that automates the feature engineering process:

```yaml
stages:
  transform_features:
    cmd: python3 src/6_transformation.py
    deps:
    - data/raw/customer_behavior/2025-08-24/customer_behavior.json
    - data/raw/customer_demographics/2025-08-24/customer_demographics.csv
    - data/raw/transactions/2025-08-24/transactions.csv
    - src/6_transformation.py
    outs:
    - data/features/features.db
    - data/processed/final_features.csv
```

## Basic DVC Commands

### Check Status
```bash
dvc status
```
Shows which files have changed and need to be updated.

### Reproduce Pipeline
```bash
dvc repro transform_features
```
Runs the feature engineering pipeline to regenerate features from raw data.

### Add New Files
```bash
dvc add path/to/file
```
Start tracking a new file with DVC.

### Remove Tracking
```bash
dvc remove path/to/file.dvc
```
Stop tracking a file with DVC.

### List Tracked Files
```bash
dvc list .
```
Shows all files tracked by DVC in the current directory.

## Workflow for Feature Updates

### 1. Update Raw Data
When you have new raw data, place it in the appropriate `data/raw/` subdirectory with a new timestamp folder.

### 2. Update Dependencies
If the data structure changes, update the `dvc.yaml` file to reflect new dependencies.

### 3. Regenerate Features
Run the pipeline to create new features:
```bash
dvc repro transform_features
```

### 4. Commit Changes
After running the pipeline, commit the changes:
```bash
git add dvc.lock dvc.yaml
git commit -m "Update features with new data"
```

## Setting Up Remote Storage (Optional)

To share data across team members or backup your data, you can set up a remote storage:

### AWS S3
```bash
dvc remote add -d myremote s3://mybucket/path
```

### Google Cloud Storage
```bash
dvc remote add -d myremote gs://mybucket/path
```

### Local Network Storage
```bash
dvc remote add -d myremote /path/to/network/storage
```

### Push/Pull Data
```bash
dvc push    # Upload data to remote
dvc pull    # Download data from remote
```

## Best Practices

1. **Always use `dvc repro`** instead of manually running scripts to ensure reproducibility
2. **Commit `dvc.lock`** after each pipeline run to lock dependency versions
3. **Use meaningful stage names** in your pipeline for clarity
4. **Document data lineage** in your pipeline dependencies
5. **Set up remote storage** for team collaboration and backup

## Troubleshooting

### Pipeline Fails
- Check that all dependencies exist
- Verify Python environment and packages
- Check file paths in the pipeline

### Data Not Updating
- Run `dvc status` to see what's changed
- Use `dvc repro` to regenerate outputs
- Check that outputs are properly specified in the pipeline

### Git Conflicts
- Resolve conflicts in `.dvc` files manually
- Use `dvc checkout` to restore data files
- Re-run pipeline after resolving conflicts

## Next Steps

1. **Set up remote storage** for team collaboration
2. **Add more pipeline stages** for data validation, model training, etc.
3. **Configure CI/CD** to automatically run pipelines on data updates
4. **Set up monitoring** for pipeline performance and data quality

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/doc/tutorials)
- [DVC Best Practices](https://dvc.org/doc/user-guide/best-practices)
- [DVC Pipeline Examples](https://dvc.org/doc/user-guide/pipelines)
