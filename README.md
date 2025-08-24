# Customer Churn Prediction Pipeline

## 🎯 Project Overview

This project implements a complete end-to-end data management pipeline for customer churn prediction. The pipeline includes data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment orchestration.

## 📁 Project Structure

```
customer-churn-pipeline/
│
├── data/                    # Data storage
│   ├── raw/                # Raw data files
│   ├── processed/          # Cleaned and processed data
│   └── features/           # Engineered features
│
├── dags/                   # Apache Airflow DAGs
├── docs/                   # Project documentation
├── logs/                   # Application and pipeline logs
├── models/                 # Trained model artifacts
├── notebooks/              # Jupyter notebooks for exploration
└── src/                    # Source code
    └── __init__.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Apache Airflow
- DVC (Data Version Control)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
```

## 🔧 Configuration

- Configure your data sources in the configuration files
- Set up Airflow connections for external services
- Update environment variables as needed

## 📊 Usage

### Running the Pipeline

1. Start Airflow webserver:
```bash
airflow webserver --port 8080
```

2. Start Airflow scheduler:
```bash
airflow scheduler
```

3. Access the Airflow UI at `http://localhost:8080`

### Data Processing

- Raw data ingestion: `python src/ingestion.py`
- Data validation: `python src/validation.py`
- Feature engineering: `python src/transformation.py`

### Model Training

- Train model: `python src/model_building.py`
- Utilities: `python src/utils.py`

## 📈 Pipeline Components

1. **Data Ingestion** (`src/ingestion.py`): Automated data collection from multiple sources
2. **Data Validation** (`src/validation.py`): Data quality checks and validation reports
3. **Feature Engineering** (`src/transformation.py`): Creating predictive features
4. **Model Building** (`src/model_building.py`): Training machine learning models
5. **Utilities** (`src/utils.py`): Helper functions and common utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support, please open an issue in the GitHub repository.

## 🔄 Version History

- **v1.0.0**: Initial project setup and basic pipeline structure
