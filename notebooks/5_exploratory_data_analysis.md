# Create Customer Churn EDA Notebook

## 🎯 **TASK OVERVIEW**
You are an expert data scientist and Python developer. Create a comprehensive Jupyter notebook (.ipynb) for exploratory data analysis (EDA) of a customer churn prediction dataset. This notebook should be production-ready, well-documented, and follow best practices for data analysis.

## 📊 **DATASET CONTEXT**
You will be working with a **credit card customer churn prediction** dataset that contains two main data sources:

### **1. Customer Demographics (`customer_demographics.csv`)**
- **customer_id**: Unique identifier for each customer
- **registration_date**: Date when customer registered
- **age**: Customer age (18-100)
- **gender**: Customer gender
- **location_city**: Customer's city
- **location_state**: Customer's state
- **income_bracket**: Customer income level
- **education_level**: Customer education
- **subscription_plan**: Type of subscription
- **customer_segment**: Customer segmentation
- **marketing_channel**: How customer was acquired
- **account_status**: Current status (Active/Churned/Suspended)
- **tenure_months**: How long customer has been with company
- **churn_date**: Date when customer churned (if applicable)
- **churn_type**: Reason for churning

### **2. Transaction Data (`transaction_data.csv`)**
- **transaction_id**: Unique transaction identifier
- **customer_id**: Links to customer demographics
- **transaction_date**: When transaction occurred
- **transaction_amount**: Transaction value
- **transaction_type**: Type of transaction
- **transaction_status**: Success/failure status
- **merchant_category**: Business category
- **payment_method**: Payment method used
- **transaction_fee**: Fee charged
- **balance_after**: Account balance after transaction
- **is_international**: International transaction flag
- **device_type**: Device used for transaction

### **3. Customer Behavior Data (`customer_behavior.json`)**
- **customer_id**: Links to demographics
- **behavior_metrics**: Various behavioral indicators
- **engagement_scores**: Customer engagement metrics
- **risk_indicators**: Risk assessment scores

## 🚀 **REQUIRED FUNCTIONALITY**

### **Section 1: Data Loading and Initial Exploration**
- Load all three data sources using pandas
- Display basic information (shape, data types, missing values)
- Show first few rows and basic statistics
- Check for data quality issues (duplicates, outliers, inconsistencies)

### **Section 2: Data Cleaning and Preprocessing**
- Handle missing values with appropriate strategies
- Convert data types (dates, categories, numerics)
- Remove or handle duplicate records
- Create derived features (e.g., days since registration, transaction frequency)
- Handle outliers using statistical methods

### **Section 3: Comprehensive Data Visualization**
Create the following visualizations using matplotlib, seaborn, and plotly:

#### **3.1 Customer Demographics Analysis**
- Age distribution (histogram + box plot)
- Gender distribution (pie chart)
- Location analysis (city/state heatmaps)
- Income bracket distribution
- Education level distribution
- Subscription plan distribution
- Account status distribution
- Tenure analysis (histogram + correlation with churn)

#### **3.2 Transaction Analysis**
- Transaction amount distribution (histogram + box plot)
- Transaction type distribution
- Payment method usage
- Merchant category analysis
- International vs domestic transactions
- Transaction frequency per customer
- Monthly/seasonal transaction patterns

#### **3.3 Churn Analysis (Target Variable)**
- Churn rate over time
- Churn by demographics (age, gender, location, income)
- Churn by transaction behavior
- Churn by subscription plan
- Churn by tenure
- Churn by engagement metrics

#### **3.4 Correlation and Relationship Analysis**
- Correlation matrix heatmap for numerical variables
- Feature importance for churn prediction
- Relationship between transaction behavior and churn
- Customer lifetime value analysis
- Risk score correlation with churn

### **Section 4: Feature Engineering Insights**
- Identify most predictive features for churn
- Create new features based on patterns discovered
- Analyze feature interactions
- Suggest feature selection strategies

### **Section 5: Statistical Analysis**
- Perform statistical tests (chi-square, t-tests, ANOVA)
- Calculate confidence intervals
- Identify significant differences between churned and active customers

### **Section 6: Business Insights and Recommendations**
- Key findings summary
- Actionable business recommendations
- Customer segmentation insights
- Risk factors for churn
- Retention strategy suggestions

## 🛠️ **TECHNICAL REQUIREMENTS**

### **Libraries to Use**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# For statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm

# For date handling
from datetime import datetime, timedelta
import calendar

# For visualization styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

### **Code Quality Standards**
- **Documentation**: Every cell should have clear markdown explanations
- **Error Handling**: Include try-catch blocks for robust execution
- **Modularity**: Break complex operations into functions
- **Performance**: Use vectorized operations, avoid loops where possible
- **Memory Management**: Handle large datasets efficiently
- **Reproducibility**: Set random seeds and document versions

### **Visualization Standards**
- **Professional Appearance**: Clean, publication-ready plots
- **Accessibility**: Use colorblind-friendly palettes
- **Consistency**: Maintain consistent styling across all plots
- **Interactivity**: Use plotly for interactive plots where beneficial
- **Annotations**: Include proper titles, labels, and legends

## 📁 **FILE STRUCTURE EXPECTATIONS**
The notebook should be organized as follows:

```python
# Cell 1: Setup and Imports
# Cell 2: Data Loading
# Cell 3: Initial Data Exploration
# Cell 4: Data Quality Assessment
# Cell 5: Data Cleaning and Preprocessing
# Cell 6: Customer Demographics Analysis
# Cell 7: Transaction Data Analysis
# Cell 8: Customer Behavior Analysis
# Cell 9: Churn Analysis (Target Variable)
# Cell 10: Correlation and Relationship Analysis
# Cell 11: Statistical Testing
# Cell 12: Feature Engineering Insights
# Cell 13: Business Insights and Recommendations
# Cell 14: Summary and Next Steps
```

## 🎨 **VISUALIZATION EXAMPLES**

### **Example 1: Churn Rate by Demographics**
```python
# Create a comprehensive visualization showing churn rates across different demographic segments
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Churn Rate Analysis by Demographics', fontsize=16, fontweight='bold')

# Age groups vs churn
age_churn = df.groupby('age_group')['account_status'].apply(lambda x: (x == 'Churned').mean())
age_churn.plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Churn Rate by Age Group')
axes[0,0].set_ylabel('Churn Rate')
axes[0,0].tick_params(axis='x', rotation=45)

# Similar plots for other demographics...
```

### **Example 2: Transaction Pattern Analysis**
```python
# Interactive plotly visualization for transaction patterns
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Transaction Amount Distribution', 'Transaction Type vs Churn',
                   'Monthly Transaction Trends', 'Customer Lifetime Value')
)

# Add traces for each subplot...
fig.update_layout(height=800, title_text="Transaction Pattern Analysis")
fig.show()
```

## 📊 **EXPECTED OUTPUTS**

### **Key Metrics to Calculate**
- Overall churn rate
- Churn rate by customer segment
- Average customer lifetime value
- Transaction frequency patterns
- Risk score distributions
- Feature importance rankings

### **Insights to Generate**
- **High-risk customer profiles** for churn
- **Retention opportunities** based on behavior patterns
- **Feature engineering recommendations** for ML models
- **Business strategy suggestions** for reducing churn

## 🔍 **QUALITY CHECKLIST**

Before submitting, ensure the notebook:
- [ ] Loads all data sources without errors
- [ ] Handles missing values appropriately
- [ ] Creates comprehensive visualizations
- [ ] Provides actionable business insights
- [ ] Uses efficient pandas operations
- [ ] Includes proper error handling
- [ ] Has clear markdown documentation
- [ ] Follows Python best practices
- [ ] Is ready for production use
- [ ] Can be executed end-to-end

## 🎯 **SUCCESS CRITERIA**

The notebook should enable stakeholders to:
1. **Understand** the customer churn patterns
2. **Identify** key risk factors for churn
3. **Develop** targeted retention strategies
4. **Prepare** data for machine learning models
5. **Make** data-driven business decisions

---

**Remember**: This is a comprehensive analysis that will be used by business stakeholders and data scientists. Focus on clarity, actionable insights, and professional presentation. The notebook should tell a compelling story about customer churn patterns and provide concrete recommendations for business improvement.
