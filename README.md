# Customer Churn Prediction

## Problem

Predict customers likely to leave telecom service.

## Tech Stack

Python, Pandas, Scikit-learn, Streamlit

## Steps

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering & Preprocessing
* Random Forest Model Training
* Feature Importance Analysis
* Dashboard Deployment using Streamlit

## Model Performance Metrics

The model performance was evaluated using standard classification metrics:

* **Accuracy:** ~85%
* **Precision:** (add value from classification report)
* **Recall:** (add value from classification report)
* **F1-Score:** (add value from classification report)

> Metrics were calculated using Scikit-learn's `classification_report` during model evaluation.

## Business Impact

Helps telecom companies identify high-risk customers and apply retention strategies to reduce customer loss.

## Project Features

* End-to-end Machine Learning pipeline
* Automated preprocessing using ColumnTransformer
* Model persistence using Joblib
* Interactive Streamlit dashboard
* Churn risk visualization with Plotly

## How to Run the Project

```bash
python train.py
streamlit run app.py
```

## Future Improvements

* Add feature importance visualization
* Deploy app online (Streamlit Cloud / Render)
* Add explainable AI (SHAP/LIME)
