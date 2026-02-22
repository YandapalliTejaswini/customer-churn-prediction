📊 Customer Churn Prediction & Analytics Dashboard


🚀 Project Overview

This project builds an end-to-end Machine Learning system to predict whether a telecom customer is likely to leave (churn).
It includes data preprocessing, model training, evaluation, and an interactive Streamlit dashboard for real-time predictions and business analytics.

The goal is to help companies identify high-risk customers and take proactive retention actions.

🎯 Problem Statement

Customer churn is a major challenge for telecom companies. Losing customers reduces revenue and increases acquisition costs.

This project predicts customer churn using historical customer data and provides insights through a visual analytics dashboard.

🧠 Machine Learning Workflow


1️⃣ Data Processing

Removed unnecessary columns (customerID)

Converted TotalCharges to numeric format

Handled missing values using forward fill

Encoded target variable (Churn → 0/1)

2️⃣ Feature Engineering

Automatic preprocessing using ColumnTransformer

StandardScaler for numerical features

OneHotEncoder for categorical features

3️⃣ Model Training

Algorithm: Random Forest Classifier

Train/Test Split: 80/20

Pipeline used for reproducibility

4️⃣ Model Evaluation

Model performance evaluated using:


Accuracy

Precision

Recall

F1-Score

Classification Report

📊 Model Performance

Metric	Score

Accuracy	79.6%

Precision	65.8%

Recall	47.9%

F1 Score	55.5%

Metrics are automatically saved and displayed in the dashboard.

💻 Dashboard Features (Streamlit)

The interactive dashboard includes:

✅ Customer churn prediction interface

✅ Churn risk probability gauge chart

✅ Customer analytics visualization

✅ Executive business summary

✅ Model performance KPI metrics

Users can input customer details and instantly view churn risk.

🏗️ Project Architecture

      User Input
           ↓
    Streamlit Dashboard
           ↓
    Saved ML Pipeline
           ↓
    Prediction + Probability
           ↓
    Analytics & Visualization


🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

Streamlit

Plotly

Joblib

📂 Project Structure


customer-churn-project/

      │
      ├── app.py                # Streamlit dashboard
      ├── train.py              # Model training pipeline
      ├── requirements.txt
      ├── README.md
      │
      ├── data/
      │   └── Telco customer dataset
      │
      ├── models/               # Generated automatically
      │   ├── churn_pipeline.pkl
      │   └── metrics.json

Model files are excluded from GitHub and generated automatically during deployment.

⚙️ Installation & Run Locally

1️⃣ Clone repository

git clone https://github.com/yandapalliTejaswini/customer-churn-prediction.git

cd customer-churn-prediction

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Train model

python train.py

4️⃣ Run dashboard


streamlit run app.py

🌐 Deployment


The application is deployed using Streamlit Cloud.

Live Demo: (Add your deployed link here)

📈 Business Impact

Identifies customers at high churn risk

Enables targeted retention strategies

Supports data-driven decision making

Improves customer lifetime value

🔮 Future Improvements

Handle class imbalance using SMOTE

Hyperparameter tuning

Feature importance visualization

Explainable AI (SHAP/LIME)

Cloud model monitoring

👨‍💻 Author

Yandapalli Tejaswini

Computer Science Engineering (Data Science)
Machine Learning & Data Analytics Enthusiast

⭐ If you like this project

Give the repository a ⭐ on GitHub!
