# Telco Customer Churn Prediction System

An end-to-end production-style Machine Learning project that predicts telecom customer churn using multiple classification algorithms, hyperparameter tuning, explainability techniques, and FastAPI deployment.

---

# Project Overview

Customer churn is one of the most critical business problems for telecom companies because retaining existing customers is significantly cheaper than acquiring new ones.

This project builds a complete ML pipeline to:

* Predict whether a customer is likely to churn
* Optimize churn prediction thresholds
* Compare multiple ML models
* Explain model predictions using SHAP
* Serve real-time predictions using FastAPI

---

# Tech Stack

## Core ML & Data Processing

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

## Visualization & Explainability

* Matplotlib
* Seaborn
* SHAP

## Deployment

* FastAPI
* Uvicorn
* Joblib

---

# Key Features

* Modular production-grade ML architecture
* Automated preprocessing pipeline
* Multiple model benchmarking
* Hyperparameter tuning with RandomizedSearchCV
* Cross-validation for robust evaluation
* Threshold optimization for business-oriented predictions
* SHAP explainability integration
* FastAPI-based inference API
* Metadata-driven inference configuration
* Serialized preprocessing + model pipelines

---

# ML Workflow

Raw Data
→ Data Cleaning
→ Feature Engineering
→ Preprocessing Pipeline
→ Model Training
→ Hyperparameter Tuning
→ Cross Validation
→ Threshold Optimization
→ SHAP Explainability
→ Best Model Selection
→ Model Serialization
→ FastAPI Deployment

---

# Models Implemented

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

---

# Evaluation Metrics

Since customer churn is an imbalanced classification problem, evaluation was focused on:

* ROC-AUC
* Precision
* Recall
* F1 Score
* Cross-validation stability

Best model achieved:

* ROC-AUC: ~0.85
* Accuracy: ~80%

---

# Project Structure

```bash
telco-churn-ml/
│
├── app/
├── data/
├── models/
├── src/
├── requirements.txt
├── run.py
└── README.md
```

---

# Running the ML Pipeline

## 1. Clone Repository

```bash
git clone https://github.com/ayushkhandelwal123/Churn-Modelling.git
cd Churn-Modelling
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Full ML Pipeline

```bash
python run.py
```

This will:

* Load and preprocess data
* Train multiple models
* Perform hyperparameter tuning
* Evaluate performance
* Optimize thresholds
* Generate SHAP explainability
* Save the best model

---

# Running FastAPI Server

## Start API

```bash
uvicorn app.main:app --reload
```

---

# Swagger API Documentation

Open:

```bash
http://127.0.0.1:8000/docs
```

---

# Example Prediction Request

```json
{
  "gender": "Male",
  "senior_citizen": "Yes",
  "partner": "Yes",
  "dependents": "Yes",
  "tenure_months": 24,
  "phone_service": "Yes",
  "multiple_lines": "Yes",
  "internet_service": "Fiber optic",
  "online_security": "Yes",
  "online_backup": "Yes",
  "device_protection": "Yes",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 100,
  "total_charges": 1100
}
```

---

# Example API Response

```json
{
  "prediction": 1,
  "probability": 0.73,
  "threshold_used": 0.45,
  "risk_level": "High"
}
```

---

# Key Learning Outcomes

* Production ML pipeline design
* Model evaluation for imbalanced datasets
* Hyperparameter optimization
* Explainable AI using SHAP
* Training-serving consistency
* FastAPI deployment architecture
* Threshold optimization for business problems

---

# Future Improvements

* Dockerization
* CI/CD integration
* MLflow experiment tracking
* Cloud deployment (AWS/GCP/Azure)
* Drift monitoring
* Streamlit frontend dashboard

---

# Author

Ayush Khandelwal

GitHub:
https://github.com/ayushkhandelwal123/Churn-Modelling
