# Telecom Customer Churn Prediction

## Live Demo

 http://13.206.196.255:8501

---

## Project Overview

Customer churn is a critical challenge in the telecommunications industry. This project leverages machine learning to predict whether a customer is likely to churn and provides actionable insights to support retention strategies.

---

## Problem Statement

The goal is to build a predictive model that identifies customers at risk of churn, enabling proactive business decisions to reduce customer attrition.

---

## Dataset

* **Source:** IBM Telco Customer Churn Dataset
* **Size:** ~7,000 customers
* **Target Variable:** `Churn` (Yes/No)

### Features Include:

* Customer tenure
* Contract type
* Monthly charges
* Internet services
* Support services
* Payment methods

---

## Models Implemented

* Logistic Regression (Baseline)
* Random Forest Classifier
* Tuned Random Forest (GridSearchCV)

---

## Model Performance

* **ROC-AUC:** ~0.75+
* Improved performance after hyperparameter tuning
* Balanced precision and recall

---

## Key Insights

* Month-to-month contracts have higher churn rates
* Customers with short tenure are more likely to churn
* Higher monthly charges increase churn probability
* Lack of TechSupport and OnlineSecurity correlates with higher churn

---

## Application

An interactive **Streamlit web app** allows users to input customer details and get real-time churn predictions.

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t churn-app .
```

### Run Container

```bash
docker run -p 8501:8501 churn-app
```

Access locally:

```
http://localhost:8501
```

---

## AWS Deployment (EC2)

The application is deployed on an AWS EC2 instance using Docker.

### Steps:

1. Launch EC2 instance (Ubuntu)
2. Install Docker
3. Transfer project files
4. Build Docker image
5. Run container on port 8501

Access live app:
http://13.206.196.255:8501

---

## Project Structure

```
customer-churn/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
```

## Tech Stack

* Python
* Scikit-learn
* Pandas
* Streamlit
* Docker
* AWS EC2




