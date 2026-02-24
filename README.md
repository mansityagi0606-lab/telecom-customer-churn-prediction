Telecom Customer Churn Prediction

Project Overview
Customer churn is a major challenge in the telecommunications industry. This project aims to predict whether a customer will churn using machine learning techniques and derive actionable business insights.

Problem Statement
The objective is to build a predictive model that identifies customers likely to churn so that the company can take proactive retention actions.

Dataset
IBM Telco Customer Churn Dataset
~7,000 customers
Target variable: Churn (Yes/No)
Features include: Customer tenure
                  Contract type
                  Monthly charges
                  Internet services
                  Support services
                  Payment methods

Models Implemented:
Logistic Regression (Baseline Model)
Random Forest Classifier
Tuned Random Forest (GridSearchCV)

Model Performance:
ROC-AUC ≈ 0.75+
Improved performance after hyperparameter tuning
Good balance between precision and recall

Key Insights:
Month-to-month contracts show higher churn rate.
Customers with short tenure are more likely to churn.
Higher monthly charges increase churn probability.
Customers without TechSupport or OnlineSecurity show higher churn risk.
