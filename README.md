# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using classification models, model evaluation, and hyperparameter tuning, with an interactive Streamlit application deployed on AWS EC2.

## Project Overview

Customer churn prediction helps businesses identify customers who may leave and develop proactive retention strategies. This project explores customer data, trains machine learning models, and evaluates their ability to identify customers at risk of churn.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Deployment:** Docker, AWS EC2

## Machine Learning Workflow

1. **Exploratory Data Analysis:** Analyzed customer characteristics, churn distribution, and relationships between features and customer exits.
2. **Data Preprocessing:** Removed irrelevant identifier columns, encoded categorical variables, and separated features from the target.
3. **Model Training:** Implemented Logistic Regression and Random Forest classifiers.
4. **Model Evaluation:** Compared precision, recall, F1-score, accuracy, and ROC-AUC.
5. **Hyperparameter Tuning:** Used GridSearchCV with five-fold cross-validation to optimize Random Forest parameters.
6. **Deployment:** Packaged the application using Docker and deployed it on AWS EC2 for interactive predictions.

## Model Performance

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Balanced Logistic Regression | 70% | 0.7663 |
| Random Forest | 86% | 0.8522 |

*Metrics are from the notebook's held-out test set. The tuned Random Forest requires a complete, consistent evaluation before its performance can be reported.*

## Key Findings

- Customer age was the most important feature in the initial Random Forest model.
- Inactive members showed a higher observed churn rate than active members.
- Customers who exited had higher average account balances in the analyzed dataset.
- Class imbalance made recall and precision important alongside accuracy.

## Run Locally

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Docker Deployment

Build and run the application using Docker:

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

## Project Structure

```text
customer-churn/
├── app.py
├── Dockerfile
├── requirements.txt
├── Customer_Churn_Project.ipynb
└── artifacts/
    ├── model.pkl
    └── preprocessor.pkl
```

## Future Improvements

- Improve churn recall through class balancing and threshold optimization.
- Build a consistent preprocessing and prediction pipeline.
- Add explainability using SHAP or permutation importance.
- Monitor model performance and prediction quality after deployment.

---

