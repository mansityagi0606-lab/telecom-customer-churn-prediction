import streamlit as st
import pickle
import pandas as pd

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

st.title("Telecom Customer Churn Prediction")

st.write("Enter customer details below:")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary
    }])

    transformed_data = preprocessor.transform(input_data)

    prediction = model.predict(transformed_data)
    probability = model.predict_proba(transformed_data)[:, 1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"Customer is likely to Churn ")
    else:
        st.success(f"Customer is likely to Stay ")

    st.write(f"Churn Probability: {probability[0]:.2f}")