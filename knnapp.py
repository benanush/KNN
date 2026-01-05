import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")  # change filename if needed

st.set_page_config(page_title="ML Model App", layout="centered")
st.title("Machine Learning Prediction App")

st.write("Enter input values to get prediction")

# ---- INPUT FIELDS (modify according to your model features) ----
feature1 = st.number_input("Age",
                             min_value=18,
                             max_value=100,
                             value=18,
                             step=1)
feature2 = st.number_input("Salary", value=0.0)

# Create input array
input_data = np.array([[feature1, feature2]])

# ---- PREDICTION ----
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
