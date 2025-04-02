import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown  # For downloading from Google Drive

# Google Drive file ID for the model
file_id = "1cEhD4f1e9tryBVqlEAs0ZIwa_gcaAYIn"  # Replace with your actual Google Drive file ID
model_path = "order_delivery_model.pkl"

# Function to download the model from Google Drive
@st.cache_resource
def load_models():
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
    return joblib.load(model_path)

rf, xgb = load_models()

# Define the ensemble prediction function
def ensemble_predict(X):
    rf_pred = rf.predict(X)
    xgb_pred = xgb.predict(X)
    return (rf_pred + xgb_pred) / 2

# Streamlit UI
st.title("📦 Order Delivery Time Prediction")
st.sidebar.header("🔢 Input Parameters")

# Input fields with reasonable ranges
purchase_dow = st.sidebar.number_input("Purchased Day of the Week", 0, 6, 3)
purchase_month = st.sidebar.number_input("Purchased Month", 1, 12, 1)
year = st.sidebar.number_input("Purchased Year", 2018, 2025, 2018)
product_size_cm3 = st.sidebar.number_input("Product Size (cm³)", 100, 50000, 9328)
product_weight_g = st.sidebar.number_input("Product Weight (g)", 100, 50000, 1800)
geolocation_state_customer = st.sidebar.number_input("Customer State", 1, 50, 10)
geolocation_state_seller = st.sidebar.number_input("Seller State", 1, 50, 20)
distance = st.sidebar.number_input("Distance (km)", 0.0, 5000.0, 475.35)

# Prediction function
def predict_wait_time():
    input_data = np.array([[  # Convert input into a 2D array
        purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
        geolocation_state_customer, geolocation_state_seller, distance
    ]])
    prediction = ensemble_predict(input_data)
    return round(prediction[0])

# Button to trigger prediction
if st.sidebar.button("🚀 Predict Wait Time"):
    with st.spinner("Predicting..."):
        result = predict_wait_time()
    st.success(f"### ⏳ Predicted Delivery Time: **{result} days**")
