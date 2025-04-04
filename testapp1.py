import streamlit as st
import pandas as pd
import numpy as np
import gdown
import joblib
import os
from PIL import Image

# Define model file path
model_path = "order_delivery_model.pkl"

# Check if the model exists, else download
if not os.path.exists(model_path):
    file_id = "1cEhD4f1e9tryBVqlEAs0ZIwa_gcaAYIn"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
rf, xgb = joblib.load(model_path)

# Define feature names
features = [
    "purchase_dow", "purchase_month", "year", "product_size_cm3", "product_weight_g", 
    "geolocation_state_customer", "geolocation_state_seller", "distance"
]

# Define the ensemble prediction function
def ensemble_predict(X):
    try:
        X_df = pd.DataFrame(X, columns=features).astype(np.float64)  # Ensure correct feature names and data types
        rf_pred = rf.predict(X_df)
        xgb_pred = xgb.predict(X_df)
        return (rf_pred + xgb_pred) / 2
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return [None]

# Streamlit UI
st.title("📦 Order Delivery Time Prediction")

# Load and display image from assets folder
image_path = "assets/supply_chain_optimisation.jpg"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, caption="Supply Chain Optimization", use_container_width=True)

st.sidebar.header("🔢 Input Parameters")

# Input fields with reasonable ranges
purchase_dow = float(st.sidebar.number_input("Purchased Day of the Week", 0, 6, 3))
purchase_month = float(st.sidebar.number_input("Purchased Month", 1, 12, 1))
year = float(st.sidebar.number_input("Purchased Year", 2018, 2025, 2018))
product_size_cm3 = float(st.sidebar.number_input("Product Size (cm³)", 100, 50000, 9328))
product_weight_g = float(st.sidebar.number_input("Product Weight (g)", 100, 50000, 1800))
geolocation_state_customer = float(st.sidebar.number_input("Customer State", 1, 50, 10))
geolocation_state_seller = float(st.sidebar.number_input("Seller State", 1, 50, 20))
distance = float(st.sidebar.number_input("Distance (km)", 0.0, 5000.0, 475.35))

# Prediction function
def predict_wait_time():
    input_data = [[
        purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
        geolocation_state_customer, geolocation_state_seller, distance
    ]]
    prediction = ensemble_predict(input_data)
    return round(prediction[0]) if prediction[0] is not None else "Error"

# Button to trigger prediction
if st.sidebar.button("🚀 Predict Wait Time"):
    with st.spinner("Predicting..."):
        result = predict_wait_time()
    if result != "Error":
        st.success(f"### ⏳ Predicted Delivery Time: **{result} days**")
