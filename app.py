import streamlit as st
import joblib
import pandas as pd

st.title("Customer Satisfaction Prediction")

# Load models
rf_model = joblib.load("models/random_forest_model.pkl")
ann_model = joblib.load("models/ann_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ("Random Forest", "ANN", "XGBoost")
)

uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.write(data.head())

    if st.button("Predict"):

        if model_choice == "Random Forest":
            preds = rf_model.predict(data)

        elif model_choice == "ANN":
            preds = ann_model.predict(data)

        else:
            preds = xgb_model.predict(data)

        st.subheader("Predicted CSAT Score")

        for p in preds:
            st.success(f"CSAT Prediction: {p}")